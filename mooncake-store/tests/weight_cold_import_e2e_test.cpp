#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <cerrno>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "real_client.h"
#include "test_server_helpers.h"
#include "weight_residency_planner.h"

namespace {
thread_local bool fail_cold_source_allocation = false;
}

extern "C" int __real_posix_memalign(void**, size_t, size_t);
extern "C" int __wrap_posix_memalign(void** pointer, size_t alignment,
                                     size_t size) {
    if (fail_cold_source_allocation && size == 4 * 1024 * 1024) {
        fail_cold_source_allocation = false;
        return ENOMEM;
    }
    return __real_posix_memalign(pointer, alignment, size);
}

namespace mooncake::testing {
namespace {

class WeightColdImportE2ETest : public ::testing::Test {
   protected:
    static constexpr size_t kMiB = 1024 * 1024;

    void Start(size_t segment_bytes) {
        auto pattern =
            (std::filesystem::current_path() / "weight-cold-e2e-XXXXXX")
                .string();
        ASSERT_NE(nullptr, ::mkdtemp(pattern.data()));
        directory_ = pattern;
        ASSERT_TRUE(server_.Start(InProcMasterConfigBuilder()
                                      .set_enable_offload(true)
                                      .set_default_kv_lease_ttl(100)
                                      .build()));
        auto ports = getFreeTcpPorts(2);
        ASSERT_EQ(ports.size(), 2u);
        owner_ = std::make_unique<RealClient>();
        ASSERT_EQ(0, owner_->setup_real("127.0.0.1:" + std::to_string(ports[0]),
                                        server_.metadata_url(), segment_bytes,
                                        64 * kMiB, "tcp", "",
                                        server_.master_address(), nullptr, "",
                                        true, directory_.string()));
        source_ = std::make_unique<RealClient>();
        ASSERT_EQ(
            0, source_->setup_real("127.0.0.1:" + std::to_string(ports[1]),
                                   server_.metadata_url(), 0, kMiB, "tcp", "",
                                   server_.master_address()));
    }

    void TearDown() override {
        for (auto* buffer : registered_) {
            EXPECT_EQ(0, source_->unregister_buffer(buffer));
        }
        source_.reset();
        owner_.reset();
        server_.Stop();
        if (!directory_.empty()) std::filesystem::remove_all(directory_);
    }

    void Import(WeightResidencyState target, bool batch, bool local,
                const std::vector<size_t>& sizes, size_t replicas = 1,
                bool paired_affinities = false, bool concurrent = false,
                bool session = false, bool revoke = false,
                bool fail_source_allocation = false) {
        ASSERT_TRUE(owner_);
        const std::string hot_control(64 * 1024, 'h');
        ASSERT_EQ(0, source_->put("hot-control", hot_control));
        WeightRevisionIdentity identity{
            .tenant_id = "default",
            .name_space = "cold-import-e2e",
            .resource_id = "weights",
            .revision =
                ::testing::UnitTest::GetInstance()->current_test_info()->name(),
            .weight_generation = 1};
        std::vector<WeightAffinityMappingEntry> layout;
        std::vector<WeightAffinityUnit> units;
        std::vector<std::string> keys;
        std::vector<void*> addresses;
        std::vector<std::string> affinity_ids;
        uint64_t total = 0;
        payloads_.resize(sizes.size());
        for (size_t index = 0; index < sizes.size(); ++index) {
            const auto key = "payload-" + std::to_string(index);
            const auto affinity =
                "tensor-" +
                std::to_string(paired_affinities ? index / 2 : index);
            keys.push_back(key);
            affinity_ids.push_back(affinity);
            layout.push_back({key, sizes[index], affinity});
            if (!units.empty() && units.back().affinity_id == affinity) {
                units.back().logical_bytes += sizes[index];
            } else {
                units.push_back({affinity, sizes[index]});
            }
            total += sizes[index];
            payloads_[index].assign(sizes[index],
                                    static_cast<char>('a' + index));
            if (session) {
                for (size_t offset = 0; offset < sizes[index]; ++offset)
                    payloads_[index][offset] =
                        static_cast<char>((offset * 17 + index) % 251);
            }
            addresses.push_back(payloads_[index].data());
            ASSERT_EQ(0,
                      source_->register_buffer(addresses.back(), sizes[index]));
            registered_.push_back(addresses.back());
        }
        auto begin = source_->begin_weight_import(
            {.identity = identity,
             .payload_group_id = {},
             .expected_payload_count = sizes.size(),
             .expected_logical_bytes = total,
             .policy = WeightStoragePolicy{.preferred_residency = target},
             .affinity_summary = {.affinity_count = units.size(),
                                  .affinity_digest =
                                      ComputeWeightAffinityMappingSha256(
                                          layout)},
             .payload_layout = layout});
        ASSERT_TRUE(begin && begin->has_value());
        const auto group = (*begin)->manifest.payload_group_id;
        ReplicateConfig config;
        config.replica_num = replicas;
        config.data_type = ObjectDataType::WEIGHT;
        auto* writer = local ? owner_.get() : source_.get();
        if (fail_source_allocation) {
            config.group_ids = std::vector<std::string>{group};
            config.residency_affinity_ids =
                std::vector<std::string>{affinity_ids[0]};
            fail_cold_source_allocation = true;
            EXPECT_LT(writer->put_from(keys[0], addresses[0], sizes[0], config),
                      0);
            EXPECT_FALSE(fail_cold_source_allocation);
            EXPECT_LT(source_->get_into(keys[0], addresses[0], sizes[0]), 0);
            config.residency_affinity_ids =
                std::vector<std::string>{affinity_ids[1]};
            EXPECT_EQ(writer->put_from(keys[1], addresses[1], sizes[1], config),
                      0);
            return;
        }
        if (session) {
            config.group_ids = std::vector<std::string>(keys.size(), group);
            config.residency_affinity_ids = affinity_ids;
            auto starts = writer->batch_put_session_start(keys, sizes, config);
            ASSERT_EQ(starts.size(), keys.size());
            for (auto status : starts) ASSERT_EQ(status, 0);
            for (size_t i = 0; i < keys.size(); ++i) {
                const size_t half = sizes[i] / 2;
                auto tail = writer->batch_put_from_multi_buffer_ranges(
                    {keys[i]}, {{static_cast<char*>(addresses[i]) + half}},
                    {{sizes[i] - half}}, {{half}});
                ASSERT_EQ(tail.size(), 1u);
                ASSERT_EQ(tail[0], sizes[i] - half);
                if (target == WeightResidencyState::COLD) {
                    auto overlap = writer->batch_put_from_multi_buffer_ranges(
                        {keys[i]}, {{static_cast<char*>(addresses[i]) + half}},
                        {{sizes[i] - half}}, {{half}});
                    ASSERT_EQ(overlap.size(), 1u);
                    ASSERT_EQ(overlap[0], sizes[i] - half);
                }
                auto invalid = writer->batch_put_from_multi_buffer_ranges(
                    {keys[i]}, {{addresses[i]}}, {{2}}, {{sizes[i] - 1}});
                ASSERT_EQ(invalid.size(), 1u);
                EXPECT_LT(invalid[0], 0);
            }
            if (revoke) {
                auto revoked = writer->batch_put_session_revoke(keys);
                ASSERT_EQ(revoked.size(), keys.size());
                for (auto status : revoked) ASSERT_EQ(status, 0);
                auto late = writer->batch_put_from_multi_buffer_ranges(
                    {keys.front()}, {{addresses.front()}}, {{1}}, {{0}});
                ASSERT_EQ(late.size(), 1u);
                EXPECT_LT(late[0], 0);
                return;
            }
            if (target == WeightResidencyState::COLD) {
                auto holes = writer->batch_put_session_end(keys);
                for (auto status : holes)
                    EXPECT_EQ(status, toInt(ErrorCode::INVALID_WRITE));
            }
            for (size_t i = 0; i < keys.size(); ++i) {
                const size_t half = sizes[i] / 2;
                const size_t quarter = half / 2;
                auto head = writer->batch_put_from_multi_buffer_ranges(
                    {keys[i]},
                    {{static_cast<char*>(addresses[i]) + quarter,
                      addresses[i]}},
                    {{half - quarter, quarter}}, {{quarter, 0}});
                ASSERT_EQ(head.size(), 1u);
                ASSERT_EQ(head[0], half);
            }
            auto ends = writer->batch_put_session_end(keys);
            ASSERT_EQ(ends.size(), keys.size());
            for (auto status : ends) ASSERT_EQ(status, 0);
        } else if (batch) {
            config.group_ids = std::vector<std::string>(keys.size(), group);
            config.residency_affinity_ids = affinity_ids;
            auto result =
                writer->batch_put_from(keys, addresses, sizes, config);
            ASSERT_EQ(result.size(), keys.size());
            if (replicas != 1) {
                for (const auto status : result)
                    EXPECT_EQ(status,
                              static_cast<int>(ErrorCode::NOT_SUPPORTED));
                return;
            }
            for (const auto status : result) ASSERT_EQ(status, 0);
        } else {
            std::vector<std::future<int>> writes;
            for (size_t index = 0; index < keys.size(); ++index) {
                config.group_ids = std::vector<std::string>{group};
                config.residency_affinity_ids =
                    std::vector<std::string>{affinity_ids[index]};
                if (concurrent) {
                    writes.push_back(std::async(std::launch::async, [&, index,
                                                                     config] {
                        return writer->put_from(keys[index], addresses[index],
                                                sizes[index], config);
                    }));
                } else {
                    ASSERT_EQ(0, writer->put_from(keys[index], addresses[index],
                                                  sizes[index], config));
                }
            }
            for (auto& write : writes) ASSERT_EQ(0, write.get());
        }
        config.group_ids = std::vector<std::string>{group};
        config.residency_affinity_ids.reset();
        config.data_type = ObjectDataType::METADATA;
        ASSERT_EQ(0,
                  source_->put(MakeWeightManifestKey(identity), "{}", config));
        auto committed = source_->commit_weight_import(
            {.identity = identity,
             .expected_metadata_generation = (*begin)->metadata_generation,
             .manifest = {
                 .manifest_key = MakeWeightManifestKey(identity),
                 .manifest_sha256 = "44136fa355b3678a1146ad16f7e8649e94fb4fc21f"
                                    "e77e8310c060f61caaff8a",
                 .payload_group_id = group,
                 .payload_keys_sha256 = ComputeWeightPayloadKeysSha256(keys),
                 .payload_count = keys.size(),
                 .logical_bytes = total}});
        ASSERT_TRUE(committed && committed->has_value());
        EXPECT_EQ((*committed)->availability, WeightAvailabilityState::READY);
        EXPECT_EQ((*committed)->residency, target);
        EXPECT_FALSE((*committed)->operation_id.has_value());
        EXPECT_TRUE((*committed)->import_layout.empty());

        std::vector<std::string> hot_affinities;
        if (target == WeightResidencyState::MIXED) {
            auto plan = PlanMixedWeightResidency(units, 0.5);
            ASSERT_TRUE(plan);
            hot_affinities = plan->hot_affinity_ids;
            EXPECT_DOUBLE_EQ(
                (*committed)->observed_hot_ratio,
                static_cast<double>(plan->hot_logical_bytes) / total);
        }
        auto placements = source_->batch_query(keys);
        ASSERT_EQ(placements.size(), keys.size());
        for (size_t index = 0; index < keys.size(); ++index) {
            ASSERT_TRUE(placements[index]);
            const bool hot =
                std::find(hot_affinities.begin(), hot_affinities.end(),
                          affinity_ids[index]) != hot_affinities.end();
            EXPECT_TRUE(std::any_of(
                placements[index]->replicas.begin(),
                placements[index]->replicas.end(), [hot](const auto& replica) {
                    return hot ? replica.is_memory_replica()
                               : replica.is_local_disk_replica();
                }));
            if (!hot) {
                EXPECT_TRUE(std::none_of(placements[index]->replicas.begin(),
                                         placements[index]->replicas.end(),
                                         [](const auto& replica) {
                                             return replica.is_memory_replica();
                                         }));
            }
            std::string actual(sizes[index], '\0');
            ASSERT_EQ(0,
                      source_->register_buffer(actual.data(), actual.size()));
            EXPECT_EQ(
                static_cast<int64_t>(actual.size()),
                source_->get_into(keys[index], actual.data(), actual.size()));
            EXPECT_EQ(actual, payloads_[index]);
            if (session) {
                auto starts = source_->batch_get_session_start({keys[index]});
                ASSERT_EQ(starts.size(), 1u);
                ASSERT_EQ(starts[0], 0);
                std::fill(actual.begin(), actual.end(), '\0');
                const size_t half = actual.size() / 2;
                auto ranges = source_->batch_get_into_multi_buffer_ranges(
                    {keys[index]}, {{actual.data() + half, actual.data()}},
                    {{actual.size() - half, half}}, {{half, 0}});
                ASSERT_EQ(ranges.size(), 1u);
                EXPECT_EQ(ranges[0], actual.size());
                EXPECT_EQ(actual, payloads_[index]);
                EXPECT_EQ(source_->batch_get_session_end({keys[index]}), 0);
            }
            EXPECT_EQ(0, source_->unregister_buffer(actual.data()));
        }
        auto manifest = source_->batch_query({MakeWeightManifestKey(identity)});
        ASSERT_EQ(manifest.size(), 1u);
        ASSERT_TRUE(manifest.front());
        ASSERT_EQ(manifest.front()->replicas.size(), 1u);
        EXPECT_TRUE(manifest.front()->replicas.front().is_memory_replica());
        std::string control_readback(hot_control.size(), '\0');
        ASSERT_EQ(0, source_->register_buffer(control_readback.data(),
                                              control_readback.size()));
        EXPECT_EQ(static_cast<int64_t>(control_readback.size()),
                  source_->get_into("hot-control", control_readback.data(),
                                    control_readback.size()));
        EXPECT_EQ(control_readback, hot_control);
        EXPECT_EQ(0, source_->unregister_buffer(control_readback.data()));
    }

    InProcMaster server_;
    std::unique_ptr<RealClient> owner_;
    std::unique_ptr<RealClient> source_;
    std::filesystem::path directory_;
    std::vector<std::string> payloads_;
    std::vector<void*> registered_;
};

TEST_F(WeightColdImportE2ETest, SourceAllocationFailureCancelsOwnerStream) {
    Start(16 * kMiB);
    Import(WeightResidencyState::COLD, false, false, {4 * kMiB, 4 * kMiB}, 1,
           false, false, false, false, true);
}

TEST_F(WeightColdImportE2ETest,
       RemoteRangedSessionSupportsMultipleColdObjects) {
    Start(16 * kMiB);
    Import(WeightResidencyState::COLD, false, false, {12 * kMiB, 12 * kMiB}, 1,
           false, false, true);
}

TEST_F(WeightColdImportE2ETest, RangedSessionRevokeFencesFurtherWrites) {
    Start(16 * kMiB);
    Import(WeightResidencyState::COLD, false, false, {12 * kMiB}, 1, false,
           false, true, true);
}

TEST_F(WeightColdImportE2ETest, LocalRangedSessionSupportsColdObjects) {
    Start(16 * kMiB);
    Import(WeightResidencyState::COLD, false, true, {12 * kMiB}, 1, false,
           false, true);
}

TEST_F(WeightColdImportE2ETest, RangedSessionSupportsMixedPlacement) {
    Start(16 * kMiB);
    Import(WeightResidencyState::MIXED, false, false, {8 * kMiB, 24 * kMiB}, 1,
           false, false, true);
}

TEST_F(WeightColdImportE2ETest, RemoteScalarBypassesHotSegmentCapacity) {
    Start(16 * kMiB);
    Import(WeightResidencyState::COLD, false, false, {32 * kMiB});
}

TEST_F(WeightColdImportE2ETest, LocalScalarBypassesHotSegmentCapacity) {
    Start(16 * kMiB);
    Import(WeightResidencyState::COLD, false, true, {32 * kMiB});
}

TEST_F(WeightColdImportE2ETest, RemoteBatchBypassesHotSegmentCapacity) {
    Start(16 * kMiB);
    Import(WeightResidencyState::COLD, true, false, {32 * kMiB, 32 * kMiB});
}

TEST_F(WeightColdImportE2ETest, MixedBatchOnlyKeepsPlannedAffinitiesHot) {
    Start(16 * kMiB);
    Import(WeightResidencyState::MIXED, true, false, {8 * kMiB, 24 * kMiB});
}

TEST_F(WeightColdImportE2ETest, MixedRemoteScalarBypassesHotSegmentCapacity) {
    Start(16 * kMiB);
    Import(WeightResidencyState::MIXED, false, false, {8 * kMiB, 24 * kMiB});
}

TEST_F(WeightColdImportE2ETest, MixedLocalScalarBypassesHotSegmentCapacity) {
    Start(16 * kMiB);
    Import(WeightResidencyState::MIXED, false, true, {8 * kMiB, 24 * kMiB});
}

TEST_F(WeightColdImportE2ETest, MixedLocalBatchBypassesHotSegmentCapacity) {
    Start(16 * kMiB);
    Import(WeightResidencyState::MIXED, true, true, {8 * kMiB, 24 * kMiB});
}

TEST_F(WeightColdImportE2ETest, MixedConcurrentHotAndColdWrites) {
    Start(20 * kMiB);
    Import(WeightResidencyState::MIXED, false, false, {8 * kMiB, 24 * kMiB}, 1,
           false, true);
}

TEST_F(WeightColdImportE2ETest, MixedLocalBatchKeepsWholeAffinitiesTogether) {
    Start(20 * kMiB);
    Import(WeightResidencyState::MIXED, true, true,
           {8 * kMiB, 8 * kMiB, 12 * kMiB, 12 * kMiB}, 1, true);
}

TEST_F(WeightColdImportE2ETest,
       ColdImportDoesNotSilentlyDropReplicaRequirements) {
    Start(16 * kMiB);
    Import(WeightResidencyState::COLD, true, false, {32 * kMiB}, 2);
}

}  // namespace
}  // namespace mooncake::testing
