#include "master_service_test_fixture.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "weight_management.h"

namespace mooncake::test {
namespace {

class WeightGroupLifecycleTest : public MasterServiceTest {
   protected:
    static WeightRevisionIdentity Identity() {
        return WeightRevisionIdentity{
            .tenant_id = "default",
            .name_space = "production",
            .resource_id = "llama-70b",
            .revision = "step-100",
            .weight_generation = 7,
        };
    }

    static std::string ManifestKey() {
        return "weights/production/llama-70b/step-100/7/manifest";
    }

    struct PayloadSpec {
        std::string key;
        uint64_t size;
        std::string affinity_id;
    };

    WeightRevisionMetadata PublishReadyWithPayloads(
        MasterService& service, const UUID& client_id,
        const std::vector<PayloadSpec>& payloads) {
        uint64_t logical_bytes = 0;
        std::vector<std::string> payload_keys;
        std::set<std::string> affinity_ids;
        for (const auto& payload : payloads) {
            logical_bytes += payload.size;
            payload_keys.push_back(payload.key);
            affinity_ids.insert(payload.affinity_id);
        }
        auto importing = service.BeginWeightImport(BeginWeightImportRequest{
            .identity = Identity(),
            .payload_group_id = {},
            .expected_payload_count = payloads.size(),
            .expected_logical_bytes = logical_bytes,
            .policy =
                WeightStoragePolicy{
                    .preferred_residency = WeightResidencyState::HOT,
                    .migration_mode = WeightMigrationMode::MANUAL,
                },
            .affinity_summary =
                WeightAffinitySummary{
                    .affinity_count = affinity_ids.size(),
                    .affinity_digest = std::string(64, 'c'),
                },
        });
        EXPECT_TRUE(importing.has_value());
        for (const auto& payload : payloads) {
            ReplicateConfig config;
            config.replica_num = 1;
            config.with_hard_pin = true;
            config.group_ids =
                std::vector<std::string>{importing->manifest.payload_group_id};
            config.residency_affinity_ids =
                std::vector<std::string>{payload.affinity_id};
            config.data_type = ObjectDataType::WEIGHT;
            PutCompletedObject(service, client_id, payload.key, config,
                               payload.size);
        }
        ReplicateConfig manifest_config;
        manifest_config.replica_num = 1;
        manifest_config.with_hard_pin = true;
        manifest_config.group_ids =
            std::vector<std::string>{importing->manifest.payload_group_id};
        manifest_config.data_type = ObjectDataType::METADATA;
        PutCompletedObject(service, client_id, ManifestKey(), manifest_config,
                           128);
        auto ready = service.CommitWeightImport(CommitWeightImportRequest{
            .identity = Identity(),
            .expected_metadata_generation = importing->metadata_generation,
            .manifest =
                WeightManifestReference{
                    .manifest_key = ManifestKey(),
                    .manifest_sha256 = std::string(64, 'a'),
                    .payload_group_id = importing->manifest.payload_group_id,
                    .payload_keys_sha256 =
                        ComputeWeightPayloadKeysSha256(payload_keys),
                    .payload_count = payloads.size(),
                    .logical_bytes = logical_bytes,
                },
        });
        EXPECT_TRUE(ready.has_value());
        return *ready;
    }

    WeightRevisionMetadata PublishReady(MasterService& service,
                                        const UUID& client_id) {
        return PublishReadyWithPayloads(
            service, client_id,
            {{.key = "payload-a", .size = 1024, .affinity_id = "affinity-a"}});
    }

    static void AddLocalDiskReplica(MasterService& service,
                                    const UUID& client_id,
                                    const std::string& key, int64_t size,
                                    const std::string& endpoint) {
        std::vector<OffloadTaskItem> tasks{
            OffloadTaskItem{.tenant_id = "default", .key = key, .size = size}};
        StorageObjectMetadata metadata;
        metadata.key_size = key.size();
        metadata.data_size = size;
        metadata.transport_endpoint = endpoint;
        ASSERT_TRUE(service.NotifyOffloadSuccess(client_id, tasks, {metadata}));
    }
};

TEST_F(WeightGroupLifecycleTest, LeaseBlocksOperationAndDelete) {
    MasterService service;
    [[maybe_unused]] const auto context = PrepareSimpleSegment(service);
    const UUID client_id = generate_uuid();
    auto ready = PublishReady(service, client_id);
    auto lease =
        service.AcquireWeightRevisionLease(AcquireWeightRevisionLeaseRequest{
            .identity = ready.identity,
            .expected_metadata_generation = ready.metadata_generation,
            .holder = "worker-0",
            .ttl_ms = 60'000,
        });
    ASSERT_TRUE(lease.has_value());

    auto operation = service.StartWeightResidencyOperation(
        StartWeightResidencyOperationRequest{
            .identity = ready.identity,
            .expected_metadata_generation = ready.metadata_generation,
            .target_residency = WeightResidencyState::COLD,
            .mixed_hot_ratio = std::nullopt,
        });
    ASSERT_TRUE(operation.has_value());
    auto migration_lease =
        service.AcquireWeightRevisionLease(AcquireWeightRevisionLeaseRequest{
            .identity = ready.identity,
            .expected_metadata_generation =
                operation->fenced_metadata_generation,
            .holder = "worker-1",
            .ttl_ms = 60'000,
        });
    ASSERT_TRUE(migration_lease.has_value());
    auto deleted = service.DeleteWeightRevision(DeleteWeightRevisionRequest{
        .identity = ready.identity,
        .expected_metadata_generation = ready.metadata_generation + 1,
    });
    ASSERT_FALSE(deleted.has_value());
    EXPECT_EQ(WeightManagementError::BUSY, deleted.error());
}

TEST_F(WeightGroupLifecycleTest, SnapshotExposesOpaqueResidencyAffinityId) {
    MasterService service;
    [[maybe_unused]] const auto context = PrepareSimpleSegment(service);
    const UUID client_id = generate_uuid();
    auto importing = service.BeginWeightImport(BeginWeightImportRequest{
        .identity = Identity(),
        .payload_group_id = {},
        .expected_payload_count = 1,
        .expected_logical_bytes = 1024,
        .policy = std::nullopt,
        .affinity_summary =
            WeightAffinitySummary{
                .affinity_count = 1,
                .affinity_digest = std::string(64, 'c'),
            },
    });
    ASSERT_TRUE(importing.has_value());

    ReplicateConfig config;
    config.replica_num = 1;
    config.group_ids =
        std::vector<std::string>{importing->manifest.payload_group_id};
    config.residency_affinity_ids =
        std::vector<std::string>{"opaque-affinity-id"};
    config.data_type = ObjectDataType::WEIGHT;
    PutCompletedObject(service, client_id, "payload-a", config, 1024);

    auto affinity_ids = GetWeightGroupAffinityIdsForTest(
        service, Identity(), importing->manifest.payload_group_id);
    ASSERT_EQ(1u, affinity_ids.size());
    EXPECT_EQ("opaque-affinity-id", affinity_ids.front());
}

TEST_F(WeightGroupLifecycleTest, OperationRemainsPendingUntilTargetObserved) {
    MasterService service;
    [[maybe_unused]] const auto context = PrepareSimpleSegment(service);
    const UUID client_id = generate_uuid();
    auto ready = PublishReady(service, client_id);

    auto started = service.StartWeightResidencyOperation(
        StartWeightResidencyOperationRequest{
            .identity = ready.identity,
            .expected_metadata_generation = ready.metadata_generation,
            .target_residency = WeightResidencyState::COLD,
            .mixed_hot_ratio = std::nullopt,
        });
    ASSERT_TRUE(started.has_value());
    EXPECT_EQ(WeightOperationKind::MIGRATING, started->kind);
    EXPECT_EQ(0, started->processed_units);
    EXPECT_EQ(1, started->total_units);
    EXPECT_EQ(*started,
              *service.QueryWeightOperation(QueryWeightOperationRequest{
                  .operation_id = started->operation_id,
              }));

    auto reconciled = service.ReconcileWeightRevision(
        ReconcileWeightRevisionRequest{.identity = ready.identity});
    ASSERT_TRUE(reconciled.has_value());
    EXPECT_TRUE(reconciled->operation_id.has_value());
    EXPECT_EQ(WeightResidencyState::HOT, reconciled->residency);
    auto progress = service.QueryWeightOperation(
        QueryWeightOperationRequest{.operation_id = started->operation_id});
    ASSERT_TRUE(progress.has_value());
    EXPECT_EQ(0, progress->processed_units);
    EXPECT_EQ(1, progress->total_units);
}

TEST_F(WeightGroupLifecycleTest, ColdOperationEvictsWholeManagedGroup) {
    MasterServiceConfig config;
    config.default_kv_lease_ttl = 0;
    config.enable_offload = true;
    config.offload_on_evict = true;
    MasterService service(config);
    const auto context = PrepareSimpleSegment(service);
    const UUID client_id = context.client_id;
    ASSERT_TRUE(service.MountLocalDiskSegment(client_id, true));
    auto ready = PublishReady(service, client_id);
    EXPECT_FALSE(IsObjectProcessingForTest(service, "payload-a"));
    AddLocalDiskReplica(service, client_id, "payload-a", 1024, "test_segment");
    auto started = service.StartWeightResidencyOperation(
        StartWeightResidencyOperationRequest{
            .identity = ready.identity,
            .expected_metadata_generation = ready.metadata_generation,
            .target_residency = WeightResidencyState::COLD,
            .mixed_hot_ratio = std::nullopt,
        });
    ASSERT_TRUE(started.has_value());

    auto reconciled = service.ReconcileWeightRevision(
        ReconcileWeightRevisionRequest{.identity = ready.identity});
    ASSERT_TRUE(reconciled.has_value());
    EXPECT_EQ(WeightAvailabilityState::READY, reconciled->availability);
    EXPECT_EQ(WeightResidencyState::COLD, reconciled->residency);
    EXPECT_FALSE(reconciled->operation_id.has_value());
    auto completed = service.QueryWeightOperation(
        QueryWeightOperationRequest{.operation_id = started->operation_id});
    ASSERT_TRUE(completed.has_value());
    EXPECT_EQ("completed", completed->message);
    EXPECT_EQ(completed->total_units, completed->processed_units);
    EXPECT_EQ(completed->total_bytes, completed->processed_bytes);

    const auto members = GetWeightGroupResidencyForTest(
        service, ready.identity, ready.manifest.payload_group_id);
    const auto manifest =
        std::find_if(members.begin(), members.end(), [](const auto& member) {
            return member.data_type == ObjectDataType::METADATA;
        });
    ASSERT_NE(members.end(), manifest);
    EXPECT_TRUE(manifest->has_memory);
}

TEST_F(WeightGroupLifecycleTest,
       ColdOperationOffloadsOnlyWeightsAndWaitsForLeaseBeforeDemotion) {
    MasterServiceConfig config;
    config.default_kv_lease_ttl = 0;
    config.enable_offload = true;
    config.offload_on_evict = true;
    MasterService service(config);
    const auto context = PrepareSimpleSegment(service);
    ASSERT_TRUE(service.MountLocalDiskSegment(context.client_id, true));
    auto ready = PublishReady(service, context.client_id);
    auto lease =
        service.AcquireWeightRevisionLease(AcquireWeightRevisionLeaseRequest{
            .identity = ready.identity,
            .expected_metadata_generation = ready.metadata_generation,
            .holder = "reader",
            .ttl_ms = 60'000,
        });
    ASSERT_TRUE(lease.has_value());
    auto started = service.StartWeightResidencyOperation(
        StartWeightResidencyOperationRequest{
            .identity = ready.identity,
            .expected_metadata_generation = ready.metadata_generation,
            .target_residency = WeightResidencyState::COLD,
            .mixed_hot_ratio = std::nullopt,
        });
    ASSERT_TRUE(started.has_value());

    ASSERT_TRUE(service.ReconcileWeightRevision(
        ReconcileWeightRevisionRequest{.identity = ready.identity}));
    auto queue = service.OffloadObjectHeartbeat(context.client_id, true);
    ASSERT_TRUE(queue.has_value());
    ASSERT_EQ(1u, queue->size());
    EXPECT_EQ("payload-a", queue->front().key);

    StorageObjectMetadata disk_metadata;
    disk_metadata.key_size = queue->front().key.size();
    disk_metadata.data_size = queue->front().size;
    disk_metadata.transport_endpoint = "test_segment";
    ASSERT_TRUE(service.NotifyOffloadSuccess(context.client_id, *queue,
                                             {disk_metadata}));
    auto leased = service.ReconcileWeightRevision(
        ReconcileWeightRevisionRequest{.identity = ready.identity});
    ASSERT_TRUE(leased.has_value());
    EXPECT_EQ(WeightResidencyState::HOT, leased->residency);
    auto leased_members = GetWeightGroupResidencyForTest(
        service, ready.identity, ready.manifest.payload_group_id);
    for (const auto& member : leased_members) {
        EXPECT_TRUE(member.has_memory);
    }

    ASSERT_TRUE(service.ReleaseWeightRevisionLease(
        ReleaseWeightRevisionLeaseRequest{.lease_id = lease->lease_id}));
    auto cold = service.ReconcileWeightRevision(
        ReconcileWeightRevisionRequest{.identity = ready.identity});
    ASSERT_TRUE(cold.has_value());
    EXPECT_EQ(WeightResidencyState::COLD, cold->residency);

    const auto members = GetWeightGroupResidencyForTest(
        service, ready.identity, ready.manifest.payload_group_id);
    for (const auto& member : members) {
        if (member.data_type == ObjectDataType::METADATA) {
            EXPECT_TRUE(member.has_memory);
        } else {
            EXPECT_FALSE(member.has_memory);
            EXPECT_TRUE(member.has_cold);
        }
    }
}

TEST_F(WeightGroupLifecycleTest,
       MixedOperationKeepsEveryAffinityWholeAndReportsActualRatio) {
    MasterServiceConfig config;
    config.default_kv_lease_ttl = 0;
    config.enable_offload = true;
    config.offload_on_evict = true;
    MasterService service(config);
    const auto context = PrepareSimpleSegment(service);
    ASSERT_TRUE(service.MountLocalDiskSegment(context.client_id, true));
    auto ready = PublishReadyWithPayloads(
        service, context.client_id,
        {
            {.key = "payload-a0", .size = 300, .affinity_id = "a"},
            {.key = "payload-a1", .size = 200, .affinity_id = "a"},
            {.key = "payload-b", .size = 400, .affinity_id = "b"},
            {.key = "payload-c", .size = 100, .affinity_id = "c"},
        });
    for (const auto& [key, size] : std::vector<std::pair<std::string, int64_t>>{
             {"payload-a0", 300},
             {"payload-a1", 200},
             {"payload-b", 400},
             {"payload-c", 100},
         }) {
        AddLocalDiskReplica(service, context.client_id, key, size,
                            "test_segment");
    }
    auto started = service.StartWeightResidencyOperation(
        StartWeightResidencyOperationRequest{
            .identity = ready.identity,
            .expected_metadata_generation = ready.metadata_generation,
            .target_residency = WeightResidencyState::MIXED,
            .mixed_hot_ratio = 0.5,
        });
    ASSERT_TRUE(started.has_value());

    auto mixed = service.ReconcileWeightRevision(
        ReconcileWeightRevisionRequest{.identity = ready.identity});

    ASSERT_TRUE(mixed.has_value());
    EXPECT_EQ(WeightResidencyState::MIXED, mixed->residency);
    EXPECT_DOUBLE_EQ(0.5, mixed->observed_hot_ratio);
    EXPECT_FALSE(mixed->operation_id.has_value());
    auto completed = service.QueryWeightOperation(
        QueryWeightOperationRequest{.operation_id = started->operation_id});
    ASSERT_TRUE(completed.has_value());
    EXPECT_EQ(3, completed->processed_units);
    EXPECT_EQ(1000, completed->processed_bytes);

    std::map<std::string, std::optional<bool>> affinity_memory;
    const auto members = GetWeightGroupResidencyForTest(
        service, ready.identity, ready.manifest.payload_group_id);
    for (const auto& member : members) {
        if (member.data_type == ObjectDataType::METADATA) {
            EXPECT_TRUE(member.has_memory);
            continue;
        }
        auto& expected = affinity_memory[member.residency_affinity_id];
        if (!expected.has_value()) {
            expected = member.has_memory;
        }
        EXPECT_EQ(*expected, member.has_memory);
    }
}

TEST_F(WeightGroupLifecycleTest, RehydrateQueuesAndCompletesWholeManagedGroup) {
    MasterServiceConfig config;
    config.default_kv_lease_ttl = 0;
    config.enable_offload = true;
    config.offload_on_evict = true;
    MasterService service(config);
    const auto context = PrepareSimpleSegment(service);
    const UUID client_id = context.client_id;
    ASSERT_TRUE(service.MountLocalDiskSegment(client_id, true));
    auto ready = PublishReady(service, client_id);
    EXPECT_FALSE(IsObjectProcessingForTest(service, "payload-a"));
    ASSERT_TRUE(service.StartWeightResidencyOperation(
        StartWeightResidencyOperationRequest{
            .identity = ready.identity,
            .expected_metadata_generation = ready.metadata_generation,
            .target_residency = WeightResidencyState::COLD,
            .mixed_hot_ratio = std::nullopt,
        }));
    ASSERT_TRUE(service.ReconcileWeightRevision(
        ReconcileWeightRevisionRequest{.identity = ready.identity}));
    auto offload = service.OffloadObjectHeartbeat(client_id, true);
    ASSERT_TRUE(offload.has_value());
    ASSERT_EQ(1u, offload->size());
    StorageObjectMetadata disk_metadata;
    disk_metadata.key_size = offload->front().key.size();
    disk_metadata.data_size = offload->front().size;
    disk_metadata.transport_endpoint = "test_segment";
    ASSERT_TRUE(
        service.NotifyOffloadSuccess(client_id, *offload, {disk_metadata}));
    auto cold = service.ReconcileWeightRevision(
        ReconcileWeightRevisionRequest{.identity = ready.identity});
    ASSERT_TRUE(cold.has_value());
    ASSERT_EQ(WeightResidencyState::COLD, cold->residency);
    EXPECT_FALSE(IsObjectProcessingForTest(service, "payload-a"));

    auto started = service.StartWeightResidencyOperation(
        StartWeightResidencyOperationRequest{
            .identity = cold->identity,
            .expected_metadata_generation = cold->metadata_generation,
            .target_residency = WeightResidencyState::HOT,
            .mixed_hot_ratio = std::nullopt,
        });
    ASSERT_TRUE(started.has_value());
    EXPECT_EQ(WeightOperationKind::MIGRATING, started->kind);
    ASSERT_TRUE(service.ReconcileWeightRevision(
        ReconcileWeightRevisionRequest{.identity = cold->identity}));

    size_t promoted = 0;
    while (promoted < 1) {
        auto pending = service.PromotionObjectHeartbeat(client_id);
        ASSERT_TRUE(pending.has_value());
        ASSERT_FALSE(pending->empty());
        for (const auto& task : *pending) {
            ASSERT_TRUE(service.PromotionAllocStart(
                client_id, task.key, TenantId(task.tenant_id), task.size, {}));
            ASSERT_TRUE(service.NotifyPromotionSuccess(
                client_id, task.key, TenantId(task.tenant_id)));
            ++promoted;
        }
    }

    auto hot = service.ReconcileWeightRevision(
        ReconcileWeightRevisionRequest{.identity = cold->identity});
    ASSERT_TRUE(hot.has_value());
    EXPECT_EQ(WeightAvailabilityState::READY, hot->availability);
    EXPECT_EQ(WeightResidencyState::HOT, hot->residency);
    EXPECT_FALSE(hot->operation_id.has_value());
    auto completed = service.QueryWeightOperation(
        QueryWeightOperationRequest{.operation_id = started->operation_id});
    ASSERT_TRUE(completed.has_value());
    EXPECT_EQ("completed", completed->message);
}

TEST_F(WeightGroupLifecycleTest,
       DeleteRemovesPayloadAndManifestThenTombstones) {
    MasterService service;
    [[maybe_unused]] const auto context = PrepareSimpleSegment(service);
    const UUID client_id = generate_uuid();
    auto ready = PublishReady(service, client_id);

    auto deleted = service.DeleteWeightRevision(DeleteWeightRevisionRequest{
        .identity = ready.identity,
        .expected_metadata_generation = ready.metadata_generation,
    });
    ASSERT_TRUE(deleted.has_value());
    EXPECT_EQ(WeightAvailabilityState::DELETED, deleted->availability);
    EXPECT_EQ(WeightResidencyState::ABSENT, deleted->residency);
    EXPECT_FALSE(
        service.ExistKey("payload-a", TenantId::Default()).value_or(false));
    EXPECT_FALSE(
        service.ExistKey(ManifestKey(), TenantId::Default()).value_or(false));
}

TEST_F(WeightGroupLifecycleTest,
       DeleteBatchesPayloadsAndKeepsManifestUntilLastBatch) {
    MasterService service;
    [[maybe_unused]] const auto context = PrepareSimpleSegment(service);
    std::vector<PayloadSpec> payloads;
    payloads.reserve(65);
    for (size_t index = 0; index < 65; ++index) {
        payloads.push_back(PayloadSpec{
            .key = "payload-" + std::to_string(index),
            .size = 1,
            .affinity_id = "affinity",
        });
    }
    auto ready = PublishReadyWithPayloads(service, context.client_id, payloads);

    auto deleting = service.DeleteWeightRevision(DeleteWeightRevisionRequest{
        .identity = ready.identity,
        .expected_metadata_generation = ready.metadata_generation,
    });

    ASSERT_TRUE(deleting.has_value());
    EXPECT_EQ(WeightAvailabilityState::DELETING, deleting->availability);
    const auto remaining = GetWeightGroupResidencyForTest(
        service, ready.identity, ready.manifest.payload_group_id);
    ASSERT_EQ(2u, remaining.size());
    EXPECT_TRUE(std::any_of(remaining.begin(), remaining.end(),
                            [](const auto& member) {
                                return member.data_type ==
                                       ObjectDataType::METADATA;
                            }));

    auto deleted = service.DeleteWeightRevision(DeleteWeightRevisionRequest{
        .identity = ready.identity,
        .expected_metadata_generation = deleting->metadata_generation,
    });
    ASSERT_TRUE(deleted.has_value());
    EXPECT_EQ(WeightAvailabilityState::DELETED, deleted->availability);
    EXPECT_EQ(WeightResidencyState::ABSENT, deleted->residency);
}

}  // namespace
}  // namespace mooncake::test
