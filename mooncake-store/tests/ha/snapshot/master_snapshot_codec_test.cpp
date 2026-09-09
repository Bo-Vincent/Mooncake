#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <msgpack.hpp>

#include "ha/snapshot/master_snapshot_codec.h"
#include "master_config.h"
#include "master_service.h"
#include "segment.h"
#include "task_manager.h"
#include "tenant_id.h"
#include "common/zstd_util.h"

namespace mooncake::ha {

class MasterSnapshotCodecTest : public ::testing::Test {
   protected:
    void SetUp() override { master_service_ = MakeMasterService(); }

    void TearDown() override { master_service_.reset(); }

    static std::unique_ptr<MasterService> MakeMasterService() {
        MasterServiceConfig config;
        config.default_kv_lease_ttl = 10000;
        config.eviction_ratio = 0.1;
        return std::make_unique<MasterService>(config);
    }

    // The fixture is befriended by MasterService, so private state access is
    // funneled through this helper (friendship is not inherited by the
    // TEST_F-generated subclasses).
    static MasterSnapshotStateView MakeStateView(MasterService& service) {
        return MasterSnapshotStateView(
            service, service.segment_manager_, service.local_ssd_manager_,
            service.nof_segment_manager_, service.task_manager_);
    }

    static WeightMetadataStore& MetadataStore(MasterService& service) {
        return service.weight_metadata_;
    }

    static WeightRevisionMetadata PublishReady(MasterService& service,
                                               WeightRevisionIdentity identity,
                                               uint64_t now_ms) {
        auto& metadata_store = MetadataStore(service);
        const auto group_id = MakeWeightPayloadGroupId(identity);
        auto begin = metadata_store.PrepareBeginImport(
            BeginWeightImportRequest{
                .identity = identity,
                .payload_group_id = group_id,
                .expected_payload_count = 1,
                .expected_logical_bytes = 1024,
                .policy = WeightStoragePolicy{
                    .preferred_residency = WeightResidencyState::HOT,
                    .migration_mode = WeightMigrationMode::MANUAL,
                },
                .affinity_summary = WeightAffinitySummary{
                    .affinity_count = 1,
                    .affinity_digest = std::string(64, 'c'),
                },
            },
            now_ms);
        EXPECT_TRUE(begin.has_value());
        EXPECT_TRUE(metadata_store.Publish(*begin).has_value());
        auto commit = metadata_store.PrepareCommitImport(
            CommitWeightImportRequest{
                .identity = identity,
                .expected_metadata_generation = 1,
                .manifest =
                    WeightManifestReference{
                        .manifest_key =
                            "weights/" + identity.name_space + "/" +
                            identity.resource_id + "/" + identity.revision +
                            "/" + std::to_string(identity.weight_generation) +
                            "/manifest",
                        .manifest_sha256 = std::string(64, 'a'),
                        .payload_group_id = group_id,
                        .payload_keys_sha256 = std::string(64, 'b'),
                        .payload_count = 1,
                        .logical_bytes = 1024,
                    },
            },
            now_ms + 1);
        EXPECT_TRUE(commit.has_value());
        auto ready = metadata_store.Publish(*commit);
        EXPECT_TRUE(ready.has_value());
        return ready.value();
    }

    static std::vector<uint8_t> RewriteWeightMetadataField(
        const std::vector<uint8_t>& metadata, bool keep_field,
        bool corrupt_field = false) {
        auto root = msgpack::unpack(
            reinterpret_cast<const char*>(metadata.data()), metadata.size());
        const auto& object = root.get();
        EXPECT_EQ(msgpack::type::MAP, object.type);
        msgpack::sbuffer buffer;
        msgpack::packer<msgpack::sbuffer> packer(&buffer);
        packer.pack_map(object.via.map.size - (keep_field ? 0 : 1));
        for (uint32_t i = 0; i < object.via.map.size; ++i) {
            const auto& item = object.via.map.ptr[i];
            const auto key = item.key.as<std::string>();
            if (key == "weight_metadata" && !keep_field) {
                continue;
            }
            packer.pack(item.key);
            if (key == "weight_metadata" && corrupt_field) {
                packer.pack_bin(3);
                packer.pack_bin_body("bad", 3);
            } else {
                packer.pack(item.val);
            }
        }
        return std::vector<uint8_t>(
            reinterpret_cast<const uint8_t*>(buffer.data()),
            reinterpret_cast<const uint8_t*>(buffer.data()) + buffer.size());
    }

    std::unique_ptr<MasterService> master_service_;
};

TEST_F(MasterSnapshotCodecTest, EncodeManifestPreservesSnapshotId) {
    std::vector<uint8_t> bytes = MasterSnapshotCodec::EncodeManifest(
        MasterSnapshotCodec::kSerializerType,
        MasterSnapshotCodec::kSerializerVersion, "snapshot-000042");
    std::string manifest(bytes.begin(), bytes.end());
    EXPECT_EQ(manifest, "messagepack|1.0.0|snapshot-000042");
}

TEST_F(MasterSnapshotCodecTest, EncodeDecodeRoundTrip) {
    MasterSnapshotCodec codec;

    MasterSnapshotStateView state_view = MakeStateView(*master_service_);

    auto encode_result = codec.Encode(state_view);
    ASSERT_TRUE(encode_result.has_value())
        << "Encode failed: " << encode_result.error().message;

    const MasterSnapshotPayloads& payloads = encode_result.value();

    // All three payload buffers must be produced.
    EXPECT_FALSE(payloads.metadata.empty());
    EXPECT_FALSE(payloads.segments.empty());
    EXPECT_FALSE(payloads.task_manager.empty());

    // Decode into a fresh service.
    auto target_service = MakeMasterService();
    auto decode_result = codec.Decode(target_service.get(), payloads);
    ASSERT_TRUE(decode_result.has_value())
        << "Decode failed: " << decode_result.error().message;
}

TEST_F(MasterSnapshotCodecTest,
       WeightMetadataRoundTripPreservesLeasesOperationsAndDerivedIndex) {
    const auto now_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    auto leased_identity = WeightRevisionIdentity{
        .tenant_id = "default",
        .name_space = "production",
        .resource_id = "llama-70b",
        .revision = "step-100",
        .weight_generation = 7,
    };
    auto operating_identity = leased_identity;
    operating_identity.revision = "step-200";
    operating_identity.weight_generation = 8;
    auto leased = PublishReady(*master_service_, leased_identity, now_ms);
    auto operating =
        PublishReady(*master_service_, operating_identity, now_ms + 10);

    auto lease =
        MetadataStore(*master_service_)
            .PrepareAcquireLease(
                AcquireWeightRevisionLeaseRequest{
                    .identity = leased_identity,
                    .expected_metadata_generation = leased.metadata_generation,
                    .holder = "worker-0",
                    .ttl_ms = 60'000,
                },
                now_ms + 20);
    ASSERT_TRUE(lease.has_value());
    ASSERT_TRUE(MetadataStore(*master_service_).Publish(*lease).has_value());

    auto operation = MetadataStore(*master_service_)
                         .PrepareStartOperation(
                             StartWeightResidencyOperationRequest{
                                 .identity = operating_identity,
                                 .expected_metadata_generation =
                                     operating.metadata_generation,
                                 .target_residency = WeightResidencyState::COLD,
                             },
                             now_ms + 30);
    ASSERT_TRUE(operation.has_value());
    ASSERT_TRUE(MetadataStore(*master_service_).Publish(*operation).has_value());
    const auto expected_snapshot = MetadataStore(*master_service_).ExportSnapshot();

    MasterSnapshotCodec codec;
    auto state_view = MakeStateView(*master_service_);
    auto encoded = codec.Encode(state_view);
    ASSERT_TRUE(encoded.has_value());
    auto target = MakeMasterService();
    ASSERT_TRUE(codec.Decode(target.get(), *encoded).has_value());

    EXPECT_EQ(expected_snapshot, MetadataStore(*target).ExportSnapshot());
    EXPECT_TRUE(MetadataStore(*target).IsManagedGroup(
        MakeWeightPayloadGroupId(leased_identity)));
    auto leased_view = target->GetWeightRevision(
        GetWeightRevisionRequest{.identity = leased_identity});
    ASSERT_TRUE(leased_view.has_value());
    EXPECT_EQ(1, leased_view->active_lease_count);
    auto operating_view = target->GetWeightRevision(
        GetWeightRevisionRequest{.identity = operating_identity});
    ASSERT_TRUE(operating_view.has_value());
    EXPECT_TRUE(operating_view->metadata.operation_id.has_value());
}

TEST_F(MasterSnapshotCodecTest, OldSnapshotWithoutWeightMetadataRestoresEmpty) {
    PublishReady(*master_service_,
                 WeightRevisionIdentity{
                     .tenant_id = "default",
                     .name_space = "production",
                     .resource_id = "llama-70b",
                     .revision = "step-100",
                     .weight_generation = 7,
                 },
                 100);
    MasterSnapshotCodec codec;
    auto state_view = MakeStateView(*master_service_);
    auto encoded = codec.Encode(state_view);
    ASSERT_TRUE(encoded.has_value());
    encoded->metadata =
        RewriteWeightMetadataField(encoded->metadata, /*keep_field=*/false);

    auto target = MakeMasterService();
    ASSERT_TRUE(codec.Decode(target.get(), *encoded).has_value());
    EXPECT_TRUE(MetadataStore(*target).ExportSnapshot().metadata.empty());
}

TEST_F(MasterSnapshotCodecTest, MalformedWeightMetadataFailsClosed) {
    MasterSnapshotCodec codec;
    auto state_view = MakeStateView(*master_service_);
    auto encoded = codec.Encode(state_view);
    ASSERT_TRUE(encoded.has_value());
    encoded->metadata = RewriteWeightMetadataField(
        encoded->metadata, /*keep_field=*/true, /*corrupt_field=*/true);

    auto target = MakeMasterService();
    auto decoded = codec.Decode(target.get(), *encoded);
    EXPECT_FALSE(decoded.has_value());
    EXPECT_TRUE(MetadataStore(*target).ExportSnapshot().metadata.empty());
}

TEST_F(MasterSnapshotCodecTest, EncodeDecodeRoundTripWithMemoryReplica) {
    // Mount a segment and store an object backed by a MEMORY replica. On
    // decode, the segment/allocator must be restored before the metadata,
    // otherwise deserializing the replica fails with SEGMENT_NOT_FOUND because
    // GetMountedSegment() cannot find its backing segment.
    constexpr size_t kSegmentBase = 0x300000000;
    constexpr size_t kSegmentSize = 1024 * 1024 * 16;  // 16MB
    const std::string kKey = "memory_replica_key";
    const TenantId& kTenant = TenantId::Default();

    Segment segment;
    segment.id = generate_uuid();
    segment.name = "codec_test_segment";
    segment.base = kSegmentBase;
    segment.size = kSegmentSize;
    segment.te_endpoint = segment.name;

    UUID client_id = generate_uuid();
    auto mount_result = master_service_->MountSegment(segment, client_id);
    ASSERT_TRUE(mount_result.has_value());

    auto put_start = master_service_->PutStart(
        client_id, kKey, kTenant,
        /*slice_length=*/1024, ReplicateConfig{.replica_num = 1});
    ASSERT_TRUE(put_start.has_value())
        << "PutStart failed: " << static_cast<int>(put_start.error());
    auto put_end =
        master_service_->PutEnd(client_id, kKey, kTenant, ReplicaType::MEMORY);
    ASSERT_TRUE(put_end.has_value())
        << "PutEnd failed: " << static_cast<int>(put_end.error());

    MasterSnapshotCodec codec;
    MasterSnapshotStateView state_view = MakeStateView(*master_service_);

    auto encode_result = codec.Encode(state_view);
    ASSERT_TRUE(encode_result.has_value())
        << "Encode failed: " << encode_result.error().message;
    EXPECT_FALSE(encode_result.value().segments.empty());

    // Decode into a fresh service. This exercises the segments-before-metadata
    // restore order.
    auto target_service = MakeMasterService();
    auto decode_result =
        codec.Decode(target_service.get(), encode_result.value());
    ASSERT_TRUE(decode_result.has_value())
        << "Decode failed: " << decode_result.error().message;

    // The MEMORY replica must be fully restored and queryable.
    auto get_result = target_service->GetReplicaList(kKey, kTenant);
    ASSERT_TRUE(get_result.has_value())
        << "GetReplicaList failed: " << static_cast<int>(get_result.error());
    EXPECT_EQ(get_result.value().replicas.size(), 1u);
}

TEST_F(MasterSnapshotCodecTest, DecodeWithCorruptPayloadFails) {
    MasterSnapshotCodec codec;

    MasterSnapshotPayloads corrupt;
    corrupt.metadata = std::vector<uint8_t>{1, 2, 3};
    corrupt.segments = std::vector<uint8_t>{4, 5, 6};
    corrupt.task_manager = std::vector<uint8_t>{7, 8, 9};

    auto decode_result = codec.Decode(master_service_.get(), corrupt);
    EXPECT_FALSE(decode_result.has_value());
    EXPECT_EQ(decode_result.error().code, ErrorCode::DESERIALIZE_FAIL);
}

TEST_F(MasterSnapshotCodecTest, DecodeWithNullService) {
    MasterSnapshotCodec codec;

    MasterSnapshotPayloads payloads;
    auto decode_result = codec.Decode(nullptr, payloads);
    EXPECT_FALSE(decode_result.has_value());
    EXPECT_EQ(decode_result.error().code, ErrorCode::INVALID_PARAMS);
}

// Regression test: a structurally valid MessagePack task-manager payload whose
// task id field has the wrong type used to throw msgpack::type_error out of
// TaskManagerSerializer::Deserialize() (the arr[0].as<std::string>() call sits
// outside the field-conversion try block). Since RestoreState() no longer
// wraps each candidate in a try/catch, an escaping exception here would abort
// restore and prevent fallback to an older healthy snapshot. Decode() must
// convert it into a SerializationError instead of throwing.
TEST_F(MasterSnapshotCodecTest, DecodeWithInvalidTaskFieldTypeReturnsError) {
    MasterSnapshotCodec codec;

    // Start from a valid encoded snapshot so the segments and metadata payloads
    // decode cleanly; we only want to corrupt the task-manager payload.
    MasterSnapshotStateView state_view = MakeStateView(*master_service_);
    auto encode_result = codec.Encode(state_view);
    ASSERT_TRUE(encode_result.has_value())
        << "Encode failed: " << encode_result.error().message;
    MasterSnapshotPayloads payloads = std::move(encode_result.value());

    // Build a structurally valid MessagePack task-manager payload: an outer
    // array of one task, the task itself a valid array with the expected field
    // count, but the id field (index 0, expected string) is an integer. This
    // unpacks cleanly and only fails at the arr[0].as<std::string>() step,
    // which used to throw msgpack::type_error out of Deserialize().
    constexpr size_t kTaskSerializedFields = 8;  // must match the serializer
    msgpack::sbuffer sbuf;
    msgpack::packer<msgpack::sbuffer> packer(&sbuf);
    packer.pack_array(1);  // one task
    packer.pack_array(kTaskSerializedFields);
    packer.pack(static_cast<int32_t>(12345));  // id: wrong type (int, not str)
    packer.pack(static_cast<int32_t>(0));      // type
    packer.pack(static_cast<int32_t>(0));      // status
    packer.pack(std::string("payload"));       // payload
    packer.pack(static_cast<int64_t>(0));      // created_at
    packer.pack(static_cast<int64_t>(0));      // last_updated_at
    packer.pack(std::string("message"));       // message
    packer.pack(std::string("assigned"));      // assigned_client

    payloads.task_manager = zstd_compress(
        reinterpret_cast<const uint8_t*>(sbuf.data()), sbuf.size(), 3);

    // Decode a fresh service. It must not throw; it must report a serialization
    // error so RestoreState() can fall back to another candidate snapshot.
    auto target_service = MakeMasterService();
    tl::expected<void, SerializationError> decode_result;
    ASSERT_NO_THROW(
        { decode_result = codec.Decode(target_service.get(), payloads); });
    EXPECT_FALSE(decode_result.has_value());
    EXPECT_EQ(decode_result.error().code, ErrorCode::DESERIALIZE_FAIL);
}

}  // namespace mooncake::ha
