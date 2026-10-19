#include "ha/snapshot/batch_oplog/codec.h"

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

namespace mooncake::test {

namespace {

struct LegacyStandbyObjectMetadata {
    UUID client_id{0, 0};
    uint64_t size{0};
    std::vector<Replica::Descriptor> replicas;
    std::string group_id;
    ObjectDataType data_type{ObjectDataType::UNKNOWN};

    YLT_REFL(LegacyStandbyObjectMetadata, client_id, size, replicas, group_id,
             data_type);
};

struct LegacyStandbyObjectEntry {
    std::string tenant_id{"default"};
    std::string key;
    LegacyStandbyObjectMetadata metadata;

    YLT_REFL(LegacyStandbyObjectEntry, tenant_id, key, metadata);
};

struct LegacyBatchOpLogSnapshotObjectChunk {
    uint64_t chunk_index{0};
    std::vector<LegacyStandbyObjectEntry> objects;

    YLT_REFL(LegacyBatchOpLogSnapshotObjectChunk, chunk_index, objects);
};

struct SessionStandbyObjectMetadata {
    UUID client_id{0, 0};
    uint64_t size{0};
    std::vector<Replica::Descriptor> replicas;
    std::string group_id;
    ObjectDataType data_type{ObjectDataType::UNKNOWN};
    struct_pack::compatible<bool, 1> hard_pinned;
    struct_pack::compatible<std::string, 2> session_id;

    YLT_REFL(SessionStandbyObjectMetadata, client_id, size, replicas, group_id,
             data_type, hard_pinned, session_id);
};

struct SessionStandbyObjectEntry {
    std::string tenant_id{"default"};
    std::string key;
    SessionStandbyObjectMetadata metadata;

    YLT_REFL(SessionStandbyObjectEntry, tenant_id, key, metadata);
};

struct SessionBatchOpLogSnapshotObjectChunk {
    uint64_t chunk_index{0};
    std::vector<SessionStandbyObjectEntry> objects;

    YLT_REFL(SessionBatchOpLogSnapshotObjectChunk, chunk_index, objects);
};

struct SessionMetadataPayload {
    UUID client_id;
    uint64_t size{0};
    std::vector<Replica::Descriptor> replicas;
    struct_pack::compatible<std::string, 1> group_id;
    struct_pack::compatible<ObjectDataType, 1> data_type;
    struct_pack::compatible<bool, 1> hard_pinned;
    struct_pack::compatible<std::string, 2> session_id;

    YLT_REFL(SessionMetadataPayload, client_id, size, replicas, group_id,
             data_type, hard_pinned, session_id);
};

}  // namespace

TEST(BatchOpLogSnapshotCodecTest, SegmentsRoundTrip) {
    std::vector<StandbySegmentInfo> segments{{
        .segment_name = "segment-1",
        .transport_endpoint = "127.0.0.1:12345",
        .capacity = 4096,
        .is_memory_segment = true,
        .file_path = "",
    }};

    auto decoded = DecodeBatchOpLogSnapshotSegments(
        EncodeBatchOpLogSnapshotSegments(segments));

    ASSERT_TRUE(decoded) << decoded.error();
    ASSERT_EQ(1u, decoded->size());
    EXPECT_EQ("segment-1", decoded->front().segment_name);
    EXPECT_EQ("127.0.0.1:12345", decoded->front().transport_endpoint);
    EXPECT_EQ(4096u, decoded->front().capacity);
    EXPECT_TRUE(decoded->front().is_memory_segment);
}

TEST(BatchOpLogSnapshotCodecTest, ObjectChunkRoundTripsAndChecksEnvelope) {
    StandbyObjectMetadata metadata;
    metadata.client_id = {1, 2};
    metadata.size = 8192;
    metadata.group_id = "group";
    metadata.hard_pinned = true;
    metadata.session_id = "session-snapshot";
    metadata.object_checksum = 0;
    metadata.residency_affinity_id = "opaque-affinity-id";
    std::vector<StandbyObjectEntry> objects{
        {.tenant_id = "tenant", .key = "key", .metadata = metadata}};
    auto encoded = EncodeBatchOpLogSnapshotObjectChunk(7, std::move(objects));

    auto decoded = DecodeBatchOpLogSnapshotObjectChunk(encoded, 7, 1);

    ASSERT_TRUE(decoded) << decoded.error();
    EXPECT_EQ(7u, decoded->chunk_index);
    ASSERT_EQ(1u, decoded->objects.size());
    EXPECT_EQ("tenant", decoded->objects[0].tenant_id);
    EXPECT_EQ("key", decoded->objects[0].key);
    EXPECT_EQ(UUID(1, 2), decoded->objects[0].metadata.client_id);
    EXPECT_EQ(8192u, decoded->objects[0].metadata.size);
    EXPECT_EQ("group", decoded->objects[0].metadata.group_id);
    EXPECT_TRUE(decoded->objects[0].metadata.hard_pinned.value_or(false));
    EXPECT_EQ(decoded->objects[0].metadata.session_id.value_or(""),
              "session-snapshot");
    ASSERT_TRUE(decoded->objects[0].metadata.object_checksum.has_value());
    EXPECT_EQ(decoded->objects[0].metadata.object_checksum.value(), 0U);
    EXPECT_EQ("opaque-affinity-id",
              decoded->objects[0].metadata.residency_affinity_id.value_or(""));

    EXPECT_FALSE(DecodeBatchOpLogSnapshotObjectChunk(encoded, 8, 1));
    EXPECT_FALSE(DecodeBatchOpLogSnapshotObjectChunk(encoded, 7, 300000000));
}

TEST(BatchOpLogSnapshotCodecTest, OldObjectChunkDefaultsHardPinnedToFalse) {
    LegacyBatchOpLogSnapshotObjectChunk old_chunk{
        0, {{"tenant", "key", {{1, 2}, 8192, {}, "group", {}}}}};
    auto old_encoded = struct_pack::serialize(old_chunk);
    std::vector<uint8_t> encoded(old_encoded.begin(), old_encoded.end());

    auto decoded = DecodeBatchOpLogSnapshotObjectChunk(encoded, 0, 1);

    ASSERT_TRUE(decoded) << decoded.error();
    EXPECT_FALSE(decoded->objects[0].metadata.hard_pinned.value_or(false));
    EXPECT_FALSE(decoded->objects[0].metadata.session_id.has_value());
    EXPECT_FALSE(decoded->objects[0].metadata.object_checksum.has_value());
}

TEST(BatchOpLogSnapshotCodecTest,
     SessionV2ChunkPreservesSessionWithoutAffinity) {
    SessionStandbyObjectMetadata metadata;
    metadata.client_id = {1, 2};
    metadata.size = 8192;
    metadata.group_id = "group";
    metadata.hard_pinned = true;
    metadata.session_id = "main-session";
    SessionBatchOpLogSnapshotObjectChunk chunk{0,
                                               {{"tenant", "key", metadata}}};
    auto bytes = struct_pack::serialize(chunk);
    std::vector<uint8_t> encoded(bytes.begin(), bytes.end());

    auto decoded = DecodeBatchOpLogSnapshotObjectChunk(encoded, 0, 1);

    ASSERT_TRUE(decoded) << decoded.error();
    ASSERT_EQ(decoded->objects.size(), 1u);
    EXPECT_EQ(decoded->objects[0].metadata.session_id.value_or(""),
              "main-session");
    EXPECT_TRUE(decoded->objects[0].metadata.hard_pinned.value_or(false));
    EXPECT_FALSE(
        decoded->objects[0].metadata.residency_affinity_id.has_value());
}

TEST(BatchOpLogSnapshotCodecTest, SessionV2PayloadUpgradesToAffinityV3) {
    SessionMetadataPayload old_payload;
    old_payload.client_id = {1, 2};
    old_payload.size = 8192;
    old_payload.group_id = "group";
    old_payload.data_type = ObjectDataType::WEIGHT;
    old_payload.hard_pinned = true;
    old_payload.session_id = "main-session";
    auto bytes = struct_pack::serialize(old_payload);

    auto decoded = struct_pack::deserialize<MetadataPayload>(bytes);

    ASSERT_TRUE(decoded);
    auto metadata = decoded->ToStandbyMetadata();
    EXPECT_EQ(metadata.session_id.value_or(""), "main-session");
    EXPECT_EQ(metadata.group_id, "group");
    EXPECT_TRUE(metadata.hard_pinned.value_or(false));
    EXPECT_TRUE(metadata.residency_affinity_id.value_or("").empty());
    decoded->residency_affinity_id = "affinity";
    auto upgraded = struct_pack::deserialize<MetadataPayload>(
        struct_pack::serialize(*decoded));
    ASSERT_TRUE(upgraded);
    EXPECT_EQ(upgraded->session_id.value_or(""), "main-session");
    EXPECT_EQ(upgraded->residency_affinity_id.value_or(""), "affinity");
}

TEST(BatchOpLogSnapshotCodecTest, RejectsTruncatedArtifacts) {
    auto segments = EncodeBatchOpLogSnapshotSegments({});
    auto objects = EncodeBatchOpLogSnapshotObjectChunk(
        0, std::vector<StandbyObjectEntry>{
               {.tenant_id = "default", .key = "key", .metadata = {}}});
    segments.pop_back();
    objects.pop_back();

    EXPECT_FALSE(DecodeBatchOpLogSnapshotSegments(segments));
    EXPECT_FALSE(DecodeBatchOpLogSnapshotObjectChunk(objects, 0, 1));
}

}  // namespace mooncake::test
