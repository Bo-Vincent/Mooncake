#include "master_service_test_fixture.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <thread>
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

    struct PayloadSpec {
        std::string key;
        uint64_t size;
        std::string affinity_id;
    };

    WeightRevisionMetadata PublishReadyWithPayloads(
        MasterService& service, const UUID& client_id,
        const std::vector<PayloadSpec>& payloads,
        WeightStoragePolicy policy =
            {
                .preferred_residency = WeightResidencyState::HOT,
                .mixed_hot_ratio = 0.5,
                .migration_mode = WeightMigrationMode::MANUAL,
            },
        WeightRevisionIdentity identity = Identity()) {
        uint64_t logical_bytes = 0;
        std::vector<std::string> payload_keys;
        std::set<std::string> affinity_ids;
        for (const auto& payload : payloads) {
            logical_bytes += payload.size;
            payload_keys.push_back(payload.key);
            affinity_ids.insert(payload.affinity_id);
        }
        auto importing = service.BeginWeightImport(BeginWeightImportRequest{
            .identity = identity,
            .payload_group_id = {},
            .expected_payload_count = payloads.size(),
            .expected_logical_bytes = logical_bytes,
            .policy = policy,
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
        const auto manifest_key = MakeWeightManifestKey(identity);
        PutCompletedObject(service, client_id, manifest_key, manifest_config,
                           128);
        auto ready = service.CommitWeightImport(CommitWeightImportRequest{
            .identity = identity,
            .expected_metadata_generation = importing->metadata_generation,
            .manifest =
                WeightManifestReference{
                    .manifest_key = manifest_key,
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

    WeightRevisionMetadata PublishReady(
        MasterService& service, const UUID& client_id,
        WeightStoragePolicy policy = {
            .preferred_residency = WeightResidencyState::HOT,
            .mixed_hot_ratio = 0.5,
            .migration_mode = WeightMigrationMode::MANUAL,
        }) {
        return PublishReadyWithPayloads(
            service, client_id,
            {{.key = "payload-a", .size = 1024, .affinity_id = "affinity-a"}},
            policy);
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

TEST_F(WeightGroupLifecycleTest, RecreatedMetadataStartsFreshMemberSelection) {
    const auto config = MasterServiceConfig::builder()
                            .set_default_kv_lease_ttl(0)
                            .set_enable_offload(true)
                            .set_root_fs_dir("/mnt/ssd")
                            .build();
    MasterService service(config);
    const auto context = PrepareSimpleSegment(service);
    ASSERT_TRUE(service.MountLocalDiskSegment(context.client_id, true));
    const std::vector<PayloadSpec> payloads{{"unit-a1", 1024, "a"},
                                            {"unit-a2", 1024, "a"},
                                            {"unit-b1", 1024, "b"},
                                            {"unit-b2", 1024, "b"}};
    const auto ready =
        PublishReadyWithPayloads(service, context.client_id, payloads);
    const auto manifest_key = MakeWeightManifestKey(ready.identity);
    const auto attach_cold = [&](const PayloadSpec& payload) {
        ASSERT_TRUE(service.PutEnd(context.client_id, payload.key,
                                   TenantId::Default(), ReplicaType::DISK));
        AddLocalDiskReplica(service, context.client_id, payload.key,
                            payload.size, "test_segment");
    };
    for (const auto& payload : payloads) attach_cold(payload);
    ASSERT_TRUE(service.StartWeightResidencyOperation({
        .identity = ready.identity,
        .expected_metadata_generation = ready.metadata_generation,
        .target_residency = WeightResidencyState::COLD,
    }));
    auto& manager = service.GetWeightStoreManager();
    const auto initial = manager.ExportSnapshot();
    ASSERT_EQ(1, initial.operations.size());
    ASSERT_TRUE(initial.lineages.has_value());
    ASSERT_EQ(1, manager.ReconcileWeightMetadataStoreOnce(
                     ready.updated_at_ms + 1, 1));

    // Observe absence before restoring the same identity and physical objects.
    WeightMetadataSnapshot empty;
    empty.schema_version = kWeightMetadataSchemaVersion;
    empty.lineages = std::vector<WeightLineageMetadata>{};
    ASSERT_TRUE(manager.RestoreSnapshot(empty));
    EXPECT_EQ(0, manager.ReconcileWeightMetadataStoreOnce(
                     ready.updated_at_ms + 2, 1));
    ReplicateConfig put;
    put.replica_num = 1;
    put.with_hard_pin = true;
    put.group_ids = std::vector<std::string>{ready.manifest.payload_group_id};
    put.data_type = ObjectDataType::WEIGHT;
    for (const auto& payload : payloads) {
        ASSERT_TRUE(service.Remove(payload.key, TenantId::Default(), true));
        put.residency_affinity_ids =
            std::vector<std::string>{payload.affinity_id};
        PutCompletedObject(service, context.client_id, payload.key, put,
                           payload.size);
        attach_cold(payload);
    }
    ASSERT_TRUE(manager.RestoreSnapshot(initial));
    ASSERT_EQ(1, manager.ReconcileWeightMetadataStoreOnce(
                     ready.updated_at_ms + 3, 1));
    for (const auto& payload : payloads) {
        const auto replicas =
            service.GetReplicaListForAdmin(payload.key, TenantId::Default());
        ASSERT_TRUE(replicas);
        const bool has_memory = std::any_of(
            replicas->replicas.begin(), replicas->replicas.end(),
            [](const auto& replica) { return replica.is_memory_replica(); });
        EXPECT_EQ(payload.affinity_id == "b", has_memory) << payload.key;
    }
    const auto manifest =
        service.GetReplicaListForAdmin(manifest_key, TenantId::Default());
    ASSERT_TRUE(manifest);
    EXPECT_TRUE(std::any_of(
        manifest->replicas.begin(), manifest->replicas.end(),
        [](const auto& replica) { return replica.is_memory_replica(); }));
}

TEST_F(WeightGroupLifecycleTest, PublicDeletePreservesDefaultMemberBudget) {
    MasterService service;
    const auto context = PrepareSimpleSegment(service);
    std::vector<PayloadSpec> payloads;
    for (size_t i = 0; i < 65; ++i) {
        payloads.push_back({"bounded-delete-" + std::to_string(i), 1024, "a"});
    }
    const auto ready =
        PublishReadyWithPayloads(service, context.client_id, payloads);
    const auto manifest_key = MakeWeightManifestKey(ready.identity);
    const auto first = service.DeleteWeightRevision({
        .identity = ready.identity,
        .expected_metadata_generation = ready.metadata_generation,
    });
    ASSERT_TRUE(first);
    EXPECT_EQ(WeightAvailabilityState::DELETING, first->availability);
    size_t remaining = 0;
    for (const auto& payload : payloads) {
        const auto exists = service.ExistKey(payload.key, TenantId::Default());
        ASSERT_TRUE(exists);
        remaining += *exists;
    }
    EXPECT_EQ(1, remaining);
    const auto manifest = service.ExistKey(manifest_key, TenantId::Default());
    ASSERT_TRUE(manifest);
    EXPECT_TRUE(*manifest);
    const auto second = service.DeleteWeightRevision({
        .identity = ready.identity,
        .expected_metadata_generation = first->metadata_generation,
    });
    ASSERT_TRUE(second);
    EXPECT_EQ(WeightAvailabilityState::DELETED, second->availability);
    for (const auto& payload : payloads) {
        const auto exists = service.ExistKey(payload.key, TenantId::Default());
        ASSERT_TRUE(exists);
        EXPECT_FALSE(*exists);
    }
    EXPECT_FALSE(
        service.ExistKey(manifest_key, TenantId::Default()).value_or(true));
}

TEST_F(WeightGroupLifecycleTest, TwoRetiringLineagesShareBoundedProgress) {
    MasterService service;
    const auto context = PrepareSimpleSegment(service);
    const WeightStoragePolicy policy{
        .preferred_residency = WeightResidencyState::HOT,
        .migration_mode = WeightMigrationMode::MANUAL};
    for (const auto& revision : {"00-idle", "01-idle"}) {
        auto identity = Identity();
        identity.revision = revision;
        PublishReadyWithPayloads(
            service, context.client_id,
            {{std::string(revision) + "-payload", 1024, "a"}}, policy,
            identity);
    }
    std::vector<std::vector<PayloadSpec>> retiring_payloads;
    uint64_t now_ms = 0;
    for (const auto& revision : {"a-retiring", "b-retiring"}) {
        auto identity = Identity();
        identity.revision = revision;
        std::vector<PayloadSpec> payloads;
        for (int i = 0; i < 5; ++i) {
            payloads.push_back(
                {std::string(revision) + "-" + std::to_string(i), 1024, "a"});
        }
        const auto base = PublishReadyWithPayloads(service, context.client_id,
                                                   payloads, policy, identity);
        const auto lease = service.AcquireWeightRevisionLease({
            .identity = base.identity,
            .expected_metadata_generation = base.metadata_generation,
            .holder = "hold-retirement",
            .ttl_ms = 60'000,
        });
        ASSERT_TRUE(lease);
        auto target = identity;
        ++target.weight_generation;
        const std::string request_id = std::string(revision) + "-upsert";
        ASSERT_TRUE(service.BeginWeightUpsert({
            .request_id = request_id,
            .mode = WeightUpsertMode::PUT_FIRST,
            .base_identity = base.identity,
            .expected_base_metadata_generation = base.metadata_generation,
            .target_identity = target,
            .import = {.identity = target,
                       .expected_payload_count = 1,
                       .expected_logical_bytes = 1024,
                       .policy = policy,
                       .affinity_summary = {.affinity_count = 1,
                                            .affinity_digest =
                                                std::string(64, 'c')}},
        }));
        const auto ready_target = PublishReadyWithPayloads(
            service, context.client_id,
            {{std::string(revision) + "-replacement", 1024, "a"}}, policy,
            target);
        const auto committed = service.CommitWeightUpsert({
            .request_id = request_id,
            .base_identity = base.identity,
            .target_identity = target,
        });
        ASSERT_TRUE(committed && committed->latest_claim);
        ASSERT_EQ(WeightUpsertPhase::RETIRING_BASE,
                  committed->latest_claim->phase);
        ASSERT_TRUE(service.ReleaseWeightRevisionLease({
            .tenant_id = identity.tenant_id,
            .lease_id = lease->lease_id,
        }));
        now_ms = std::max(now_ms, ready_target.updated_at_ms + 1);
        retiring_payloads.push_back(payloads);
    }
    const auto remaining = [&](const std::vector<PayloadSpec>& payloads) {
        size_t count = 0;
        for (const auto& payload : payloads) {
            const auto exists =
                service.ExistKey(payload.key, TenantId::Default());
            EXPECT_TRUE(exists);
            count += exists.value_or(false);
        }
        return count;
    };
    for (const auto& payloads : retiring_payloads)
        ASSERT_EQ(5, remaining(payloads));
    // Idle revisions consume the revision-first tick; both active lineages
    // must make physical progress on the two lineage-first ticks.
    for (int tick = 0; tick < 3; ++tick) {
        EXPECT_LE(
            MasterServiceTestPeer(service).RunWeightReconciliationForTesting(
                now_ms + tick, 2),
            2);
    }
    EXPECT_EQ(3, remaining(retiring_payloads[0]));
    EXPECT_EQ(3, remaining(retiring_payloads[1]));
}

TEST_F(WeightGroupLifecycleTest,
       ExpiredLeaseTrafficLeavesBothCleanupPhasesProgress) {
    MasterService service;
    const auto context = PrepareSimpleSegment(service);
    const WeightStoragePolicy policy{
        .preferred_residency = WeightResidencyState::HOT,
        .migration_mode = WeightMigrationMode::MANUAL};
    auto identity = Identity();
    identity.revision = "a-retiring-with-lease-traffic";
    const std::vector<PayloadSpec> payloads{
        {"lease-traffic-retire-a", 1024, "a"},
        {"lease-traffic-retire-b", 1024, "a"},
        {"lease-traffic-retire-c", 1024, "a"}};
    const auto base = PublishReadyWithPayloads(service, context.client_id,
                                               payloads, policy, identity);
    const auto hold = service.AcquireWeightRevisionLease({
        .identity = identity,
        .expected_metadata_generation = base.metadata_generation,
        .holder = "hold-retirement",
        .ttl_ms = 60'000,
    });
    ASSERT_TRUE(hold);
    auto target = identity;
    ++target.weight_generation;
    ASSERT_TRUE(service.BeginWeightUpsert({
        .request_id = "lease-traffic-retirement",
        .mode = WeightUpsertMode::PUT_FIRST,
        .base_identity = identity,
        .expected_base_metadata_generation = base.metadata_generation,
        .target_identity = target,
        .import = {.identity = target,
                   .expected_payload_count = 1,
                   .expected_logical_bytes = 1024,
                   .policy = policy,
                   .affinity_summary = {.affinity_count = 1,
                                        .affinity_digest =
                                            std::string(64, 'c')}},
    }));
    PublishReadyWithPayloads(service, context.client_id,
                             {{"lease-traffic-target", 1024, "a"}}, policy,
                             target);
    const auto committed = service.CommitWeightUpsert({
        .request_id = "lease-traffic-retirement",
        .base_identity = identity,
        .target_identity = target,
    });
    ASSERT_TRUE(committed && committed->latest_claim);
    ASSERT_EQ(WeightUpsertPhase::RETIRING_BASE, committed->latest_claim->phase);
    ASSERT_TRUE(service.ReleaseWeightRevisionLease(
        {.tenant_id = identity.tenant_id, .lease_id = hold->lease_id}));
    auto damaged_identity = Identity();
    damaged_identity.revision = "b-damaged-with-lease-traffic";
    const auto damaged = PublishReadyWithPayloads(
        service, context.client_id, {{"lease-traffic-damaged", 1024, "a"}},
        policy, damaged_identity);
    ASSERT_TRUE(MasterServiceTestPeer(service).DropWeightGroupMemberForTesting(
        damaged.identity, "lease-traffic-damaged"));
    auto healthy_identity = Identity();
    healthy_identity.revision = "z-healthy-with-lease-traffic";
    const auto healthy = PublishReadyWithPayloads(
        service, context.client_id, {{"lease-traffic-healthy", 1024, "a"}},
        policy, healthy_identity);
    size_t previously_released = 0;
    for (size_t tick = 0; tick < 16; ++tick) {
        const auto lease = service.AcquireWeightRevisionLease({
            .identity = healthy.identity,
            .expected_metadata_generation = healthy.metadata_generation,
            .holder = "continuous-" + std::to_string(tick),
            .ttl_ms = 60'000,
        });
        ASSERT_TRUE(lease);
        EXPECT_LE(
            MasterServiceTestPeer(service).RunWeightReconciliationForTesting(
                lease->expires_at_ms + 1, 1),
            1);
        if ((tick + 1) % 4 == 0) {
            const auto snapshot =
                service.GetWeightStoreManager().ExportSnapshot();
            ASSERT_LE(snapshot.leases.size(), tick + 1);
            const auto released = tick + 1 - snapshot.leases.size();
            EXPECT_GT(released, previously_released);
            previously_released = released;
        }
    }
    size_t remaining = 0;
    for (const auto& payload : payloads) {
        const auto exists = service.ExistKey(payload.key, TenantId::Default());
        ASSERT_TRUE(exists);
        remaining += *exists;
    }
    EXPECT_LT(remaining, payloads.size());
    const auto observed =
        service.GetWeightRevision({.identity = damaged.identity});
    ASSERT_TRUE(observed);
    EXPECT_EQ(WeightAvailabilityState::DEGRADED,
              observed->metadata.availability);
}

TEST_F(WeightGroupLifecycleTest, BlockedRetirementDoesNotStarveOtherLineages) {
    MasterService service;
    const auto source = PrepareSimpleSegment(service);
    const WeightStoragePolicy policy{
        .preferred_residency = WeightResidencyState::HOT,
        .mixed_hot_ratio = 0.5,
        .migration_mode = WeightMigrationMode::MANUAL,
    };
    auto first_identity = Identity();
    first_identity.resource_id = "a-blocked-retirement";
    auto second_identity = Identity();
    second_identity.resource_id = "z-progressing-retirement";
    const auto first = PublishReadyWithPayloads(
        service, source.client_id, {{"blocked-payload", 1024, "a"}}, policy,
        first_identity);
    const auto second = PublishReadyWithPayloads(
        service, source.client_id, {{"progressing-payload", 1024, "b"}}, policy,
        second_identity);
    [[maybe_unused]] const auto target_segment = PrepareSimpleSegment(
        service, "copy-target", kDefaultSegmentBase + kDefaultSegmentSize);
    const auto copy =
        service.CopyStart(source.client_id, "blocked-payload",
                          TenantId::Default(), "test_segment", {"copy-target"});
    ASSERT_TRUE(copy);
    ASSERT_TRUE(service.CopyStart(source.client_id, "progressing-payload",
                                  TenantId::Default(), "test_segment",
                                  {"copy-target"}));

    for (const auto& base : {first, second}) {
        auto target = base.identity;
        ++target.weight_generation;
        const auto begun = service.BeginWeightUpsert({
            .request_id = base.identity.resource_id,
            .mode = WeightUpsertMode::DELETE_FIRST,
            .base_identity = base.identity,
            .expected_base_metadata_generation = base.metadata_generation,
            .target_identity = target,
            .import = {.identity = target,
                       .expected_payload_count = 1,
                       .expected_logical_bytes = 1024,
                       .policy = policy,
                       .affinity_summary = {.affinity_count = 1,
                                            .affinity_digest =
                                                std::string(64, 'c')}},
        });
        ASSERT_FALSE(begun);
        EXPECT_EQ(WeightManagementError::BUSY, begun.error());
        const auto lineage = service.GetWeightLineage({
            .identity = ToWeightLineageIdentity(base.identity),
        });
        ASSERT_TRUE(lineage);
        ASSERT_TRUE(lineage->latest_claim);
        EXPECT_EQ(WeightUpsertPhase::DELETING_BASE,
                  lineage->latest_claim->phase);
        const auto deleting =
            service.GetWeightRevision({.identity = base.identity});
        ASSERT_TRUE(deleting);
        ASSERT_EQ(WeightAvailabilityState::DELETING,
                  deleting->metadata.availability);
    }
    ASSERT_TRUE(service.CopyRevoke(source.client_id, "progressing-payload",
                                   TenantId::Default()));

    auto ordinary_identity = Identity();
    ordinary_identity.resource_id = "ordinary-aborted-import";
    const auto ordinary = service.BeginWeightImport({
        .identity = ordinary_identity,
        .expected_payload_count = 1,
        .expected_logical_bytes = 1024,
        .policy = policy,
        .affinity_summary = {.affinity_count = 1,
                             .affinity_digest = std::string(64, 'c')},
    });
    ASSERT_TRUE(ordinary);
    ReplicateConfig ordinary_put;
    ordinary_put.replica_num = 1;
    ordinary_put.with_hard_pin = true;
    ordinary_put.group_ids =
        std::vector<std::string>{ordinary->manifest.payload_group_id};
    ordinary_put.residency_affinity_ids = std::vector<std::string>{"ordinary"};
    ordinary_put.data_type = ObjectDataType::WEIGHT;
    PutCompletedObject(service, source.client_id, "ordinary-payload",
                       ordinary_put, 1024);
    ASSERT_TRUE(service.AbortWeightImport({
        .identity = ordinary_identity,
        .expected_metadata_generation = ordinary->metadata_generation,
    }));

    // Keep a real CopyStart unfinished while independent retirement advances.
    // Independent lineage and revision cleanup must progress within sixteen
    // ticks.
    for (size_t tick = 0; tick < 16; ++tick) {
        MasterServiceTestPeer(service).RunWeightReconciliationForTesting(
            first.updated_at_ms + tick + 1, 1);
    }
    const auto blocked =
        service.GetWeightRevision({.identity = first.identity});
    const auto progressed =
        service.GetWeightRevision({.identity = second.identity});
    ASSERT_TRUE(blocked);
    ASSERT_TRUE(progressed);
    EXPECT_EQ(WeightAvailabilityState::DELETING,
              blocked->metadata.availability);
    EXPECT_TRUE(service.ExistKey("blocked-payload", TenantId::Default())
                    .value_or(false));
    EXPECT_EQ(WeightAvailabilityState::DELETED,
              progressed->metadata.availability);
    EXPECT_FALSE(service.ExistKey("progressing-payload", TenantId::Default())
                     .value_or(true));
    const auto ordinary_view =
        service.GetWeightRevision({.identity = ordinary_identity});
    ASSERT_TRUE(ordinary_view);
    EXPECT_EQ(WeightAvailabilityState::DELETED,
              ordinary_view->metadata.availability);
    EXPECT_FALSE(service.ExistKey("ordinary-payload", TenantId::Default())
                     .value_or(true));
    EXPECT_TRUE(service.CopyRevoke(source.client_id, "blocked-payload",
                                   TenantId::Default()));
}

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

TEST_F(WeightGroupLifecycleTest,
       MissingPayloadPreservesObservedMixedResidency) {
    MasterServiceConfig config;
    config.default_kv_lease_ttl = 0;
    config.enable_offload = true;
    config.offload_on_evict = true;
    MasterService service(config);
    const auto context = PrepareSimpleSegment(service);
    ASSERT_TRUE(service.MountLocalDiskSegment(context.client_id, true));
    const auto ready = PublishReadyWithPayloads(service, context.client_id,
                                                {{"mixed-a", 1024, "a"},
                                                 {"mixed-b", 1024, "b"},
                                                 {"mixed-c", 1024, "c"}});
    const auto operation = service.StartWeightResidencyOperation({
        .identity = ready.identity,
        .expected_metadata_generation = ready.metadata_generation,
        .target_residency = WeightResidencyState::COLD,
        .mixed_hot_ratio = std::nullopt,
    });
    ASSERT_TRUE(operation);
    ASSERT_TRUE(service.ReconcileWeightRevision({.identity = ready.identity}));
    const auto tasks = service.OffloadObjectHeartbeat(context.client_id, true);
    ASSERT_TRUE(tasks);
    ASSERT_EQ(3u, tasks->size());
    StorageObjectMetadata disk{};
    disk.key_size = tasks->front().key.size();
    disk.data_size = tasks->front().size;
    disk.transport_endpoint = "test_segment";
    ASSERT_TRUE(service.NotifyOffloadSuccess(context.client_id,
                                             {tasks->front()}, {disk}));
    ASSERT_TRUE(service.ReconcileWeightRevision({.identity = ready.identity}));
    ASSERT_TRUE(
        service.BatchReplicaClear({(*tasks)[1].key}, context.client_id, ""));
    const auto observed =
        service.ReconcileWeightRevision({.identity = ready.identity});
    ASSERT_TRUE(observed);
    EXPECT_EQ(WeightAvailabilityState::DEGRADED, observed->availability);
    EXPECT_EQ(WeightResidencyState::MIXED, observed->residency);
    EXPECT_DOUBLE_EQ(0.5, observed->observed_hot_ratio);
    EXPECT_EQ(operation->operation_id, observed->operation_id);
    const auto progress = service.QueryWeightOperation({
        .operation_id = operation->operation_id,
    });
    ASSERT_TRUE(progress);
    EXPECT_NE("completed", progress->message);
}

TEST_F(WeightGroupLifecycleTest, RetirementSharesDeletionMemberBudget) {
    MasterService service;
    const auto context = PrepareSimpleSegment(service);
    const auto base = PublishReadyWithPayloads(
        service, context.client_id,
        {{"retire-a", 1024, "a"}, {"retire-b", 1024, "b"}});
    const auto lease = service.AcquireWeightRevisionLease({
        .identity = base.identity,
        .expected_metadata_generation = base.metadata_generation,
        .holder = "retirement-reader",
        .ttl_ms = 60'000,
    });
    ASSERT_TRUE(lease);
    auto target = base.identity;
    ++target.weight_generation;
    ASSERT_TRUE(service.BeginWeightUpsert({
        .request_id = "bounded-retirement",
        .mode = WeightUpsertMode::PUT_FIRST,
        .base_identity = base.identity,
        .expected_base_metadata_generation = base.metadata_generation,
        .target_identity = target,
        .import = {.identity = target,
                   .expected_payload_count = 1,
                   .expected_logical_bytes = 1024,
                   .policy =
                       WeightStoragePolicy{
                           .preferred_residency = WeightResidencyState::HOT,
                           .migration_mode = WeightMigrationMode::MANUAL},
                   .affinity_summary = {.affinity_count = 1,
                                        .affinity_digest =
                                            std::string(64, 'c')}},
    }));
    PublishReadyWithPayloads(service, context.client_id,
                             {{"replacement", 1024, "replacement"}},
                             {.preferred_residency = WeightResidencyState::HOT,
                              .migration_mode = WeightMigrationMode::MANUAL},
                             target);
    const auto committed = service.CommitWeightUpsert({
        .request_id = "bounded-retirement",
        .base_identity = base.identity,
        .target_identity = target,
    });
    ASSERT_TRUE(committed && committed->latest_claim);
    ASSERT_EQ(WeightUpsertPhase::RETIRING_BASE, committed->latest_claim->phase);
    ASSERT_TRUE(service.ReleaseWeightRevisionLease({
        .tenant_id = base.identity.tenant_id,
        .lease_id = lease->lease_id,
    }));
    auto ordinary = base.identity;
    ordinary.revision = "ordinary-deletion";
    const auto importing = service.BeginWeightImport({
        .identity = ordinary,
        .expected_payload_count = 2,
        .expected_logical_bytes = 2048,
        .affinity_summary = {.affinity_count = 2,
                             .affinity_digest = std::string(64, 'c')},
    });
    ASSERT_TRUE(importing);
    ReplicateConfig put;
    put.replica_num = 1;
    put.with_hard_pin = true;
    put.group_ids =
        std::vector<std::string>{importing->manifest.payload_group_id};
    put.residency_affinity_ids = std::vector<std::string>{"ordinary"};
    put.data_type = ObjectDataType::WEIGHT;
    for (const auto& key : {"ordinary-a", "ordinary-b"}) {
        PutCompletedObject(service, context.client_id, key, put, 1024);
    }
    put.data_type = ObjectDataType::METADATA;
    PutCompletedObject(service, context.client_id,
                       MakeWeightManifestKey(ordinary), put, 128);
    ASSERT_TRUE(service.AbortWeightImport({
        .identity = ordinary,
        .expected_metadata_generation = importing->metadata_generation,
    }));
    const std::vector<std::string> keys{
        "retire-a",   "retire-b",   MakeWeightManifestKey(base.identity),
        "ordinary-a", "ordinary-b", MakeWeightManifestKey(ordinary)};
    auto exists = [&](const std::string& key) {
        return service.ExistKey(key, TenantId::Default()).value_or(false);
    };
    ASSERT_EQ(keys.size(), std::count_if(keys.begin(), keys.end(), exists));
    size_t previous = keys.size();
    for (size_t round = 0; round < 16 && previous != 0; ++round) {
        MasterServiceTestPeer(service).RunWeightReconciliationForTesting(
            importing->updated_at_ms + round + 1, 1);
        const auto remaining = static_cast<size_t>(
            std::count_if(keys.begin(), keys.end(), exists));
        EXPECT_LE(previous - remaining, 1u);
        if (round == 0) EXPECT_EQ(previous - remaining, 1u);
        if (exists("retire-a") || exists("retire-b")) {
            EXPECT_TRUE(exists(MakeWeightManifestKey(base.identity)));
        }
        if (exists("ordinary-a") || exists("ordinary-b")) {
            EXPECT_TRUE(exists(MakeWeightManifestKey(ordinary)));
        }
        previous = remaining;
    }
    EXPECT_EQ(0u, previous);
    const auto retired = service.GetWeightRevision({.identity = base.identity});
    ASSERT_TRUE(retired);
    EXPECT_EQ(WeightAvailabilityState::DELETED, retired->metadata.availability);
}

TEST_F(WeightGroupLifecycleTest, NewLineageClaimRevalidatesCurrentBase) {
    MasterService service;
    const auto context = PrepareSimpleSegment(service);
    auto base = PublishReady(service, context.client_id);
    auto retirement_lease =
        service.AcquireWeightRevisionLease(AcquireWeightRevisionLeaseRequest{
            .identity = base.identity,
            .expected_metadata_generation = base.metadata_generation,
            .holder = "retiring-reader",
            .ttl_ms = 60'000,
        });
    ASSERT_TRUE(retirement_lease.has_value());
    auto target_identity = base.identity;
    target_identity.weight_generation = 8;
    auto begin = service.BeginWeightUpsert(BeginWeightUpsertRequest{
        .request_id = "replace-7-with-8",
        .mode = WeightUpsertMode::PUT_FIRST,
        .base_identity = base.identity,
        .expected_base_metadata_generation = base.metadata_generation,
        .target_identity = target_identity,
        .import =
            BeginWeightImportRequest{
                .identity = target_identity,
                .payload_group_id = {},
                .expected_payload_count = 1,
                .expected_logical_bytes = 2048,
                .policy =
                    WeightStoragePolicy{
                        .preferred_residency = WeightResidencyState::HOT,
                        .migration_mode = WeightMigrationMode::MANUAL,
                    },
                .affinity_summary =
                    WeightAffinitySummary{
                        .affinity_count = 1,
                        .affinity_digest = std::string(64, 'd'),
                    },
            },
    });
    ASSERT_TRUE(begin.has_value());

    ReplicateConfig payload_config;
    payload_config.replica_num = 1;
    payload_config.with_hard_pin = true;
    payload_config.group_ids =
        std::vector<std::string>{begin->manifest.payload_group_id};
    payload_config.residency_affinity_ids =
        std::vector<std::string>{"affinity-b"};
    payload_config.data_type = ObjectDataType::WEIGHT;
    PutCompletedObject(service, context.client_id, "payload-b", payload_config,
                       2048);
    ReplicateConfig manifest_config;
    manifest_config.replica_num = 1;
    manifest_config.with_hard_pin = true;
    manifest_config.group_ids =
        std::vector<std::string>{begin->manifest.payload_group_id};
    manifest_config.data_type = ObjectDataType::METADATA;
    PutCompletedObject(service, context.client_id,
                       MakeWeightManifestKey(target_identity), manifest_config,
                       128);
    auto target = service.CommitWeightImport(CommitWeightImportRequest{
        .identity = target_identity,
        .expected_metadata_generation = begin->metadata_generation,
        .manifest =
            WeightManifestReference{
                .manifest_key = MakeWeightManifestKey(target_identity),
                .manifest_sha256 = std::string(64, 'a'),
                .payload_group_id = begin->manifest.payload_group_id,
                .payload_keys_sha256 =
                    ComputeWeightPayloadKeysSha256({"payload-b"}),
                .payload_count = 1,
                .logical_bytes = 2048,
            },
    });
    ASSERT_TRUE(target.has_value());
    auto committed = service.CommitWeightUpsert(CommitWeightUpsertRequest{
        .request_id = "replace-7-with-8",
        .base_identity = base.identity,
        .target_identity = target_identity,
    });
    ASSERT_TRUE(committed.has_value());
    EXPECT_EQ(8, committed->committed_weight_generation);
    ASSERT_TRUE(committed->latest_claim.has_value());
    EXPECT_EQ(WeightUpsertPhase::RETIRING_BASE, committed->latest_claim->phase);

    auto fenced_policy = service.UpdateWeightPolicy(UpdateWeightPolicyRequest{
        .identity = base.identity,
        .expected_metadata_generation = base.metadata_generation,
        .policy =
            WeightStoragePolicy{
                .preferred_residency = WeightResidencyState::COLD,
                .migration_mode = WeightMigrationMode::MANUAL,
            },
    });
    ASSERT_FALSE(fenced_policy.has_value());
    EXPECT_EQ(WeightManagementError::BUSY, fenced_policy.error());
    auto fenced_migration = service.StartWeightResidencyOperation(
        StartWeightResidencyOperationRequest{
            .identity = base.identity,
            .expected_metadata_generation = base.metadata_generation,
            .target_residency = WeightResidencyState::COLD,
        });
    ASSERT_FALSE(fenced_migration.has_value());
    EXPECT_EQ(WeightManagementError::BUSY, fenced_migration.error());
    ASSERT_TRUE(
        service
            .ReleaseWeightRevisionLease(ReleaseWeightRevisionLeaseRequest{
                .lease_id = retirement_lease->lease_id,
            })
            .has_value());
    committed = service.CommitWeightUpsert(CommitWeightUpsertRequest{
        .request_id = "replace-7-with-8",
        .base_identity = base.identity,
        .target_identity = target_identity,
    });
    ASSERT_TRUE(committed.has_value());
    ASSERT_TRUE(committed->latest_claim.has_value());
    EXPECT_EQ(WeightUpsertPhase::COMPLETED, committed->latest_claim->phase);

    auto successor = target_identity;
    successor.weight_generation = 9;
    auto next_request = BeginWeightUpsertRequest{
        .request_id = "replace-8-with-9",
        .mode = WeightUpsertMode::DELETE_FIRST,
        .base_identity = target_identity,
        .expected_base_metadata_generation = target->metadata_generation + 1,
        .target_identity = successor,
        .import =
            BeginWeightImportRequest{
                .identity = successor,
                .payload_group_id = {},
                .expected_payload_count = 1,
                .expected_logical_bytes = 2048,
                .policy =
                    WeightStoragePolicy{
                        .preferred_residency = WeightResidencyState::HOT,
                        .migration_mode = WeightMigrationMode::MANUAL,
                    },
                .affinity_summary =
                    WeightAffinitySummary{
                        .affinity_count = 1,
                        .affinity_digest = std::string(64, 'e'),
                    },
            },
    };
    auto stale = service.BeginWeightUpsert(next_request);
    ASSERT_FALSE(stale.has_value());
    EXPECT_EQ(WeightManagementError::STALE_GENERATION, stale.error());

    next_request.expected_base_metadata_generation =
        target->metadata_generation;
    auto lease =
        service.AcquireWeightRevisionLease(AcquireWeightRevisionLeaseRequest{
            .identity = target_identity,
            .expected_metadata_generation = target->metadata_generation,
            .holder = "worker-0",
            .ttl_ms = 60'000,
        });
    ASSERT_TRUE(lease.has_value());
    auto busy = service.BeginWeightUpsert(next_request);
    ASSERT_FALSE(busy.has_value());
    EXPECT_EQ(WeightManagementError::BUSY, busy.error());

    ASSERT_TRUE(
        service
            .ReleaseWeightRevisionLease(ReleaseWeightRevisionLeaseRequest{
                .lease_id = lease->lease_id,
            })
            .has_value());
    auto deleted = service.DeleteWeightRevision(DeleteWeightRevisionRequest{
        .identity = target_identity,
        .expected_metadata_generation = target->metadata_generation,
    });
    ASSERT_TRUE(deleted.has_value());
    ASSERT_EQ(WeightAvailabilityState::DELETED, deleted->availability);
    next_request.mode = WeightUpsertMode::PUT_FIRST;
    next_request.expected_base_metadata_generation =
        deleted->metadata_generation;
    auto not_ready = service.BeginWeightUpsert(next_request);
    ASSERT_FALSE(not_ready.has_value());
    EXPECT_EQ(WeightManagementError::NOT_READY, not_ready.error());
}

TEST_F(WeightGroupLifecycleTest,
       ConflictingTargetDoesNotLeaveBlockingLineageClaim) {
    MasterService service;
    const auto context = PrepareSimpleSegment(service);
    auto base = PublishReady(service, context.client_id);
    auto occupied_identity = base.identity;
    occupied_identity.weight_generation = 8;
    PublishReadyWithPayloads(
        service, context.client_id,
        {{.key = "occupied-payload",
          .size = 512,
          .affinity_id = "occupied-affinity"}},
        WeightStoragePolicy{
            .preferred_residency = WeightResidencyState::HOT,
            .mixed_hot_ratio = 0.5,
            .migration_mode = WeightMigrationMode::MANUAL,
        },
        occupied_identity);

    auto request = BeginWeightUpsertRequest{
        .request_id = "conflicting-target",
        .mode = WeightUpsertMode::PUT_FIRST,
        .base_identity = base.identity,
        .expected_base_metadata_generation = base.metadata_generation,
        .target_identity = occupied_identity,
        .import =
            BeginWeightImportRequest{
                .identity = occupied_identity,
                .expected_payload_count = 1,
                .expected_logical_bytes = 1024,
                .policy =
                    WeightStoragePolicy{
                        .preferred_residency = WeightResidencyState::HOT,
                        .mixed_hot_ratio = 0.5,
                        .migration_mode = WeightMigrationMode::MANUAL,
                    },
                .affinity_summary =
                    WeightAffinitySummary{
                        .affinity_count = 1,
                        .affinity_digest = std::string(64, 'd'),
                    },
            },
    };
    auto conflicting = service.BeginWeightUpsert(request);
    ASSERT_FALSE(conflicting.has_value());
    EXPECT_EQ(WeightManagementError::CONFLICT, conflicting.error());
    EXPECT_FALSE(service
                     .GetWeightLineage(GetWeightLineageRequest{
                         .identity = ToWeightLineageIdentity(base.identity),
                     })
                     .has_value());

    request.request_id = "valid-target";
    request.target_identity.weight_generation = 9;
    request.import.identity = request.target_identity;
    auto valid = service.BeginWeightUpsert(request);
    ASSERT_TRUE(valid.has_value())
        << "error=" << static_cast<int>(valid.error());
    EXPECT_EQ(9, valid->identity.weight_generation);
}

TEST_F(WeightGroupLifecycleTest,
       ReconciliationFinishesPutFirstAbortAfterLineagePublishGap) {
    for (const bool finish_target_delete_before_recovery : {false, true}) {
        MasterService service;
        const auto context = PrepareSimpleSegment(service);
        auto base = PublishReady(service, context.client_id);
        auto target_identity = base.identity;
        target_identity.weight_generation = 8;
        auto target = service.BeginWeightUpsert(BeginWeightUpsertRequest{
            .request_id = "interrupted-abort",
            .mode = WeightUpsertMode::PUT_FIRST,
            .base_identity = base.identity,
            .expected_base_metadata_generation = base.metadata_generation,
            .target_identity = target_identity,
            .import =
                BeginWeightImportRequest{
                    .identity = target_identity,
                    .expected_payload_count = 1,
                    .expected_logical_bytes = 1024,
                    .policy =
                        WeightStoragePolicy{
                            .preferred_residency = WeightResidencyState::HOT,
                            .mixed_hot_ratio = 0.5,
                            .migration_mode = WeightMigrationMode::MANUAL,
                        },
                    .affinity_summary =
                        WeightAffinitySummary{
                            .affinity_count = 1,
                            .affinity_digest = std::string(64, 'd'),
                        },
                },
        });
        ASSERT_TRUE(target.has_value());
        auto deleting = service.AbortWeightImport(AbortWeightImportRequest{
            .identity = target_identity,
            .expected_metadata_generation = target->metadata_generation,
        });
        ASSERT_TRUE(deleting.has_value());
        ASSERT_EQ(WeightAvailabilityState::DELETING, deleting->availability);
        if (finish_target_delete_before_recovery) {
            auto deleted = service.ReconcileWeightRevision(
                ReconcileWeightRevisionRequest{.identity = target_identity});
            ASSERT_TRUE(deleted.has_value());
            ASSERT_EQ(WeightAvailabilityState::DELETED, deleted->availability);
        }

        EXPECT_GT(
            MasterServiceTestPeer(service).RunWeightReconciliationForTesting(
                deleting->updated_at_ms + 1, 8),
            0);
        auto deleted = service.GetWeightRevision(
            GetWeightRevisionRequest{.identity = target_identity});
        ASSERT_TRUE(deleted.has_value());
        EXPECT_EQ(WeightAvailabilityState::DELETED,
                  deleted->metadata.availability);
        auto lineage = service.GetWeightLineage(GetWeightLineageRequest{
            .identity = ToWeightLineageIdentity(base.identity),
        });
        ASSERT_TRUE(lineage.has_value());
        ASSERT_TRUE(lineage->latest_claim.has_value());
        EXPECT_EQ(WeightUpsertPhase::ABORTED, lineage->latest_claim->phase);
    }
}

TEST_F(WeightGroupLifecycleTest, InvalidAbortCannotMutateTargetImport) {
    MasterService service;
    const auto context = PrepareSimpleSegment(service);
    auto base = PublishReady(service, context.client_id);
    auto target = base.identity;
    target.weight_generation = 8;
    auto importing = service.BeginWeightUpsert(BeginWeightUpsertRequest{
        .request_id = "replace-7-with-8",
        .mode = WeightUpsertMode::PUT_FIRST,
        .base_identity = base.identity,
        .expected_base_metadata_generation = base.metadata_generation,
        .target_identity = target,
        .import =
            BeginWeightImportRequest{
                .identity = target,
                .payload_group_id = {},
                .expected_payload_count = 1,
                .expected_logical_bytes = 1024,
                .policy =
                    WeightStoragePolicy{
                        .preferred_residency = WeightResidencyState::HOT,
                        .migration_mode = WeightMigrationMode::MANUAL,
                    },
                .affinity_summary =
                    WeightAffinitySummary{
                        .affinity_count = 1,
                        .affinity_digest = std::string(64, 'd'),
                    },
            },
    });
    ASSERT_TRUE(importing.has_value());

    auto rejected = service.AbortWeightUpsert(AbortWeightUpsertRequest{
        .request_id = "wrong-request",
        .base_identity = base.identity,
        .target_identity = target,
    });
    ASSERT_FALSE(rejected.has_value());
    EXPECT_EQ(WeightManagementError::CONFLICT, rejected.error());
    auto target_view =
        service.GetWeightRevision(GetWeightRevisionRequest{.identity = target});
    ASSERT_TRUE(target_view.has_value());
    EXPECT_EQ(WeightAvailabilityState::IMPORTING,
              target_view->metadata.availability);
}

TEST_F(WeightGroupLifecycleTest, ReconciliationRecreatesMissingUpsertTarget) {
    for (const auto phase : {WeightUpsertPhase::PREPARING_TARGET,
                             WeightUpsertPhase::TARGET_IMPORTING}) {
        MasterService service;
        auto base = Identity();
        auto target = base;
        target.weight_generation = 8;
        WeightMetadataSnapshot snapshot;
        snapshot.metadata.push_back(WeightRevisionMetadata{
            .identity = base,
            .manifest =
                WeightManifestReference{
                    .manifest_key = MakeWeightManifestKey(base),
                    .manifest_sha256 = std::string(64, 'a'),
                    .payload_group_id = MakeWeightPayloadGroupId(base),
                    .payload_keys_sha256 = std::string(64, 'b'),
                    .payload_count = 1,
                    .logical_bytes = 1024,
                },
            .policy =
                WeightStoragePolicy{
                    .preferred_residency = WeightResidencyState::HOT,
                    .migration_mode = WeightMigrationMode::MANUAL,
                },
            .availability = phase == WeightUpsertPhase::PREPARING_TARGET
                                ? WeightAvailabilityState::READY
                                : WeightAvailabilityState::DELETED,
            .residency = phase == WeightUpsertPhase::PREPARING_TARGET
                             ? WeightResidencyState::HOT
                             : WeightResidencyState::ABSENT,
            .affinity_count = 1,
            .affinity_digest = std::string(64, 'c'),
            .observed_hot_ratio =
                phase == WeightUpsertPhase::PREPARING_TARGET ? 1.0 : 0.0,
            .metadata_generation =
                phase == WeightUpsertPhase::PREPARING_TARGET ? 2ULL : 4ULL,
            .created_at_ms = 100,
            .updated_at_ms = 100,
            .last_accessed_at_ms = 100,
        });
        snapshot.lineages = std::vector<WeightLineageMetadata>{
            WeightLineageMetadata{
                .identity = ToWeightLineageIdentity(base),
                .lineage_metadata_generation = 1,
                .committed_weight_generation = base.weight_generation,
                .latest_claim =
                    WeightUpsertClaim{
                        .request_id = "replace-7-with-8",
                        .base_identity = base,
                        .target_identity = target,
                        .mode = phase == WeightUpsertPhase::PREPARING_TARGET
                                    ? WeightUpsertMode::PUT_FIRST
                                    : WeightUpsertMode::DELETE_FIRST,
                        .phase = phase,
                        .expected_base_metadata_generation = 2,
                        .import =
                            WeightUpsertImportSummary{
                                .payload_group_id =
                                    MakeWeightPayloadGroupId(target),
                                .expected_payload_count = 1,
                                .expected_logical_bytes = 1024,
                                .policy =
                                    WeightStoragePolicy{
                                        .preferred_residency =
                                            WeightResidencyState::HOT,
                                        .migration_mode =
                                            WeightMigrationMode::MANUAL,
                                    },
                                .affinity_summary =
                                    WeightAffinitySummary{
                                        .affinity_count = 1,
                                        .affinity_digest = std::string(64, 'd'),
                                    },
                            },
                        .created_at_ms = 100,
                        .updated_at_ms = 100,
                    },
            },
        };
        ASSERT_TRUE(service.GetWeightStoreManager()
                        .RestoreSnapshot(snapshot)
                        .has_value());
        EXPECT_EQ(
            1, MasterServiceTestPeer(service).RunWeightReconciliationForTesting(
                   200, 1));
        auto recreated = service.GetWeightRevision(
            GetWeightRevisionRequest{.identity = target});
        ASSERT_TRUE(recreated.has_value());
        EXPECT_EQ(WeightAvailabilityState::IMPORTING,
                  recreated->metadata.availability);
    }
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
        .policy =
            WeightStoragePolicy{
                .preferred_residency = WeightResidencyState::HOT,
                .migration_mode = WeightMigrationMode::MANUAL,
            },
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

TEST_F(WeightGroupLifecycleTest,
       RejectsColdOperationWithoutReadableColdCopies) {
    MasterService service;
    [[maybe_unused]] const auto context = PrepareSimpleSegment(service);
    const auto ready = PublishReady(service, context.client_id);
    const auto started = service.StartWeightResidencyOperation({
        .identity = ready.identity,
        .expected_metadata_generation = ready.metadata_generation,
        .target_residency = WeightResidencyState::COLD,
    });
    ASSERT_FALSE(started.has_value());
    EXPECT_EQ(WeightManagementError::NOT_READY, started.error());
}

TEST_F(WeightGroupLifecycleTest, DeleteCancelsPendingResidencyOperation) {
    MasterService service;
    [[maybe_unused]] const auto context = PrepareSimpleSegment(service);
    const auto ready = PublishReady(service, context.client_id);
    const auto started = service.StartWeightResidencyOperation({
        .identity = ready.identity,
        .expected_metadata_generation = ready.metadata_generation,
        .target_residency = WeightResidencyState::HOT,
    });
    ASSERT_TRUE(started.has_value());
    const auto deleted = service.DeleteWeightRevision({
        .identity = ready.identity,
        .expected_metadata_generation = started->fenced_metadata_generation,
    });
    ASSERT_TRUE(deleted.has_value());
    EXPECT_EQ(WeightAvailabilityState::DELETED, deleted->availability);
    const auto operation = service.QueryWeightOperation({
        .operation_id = started->operation_id,
    });
    ASSERT_TRUE(operation.has_value());
    EXPECT_EQ("cancelled", operation->message);
}

TEST_F(WeightGroupLifecycleTest, DeleteCancelsOperationAfterMemberLoss) {
    MasterService service;
    const auto context = PrepareSimpleSegment(service);
    const auto ready = PublishReady(service, context.client_id);
    const auto started = service.StartWeightResidencyOperation({
        .identity = ready.identity,
        .expected_metadata_generation = ready.metadata_generation,
        .target_residency = WeightResidencyState::HOT,
    });
    ASSERT_TRUE(started);
    ASSERT_TRUE(MasterServiceTestPeer(service).DropWeightGroupMemberForTesting(
        ready.identity, "payload-a"));
    const auto damaged =
        service.ReconcileWeightRevision({.identity = ready.identity});
    ASSERT_TRUE(damaged);
    ASSERT_EQ(WeightAvailabilityState::DEGRADED, damaged->availability);
    const auto stale = service.DeleteWeightRevision({
        .identity = ready.identity,
        .expected_metadata_generation = ready.metadata_generation,
    });
    ASSERT_FALSE(stale);
    EXPECT_EQ(WeightManagementError::STALE_GENERATION, stale.error());
    const auto before =
        service.QueryWeightOperation({.operation_id = started->operation_id});
    ASSERT_TRUE(before);
    const auto deleted = service.DeleteWeightRevision({
        .identity = ready.identity,
        .expected_metadata_generation = damaged->metadata_generation,
    });
    ASSERT_TRUE(deleted);
    EXPECT_EQ(WeightAvailabilityState::DELETED, deleted->availability);
    const auto after =
        service.QueryWeightOperation({.operation_id = started->operation_id});
    ASSERT_TRUE(after);
    EXPECT_EQ("cancelled", after->message);
    EXPECT_EQ(before->target_residency, after->target_residency);
    EXPECT_EQ(before->processed_members, after->processed_members);
    EXPECT_EQ(before->cursor, after->cursor);
    EXPECT_TRUE(service.DeleteWeightRevision({
        .identity = ready.identity,
        .expected_metadata_generation = deleted->metadata_generation,
    }));
}

TEST_F(WeightGroupLifecycleTest, OperationRemainsPendingUntilTargetObserved) {
    MasterService service(MasterServiceConfig::builder()
                              .set_root_fs_dir("/mnt/ssd")
                              .set_default_kv_lease_ttl(60'000)
                              .build());
    [[maybe_unused]] const auto context = PrepareSimpleSegment(service);
    const UUID client_id = generate_uuid();
    auto ready = PublishReady(service, client_id);

    for (const auto& key : {std::string("payload-a"), ManifestKey()}) {
        ASSERT_TRUE(service.PutEnd(client_id, key, TenantId::Default(),
                                   ReplicaType::DISK));
        ASSERT_TRUE(service.GetReplicaList(key, TenantId::Default()));
    }
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
       PublicLeaseAllowsMigrationButPreventsPhysicalDemotion) {
    MasterServiceConfig config;
    config.default_kv_lease_ttl = 0;
    config.enable_offload = true;
    config.offload_on_evict = true;
    MasterService service(config);
    const auto context = PrepareSimpleSegment(service);
    ASSERT_TRUE(service.MountLocalDiskSegment(context.client_id, true));
    auto ready = PublishReady(service, context.client_id);
    AddLocalDiskReplica(service, context.client_id, "payload-a", 1024,
                        "test_segment");
    auto lease =
        service.AcquireWeightRevisionLease(AcquireWeightRevisionLeaseRequest{
            .identity = ready.identity,
            .expected_metadata_generation = ready.metadata_generation,
            .holder = "reader",
            .ttl_ms = 60'000,
        });
    ASSERT_TRUE(lease.has_value());
    auto operation = service.StartWeightResidencyOperation(
        StartWeightResidencyOperationRequest{
            .identity = ready.identity,
            .expected_metadata_generation = ready.metadata_generation,
            .target_residency = WeightResidencyState::COLD,
        });
    ASSERT_TRUE(operation.has_value());
    auto pending = service.ReconcileWeightRevision(
        ReconcileWeightRevisionRequest{.identity = ready.identity});
    ASSERT_TRUE(pending.has_value());
    EXPECT_EQ(WeightResidencyState::HOT, pending->residency);
    EXPECT_TRUE(pending->operation_id.has_value());
    auto payload = service.GetReplicaList("payload-a", TenantId::Default());
    ASSERT_TRUE(payload.has_value());
    bool has_memory = false;
    for (const auto& replica : payload->replicas) {
        has_memory = has_memory || replica.is_memory_replica();
    }
    EXPECT_TRUE(has_memory);
    ASSERT_TRUE(service.ReleaseWeightRevisionLease(
        ReleaseWeightRevisionLeaseRequest{.lease_id = lease->lease_id}));
    auto cold = service.ReconcileWeightRevision(
        ReconcileWeightRevisionRequest{.identity = ready.identity});
    ASSERT_TRUE(cold.has_value());
    EXPECT_EQ(WeightResidencyState::COLD, cold->residency);
    EXPECT_FALSE(cold->operation_id.has_value());
    payload = service.GetReplicaList("payload-a", TenantId::Default());
    ASSERT_TRUE(payload.has_value());
    ASSERT_FALSE(payload->replicas.empty());
    for (const auto& replica : payload->replicas) {
        EXPECT_FALSE(replica.is_memory_replica());
    }
    auto manifest = service.GetReplicaList(
        MakeWeightManifestKey(ready.identity), TenantId::Default());
    ASSERT_TRUE(manifest.has_value());
    bool manifest_has_memory = false;
    for (const auto& replica : manifest->replicas) {
        manifest_has_memory =
            manifest_has_memory || replica.is_memory_replica();
    }
    EXPECT_TRUE(manifest_has_memory);
}

TEST_F(WeightGroupLifecycleTest,
       ReadableColdMembersDetermineDegradedResidency) {
    MasterService service(MasterServiceConfig::builder()
                              .set_default_kv_lease_ttl(0)
                              .set_root_fs_dir("/mnt/ssd")
                              .build());
    const auto context = PrepareSimpleSegment(service);
    const auto ready = PublishReady(service, context.client_id);
    for (const auto& key : {std::string("payload-a"), ManifestKey()}) {
        ASSERT_TRUE(service.PutEnd(context.client_id, key, TenantId::Default(),
                                   ReplicaType::DISK));
    }
    ASSERT_TRUE(service.StartWeightResidencyOperation({
        .identity = ready.identity,
        .expected_metadata_generation = ready.metadata_generation,
        .target_residency = WeightResidencyState::COLD,
    }));
    const auto cold =
        service.ReconcileWeightRevision({.identity = ready.identity});
    ASSERT_TRUE(cold);
    ASSERT_EQ(WeightOperationState::NONE, cold->operation);
    ASSERT_EQ(WeightResidencyState::COLD, cold->residency);
    for (const auto& key : {std::string("payload-a"), ManifestKey()}) {
        {
            MasterServiceTestPeer::MetadataAccessorRW member(
                &service, MasterServiceTestPeer::ObjectIdentity{
                              TenantId::Default(), key});
            ASSERT_TRUE(member.Exists());
            member.Get().VisitReplicas(
                [](const Replica&) { return true; },
                [](Replica& replica) { replica.mark_processing(); });
        }
        const auto observed =
            service.ReconcileWeightRevision({.identity = ready.identity});
        ASSERT_TRUE(observed);
        EXPECT_EQ(WeightAvailabilityState::DEGRADED, observed->availability);
        EXPECT_EQ(key == "payload-a" ? WeightResidencyState::COLD
                                     : WeightResidencyState::ABSENT,
                  observed->residency);
    }
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
       ColdWriteFailureStaysReadyAndOperationRemainsRetryable) {
    MasterServiceConfig config;
    config.default_kv_lease_ttl = 0;
    config.enable_offload = true;
    config.offload_on_evict = true;
    MasterService service(config);
    const auto context = PrepareSimpleSegment(service);
    ASSERT_TRUE(service.MountLocalDiskSegment(context.client_id, true));
    auto ready = PublishReady(service, context.client_id);
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
    auto failed_task = service.OffloadObjectHeartbeat(context.client_id, true);
    ASSERT_TRUE(failed_task.has_value());
    ASSERT_EQ(1u, failed_task->size());
    ASSERT_TRUE(service.NotifyOffloadSuccess(
        context.client_id, *failed_task,
        {StorageObjectMetadata{-1, 0, 0, -1, ""}}));

    auto failed = service.GetWeightRevision(
        GetWeightRevisionRequest{.identity = ready.identity});
    ASSERT_TRUE(failed.has_value());
    EXPECT_EQ(WeightAvailabilityState::READY, failed->metadata.availability);
    EXPECT_EQ(WeightResidencyState::HOT, failed->metadata.residency);
    ASSERT_TRUE(failed->metadata.operation_id.has_value());
    auto failed_operation =
        service.QueryWeightOperation(QueryWeightOperationRequest{
            .operation_id = *failed->metadata.operation_id,
        });
    ASSERT_TRUE(failed_operation.has_value());
    EXPECT_EQ("cold replica write failed", failed_operation->message);
    const auto readable_members = GetWeightGroupResidencyForTest(
        service, ready.identity, ready.manifest.payload_group_id);
    for (const auto& member : readable_members) {
        EXPECT_TRUE(member.has_memory);
    }

    ASSERT_TRUE(service.ReconcileWeightRevision(
        ReconcileWeightRevisionRequest{.identity = ready.identity}));
    auto retry_task = service.OffloadObjectHeartbeat(context.client_id, true);
    ASSERT_TRUE(retry_task.has_value());
    ASSERT_EQ(1u, retry_task->size());
    StorageObjectMetadata disk_metadata;
    disk_metadata.key_size = retry_task->front().key.size();
    disk_metadata.data_size = retry_task->front().size;
    disk_metadata.transport_endpoint = "test_segment";
    ASSERT_TRUE(service.NotifyOffloadSuccess(context.client_id, *retry_task,
                                             {disk_metadata}));
    auto cold = service.ReconcileWeightRevision(
        ReconcileWeightRevisionRequest{.identity = ready.identity});
    ASSERT_TRUE(cold.has_value());
    EXPECT_EQ(WeightResidencyState::COLD, cold->residency);
    EXPECT_FALSE(cold->operation_id.has_value());
}

TEST_F(WeightGroupLifecycleTest,
       AutoAccessPersistsPromotionBeforeAcquiringLease) {
    MasterServiceConfig config;
    config.default_kv_lease_ttl = 0;
    config.enable_offload = true;
    config.offload_on_evict = true;
    config.weight_migration_cooldown_ms = 0;
    MasterService service(config);
    const auto context = PrepareSimpleSegment(service);
    ASSERT_TRUE(service.MountLocalDiskSegment(context.client_id, true));
    auto ready =
        PublishReady(service, context.client_id,
                     WeightStoragePolicy{
                         .preferred_residency = WeightResidencyState::HOT,
                         .mixed_hot_ratio = 0.5,
                         .migration_mode = WeightMigrationMode::AUTO,
                     });
    AddLocalDiskReplica(service, context.client_id, "payload-a", 1024,
                        "test_segment");
    ASSERT_TRUE(service.StartWeightResidencyOperation(
        StartWeightResidencyOperationRequest{
            .identity = ready.identity,
            .expected_metadata_generation = ready.metadata_generation,
            .target_residency = WeightResidencyState::COLD,
            .mixed_hot_ratio = std::nullopt,
        }));
    auto cold = service.ReconcileWeightRevision(
        ReconcileWeightRevisionRequest{.identity = ready.identity});
    ASSERT_TRUE(cold.has_value());
    ASSERT_EQ(WeightResidencyState::COLD, cold->residency);

    auto stale_lease =
        service.AcquireWeightRevisionLease(AcquireWeightRevisionLeaseRequest{
            .identity = cold->identity,
            .expected_metadata_generation = cold->metadata_generation - 1,
            .holder = "stale-reader",
            .ttl_ms = 60'000,
        });
    ASSERT_FALSE(stale_lease.has_value());
    EXPECT_EQ(WeightManagementError::STALE_GENERATION, stale_lease.error());
    auto unchanged = service.GetWeightRevision(
        GetWeightRevisionRequest{.identity = cold->identity});
    ASSERT_TRUE(unchanged.has_value());
    EXPECT_FALSE(unchanged->metadata.operation_id.has_value());

    auto lease =
        service.AcquireWeightRevisionLease(AcquireWeightRevisionLeaseRequest{
            .identity = cold->identity,
            .expected_metadata_generation = cold->metadata_generation,
            .holder = "reader",
            .ttl_ms = 60'000,
        });

    ASSERT_TRUE(lease.has_value());
    EXPECT_EQ(cold->metadata_generation + 1, lease->fenced_metadata_generation);
    auto view = service.GetWeightRevision(
        GetWeightRevisionRequest{.identity = cold->identity});
    ASSERT_TRUE(view.has_value());
    EXPECT_EQ(1, view->active_lease_count);
    EXPECT_EQ(WeightResidencyState::COLD, view->metadata.residency);
    ASSERT_TRUE(view->metadata.operation_id.has_value());
    auto operation = service.QueryWeightOperation(QueryWeightOperationRequest{
        .operation_id = *view->metadata.operation_id,
    });
    ASSERT_TRUE(operation.has_value());
    EXPECT_EQ(WeightResidencyState::HOT, operation->target_residency);
}

TEST_F(WeightGroupLifecycleTest,
       ExistingOffloadReportsFailureToWeightOperation) {
    MasterServiceConfig config;
    config.default_kv_lease_ttl = 0;
    config.enable_offload = true;
    config.offload_on_evict = false;
    MasterService service(config);
    const auto context = PrepareSimpleSegment(service);
    ASSERT_TRUE(service.MountLocalDiskSegment(context.client_id, true));
    auto ready = PublishReady(service, context.client_id);
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

    auto failed_tasks = service.OffloadObjectHeartbeat(context.client_id, true);
    ASSERT_TRUE(failed_tasks.has_value());
    ASSERT_EQ(2u, failed_tasks->size());
    std::vector<StorageObjectMetadata> failures(
        failed_tasks->size(), StorageObjectMetadata{-1, 0, 0, -1, ""});
    ASSERT_TRUE(service.NotifyOffloadSuccess(context.client_id, *failed_tasks,
                                             failures));

    auto operation = service.QueryWeightOperation(
        QueryWeightOperationRequest{.operation_id = started->operation_id});
    ASSERT_TRUE(operation.has_value());
    EXPECT_EQ("cold replica write failed", operation->message);
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

TEST_F(WeightGroupLifecycleTest,
       MigrationBatchLimitsKeepOversizedAffinityWholeAndMakeProgress) {
    MasterServiceConfig config;
    config.default_kv_lease_ttl = 0;
    config.enable_offload = true;
    config.offload_on_evict = true;
    config.weight_migration_max_members_per_round = 1;
    config.weight_migration_max_bytes_per_round = 1000;
    MasterService service(config);
    const auto context = PrepareSimpleSegment(service);
    ASSERT_TRUE(service.MountLocalDiskSegment(context.client_id, true));
    auto ready = PublishReadyWithPayloads(
        service, context.client_id,
        {
            {.key = "payload-a0", .size = 600, .affinity_id = "a"},
            {.key = "payload-a1", .size = 600, .affinity_id = "a"},
            {.key = "payload-b", .size = 500, .affinity_id = "b"},
        });
    for (const auto& [key, size] : std::vector<std::pair<std::string, int64_t>>{
             {"payload-a0", 600},
             {"payload-a1", 600},
             {"payload-b", 500},
         }) {
        AddLocalDiskReplica(service, context.client_id, key, size,
                            "test_segment");
    }
    auto started = service.StartWeightResidencyOperation(
        StartWeightResidencyOperationRequest{
            .identity = ready.identity,
            .expected_metadata_generation = ready.metadata_generation,
            .target_residency = WeightResidencyState::COLD,
            .mixed_hot_ratio = std::nullopt,
        });
    ASSERT_TRUE(started.has_value());

    auto partial = service.ReconcileWeightRevision(
        ReconcileWeightRevisionRequest{.identity = ready.identity});

    ASSERT_TRUE(partial.has_value());
    EXPECT_EQ(WeightResidencyState::MIXED, partial->residency);
    ASSERT_TRUE(partial->operation_id.has_value());
    const auto partial_members = GetWeightGroupResidencyForTest(
        service, ready.identity, ready.manifest.payload_group_id);
    for (const auto& member : partial_members) {
        if (member.residency_affinity_id == "a") {
            EXPECT_FALSE(member.has_memory);
        } else if (member.residency_affinity_id == "b") {
            EXPECT_TRUE(member.has_memory);
        }
    }
    auto operation = service.QueryWeightOperation(QueryWeightOperationRequest{
        .operation_id = started->operation_id,
    });
    ASSERT_TRUE(operation.has_value());
    EXPECT_EQ(1, operation->processed_units);
    EXPECT_EQ(1200, operation->processed_bytes);

    auto cold = service.ReconcileWeightRevision(
        ReconcileWeightRevisionRequest{.identity = ready.identity});
    ASSERT_TRUE(cold.has_value());
    EXPECT_EQ(WeightResidencyState::COLD, cold->residency);
    EXPECT_FALSE(cold->operation_id.has_value());
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

TEST_F(WeightGroupLifecycleTest, UnitBudgetBoundsEvictionAndPromotion) {
    const auto config = MasterServiceConfig::builder()
                            .set_default_kv_lease_ttl(0)
                            .set_enable_offload(true)
                            .set_root_fs_dir("/mnt/ssd")
                            .build();
    MasterService service(config);
    const auto context = PrepareSimpleSegment(service);
    const auto client_id = context.client_id;
    ASSERT_TRUE(service.MountLocalDiskSegment(client_id, true));
    const std::vector<std::string> keys{"unit-a-1", "unit-a-2", "unit-b-1",
                                        "unit-b-2"};
    const auto ready = PublishReadyWithPayloads(service, client_id,
                                                {{keys[0], 1024, "unit-a"},
                                                 {keys[1], 1024, "unit-a"},
                                                 {keys[2], 1024, "unit-b"},
                                                 {keys[3], 1024, "unit-b"}});
    for (const auto& key : keys) {
        ASSERT_TRUE(service.PutEnd(client_id, key, TenantId::Default(),
                                   ReplicaType::DISK));
        AddLocalDiskReplica(service, client_id, key, 1024, "test_segment");
    }
    ASSERT_TRUE(service.StartWeightResidencyOperation({
        .identity = ready.identity,
        .expected_metadata_generation = ready.metadata_generation,
        .target_residency = WeightResidencyState::COLD,
        .mixed_hot_ratio = std::nullopt,
    }));
    auto has_memory = [&](const std::string& key) {
        const auto replicas =
            service.GetReplicaListForAdmin(key, TenantId::Default());
        EXPECT_TRUE(replicas);
        return replicas &&
               std::any_of(replicas->replicas.begin(), replicas->replicas.end(),
                           [](const auto& replica) {
                               return replica.is_memory_replica();
                           });
    };
    MasterServiceTestPeer(service).RunWeightReconciliationForTesting(
        ready.updated_at_ms + 1, 1);
    EXPECT_EQ(2, std::count_if(keys.begin(), keys.end(), has_memory));
    EXPECT_EQ(has_memory(keys[0]), has_memory(keys[1]));
    EXPECT_EQ(has_memory(keys[2]), has_memory(keys[3]));
    EXPECT_TRUE(has_memory(MakeWeightManifestKey(ready.identity)));
    MasterServiceTestPeer(service).RunWeightReconciliationForTesting(
        ready.updated_at_ms + 2, 1);
    const auto cold = service.GetWeightRevision({.identity = ready.identity});
    ASSERT_TRUE(cold);
    ASSERT_EQ(WeightResidencyState::COLD, cold->metadata.residency);
    EXPECT_EQ(0, std::count_if(keys.begin(), keys.end(), has_memory));
    EXPECT_TRUE(has_memory(MakeWeightManifestKey(ready.identity)));
    ASSERT_TRUE(service.StartWeightResidencyOperation({
        .identity = ready.identity,
        .expected_metadata_generation = cold->metadata.metadata_generation,
        .target_residency = WeightResidencyState::HOT,
        .mixed_hot_ratio = std::nullopt,
    }));
    std::string first_affinity;
    for (int i = 0; i < 2; ++i) {
        MasterServiceTestPeer(service).RunWeightReconciliationForTesting(
            ready.updated_at_ms + 3 + i, 1);
        std::vector<PromotionTaskItem> pending;
        for (int heartbeat = 0; heartbeat < 2; ++heartbeat) {
            const auto tasks = service.PromotionObjectHeartbeat(client_id);
            ASSERT_TRUE(tasks);
            ASSERT_EQ(1u, tasks->size());
            pending.push_back(tasks->front());
        }
        const auto extra = service.PromotionObjectHeartbeat(client_id);
        ASSERT_TRUE(extra);
        EXPECT_TRUE(extra->empty());
        const auto affinity = pending.front().key.substr(0, 6);
        EXPECT_EQ(affinity, pending.back().key.substr(0, 6));
        if (i == 0)
            first_affinity = affinity;
        else
            EXPECT_NE(first_affinity, affinity);
        for (const auto& task : pending) {
            ASSERT_TRUE(service.PromotionAllocStart(
                client_id, task.key, TenantId(task.tenant_id), task.size, {}));
            ASSERT_TRUE(service.NotifyPromotionSuccess(
                client_id, task.key, TenantId(task.tenant_id)));
        }
        EXPECT_EQ(static_cast<size_t>((i + 1) * 2),
                  std::count_if(keys.begin(), keys.end(), has_memory));
        EXPECT_TRUE(has_memory(MakeWeightManifestKey(ready.identity)));
    }
    MasterServiceTestPeer(service).RunWeightReconciliationForTesting(
        ready.updated_at_ms + 5, 1);
    const auto hot = service.GetWeightRevision({.identity = ready.identity});
    ASSERT_TRUE(hot);
    EXPECT_EQ(WeightResidencyState::HOT, hot->metadata.residency);
    EXPECT_FALSE(hot->metadata.operation_id.has_value());
}

TEST_F(WeightGroupLifecycleTest, PendingOperationReflectsLostReadableMembers) {
    MasterService service;
    const auto segment = PrepareSimpleSegment(service);
    const auto ready = PublishReady(service, segment.client_id);
    const auto started = service.StartWeightResidencyOperation({
        .identity = ready.identity,
        .expected_metadata_generation = ready.metadata_generation,
        .target_residency = WeightResidencyState::HOT,
    });
    ASSERT_TRUE(started);
    ASSERT_TRUE(service.UnmountSegment(segment.segment_id, segment.client_id));
    for (const auto& key :
         {std::string("payload-a"), MakeWeightManifestKey(ready.identity)}) {
        const auto readable = service.ExistKey(key, TenantId::Default());
        ASSERT_TRUE(readable);
        ASSERT_FALSE(*readable);
    }

    const auto reconciled =
        service.ReconcileWeightRevision({.identity = ready.identity});
    ASSERT_TRUE(reconciled);
    EXPECT_EQ(WeightAvailabilityState::DEGRADED, reconciled->availability);
    EXPECT_EQ(WeightResidencyState::ABSENT, reconciled->residency);
    ASSERT_TRUE(reconciled->operation_id.has_value());
    EXPECT_EQ(started->operation_id, reconciled->operation_id);
    EXPECT_EQ(started->fenced_metadata_generation + 1,
              reconciled->metadata_generation);
    const auto view = service.GetWeightRevision({.identity = ready.identity});
    ASSERT_TRUE(view);
    EXPECT_EQ(*reconciled, view->metadata);
    const auto progress = service.QueryWeightOperation({
        .operation_id = started->operation_id,
    });
    ASSERT_TRUE(progress);
    EXPECT_EQ(WeightOperationKind::MIGRATING, progress->kind);
    EXPECT_EQ(WeightResidencyState::COLD, progress->target_residency);
    EXPECT_EQ(reconciled->metadata_generation,
              progress->fenced_metadata_generation);
    EXPECT_NE("completed", progress->message);

    const auto retry =
        service.ReconcileWeightRevision({.identity = ready.identity});
    ASSERT_TRUE(retry);
    EXPECT_EQ(*reconciled, *retry);
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
    EXPECT_FALSE(service
                     .ExistKey(MakeWeightManifestKey(ready.identity),
                               TenantId::Default())
                     .value_or(false));
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
    EXPECT_TRUE(
        std::any_of(remaining.begin(), remaining.end(), [](const auto& member) {
            return member.data_type == ObjectDataType::METADATA;
        }));

    auto deleted = service.DeleteWeightRevision(DeleteWeightRevisionRequest{
        .identity = ready.identity,
        .expected_metadata_generation = deleting->metadata_generation,
    });
    ASSERT_TRUE(deleted.has_value());
    EXPECT_EQ(WeightAvailabilityState::DELETED, deleted->availability);
    EXPECT_EQ(WeightResidencyState::ABSENT, deleted->residency);
}

TEST_F(WeightGroupLifecycleTest, ExpiredLeaseDeleteSnapshotRemainsRestorable) {
    MasterService service;
    const auto context = PrepareSimpleSegment(service);
    const auto ready = PublishReady(service, context.client_id);
    auto lease =
        service.AcquireWeightRevisionLease(AcquireWeightRevisionLeaseRequest{
            .identity = ready.identity,
            .expected_metadata_generation = ready.metadata_generation,
            .holder = "snapshot-reader",
            .ttl_ms = 5,
        });
    ASSERT_TRUE(lease.has_value());
    std::this_thread::sleep_until(std::chrono::system_clock::time_point{
        std::chrono::milliseconds(lease->expires_at_ms + 1)});

    auto deleted = service.DeleteWeightRevision(DeleteWeightRevisionRequest{
        .identity = ready.identity,
        .expected_metadata_generation = ready.metadata_generation,
    });
    ASSERT_TRUE(deleted.has_value());
    ASSERT_EQ(WeightAvailabilityState::DELETED, deleted->availability);
    const auto snapshot =
        MasterServiceTestPeer::WeightMetadata(service).ExportSnapshot();
    EXPECT_TRUE(snapshot.leases.empty());

    WeightMetadataStore restored;
    auto result = restored.RestoreSnapshot(snapshot);
    EXPECT_TRUE(result.has_value())
        << "A snapshot emitted after a successful public delete must restore";
}

TEST_F(WeightGroupLifecycleTest, CommitRejectsMismatchedAffinityCount) {
    struct Case {
        uint64_t declared_count;
        std::vector<std::string> affinity_ids;
        bool accepted;
    };
    for (const auto& test_case : std::vector<Case>{
             {2, {"shared-affinity", "shared-affinity"}, false},
             {1, {"", ""}, false},
             {2, {"affinity-a", ""}, false},
             {1, {"shared-affinity", "shared-affinity"}, true},
             {2, {"affinity-a", "affinity-b"}, true},
         }) {
        SCOPED_TRACE(test_case.declared_count);
        SCOPED_TRACE(test_case.affinity_ids[0] + "/" +
                     test_case.affinity_ids[1]);
        MasterService service;
        const auto context = PrepareSimpleSegment(service);
        const auto identity = Identity();
        const std::vector<std::string> keys{"payload-a", "payload-b"};
        auto importing = service.BeginWeightImport(BeginWeightImportRequest{
            .identity = identity,
            .payload_group_id = {},
            .expected_payload_count = 2,
            .expected_logical_bytes = 2048,
            .policy =
                WeightStoragePolicy{
                    .preferred_residency = WeightResidencyState::COLD,
                    .mixed_hot_ratio = 0.5,
                    .migration_mode = WeightMigrationMode::MANUAL,
                },
            .affinity_summary =
                WeightAffinitySummary{
                    .affinity_count = test_case.declared_count,
                    .affinity_digest = std::string(64, 'c'),
                },
        });
        ASSERT_TRUE(importing.has_value());
        ReplicateConfig payload_config;
        payload_config.replica_num = 1;
        payload_config.with_hard_pin = true;
        payload_config.group_ids =
            std::vector<std::string>{importing->manifest.payload_group_id};
        payload_config.data_type = ObjectDataType::WEIGHT;
        for (size_t index = 0; index < keys.size(); ++index) {
            payload_config.residency_affinity_ids =
                std::vector<std::string>{test_case.affinity_ids[index]};
            PutCompletedObject(service, context.client_id, keys[index],
                               payload_config, 1024);
        }
        ReplicateConfig manifest_config;
        manifest_config.replica_num = 1;
        manifest_config.with_hard_pin = true;
        manifest_config.group_ids =
            std::vector<std::string>{importing->manifest.payload_group_id};
        manifest_config.data_type = ObjectDataType::METADATA;
        const auto manifest_key = MakeWeightManifestKey(identity);
        PutCompletedObject(service, context.client_id, manifest_key,
                           manifest_config, 128);
        auto committed = service.CommitWeightImport(CommitWeightImportRequest{
            .identity = identity,
            .expected_metadata_generation = importing->metadata_generation,
            .manifest =
                WeightManifestReference{
                    .manifest_key = manifest_key,
                    .manifest_sha256 = std::string(64, 'a'),
                    .payload_group_id = importing->manifest.payload_group_id,
                    .payload_keys_sha256 = ComputeWeightPayloadKeysSha256(keys),
                    .payload_count = 2,
                    .logical_bytes = 2048,
                },
        });
        if (!test_case.accepted) {
            ASSERT_FALSE(committed.has_value());
            EXPECT_EQ(WeightManagementError::CONFLICT, committed.error());
            auto view = service.GetWeightRevision(
                GetWeightRevisionRequest{.identity = identity});
            ASSERT_TRUE(view.has_value());
            EXPECT_EQ(WeightAvailabilityState::IMPORTING,
                      view->metadata.availability);
            EXPECT_FALSE(view->metadata.operation_id.has_value());
            continue;
        }
        ASSERT_TRUE(committed.has_value());
        ASSERT_TRUE(committed->operation_id.has_value());
        for (const auto& key : keys) {
            AddLocalDiskReplica(service, context.client_id, key, 1024,
                                "127.0.0.1:19001");
        }
        auto reconciled = service.ReconcileWeightRevision(
            ReconcileWeightRevisionRequest{.identity = identity});
        ASSERT_TRUE(reconciled.has_value());
        auto operation =
            service.QueryWeightOperation(QueryWeightOperationRequest{
                .operation_id = *committed->operation_id,
            });
        ASSERT_TRUE(operation.has_value());
        EXPECT_EQ("completed", operation->message)
            << "An accepted import must converge after every payload becomes "
               "cold; "
            << "processed=" << operation->processed_units
            << " total=" << operation->total_units;
    }
}

}  // namespace
}  // namespace mooncake::test
