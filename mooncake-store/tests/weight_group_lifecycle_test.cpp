#include "master_service_test_fixture.h"

#include <gtest/gtest.h>

#include <string>
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

    WeightRevisionMetadata PublishReady(MasterService& service,
                                        const UUID& client_id) {
        auto importing = service.BeginWeightImport(BeginWeightImportRequest{
            .identity = Identity(),
            .payload_group_id = {},
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
        });
        EXPECT_TRUE(importing.has_value());
        ReplicateConfig config;
        config.replica_num = 1;
        config.with_hard_pin = true;
        config.group_ids =
            std::vector<std::string>{importing->manifest.payload_group_id};
        config.data_type = ObjectDataType::WEIGHT;
        PutCompletedObject(service, client_id, "payload-a", config, 1024);
        config.data_type = ObjectDataType::METADATA;
        PutCompletedObject(service, client_id, ManifestKey(), config, 128);
        auto ready = service.CommitWeightImport(CommitWeightImportRequest{
            .identity = Identity(),
            .expected_metadata_generation = importing->metadata_generation,
            .manifest =
                WeightManifestReference{
                    .manifest_key = ManifestKey(),
                    .manifest_sha256 = std::string(64, 'a'),
                    .payload_group_id = importing->manifest.payload_group_id,
                    .payload_keys_sha256 =
                        ComputeWeightPayloadKeysSha256({"payload-a"}),
                    .payload_count = 1,
                    .logical_bytes = 1024,
                },
        });
        EXPECT_TRUE(ready.has_value());
        return *ready;
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
    auto lease = service.AcquireWeightRevisionLease(
        AcquireWeightRevisionLeaseRequest{
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
        });
    ASSERT_TRUE(operation.has_value());
    auto deleted = service.DeleteWeightRevision(DeleteWeightRevisionRequest{
        .identity = ready.identity,
        .expected_metadata_generation = ready.metadata_generation + 1,
    });
    ASSERT_FALSE(deleted.has_value());
    EXPECT_EQ(WeightManagementError::BUSY, deleted.error());
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
    auto config = MasterServiceConfig::builder()
                      .set_default_kv_lease_ttl(0)
                      .set_root_fs_dir("/mnt/ssd")
                      .build();
    MasterService service(config);
    [[maybe_unused]] const auto context = PrepareSimpleSegment(service);
    const UUID client_id = generate_uuid();
    auto ready = PublishReady(service, client_id);
    ASSERT_TRUE(service
                    .PutEnd(client_id, "payload-a", TenantId::Default(),
                            ReplicaType::DISK)
                    .has_value());
    ASSERT_TRUE(service
                    .PutEnd(client_id, ManifestKey(), TenantId::Default(),
                            ReplicaType::DISK)
                    .has_value());
    auto started = service.StartWeightResidencyOperation(
        StartWeightResidencyOperationRequest{
            .identity = ready.identity,
            .expected_metadata_generation = ready.metadata_generation,
            .target_residency = WeightResidencyState::COLD,
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
}

TEST_F(WeightGroupLifecycleTest, RehydrateQueuesAndCompletesWholeManagedGroup) {
    auto config = MasterServiceConfig::builder()
                      .set_default_kv_lease_ttl(0)
                      .set_enable_offload(true)
                      .set_root_fs_dir("/mnt/ssd")
                      .build();
    MasterService service(config);
    [[maybe_unused]] const auto context = PrepareSimpleSegment(service);
    const UUID client_id = generate_uuid();
    ASSERT_TRUE(service.MountLocalDiskSegment(client_id, true));
    auto ready = PublishReady(service, client_id);
    ASSERT_TRUE(service
                    .PutEnd(client_id, "payload-a", TenantId::Default(),
                            ReplicaType::DISK)
                    .has_value());
    ASSERT_TRUE(service
                    .PutEnd(client_id, ManifestKey(), TenantId::Default(),
                            ReplicaType::DISK)
                    .has_value());
    AddLocalDiskReplica(service, client_id, "payload-a", 1024, "test_segment");
    AddLocalDiskReplica(service, client_id, ManifestKey(), 128, "test_segment");
    ASSERT_TRUE(service.StartWeightResidencyOperation(
        StartWeightResidencyOperationRequest{
            .identity = ready.identity,
            .expected_metadata_generation = ready.metadata_generation,
            .target_residency = WeightResidencyState::COLD,
        }));
    auto cold = service.ReconcileWeightRevision(
        ReconcileWeightRevisionRequest{.identity = ready.identity});
    ASSERT_TRUE(cold.has_value());
    ASSERT_EQ(WeightResidencyState::COLD, cold->residency);

    auto started = service.StartWeightResidencyOperation(
        StartWeightResidencyOperationRequest{
            .identity = cold->identity,
            .expected_metadata_generation = cold->metadata_generation,
            .target_residency = WeightResidencyState::HOT,
        });
    ASSERT_TRUE(started.has_value());
    EXPECT_EQ(WeightOperationKind::MIGRATING, started->kind);
    ASSERT_TRUE(service.ReconcileWeightRevision(
        ReconcileWeightRevisionRequest{.identity = cold->identity}));

    size_t promoted = 0;
    while (promoted < 2) {
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

TEST_F(WeightGroupLifecycleTest, DeleteRemovesPayloadAndManifestThenTombstones) {
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
    EXPECT_FALSE(service.ExistKey("payload-a", TenantId::Default())
                     .value_or(false));
    EXPECT_FALSE(service.ExistKey(ManifestKey(), TenantId::Default())
                     .value_or(false));
}

}  // namespace
}  // namespace mooncake::test
