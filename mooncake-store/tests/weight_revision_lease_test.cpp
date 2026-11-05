#include <gtest/gtest.h>

#include <string>

#include "weight_metadata_store.h"

namespace mooncake::test {
namespace {

WeightRevisionIdentity Identity() {
    return WeightRevisionIdentity{
        .tenant_id = "default",
        .name_space = "production",
        .resource_id = "llama-70b",
        .revision = "step-100",
        .weight_generation = 7,
    };
}

WeightRevisionMetadata PublishReady(WeightMetadataStore& metadata_store,
                                    uint64_t now_ms) {
    const auto identity = Identity();
    const auto group_id = MakeWeightPayloadGroupId(identity);
    auto begin = metadata_store.PrepareBeginImport(
        BeginWeightImportRequest{
            .identity = identity,
            .payload_group_id = group_id,
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
                        "weights/production/llama-70b/step-100/7/manifest",
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

TEST(WeightRevisionLeaseTest, AcquireRetryRenewAndReleaseAreIdempotent) {
    WeightMetadataStore metadata_store;
    const auto ready = PublishReady(metadata_store, 100);
    const AcquireWeightRevisionLeaseRequest request{
        .identity = ready.identity,
        .expected_metadata_generation = ready.metadata_generation,
        .holder = "worker-0",
        .ttl_ms = 1000,
    };

    auto acquire = metadata_store.PrepareAcquireLease(request, 200);
    ASSERT_TRUE(acquire.has_value());
    auto lease = metadata_store.Publish(*acquire);
    ASSERT_TRUE(lease.has_value());

    auto retry = metadata_store.PrepareAcquireLease(request, 201);
    ASSERT_TRUE(retry.has_value());
    EXPECT_TRUE(retry->no_op);
    auto retried_lease = metadata_store.Publish(*retry);
    ASSERT_TRUE(retried_lease.has_value());
    EXPECT_EQ(*lease, *retried_lease);

    auto renew = metadata_store.PrepareRenewLease(
        RenewWeightRevisionLeaseRequest{
            .lease_id = lease->lease_id,
            .ttl_ms = 2000,
        },
        300);
    ASSERT_TRUE(renew.has_value());
    auto renewed = metadata_store.Publish(*renew);
    ASSERT_TRUE(renewed.has_value());
    EXPECT_EQ(2300u, renewed->expires_at_ms);

    auto release = metadata_store.PrepareReleaseLease(
        ReleaseWeightRevisionLeaseRequest{.lease_id = lease->lease_id});
    ASSERT_TRUE(release.has_value());
    ASSERT_TRUE(metadata_store.Publish(*release).has_value());
    auto release_retry = metadata_store.PrepareReleaseLease(
        ReleaseWeightRevisionLeaseRequest{.lease_id = lease->lease_id});
    ASSERT_TRUE(release_retry.has_value());
    EXPECT_TRUE(release_retry->no_op);
    EXPECT_TRUE(metadata_store.Publish(*release_retry).has_value());
}

TEST(WeightRevisionLeaseTest, RejectsStaleGenerationAndExpiredRenewal) {
    WeightMetadataStore metadata_store;
    const auto ready = PublishReady(metadata_store, 100);
    auto stale = metadata_store.PrepareAcquireLease(
        AcquireWeightRevisionLeaseRequest{
            .identity = ready.identity,
            .expected_metadata_generation = ready.metadata_generation - 1,
            .holder = "worker-0",
            .ttl_ms = 1000,
        },
        200);
    ASSERT_FALSE(stale.has_value());
    EXPECT_EQ(WeightManagementError::STALE_GENERATION, stale.error());

    auto acquire = metadata_store.PrepareAcquireLease(
        AcquireWeightRevisionLeaseRequest{
            .identity = ready.identity,
            .expected_metadata_generation = ready.metadata_generation,
            .holder = "worker-0",
            .ttl_ms = 100,
        },
        200);
    ASSERT_TRUE(acquire.has_value());
    auto lease = metadata_store.Publish(*acquire);
    ASSERT_TRUE(lease.has_value());
    auto expired = metadata_store.PrepareRenewLease(
        RenewWeightRevisionLeaseRequest{
            .lease_id = lease->lease_id,
            .ttl_ms = 100,
        },
        300);
    ASSERT_FALSE(expired.has_value());
    EXPECT_EQ(WeightManagementError::LEASE_EXPIRED, expired.error());
}

TEST(WeightRevisionLeaseTest, OlderFenceProtectsAdvancedMetadataGeneration) {
    WeightMetadataStore metadata_store;
    auto ready = PublishReady(metadata_store, 100);
    auto snapshot = metadata_store.ExportSnapshot();
    snapshot.metadata.front().metadata_generation = 3;
    snapshot.metadata.front().updated_at_ms = 300;
    snapshot.leases.push_back(WeightRevisionLease{
        .lease_id = 1,
        .identity = ready.identity,
        .holder = "worker-0",
        .expires_at_ms = 1000,
        .fenced_metadata_generation = 2,
    });
    snapshot.next_lease_id = 2;
    ASSERT_TRUE(metadata_store.RestoreSnapshot(snapshot).has_value());

    auto view = metadata_store.Get(ready.identity, 500);
    ASSERT_TRUE(view.has_value());
    EXPECT_EQ(1u, view->active_lease_count);
    EXPECT_TRUE(metadata_store.HasActiveLease(ready.identity, 3, 500));
}

TEST(WeightRevisionLeaseTest, PutFirstFencesBaseAfterTargetCommit) {
    WeightMetadataStore metadata_store;
    const auto ready = PublishReady(metadata_store, 100);
    auto lease_mutation = metadata_store.PrepareAcquireLease(
        AcquireWeightRevisionLeaseRequest{
            .identity = ready.identity,
            .expected_metadata_generation = ready.metadata_generation,
            .holder = "worker-0",
            .ttl_ms = 1000,
        },
        200);
    ASSERT_TRUE(lease_mutation.has_value());
    auto lease = metadata_store.Publish(*lease_mutation);
    ASSERT_TRUE(lease.has_value());

    auto target = ready.identity;
    target.weight_generation = 8;
    const auto target_group = MakeWeightPayloadGroupId(target);
    const BeginWeightUpsertRequest upsert{
        .request_id = "request-1",
        .mode = WeightUpsertMode::PUT_FIRST,
        .base_identity = ready.identity,
        .expected_base_metadata_generation = ready.metadata_generation,
        .target_identity = target,
        .import =
            BeginWeightImportRequest{
                .identity = target,
                .payload_group_id = target_group,
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
    };
    auto claim = metadata_store.PrepareBeginUpsert(upsert, 210);
    ASSERT_TRUE(claim.has_value());
    ASSERT_TRUE(metadata_store.Publish(*claim).has_value());

    auto still_allowed = metadata_store.PrepareAcquireLease(
        AcquireWeightRevisionLeaseRequest{
            .identity = ready.identity,
            .expected_metadata_generation = ready.metadata_generation,
            .holder = "worker-1",
            .ttl_ms = 1000,
        },
        220);
    ASSERT_TRUE(still_allowed.has_value());

    auto target_begin = metadata_store.PrepareBeginImport(upsert.import, 230);
    ASSERT_TRUE(target_begin.has_value());
    auto importing = metadata_store.Publish(*target_begin);
    ASSERT_TRUE(importing.has_value());
    auto target_commit = metadata_store.PrepareCommitImport(
        CommitWeightImportRequest{
            .identity = target,
            .expected_metadata_generation = importing->metadata_generation,
            .manifest =
                WeightManifestReference{
                    .manifest_key =
                        "weights/production/llama-70b/step-100/8/manifest",
                    .manifest_sha256 = std::string(64, 'a'),
                    .payload_group_id = target_group,
                    .payload_keys_sha256 = std::string(64, 'b'),
                    .payload_count = 1,
                    .logical_bytes = 1024,
                },
        },
        240);
    ASSERT_TRUE(target_commit.has_value());
    ASSERT_TRUE(metadata_store.Publish(*target_commit).has_value());
    auto retire = metadata_store.PrepareCommitUpsert(
        CommitWeightUpsertRequest{
            .request_id = "request-1",
            .base_identity = ready.identity,
            .target_identity = target,
        },
        250);
    ASSERT_TRUE(retire.has_value());
    ASSERT_TRUE(metadata_store.Publish(*retire).has_value());

    auto fenced_acquire = metadata_store.PrepareAcquireLease(
        AcquireWeightRevisionLeaseRequest{
            .identity = ready.identity,
            .expected_metadata_generation = ready.metadata_generation,
            .holder = "worker-2",
            .ttl_ms = 1000,
        },
        260);
    ASSERT_FALSE(fenced_acquire.has_value());
    EXPECT_EQ(WeightManagementError::BUSY, fenced_acquire.error());
    auto fenced_renew = metadata_store.PrepareRenewLease(
        RenewWeightRevisionLeaseRequest{
            .lease_id = lease->lease_id,
            .ttl_ms = 1000,
        },
        260);
    ASSERT_FALSE(fenced_renew.has_value());
    EXPECT_EQ(WeightManagementError::BUSY, fenced_renew.error());

    auto fenced_policy = metadata_store.PrepareUpdatePolicy(
        UpdateWeightPolicyRequest{
            .identity = ready.identity,
            .expected_metadata_generation = ready.metadata_generation,
            .policy =
                WeightStoragePolicy{
                    .preferred_residency = WeightResidencyState::COLD,
                    .migration_mode = WeightMigrationMode::MANUAL,
                },
        },
        260);
    ASSERT_FALSE(fenced_policy.has_value());
    EXPECT_EQ(WeightManagementError::BUSY, fenced_policy.error());
    auto fenced_migration = metadata_store.PrepareStartOperation(
        StartWeightResidencyOperationRequest{
            .identity = ready.identity,
            .expected_metadata_generation = ready.metadata_generation,
            .target_residency = WeightResidencyState::COLD,
        },
        260);
    ASSERT_FALSE(fenced_migration.has_value());
    EXPECT_EQ(WeightManagementError::BUSY, fenced_migration.error());
}

}  // namespace
}  // namespace mooncake::test
