#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#include "weight_catalog.h"

namespace mooncake {
namespace {

WeightRevisionIdentity Identity(std::string revision = "step-100",
                                uint64_t generation = 7) {
    return WeightRevisionIdentity{
        .tenant_id = "tenant-a",
        .name_space = "production",
        .resource_id = "llama-70b",
        .revision = std::move(revision),
        .weight_generation = generation,
    };
}

BeginWeightImportRequest BeginRequest(
    const WeightRevisionIdentity& identity = Identity()) {
    return BeginWeightImportRequest{
        .identity = identity,
        .payload_group_id = "weight-group-7",
        .expected_payload_count = 3,
        .expected_logical_bytes = 4096,
    };
}

WeightManifestReference Manifest() {
    return WeightManifestReference{
        .manifest_key = "weights/production/llama-70b/step-100/7/manifest",
        .manifest_sha256 = std::string(64, 'a'),
        .payload_group_id = "weight-group-7",
        .payload_keys_sha256 = std::string(64, 'b'),
        .payload_count = 3,
        .logical_bytes = 4096,
    };
}

WeightRevisionMetadata PublishBegin(WeightCatalog& catalog,
                                    const BeginWeightImportRequest& request,
                                    uint64_t now_ms = 100) {
    auto candidate = catalog.PrepareBeginImport(request, now_ms);
    EXPECT_TRUE(candidate.has_value());
    auto published = catalog.Publish(*candidate);
    EXPECT_TRUE(published.has_value());
    return *published;
}

WeightRevisionMetadata PublishReady(WeightCatalog& catalog) {
    auto importing = PublishBegin(catalog, BeginRequest());
    auto candidate = catalog.PrepareCommitImport(
        CommitWeightImportRequest{
            .identity = importing.identity,
            .expected_metadata_generation = importing.metadata_generation,
            .manifest = Manifest(),
        },
        200);
    EXPECT_TRUE(candidate.has_value());
    auto published = catalog.Publish(*candidate);
    EXPECT_TRUE(published.has_value());
    return *published;
}

TEST(WeightCatalogTest, BeginIsIdempotentAndRejectsConflicts) {
    WeightCatalog catalog;
    auto first = PublishBegin(catalog, BeginRequest());
    EXPECT_EQ(WeightAvailabilityState::IMPORTING, first.availability);
    EXPECT_EQ(1, first.metadata_generation);

    auto retry = catalog.PrepareBeginImport(BeginRequest(), 150);
    ASSERT_TRUE(retry.has_value());
    EXPECT_TRUE(retry->no_op);
    auto retried = catalog.Publish(*retry);
    ASSERT_TRUE(retried.has_value());
    EXPECT_EQ(first, *retried);

    auto conflicting = BeginRequest();
    conflicting.payload_group_id = "different-group";
    auto rejected = catalog.PrepareBeginImport(conflicting, 160);
    ASSERT_FALSE(rejected.has_value());
    EXPECT_EQ(WeightCatalogError::CONFLICT, rejected.error());
}

TEST(WeightCatalogTest, CommitUsesCasAndIsRetryableAfterResponseLoss) {
    WeightCatalog catalog;
    auto importing = PublishBegin(catalog, BeginRequest());
    auto request = CommitWeightImportRequest{
        .identity = importing.identity,
        .expected_metadata_generation = importing.metadata_generation,
        .manifest = Manifest(),
    };
    auto candidate = catalog.PrepareCommitImport(request, 200);
    ASSERT_TRUE(candidate.has_value());

    auto stale = request;
    stale.expected_metadata_generation = importing.metadata_generation + 1;
    auto rejected = catalog.PrepareCommitImport(stale, 200);
    ASSERT_FALSE(rejected.has_value());
    EXPECT_EQ(WeightCatalogError::STALE_GENERATION, rejected.error());

    auto published = catalog.Publish(*candidate);
    ASSERT_TRUE(published.has_value());
    EXPECT_EQ(WeightAvailabilityState::READY, published->availability);
    EXPECT_EQ(2, published->metadata_generation);

    auto response_lost_retry = catalog.PrepareCommitImport(request, 300);
    ASSERT_TRUE(response_lost_retry.has_value());
    EXPECT_TRUE(response_lost_retry->no_op);
    auto retry_result = catalog.Publish(*response_lost_retry);
    ASSERT_TRUE(retry_result.has_value());
    EXPECT_EQ(*published, *retry_result);
}

TEST(WeightCatalogTest, LookupAndPaginationAreExactAndDeterministic) {
    WeightCatalog catalog;
    for (const auto& [revision, generation] :
         std::vector<std::pair<std::string, uint64_t>>{
             {"step-20", 2}, {"step-10", 3}, {"step-10", 1}}) {
        auto request = BeginRequest(Identity(revision, generation));
        request.payload_group_id = revision + "-" + std::to_string(generation);
        PublishBegin(catalog, request);
    }

    auto exact = catalog.Get(Identity("step-10", 3), 200);
    ASSERT_TRUE(exact.has_value());
    EXPECT_EQ(3, exact->metadata.identity.weight_generation);

    ListWeightRevisionsRequest request{
        .tenant_id = "tenant-a",
        .name_space = "production",
        .resource_id = "llama-70b",
        .page_token = {},
        .limit = 2,
    };
    auto first = catalog.List(request, 200);
    ASSERT_TRUE(first.has_value());
    ASSERT_EQ(2, first->revisions.size());
    EXPECT_EQ("step-10", first->revisions[0].metadata.identity.revision);
    EXPECT_EQ(1, first->revisions[0].metadata.identity.weight_generation);
    EXPECT_EQ("step-10", first->revisions[1].metadata.identity.revision);
    EXPECT_EQ(3, first->revisions[1].metadata.identity.weight_generation);
    ASSERT_FALSE(first->next_page_token.empty());

    request.page_token = first->next_page_token;
    auto second = catalog.List(request, 200);
    ASSERT_TRUE(second.has_value());
    ASSERT_EQ(1, second->revisions.size());
    EXPECT_EQ("step-20", second->revisions[0].metadata.identity.revision);
    EXPECT_TRUE(second->next_page_token.empty());
}

TEST(WeightCatalogTest, LeaseExpiryIsGenerationFencedAndIdempotent) {
    WeightCatalog catalog;
    auto ready = PublishReady(catalog);
    auto candidate = catalog.PrepareAcquireLease(
        AcquireWeightRevisionLeaseRequest{
            .identity = ready.identity,
            .expected_metadata_generation = ready.metadata_generation,
            .holder = "worker-1",
            .ttl_ms = 50,
        },
        300);
    ASSERT_TRUE(candidate.has_value());
    auto lease = catalog.Publish(*candidate);
    ASSERT_TRUE(lease.has_value());
    EXPECT_TRUE(
        catalog.HasActiveLease(ready.identity, ready.metadata_generation, 349));
    EXPECT_FALSE(catalog.HasActiveLease(ready.identity,
                                        ready.metadata_generation + 1, 349));

    auto expired = catalog.PrepareExpireLeases(350);
    ASSERT_EQ(1, expired.size());
    ASSERT_TRUE(catalog.Publish(expired.front()).has_value());
    EXPECT_FALSE(
        catalog.HasActiveLease(ready.identity, ready.metadata_generation, 350));
    EXPECT_TRUE(catalog.PrepareExpireLeases(350).empty());
}

TEST(WeightCatalogTest, ExcludesConcurrentResidencyOperations) {
    WeightCatalog catalog;
    auto ready = PublishReady(catalog);
    auto first = catalog.PrepareStartOperation(
        StartWeightResidencyOperationRequest{
            .identity = ready.identity,
            .expected_metadata_generation = ready.metadata_generation,
            .target_residency = WeightResidencyState::COLD,
        },
        300);
    ASSERT_TRUE(first.has_value());
    auto operation = catalog.Publish(*first);
    ASSERT_TRUE(operation.has_value());
    EXPECT_EQ(WeightOperationState::EVICTING, operation->operation);

    auto conflicting = catalog.PrepareStartOperation(
        StartWeightResidencyOperationRequest{
            .identity = ready.identity,
            .expected_metadata_generation = ready.metadata_generation + 1,
            .target_residency = WeightResidencyState::HOT,
        },
        301);
    ASSERT_FALSE(conflicting.has_value());
    EXPECT_EQ(WeightCatalogError::BUSY, conflicting.error());
}

TEST(WeightCatalogTest, OnlyOneConcurrentCasCandidatePublishes) {
    WeightCatalog catalog;
    auto importing = PublishBegin(catalog, BeginRequest());
    auto request = CommitWeightImportRequest{
        .identity = importing.identity,
        .expected_metadata_generation = importing.metadata_generation,
        .manifest = Manifest(),
    };
    auto first = catalog.PrepareCommitImport(request, 200);
    auto second = catalog.PrepareCommitImport(request, 201);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());

    std::atomic<int> successes{0};
    std::vector<std::thread> threads;
    threads.emplace_back([&] {
        if (catalog.Publish(*first).has_value()) {
            ++successes;
        }
    });
    threads.emplace_back([&] {
        if (catalog.Publish(*second).has_value()) {
            ++successes;
        }
    });
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_EQ(1, successes.load());
}

}  // namespace
}  // namespace mooncake
