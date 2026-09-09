#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "weight_residency_planner.h"

namespace mooncake::test {
namespace {

TEST(WeightResidencyPlannerTest, PicksCompleteAffinitiesClosestToMixedTarget) {
    const std::vector<WeightAffinityUnit> units{
        {.affinity_id = "a", .logical_bytes = 500},
        {.affinity_id = "b", .logical_bytes = 400},
        {.affinity_id = "c", .logical_bytes = 100},
    };

    auto plan = PlanMixedWeightResidency(units, 0.5);

    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(500, plan->hot_logical_bytes);
    EXPECT_EQ(std::vector<std::string>{"a"}, plan->hot_affinity_ids);
}

TEST(WeightResidencyPlannerTest, IsDeterministicAcrossInputOrder) {
    const std::vector<WeightAffinityUnit> first{
        {.affinity_id = "a", .logical_bytes = 600},
        {.affinity_id = "b", .logical_bytes = 400},
        {.affinity_id = "c", .logical_bytes = 200},
    };
    const std::vector<WeightAffinityUnit> second{first.rbegin(), first.rend()};

    auto first_plan = PlanMixedWeightResidency(first, 0.5);
    auto second_plan = PlanMixedWeightResidency(second, 0.5);

    ASSERT_TRUE(first_plan.has_value());
    ASSERT_TRUE(second_plan.has_value());
    EXPECT_EQ(*first_plan, *second_plan);
}

TEST(WeightResidencyPlannerTest, ImprovesBySwappingWholeAffinities) {
    const std::vector<WeightAffinityUnit> units{
        {.affinity_id = "two", .logical_bytes = 2},
        {.affinity_id = "three", .logical_bytes = 3},
        {.affinity_id = "four", .logical_bytes = 4},
    };

    auto plan = PlanMixedWeightResidency(units, 0.6);

    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(5, plan->hot_logical_bytes);
    EXPECT_EQ((std::vector<std::string>{"three", "two"}),
              plan->hot_affinity_ids);
}

TEST(WeightResidencyPlannerTest, RejectsMixedForSingleAffinity) {
    auto plan = PlanMixedWeightResidency(
        {{.affinity_id = "only", .logical_bytes = 1024}}, 0.5);

    ASSERT_FALSE(plan.has_value());
    EXPECT_EQ(WeightManagementError::POLICY_UNSATISFIABLE, plan.error());
}

WeightRevisionMetadata AutoMetadata(WeightResidencyState residency) {
    WeightRevisionMetadata metadata;
    metadata.policy = WeightStoragePolicy{
        .preferred_residency = WeightResidencyState::HOT,
        .mixed_hot_ratio = 0.5,
        .migration_mode = WeightMigrationMode::AUTO,
    };
    metadata.availability = WeightAvailabilityState::READY;
    metadata.residency = residency;
    metadata.affinity_count = 2;
    metadata.observed_hot_ratio =
        residency == WeightResidencyState::HOT ? 1.0 : 0.0;
    metadata.metadata_generation = 2;
    metadata.created_at_ms = 100;
    metadata.updated_at_ms = 100;
    metadata.last_accessed_at_ms = 100;
    return metadata;
}

TEST(WeightResidencyPlannerTest, AutoPressureStepsTowardCold) {
    auto hot = AutoMetadata(WeightResidencyState::HOT);
    auto mixed = AutoMetadata(WeightResidencyState::MIXED);

    EXPECT_EQ((WeightAutoMigrationTarget{
                  .residency = WeightResidencyState::MIXED,
                  .mixed_hot_ratio = 0.5,
              }),
              PlanAutomaticWeightMigration(
                  hot, 0, WeightAutoMigrationSignal::MEMORY_PRESSURE, 200, 50));
    EXPECT_EQ((WeightAutoMigrationTarget{
                  .residency = WeightResidencyState::COLD,
                  .mixed_hot_ratio = std::nullopt,
              }),
              PlanAutomaticWeightMigration(
                  mixed, 0, WeightAutoMigrationSignal::MEMORY_PRESSURE, 200,
                  50));

    hot.affinity_count = 1;
    EXPECT_EQ((WeightAutoMigrationTarget{
                  .residency = WeightResidencyState::COLD,
                  .mixed_hot_ratio = std::nullopt,
              }),
              PlanAutomaticWeightMigration(
                  hot, 0, WeightAutoMigrationSignal::MEMORY_PRESSURE, 200, 50));
}

TEST(WeightResidencyPlannerTest, AutoAccessPromotesTowardPreferred) {
    auto cold = AutoMetadata(WeightResidencyState::COLD);
    auto mixed = AutoMetadata(WeightResidencyState::MIXED);
    mixed.observed_hot_ratio = 0.5;

    EXPECT_EQ((WeightAutoMigrationTarget{
                  .residency = WeightResidencyState::HOT,
                  .mixed_hot_ratio = std::nullopt,
              }),
              PlanAutomaticWeightMigration(
                  cold, 0, WeightAutoMigrationSignal::ACCESS, 200, 50));
    EXPECT_EQ((WeightAutoMigrationTarget{
                  .residency = WeightResidencyState::HOT,
                  .mixed_hot_ratio = std::nullopt,
              }),
              PlanAutomaticWeightMigration(
                  cold, 1, WeightAutoMigrationSignal::ACCESS, 200, 50));
    EXPECT_EQ((WeightAutoMigrationTarget{
                  .residency = WeightResidencyState::HOT,
                  .mixed_hot_ratio = std::nullopt,
              }),
              PlanAutomaticWeightMigration(
                  mixed, 0, WeightAutoMigrationSignal::CAPACITY_AVAILABLE, 200,
                  50));
}

TEST(WeightResidencyPlannerTest, AutoDecisionEnforcesEligibilityAndCooldown) {
    auto metadata = AutoMetadata(WeightResidencyState::HOT);
    EXPECT_FALSE(PlanAutomaticWeightMigration(
                     metadata, 1, WeightAutoMigrationSignal::MEMORY_PRESSURE,
                     200, 50)
                     .has_value());
    EXPECT_FALSE(PlanAutomaticWeightMigration(
                     metadata, 0, WeightAutoMigrationSignal::MEMORY_PRESSURE,
                     149, 50)
                     .has_value());
    metadata.last_accessed_at_ms = 190;
    EXPECT_FALSE(PlanAutomaticWeightMigration(
                     metadata, 0, WeightAutoMigrationSignal::MEMORY_PRESSURE,
                     200, 50)
                     .has_value());
    metadata.last_accessed_at_ms = 100;

    metadata.policy.migration_mode = WeightMigrationMode::MANUAL;
    EXPECT_FALSE(PlanAutomaticWeightMigration(
                     metadata, 0, WeightAutoMigrationSignal::MEMORY_PRESSURE,
                     200, 50)
                     .has_value());
    metadata.policy.migration_mode = WeightMigrationMode::AUTO;
    metadata.operation_id = 9;
    EXPECT_FALSE(PlanAutomaticWeightMigration(
                     metadata, 0, WeightAutoMigrationSignal::MEMORY_PRESSURE,
                     200, 50)
                     .has_value());
    metadata.operation_id.reset();
    metadata.availability = WeightAvailabilityState::DEGRADED;
    EXPECT_FALSE(PlanAutomaticWeightMigration(
                     metadata, 0, WeightAutoMigrationSignal::MEMORY_PRESSURE,
                     200, 50)
                     .has_value());
}

TEST(WeightResidencyPlannerTest, AutoCandidatesUsePersistentAccessOrder) {
    auto older = AutoMetadata(WeightResidencyState::HOT);
    auto newer = older;
    older.identity.revision = "older";
    newer.identity.revision = "newer";
    older.last_accessed_at_ms = 200;
    newer.last_accessed_at_ms = 300;
    older.updated_at_ms = 400;
    newer.updated_at_ms = 100;

    EXPECT_TRUE(WeightAutoMigrationCandidateLess(older, newer));
    EXPECT_FALSE(WeightAutoMigrationCandidateLess(newer, older));

    newer.last_accessed_at_ms = older.last_accessed_at_ms;
    older.manifest.logical_bytes = 1024;
    newer.manifest.logical_bytes = 2048;
    EXPECT_TRUE(WeightAutoMigrationCandidateLess(newer, older));
}

}  // namespace
}  // namespace mooncake::test
