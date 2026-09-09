#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <ylt/util/tl/expected.hpp>
#include "weight_management.h"

namespace mooncake {

struct WeightAffinityUnit {
    std::string affinity_id;
    uint64_t logical_bytes{0};

    friend bool operator==(const WeightAffinityUnit&,
                           const WeightAffinityUnit&) = default;
};

struct WeightMixedResidencyPlan {
    std::vector<std::string> hot_affinity_ids;
    uint64_t hot_logical_bytes{0};
    uint64_t total_logical_bytes{0};

    friend bool operator==(const WeightMixedResidencyPlan&,
                           const WeightMixedResidencyPlan&) = default;
};

enum class WeightAutoMigrationSignal : uint8_t {
    MEMORY_PRESSURE = 0,
    CAPACITY_AVAILABLE = 1,
    ACCESS = 2,
};

struct WeightAutoMigrationTarget {
    WeightResidencyState residency{WeightResidencyState::UNKNOWN};
    std::optional<double> mixed_hot_ratio;

    friend bool operator==(const WeightAutoMigrationTarget&,
                           const WeightAutoMigrationTarget&) = default;
};

tl::expected<WeightMixedResidencyPlan, WeightManagementError>
PlanMixedWeightResidency(const std::vector<WeightAffinityUnit>& units,
                         double target_hot_ratio);

std::optional<WeightAutoMigrationTarget> PlanAutomaticWeightMigration(
    const WeightRevisionMetadata& metadata, uint64_t active_lease_count,
    WeightAutoMigrationSignal signal, uint64_t now_ms,
    uint64_t cooldown_ms);

bool WeightAutoMigrationCandidateLess(const WeightRevisionMetadata& lhs,
                                      const WeightRevisionMetadata& rhs);

}  // namespace mooncake
