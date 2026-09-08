#pragma once

#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <ylt/util/tl/expected.hpp>

#include "weight_management.h"

namespace mooncake {

enum class WeightCatalogError : uint8_t {
    INVALID_ARGUMENT = 1,
    NOT_FOUND = 2,
    CONFLICT = 3,
    STALE_GENERATION = 4,
    NOT_READY = 5,
    BUSY = 6,
    LEASE_EXPIRED = 7,
    GENERATION_EXHAUSTED = 8,
    DURABILITY_FAILED = 9,
};

enum class WeightCatalogMutationKind : uint8_t {
    UPSERT = 0,
    ERASE = 1,
};

struct WeightCatalogMutation {
    WeightCatalogMutationKind kind{WeightCatalogMutationKind::UPSERT};
    WeightRevisionIdentity identity;
    std::optional<WeightRevisionMetadata> previous;
    std::optional<WeightRevisionMetadata> next;
    bool no_op{false};
};

struct WeightLeaseMutation {
    WeightCatalogMutationKind kind{WeightCatalogMutationKind::UPSERT};
    uint64_t lease_id{0};
    std::optional<WeightRevisionLease> previous;
    std::optional<WeightRevisionLease> next;
    bool no_op{false};
};

struct WeightOperationMutation {
    WeightCatalogMutation metadata;
    std::optional<WeightResidencyOperation> previous;
    std::optional<WeightResidencyOperation> next;
    bool no_op{false};
};

struct WeightCatalogSnapshot {
    uint32_t schema_version{1};
    std::vector<WeightRevisionMetadata> metadata;
    std::vector<WeightRevisionLease> leases;
    std::vector<WeightResidencyOperation> operations;
    uint64_t next_lease_id{1};
    uint64_t next_operation_id{1};

    friend bool operator==(const WeightCatalogSnapshot&,
                           const WeightCatalogSnapshot&) = default;
};
YLT_REFL(WeightCatalogSnapshot, schema_version, metadata, leases, operations,
         next_lease_id, next_operation_id);

class WeightCatalog {
   public:
    template <typename T>
    using Result = tl::expected<T, WeightCatalogError>;

    Result<WeightCatalogMutation> PrepareBeginImport(
        const BeginWeightImportRequest& request, uint64_t now_ms) const;
    Result<WeightCatalogMutation> PrepareCommitImport(
        const CommitWeightImportRequest& request, uint64_t now_ms) const;
    Result<WeightCatalogMutation> PrepareAbortImport(
        const AbortWeightImportRequest& request, uint64_t now_ms) const;
    Result<WeightRevisionMetadata> Publish(
        const WeightCatalogMutation& mutation);

    Result<WeightRevisionView> Get(const WeightRevisionIdentity& identity,
                                   uint64_t now_ms) const;
    Result<ListWeightRevisionsResponse> List(
        const ListWeightRevisionsRequest& request, uint64_t now_ms) const;

    Result<WeightLeaseMutation> PrepareAcquireLease(
        const AcquireWeightRevisionLeaseRequest& request, uint64_t now_ms);
    Result<WeightLeaseMutation> PrepareRenewLease(
        const RenewWeightRevisionLeaseRequest& request, uint64_t now_ms) const;
    Result<WeightLeaseMutation> PrepareReleaseLease(
        const ReleaseWeightRevisionLeaseRequest& request) const;
    std::vector<WeightLeaseMutation> PrepareExpireLeases(uint64_t now_ms) const;
    Result<WeightRevisionLease> Publish(const WeightLeaseMutation& mutation);
    bool HasActiveLease(const WeightRevisionIdentity& identity,
                        uint64_t metadata_generation, uint64_t now_ms) const;

    Result<WeightOperationMutation> PrepareStartOperation(
        const StartWeightResidencyOperationRequest& request, uint64_t now_ms);
    Result<WeightResidencyOperation> Publish(
        const WeightOperationMutation& mutation);

    bool IsManagedGroup(const std::string& payload_group_id) const;
    WeightCatalogSnapshot ExportSnapshot() const;
    Result<void> RestoreSnapshot(const WeightCatalogSnapshot& snapshot);
    void Clear();

   private:
    static std::string MakePageToken(const WeightRevisionIdentity& identity);
    static Result<std::pair<std::string, uint64_t>> ParsePageToken(
        const std::string& page_token);
    static uint64_t AddTtl(uint64_t now_ms, uint64_t ttl_ms);
    static WeightOperationState OperationForTarget(WeightResidencyState target);

    uint64_t CountActiveLeasesLocked(const WeightRevisionMetadata& metadata,
                                     uint64_t now_ms,
                                     std::optional<uint64_t>* nearest) const;

    mutable std::mutex mutex_;
    std::map<WeightRevisionIdentity, WeightRevisionMetadata> revisions_;
    std::map<std::string, WeightRevisionIdentity> group_index_;
    std::unordered_map<uint64_t, WeightRevisionLease> leases_;
    std::unordered_map<uint64_t, WeightResidencyOperation> operations_;
    uint64_t next_lease_id_{1};
    uint64_t next_operation_id_{1};
};

}  // namespace mooncake
