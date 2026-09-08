#include "weight_catalog.h"

#include <algorithm>
#include <charconv>
#include <limits>
#include <memory>
#include <sstream>
#include <string_view>

#include <openssl/evp.h>

namespace mooncake {
namespace {

constexpr uint32_t kMaxListLimit = 1000;
constexpr uint64_t kMaxLeaseTtlMs = 24ULL * 60 * 60 * 1000;
constexpr char kPageTokenSeparator = '\x1f';

bool SameImport(const WeightRevisionMetadata& metadata,
                const BeginWeightImportRequest& request) {
    return metadata.identity == request.identity &&
           metadata.manifest.payload_group_id == request.payload_group_id &&
           metadata.manifest.payload_count == request.expected_payload_count &&
           metadata.manifest.logical_bytes == request.expected_logical_bytes;
}

bool SameManifest(const WeightRevisionMetadata& metadata,
                  const WeightManifestReference& manifest) {
    return metadata.manifest == manifest;
}

std::string Sha256Hex(const std::vector<std::string_view>& chunks) {
    using Context = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
    Context context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!context ||
        EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
        return {};
    }
    for (const auto chunk : chunks) {
        if (EVP_DigestUpdate(context.get(), chunk.data(), chunk.size()) != 1) {
            return {};
        }
    }
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_size = 0;
    if (EVP_DigestFinal_ex(context.get(), digest, &digest_size) != 1 ||
        digest_size != 32) {
        return {};
    }
    static constexpr char kHex[] = "0123456789abcdef";
    std::string encoded(digest_size * 2, '0');
    for (unsigned int i = 0; i < digest_size; ++i) {
        encoded[2 * i] = kHex[digest[i] >> 4];
        encoded[2 * i + 1] = kHex[digest[i] & 0x0f];
    }
    return encoded;
}

void AppendLengthPrefixed(std::vector<std::string>* storage,
                          std::vector<std::string_view>* chunks,
                          std::string_view value) {
    storage->push_back(std::to_string(value.size()));
    chunks->push_back(storage->back());
    chunks->push_back(":");
    chunks->push_back(value);
    chunks->push_back("\n");
}

}  // namespace

std::string ComputeWeightPayloadKeysSha256(
    const std::vector<std::string>& payload_keys) {
    auto sorted_keys = payload_keys;
    std::sort(sorted_keys.begin(), sorted_keys.end());
    std::vector<std::string> lengths;
    lengths.reserve(sorted_keys.size());
    std::vector<std::string_view> chunks;
    chunks.reserve(sorted_keys.size() * 4);
    for (const auto& key : sorted_keys) {
        AppendLengthPrefixed(&lengths, &chunks, key);
    }
    return Sha256Hex(chunks);
}

std::string MakeWeightPayloadGroupId(const WeightRevisionIdentity& identity) {
    if (!ValidateWeightRevisionIdentity(identity).ok()) {
        return {};
    }
    std::vector<std::string> lengths;
    lengths.reserve(5);
    std::vector<std::string_view> chunks;
    chunks.reserve(20);
    AppendLengthPrefixed(&lengths, &chunks, identity.tenant_id);
    AppendLengthPrefixed(&lengths, &chunks, identity.name_space);
    AppendLengthPrefixed(&lengths, &chunks, identity.resource_id);
    AppendLengthPrefixed(&lengths, &chunks, identity.revision);
    const auto generation = std::to_string(identity.weight_generation);
    AppendLengthPrefixed(&lengths, &chunks, generation);
    const auto digest = Sha256Hex(chunks);
    return digest.empty() ? std::string() : "weight:" + digest;
}

WeightCatalog::Result<WeightCatalogMutation> WeightCatalog::PrepareBeginImport(
    const BeginWeightImportRequest& request, uint64_t now_ms) const {
    if (!ValidateWeightRevisionIdentity(request.identity).ok() ||
        !IsValidWeightComponent(request.payload_group_id) ||
        request.expected_payload_count == 0 ||
        request.expected_logical_bytes == 0) {
        return tl::make_unexpected(WeightCatalogError::INVALID_ARGUMENT);
    }

    std::lock_guard lock(mutex_);
    const auto group = group_index_.find(request.payload_group_id);
    if (group != group_index_.end() && group->second != request.identity) {
        return tl::make_unexpected(WeightCatalogError::CONFLICT);
    }

    const auto current = revisions_.find(request.identity);
    if (current != revisions_.end()) {
        if ((current->second.availability ==
                 WeightAvailabilityState::IMPORTING ||
             current->second.availability == WeightAvailabilityState::READY) &&
            SameImport(current->second, request)) {
            return WeightCatalogMutation{
                .identity = request.identity,
                .previous = current->second,
                .next = current->second,
                .no_op = true,
            };
        }
        return tl::make_unexpected(WeightCatalogError::CONFLICT);
    }

    WeightRevisionMetadata metadata{
        .identity = request.identity,
        .manifest =
            WeightManifestReference{
                .payload_group_id = request.payload_group_id,
                .payload_count = request.expected_payload_count,
                .logical_bytes = request.expected_logical_bytes,
            },
        .availability = WeightAvailabilityState::IMPORTING,
        .residency = WeightResidencyState::UNKNOWN,
        .operation = WeightOperationState::NONE,
        .metadata_generation = 1,
        .created_at_ms = now_ms,
        .updated_at_ms = now_ms,
    };
    return WeightCatalogMutation{
        .identity = request.identity,
        .next = std::move(metadata),
    };
}

WeightCatalog::Result<WeightCatalogMutation> WeightCatalog::PrepareCommitImport(
    const CommitWeightImportRequest& request, uint64_t now_ms) const {
    if (!ValidateWeightRevisionIdentity(request.identity).ok() ||
        !ValidateWeightManifestReference(request.manifest).ok() ||
        request.expected_metadata_generation == 0) {
        return tl::make_unexpected(WeightCatalogError::INVALID_ARGUMENT);
    }

    std::lock_guard lock(mutex_);
    const auto current = revisions_.find(request.identity);
    if (current == revisions_.end()) {
        return tl::make_unexpected(WeightCatalogError::NOT_FOUND);
    }
    if (current->second.availability == WeightAvailabilityState::READY) {
        if (SameManifest(current->second, request.manifest)) {
            return WeightCatalogMutation{
                .identity = request.identity,
                .previous = current->second,
                .next = current->second,
                .no_op = true,
            };
        }
        return tl::make_unexpected(WeightCatalogError::CONFLICT);
    }
    if (current->second.metadata_generation !=
        request.expected_metadata_generation) {
        return tl::make_unexpected(WeightCatalogError::STALE_GENERATION);
    }
    if (current->second.availability != WeightAvailabilityState::IMPORTING) {
        return tl::make_unexpected(WeightCatalogError::CONFLICT);
    }
    if (current->second.manifest.payload_group_id !=
            request.manifest.payload_group_id ||
        current->second.manifest.payload_count !=
            request.manifest.payload_count ||
        current->second.manifest.logical_bytes !=
            request.manifest.logical_bytes) {
        return tl::make_unexpected(WeightCatalogError::CONFLICT);
    }
    if (!CanAdvanceWeightMetadataGeneration(
            current->second.metadata_generation)) {
        return tl::make_unexpected(WeightCatalogError::GENERATION_EXHAUSTED);
    }

    auto next = current->second;
    next.manifest = request.manifest;
    next.availability = WeightAvailabilityState::READY;
    next.residency = WeightResidencyState::HOT;
    ++next.metadata_generation;
    next.updated_at_ms = now_ms;
    return WeightCatalogMutation{
        .identity = request.identity,
        .previous = current->second,
        .next = std::move(next),
    };
}

WeightCatalog::Result<WeightCatalogMutation> WeightCatalog::PrepareAbortImport(
    const AbortWeightImportRequest& request, uint64_t now_ms) const {
    if (!ValidateWeightRevisionIdentity(request.identity).ok() ||
        request.expected_metadata_generation == 0) {
        return tl::make_unexpected(WeightCatalogError::INVALID_ARGUMENT);
    }

    std::lock_guard lock(mutex_);
    const auto current = revisions_.find(request.identity);
    if (current == revisions_.end()) {
        return tl::make_unexpected(WeightCatalogError::NOT_FOUND);
    }
    if (current->second.availability == WeightAvailabilityState::DELETING) {
        return WeightCatalogMutation{
            .identity = request.identity,
            .previous = current->second,
            .next = current->second,
            .no_op = true,
        };
    }
    if (current->second.metadata_generation !=
        request.expected_metadata_generation) {
        return tl::make_unexpected(WeightCatalogError::STALE_GENERATION);
    }
    if (current->second.availability != WeightAvailabilityState::IMPORTING) {
        return tl::make_unexpected(WeightCatalogError::CONFLICT);
    }
    if (!CanAdvanceWeightMetadataGeneration(
            current->second.metadata_generation)) {
        return tl::make_unexpected(WeightCatalogError::GENERATION_EXHAUSTED);
    }

    auto next = current->second;
    next.availability = WeightAvailabilityState::DELETING;
    ++next.metadata_generation;
    next.updated_at_ms = now_ms;
    return WeightCatalogMutation{
        .identity = request.identity,
        .previous = current->second,
        .next = std::move(next),
    };
}

WeightCatalog::Result<WeightRevisionMetadata> WeightCatalog::Publish(
    const WeightCatalogMutation& mutation) {
    std::lock_guard lock(mutex_);
    const auto current = revisions_.find(mutation.identity);
    if (mutation.previous.has_value()) {
        if (current == revisions_.end() ||
            current->second != *mutation.previous) {
            return tl::make_unexpected(WeightCatalogError::STALE_GENERATION);
        }
    } else if (current != revisions_.end()) {
        return tl::make_unexpected(WeightCatalogError::STALE_GENERATION);
    }

    if (mutation.no_op) {
        return current->second;
    }
    if (mutation.kind == WeightCatalogMutationKind::ERASE ||
        !mutation.next.has_value()) {
        if (current == revisions_.end()) {
            return tl::make_unexpected(WeightCatalogError::NOT_FOUND);
        }
        auto removed = current->second;
        group_index_.erase(removed.manifest.payload_group_id);
        revisions_.erase(current);
        return removed;
    }

    const auto& next = *mutation.next;
    if (!ValidateWeightRevisionMetadata(next).ok()) {
        return tl::make_unexpected(WeightCatalogError::INVALID_ARGUMENT);
    }
    const auto group = group_index_.find(next.manifest.payload_group_id);
    if (group != group_index_.end() && group->second != next.identity) {
        return tl::make_unexpected(WeightCatalogError::CONFLICT);
    }
    revisions_[next.identity] = next;
    group_index_[next.manifest.payload_group_id] = next.identity;
    return next;
}

WeightCatalog::Result<WeightRevisionView> WeightCatalog::Get(
    const WeightRevisionIdentity& identity, uint64_t now_ms) const {
    if (!ValidateWeightRevisionIdentity(identity).ok()) {
        return tl::make_unexpected(WeightCatalogError::INVALID_ARGUMENT);
    }
    std::lock_guard lock(mutex_);
    const auto current = revisions_.find(identity);
    if (current == revisions_.end()) {
        return tl::make_unexpected(WeightCatalogError::NOT_FOUND);
    }
    std::optional<uint64_t> nearest;
    const auto count =
        CountActiveLeasesLocked(current->second, now_ms, &nearest);
    return WeightRevisionView{
        .metadata = current->second,
        .active_lease_count = count,
        .nearest_lease_expiry_ms = nearest,
    };
}

WeightCatalog::Result<ListWeightRevisionsResponse> WeightCatalog::List(
    const ListWeightRevisionsRequest& request, uint64_t now_ms) const {
    if (request.tenant_id.empty() || !TenantId(request.tenant_id).IsValid() ||
        !IsValidWeightComponent(request.name_space) ||
        !IsValidWeightComponent(request.resource_id) || request.limit == 0 ||
        request.limit > kMaxListLimit) {
        return tl::make_unexpected(WeightCatalogError::INVALID_ARGUMENT);
    }

    std::optional<std::pair<std::string, uint64_t>> cursor;
    if (!request.page_token.empty()) {
        auto parsed = ParsePageToken(request.page_token);
        if (!parsed.has_value()) {
            return tl::make_unexpected(parsed.error());
        }
        cursor = *parsed;
    }

    std::lock_guard lock(mutex_);
    ListWeightRevisionsResponse response;
    for (const auto& [identity, metadata] : revisions_) {
        if (identity.tenant_id != request.tenant_id ||
            identity.name_space != request.name_space ||
            identity.resource_id != request.resource_id) {
            continue;
        }
        if (cursor.has_value() &&
            std::tie(identity.revision, identity.weight_generation) <=
                std::tie(cursor->first, cursor->second)) {
            continue;
        }
        if (response.revisions.size() == request.limit) {
            response.next_page_token =
                MakePageToken(response.revisions.back().metadata.identity);
            break;
        }
        std::optional<uint64_t> nearest;
        const auto count = CountActiveLeasesLocked(metadata, now_ms, &nearest);
        response.revisions.push_back(WeightRevisionView{
            .metadata = metadata,
            .active_lease_count = count,
            .nearest_lease_expiry_ms = nearest,
        });
    }
    return response;
}

WeightCatalog::Result<WeightLeaseMutation> WeightCatalog::PrepareAcquireLease(
    const AcquireWeightRevisionLeaseRequest& request, uint64_t now_ms) {
    if (!ValidateWeightRevisionIdentity(request.identity).ok() ||
        request.expected_metadata_generation == 0 ||
        !IsValidWeightComponent(request.holder) || request.ttl_ms == 0 ||
        request.ttl_ms > kMaxLeaseTtlMs) {
        return tl::make_unexpected(WeightCatalogError::INVALID_ARGUMENT);
    }

    std::lock_guard lock(mutex_);
    const auto current = revisions_.find(request.identity);
    if (current == revisions_.end()) {
        return tl::make_unexpected(WeightCatalogError::NOT_FOUND);
    }
    if (current->second.metadata_generation !=
        request.expected_metadata_generation) {
        return tl::make_unexpected(WeightCatalogError::STALE_GENERATION);
    }
    if (current->second.availability != WeightAvailabilityState::READY) {
        return tl::make_unexpected(WeightCatalogError::NOT_READY);
    }
    if (current->second.operation != WeightOperationState::NONE) {
        return tl::make_unexpected(WeightCatalogError::BUSY);
    }
    if (next_lease_id_ == 0 ||
        next_lease_id_ == std::numeric_limits<uint64_t>::max()) {
        return tl::make_unexpected(WeightCatalogError::GENERATION_EXHAUSTED);
    }
    const uint64_t lease_id = next_lease_id_++;
    return WeightLeaseMutation{
        .lease_id = lease_id,
        .next =
            WeightRevisionLease{
                .lease_id = lease_id,
                .identity = request.identity,
                .holder = request.holder,
                .expires_at_ms = AddTtl(now_ms, request.ttl_ms),
                .fenced_metadata_generation =
                    request.expected_metadata_generation,
            },
    };
}

WeightCatalog::Result<WeightLeaseMutation> WeightCatalog::PrepareRenewLease(
    const RenewWeightRevisionLeaseRequest& request, uint64_t now_ms) const {
    if (request.lease_id == 0 || request.ttl_ms == 0 ||
        request.ttl_ms > kMaxLeaseTtlMs) {
        return tl::make_unexpected(WeightCatalogError::INVALID_ARGUMENT);
    }
    std::lock_guard lock(mutex_);
    const auto current = leases_.find(request.lease_id);
    if (current == leases_.end()) {
        return tl::make_unexpected(WeightCatalogError::NOT_FOUND);
    }
    if (current->second.expires_at_ms <= now_ms) {
        return tl::make_unexpected(WeightCatalogError::LEASE_EXPIRED);
    }
    auto next = current->second;
    next.expires_at_ms = AddTtl(now_ms, request.ttl_ms);
    return WeightLeaseMutation{
        .lease_id = request.lease_id,
        .previous = current->second,
        .next = std::move(next),
    };
}

WeightCatalog::Result<WeightLeaseMutation> WeightCatalog::PrepareReleaseLease(
    const ReleaseWeightRevisionLeaseRequest& request) const {
    if (request.lease_id == 0) {
        return tl::make_unexpected(WeightCatalogError::INVALID_ARGUMENT);
    }
    std::lock_guard lock(mutex_);
    const auto current = leases_.find(request.lease_id);
    if (current == leases_.end()) {
        return WeightLeaseMutation{
            .kind = WeightCatalogMutationKind::ERASE,
            .lease_id = request.lease_id,
            .no_op = true,
        };
    }
    return WeightLeaseMutation{
        .kind = WeightCatalogMutationKind::ERASE,
        .lease_id = request.lease_id,
        .previous = current->second,
    };
}

std::vector<WeightLeaseMutation> WeightCatalog::PrepareExpireLeases(
    uint64_t now_ms) const {
    std::lock_guard lock(mutex_);
    std::vector<WeightLeaseMutation> expired;
    for (const auto& [lease_id, lease] : leases_) {
        if (lease.expires_at_ms <= now_ms) {
            expired.push_back(WeightLeaseMutation{
                .kind = WeightCatalogMutationKind::ERASE,
                .lease_id = lease_id,
                .previous = lease,
            });
        }
    }
    std::sort(expired.begin(), expired.end(),
              [](const auto& lhs, const auto& rhs) {
                  return lhs.lease_id < rhs.lease_id;
              });
    return expired;
}

WeightCatalog::Result<WeightRevisionLease> WeightCatalog::Publish(
    const WeightLeaseMutation& mutation) {
    std::lock_guard lock(mutex_);
    const auto current = leases_.find(mutation.lease_id);
    if (mutation.no_op) {
        return WeightRevisionLease{.lease_id = mutation.lease_id};
    }
    if (mutation.previous.has_value()) {
        if (current == leases_.end() || current->second != *mutation.previous) {
            return tl::make_unexpected(WeightCatalogError::STALE_GENERATION);
        }
    } else if (current != leases_.end()) {
        return tl::make_unexpected(WeightCatalogError::CONFLICT);
    }

    if (mutation.kind == WeightCatalogMutationKind::ERASE ||
        !mutation.next.has_value()) {
        auto removed = current->second;
        leases_.erase(current);
        return removed;
    }

    const auto& next = *mutation.next;
    const auto revision = revisions_.find(next.identity);
    if (revision == revisions_.end()) {
        return tl::make_unexpected(WeightCatalogError::NOT_FOUND);
    }
    if (revision->second.metadata_generation !=
        next.fenced_metadata_generation) {
        return tl::make_unexpected(WeightCatalogError::STALE_GENERATION);
    }
    if (revision->second.availability != WeightAvailabilityState::READY) {
        return tl::make_unexpected(WeightCatalogError::NOT_READY);
    }
    if (revision->second.operation != WeightOperationState::NONE) {
        return tl::make_unexpected(WeightCatalogError::BUSY);
    }
    leases_[next.lease_id] = next;
    return next;
}

bool WeightCatalog::HasActiveLease(const WeightRevisionIdentity& identity,
                                   uint64_t metadata_generation,
                                   uint64_t now_ms) const {
    std::lock_guard lock(mutex_);
    for (const auto& [lease_id, lease] : leases_) {
        static_cast<void>(lease_id);
        if (lease.identity == identity &&
            lease.fenced_metadata_generation == metadata_generation &&
            lease.expires_at_ms > now_ms) {
            return true;
        }
    }
    return false;
}

WeightCatalog::Result<WeightOperationMutation>
WeightCatalog::PrepareStartOperation(
    const StartWeightResidencyOperationRequest& request, uint64_t now_ms) {
    if (!ValidateWeightRevisionIdentity(request.identity).ok() ||
        request.expected_metadata_generation == 0 ||
        (request.target_residency != WeightResidencyState::HOT &&
         request.target_residency != WeightResidencyState::COLD)) {
        return tl::make_unexpected(WeightCatalogError::INVALID_ARGUMENT);
    }
    std::lock_guard lock(mutex_);
    const auto current = revisions_.find(request.identity);
    if (current == revisions_.end()) {
        return tl::make_unexpected(WeightCatalogError::NOT_FOUND);
    }
    if (current->second.operation != WeightOperationState::NONE) {
        const auto operation = operations_.find(current->second.operation_id);
        if (operation != operations_.end() &&
            operation->second.target_residency == request.target_residency) {
            return WeightOperationMutation{
                .metadata =
                    WeightCatalogMutation{
                        .identity = request.identity,
                        .previous = current->second,
                        .next = current->second,
                        .no_op = true,
                    },
                .previous = operation->second,
                .next = operation->second,
                .no_op = true,
            };
        }
        return tl::make_unexpected(WeightCatalogError::BUSY);
    }
    if (current->second.metadata_generation !=
        request.expected_metadata_generation) {
        return tl::make_unexpected(WeightCatalogError::STALE_GENERATION);
    }
    if (current->second.availability != WeightAvailabilityState::READY &&
        current->second.availability != WeightAvailabilityState::DEGRADED) {
        return tl::make_unexpected(WeightCatalogError::NOT_READY);
    }
    if (!CanAdvanceWeightMetadataGeneration(
            current->second.metadata_generation) ||
        next_operation_id_ == 0 ||
        next_operation_id_ == std::numeric_limits<uint64_t>::max()) {
        return tl::make_unexpected(WeightCatalogError::GENERATION_EXHAUSTED);
    }

    const uint64_t operation_id = next_operation_id_++;
    auto next_metadata = current->second;
    next_metadata.operation = OperationForTarget(request.target_residency);
    next_metadata.operation_id = operation_id;
    ++next_metadata.metadata_generation;
    next_metadata.updated_at_ms = now_ms;
    WeightResidencyOperation next_operation{
        .operation_id = operation_id,
        .identity = request.identity,
        .operation = next_metadata.operation,
        .target_residency = request.target_residency,
        .fenced_metadata_generation = next_metadata.metadata_generation,
        .started_at_ms = now_ms,
        .updated_at_ms = now_ms,
    };
    return WeightOperationMutation{
        .metadata =
            WeightCatalogMutation{
                .identity = request.identity,
                .previous = current->second,
                .next = std::move(next_metadata),
            },
        .next = std::move(next_operation),
    };
}

WeightCatalog::Result<WeightResidencyOperation> WeightCatalog::Publish(
    const WeightOperationMutation& mutation) {
    std::lock_guard lock(mutex_);
    const auto current = revisions_.find(mutation.metadata.identity);
    if (!mutation.metadata.previous.has_value() ||
        current == revisions_.end() ||
        current->second != *mutation.metadata.previous) {
        return tl::make_unexpected(WeightCatalogError::STALE_GENERATION);
    }
    if (mutation.no_op) {
        if (!mutation.next.has_value()) {
            return tl::make_unexpected(WeightCatalogError::NOT_FOUND);
        }
        return *mutation.next;
    }
    if (!mutation.metadata.next.has_value() || !mutation.next.has_value()) {
        return tl::make_unexpected(WeightCatalogError::INVALID_ARGUMENT);
    }
    const auto operation = operations_.find(mutation.next->operation_id);
    if (operation != operations_.end()) {
        return tl::make_unexpected(WeightCatalogError::CONFLICT);
    }
    revisions_[mutation.metadata.identity] = *mutation.metadata.next;
    operations_[mutation.next->operation_id] = *mutation.next;
    return *mutation.next;
}

bool WeightCatalog::IsManagedGroup(const std::string& payload_group_id) const {
    std::lock_guard lock(mutex_);
    return group_index_.contains(payload_group_id);
}

std::string WeightCatalog::MakePageToken(
    const WeightRevisionIdentity& identity) {
    return identity.revision + kPageTokenSeparator +
           std::to_string(identity.weight_generation);
}

WeightCatalog::Result<std::pair<std::string, uint64_t>>
WeightCatalog::ParsePageToken(const std::string& page_token) {
    const auto separator = page_token.rfind(kPageTokenSeparator);
    if (separator == std::string::npos || separator == 0 ||
        separator + 1 == page_token.size()) {
        return tl::make_unexpected(WeightCatalogError::INVALID_ARGUMENT);
    }
    uint64_t generation = 0;
    const auto value = std::string_view(page_token).substr(separator + 1);
    const auto [end, error] =
        std::from_chars(value.data(), value.data() + value.size(), generation);
    if (error != std::errc() || end != value.data() + value.size() ||
        generation == 0) {
        return tl::make_unexpected(WeightCatalogError::INVALID_ARGUMENT);
    }
    return std::make_pair(page_token.substr(0, separator), generation);
}

uint64_t WeightCatalog::AddTtl(uint64_t now_ms, uint64_t ttl_ms) {
    if (ttl_ms > std::numeric_limits<uint64_t>::max() - now_ms) {
        return std::numeric_limits<uint64_t>::max();
    }
    return now_ms + ttl_ms;
}

WeightOperationState WeightCatalog::OperationForTarget(
    WeightResidencyState target) {
    return target == WeightResidencyState::COLD
               ? WeightOperationState::EVICTING
               : WeightOperationState::REHYDRATING;
}

uint64_t WeightCatalog::CountActiveLeasesLocked(
    const WeightRevisionMetadata& metadata, uint64_t now_ms,
    std::optional<uint64_t>* nearest) const {
    uint64_t count = 0;
    for (const auto& [lease_id, lease] : leases_) {
        static_cast<void>(lease_id);
        if (lease.identity != metadata.identity ||
            lease.fenced_metadata_generation != metadata.metadata_generation ||
            lease.expires_at_ms <= now_ms) {
            continue;
        }
        ++count;
        if (!nearest->has_value() || lease.expires_at_ms < **nearest) {
            *nearest = lease.expires_at_ms;
        }
    }
    return count;
}

}  // namespace mooncake
