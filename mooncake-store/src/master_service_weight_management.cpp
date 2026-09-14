#include "master_service.h"

#include <map>

#include "weight_residency_planner.h"

namespace mooncake {

MasterService::ObjectOperationLock
MasterService::AcquireWeightGroupOperationLock(const TenantId& tenant_id,
                                               const std::string& group_id) {
    const auto scoped_group = tenant_id.MakeScopedKey(group_id);
    const auto stripe_idx =
        std::hash<std::string>{}(scoped_group) % kObjectOperationLockStripes;
    return {std::unique_lock<std::mutex>(
        weight_group_operation_locks_[stripe_idx])};
}

MasterService::ObjectOperationLock
MasterService::AcquireWeightLineageOperationLock(
    const WeightLineageIdentity& identity) {
    const auto key = identity.tenant_id + "\x1f" + identity.name_space +
                     "\x1f" + identity.resource_id + "\x1f" + identity.revision;
    const auto stripe_idx =
        std::hash<std::string>{}(key) % kObjectOperationLockStripes;
    return {std::unique_lock<std::mutex>(
        weight_lineage_operation_locks_[stripe_idx])};
}

WeightMetadataStore::Result<WeightRevisionMetadata>
MasterService::BeginWeightImport(const BeginWeightImportRequest& request) {
    auto normalized = request;
    const auto canonical_group = MakeWeightPayloadGroupId(request.identity);
    if (canonical_group.empty() ||
        (!request.payload_group_id.empty() &&
         request.payload_group_id != canonical_group)) {
        return tl::make_unexpected(WeightManagementError::INVALID_ARGUMENT);
    }
    normalized.payload_group_id = canonical_group;
    if (!normalized.policy.has_value()) {
        normalized.policy = default_weight_storage_policy_;
    }
    const auto now_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    auto mutation = weight_metadata_.PrepareBeginImport(normalized, now_ms);
    if (!mutation) {
        return tl::make_unexpected(mutation.error());
    }
    return PersistAndPublishWeightMutation(*mutation);
}

WeightMetadataStore::Result<WeightRevisionMetadata>
MasterService::CommitWeightImport(const CommitWeightImportRequest& request) {
    const auto canonical_group = MakeWeightPayloadGroupId(request.identity);
    if (canonical_group.empty() ||
        request.manifest.payload_group_id != canonical_group) {
        return tl::make_unexpected(WeightManagementError::INVALID_ARGUMENT);
    }
    [[maybe_unused]] auto group_operation_lock =
        AcquireWeightGroupOperationLock(TenantId(request.identity.tenant_id),
                                        canonical_group);
    const auto now_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());

    auto mutation = weight_metadata_.PrepareCommitImport(request, now_ms);
    if (!mutation) {
        return tl::make_unexpected(mutation.error());
    }
    if (mutation->no_op) {
        return PersistAndPublishWeightMutation(*mutation);
    }
    auto validation = ValidateWeightGroupForCommit(request);
    if (!validation) {
        return tl::make_unexpected(validation.error());
    }

    mutation = weight_metadata_.PrepareCommitImport(request, now_ms);
    if (!mutation) {
        return tl::make_unexpected(mutation.error());
    }
    return PersistAndPublishWeightMutation(*mutation);
}

WeightMetadataStore::Result<WeightRevisionMetadata>
MasterService::AbortWeightImport(const AbortWeightImportRequest& request) {
    const auto now_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    auto mutation = weight_metadata_.PrepareAbortImport(request, now_ms);
    if (!mutation) {
        return tl::make_unexpected(mutation.error());
    }
    return PersistAndPublishWeightMutation(*mutation);
}

WeightMetadataStore::Result<WeightRevisionView>
MasterService::GetWeightRevision(
    const GetWeightRevisionRequest& request) const {
    const auto now_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    return weight_metadata_.Get(request.identity, now_ms);
}

WeightMetadataStore::Result<ListWeightRevisionsResponse>
MasterService::ListWeightRevisions(
    const ListWeightRevisionsRequest& request) const {
    const auto now_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    return weight_metadata_.List(request, now_ms);
}

WeightMetadataStore::Result<WeightRevisionMetadata>
MasterService::UpdateWeightPolicy(const UpdateWeightPolicyRequest& request) {
    const auto canonical_group = MakeWeightPayloadGroupId(request.identity);
    if (canonical_group.empty()) {
        return tl::make_unexpected(WeightManagementError::INVALID_ARGUMENT);
    }
    [[maybe_unused]] auto lineage_operation_lock =
        AcquireWeightLineageOperationLock(
            ToWeightLineageIdentity(request.identity));
    [[maybe_unused]] auto group_operation_lock =
        AcquireWeightGroupOperationLock(TenantId(request.identity.tenant_id),
                                        canonical_group);
    const auto now_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    auto mutation = weight_metadata_.PrepareUpdatePolicy(request, now_ms);
    if (!mutation) {
        return tl::make_unexpected(mutation.error());
    }
    return PersistAndPublishWeightMutation(*mutation);
}

WeightMetadataStore::Result<WeightRevisionMetadata>
MasterService::BeginWeightUpsert(const BeginWeightUpsertRequest& request) {
    auto normalized = request;
    const auto canonical_group =
        MakeWeightPayloadGroupId(request.target_identity);
    if (canonical_group.empty() ||
        request.import.identity != request.target_identity ||
        (!request.import.payload_group_id.empty() &&
         request.import.payload_group_id != canonical_group)) {
        return tl::make_unexpected(WeightManagementError::INVALID_ARGUMENT);
    }
    normalized.import.payload_group_id = canonical_group;
    if (!normalized.import.policy.has_value()) {
        normalized.import.policy = default_weight_storage_policy_;
    }
    [[maybe_unused]] auto lineage_lock = AcquireWeightLineageOperationLock(
        ToWeightLineageIdentity(request.base_identity));
    const auto now_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());

    auto existing = weight_metadata_.GetLineage(
        ToWeightLineageIdentity(request.base_identity));
    const bool same_request =
        existing && existing->latest_claim.has_value() &&
        existing->latest_claim->request_id == request.request_id &&
        existing->latest_claim->base_identity == request.base_identity &&
        existing->latest_claim->target_identity == request.target_identity;
    if (!same_request) {
        auto base = weight_metadata_.Get(request.base_identity, now_ms);
        if (!base) {
            return tl::make_unexpected(base.error());
        }
        if (base->metadata.metadata_generation !=
            request.expected_base_metadata_generation) {
            return tl::make_unexpected(WeightManagementError::STALE_GENERATION);
        }
        if (base->metadata.availability != WeightAvailabilityState::READY) {
            return tl::make_unexpected(WeightManagementError::NOT_READY);
        }
        if (request.mode == WeightUpsertMode::DELETE_FIRST &&
            (base->active_lease_count != 0 ||
             base->metadata.operation_id.has_value())) {
            return tl::make_unexpected(WeightManagementError::BUSY);
        }
    }
    auto import = weight_metadata_.PrepareBeginImport(
        normalized.import, now_ms,
        request.mode == WeightUpsertMode::DELETE_FIRST);
    if (!import) {
        return tl::make_unexpected(import.error());
    }
    auto claim = weight_metadata_.PrepareBeginUpsert(normalized, now_ms);
    if (!claim) {
        return tl::make_unexpected(claim.error());
    }
    if (!weight_management_mutations_enabled_ ||
        !weight_lineage_mutations_enabled_) {
        return tl::make_unexpected(WeightManagementError::DURABILITY_FAILED);
    }
    auto lineage = PersistAndPublishWeightLineageMutation(*claim);
    if (!lineage) {
        return tl::make_unexpected(lineage.error());
    }

    if (lineage->latest_claim->mode == WeightUpsertMode::DELETE_FIRST &&
        lineage->latest_claim->phase == WeightUpsertPhase::DELETING_BASE) {
        auto base = weight_metadata_.Get(request.base_identity, now_ms);
        if (base &&
            base->metadata.availability != WeightAvailabilityState::DELETED) {
            auto deleted = DeleteWeightRevisionInternal(
                DeleteWeightRevisionRequest{
                    .identity = request.base_identity,
                    .expected_metadata_generation =
                        base->metadata.metadata_generation,
                },
                true);
            if (!deleted ||
                deleted->availability != WeightAvailabilityState::DELETED) {
                return tl::make_unexpected(WeightManagementError::BUSY);
            }
        }
        auto advanced = weight_metadata_.PrepareAdvanceUpsert(
            lineage->identity, request.request_id,
            WeightUpsertPhase::TARGET_IMPORTING, now_ms);
        if (!advanced) {
            return tl::make_unexpected(advanced.error());
        }
        lineage = PersistAndPublishWeightLineageMutation(*advanced);
        if (!lineage) {
            return tl::make_unexpected(lineage.error());
        }
    }
    import = weight_metadata_.PrepareBeginImport(
        normalized.import, now_ms,
        lineage->latest_claim->mode == WeightUpsertMode::DELETE_FIRST &&
            lineage->latest_claim->phase ==
                WeightUpsertPhase::TARGET_IMPORTING);
    if (!import) {
        return tl::make_unexpected(import.error());
    }
    return PersistAndPublishWeightMutation(*import);
}

WeightMetadataStore::Result<WeightLineageMetadata>
MasterService::CommitWeightUpsert(const CommitWeightUpsertRequest& request) {
    [[maybe_unused]] auto lineage_lock = AcquireWeightLineageOperationLock(
        ToWeightLineageIdentity(request.base_identity));
    const auto now_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    auto mutation = weight_metadata_.PrepareCommitUpsert(request, now_ms);
    if (!mutation) {
        return tl::make_unexpected(mutation.error());
    }
    if (!weight_management_mutations_enabled_ ||
        !weight_lineage_mutations_enabled_) {
        return tl::make_unexpected(WeightManagementError::DURABILITY_FAILED);
    }
    auto lineage = PersistAndPublishWeightLineageMutation(*mutation);
    if (!lineage || !lineage->latest_claim.has_value() ||
        lineage->latest_claim->phase != WeightUpsertPhase::RETIRING_BASE) {
        return lineage;
    }
    auto base = weight_metadata_.Get(request.base_identity, now_ms);
    if (base && base->active_lease_count == 0 &&
        !base->metadata.operation_id.has_value()) {
        auto deleted = DeleteWeightRevisionInternal(
            DeleteWeightRevisionRequest{
                .identity = request.base_identity,
                .expected_metadata_generation =
                    base->metadata.metadata_generation,
            },
            true);
        if (deleted &&
            deleted->availability == WeightAvailabilityState::DELETED) {
            auto completed = weight_metadata_.PrepareAdvanceUpsert(
                lineage->identity, request.request_id,
                WeightUpsertPhase::COMPLETED, now_ms);
            if (completed) {
                return PersistAndPublishWeightLineageMutation(*completed);
            }
        }
    }
    return lineage;
}

WeightMetadataStore::Result<WeightLineageMetadata>
MasterService::AbortWeightUpsert(const AbortWeightUpsertRequest& request) {
    [[maybe_unused]] auto lineage_lock = AcquireWeightLineageOperationLock(
        ToWeightLineageIdentity(request.base_identity));
    const auto now_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    auto mutation = weight_metadata_.PrepareAbortUpsert(request, now_ms);
    if (!mutation) {
        return tl::make_unexpected(mutation.error());
    }
    if (!weight_management_mutations_enabled_ ||
        !weight_lineage_mutations_enabled_) {
        return tl::make_unexpected(WeightManagementError::DURABILITY_FAILED);
    }
    auto target = weight_metadata_.Get(request.target_identity, now_ms);
    if (target &&
        target->metadata.availability == WeightAvailabilityState::IMPORTING) {
        auto aborted = AbortWeightImport(AbortWeightImportRequest{
            .identity = request.target_identity,
            .expected_metadata_generation =
                target->metadata.metadata_generation,
        });
        if (!aborted) {
            return tl::make_unexpected(aborted.error());
        }
    }
    return PersistAndPublishWeightLineageMutation(*mutation);
}

WeightMetadataStore::Result<WeightLineageMetadata>
MasterService::GetWeightLineage(const GetWeightLineageRequest& request) const {
    return weight_metadata_.GetLineage(request.identity);
}

WeightMetadataStore::Result<WeightRevisionLease>
MasterService::AcquireWeightRevisionLease(
    const AcquireWeightRevisionLeaseRequest& request) {
    const auto canonical_group = MakeWeightPayloadGroupId(request.identity);
    if (canonical_group.empty()) {
        return tl::make_unexpected(WeightManagementError::INVALID_ARGUMENT);
    }
    [[maybe_unused]] auto lineage_operation_lock =
        AcquireWeightLineageOperationLock(
            ToWeightLineageIdentity(request.identity));
    [[maybe_unused]] auto group_operation_lock =
        AcquireWeightGroupOperationLock(TenantId(request.identity.tenant_id),
                                        canonical_group);
    const auto now_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    auto normalized = request;
    auto current = weight_metadata_.Get(request.identity, now_ms);
    if (!weight_metadata_.IsWeightRevisionMutationFenced(request.identity) &&
        current &&
        current->metadata.metadata_generation ==
            request.expected_metadata_generation) {
        auto target = PlanAutomaticWeightMigration(
            current->metadata, current->active_lease_count,
            WeightAutoMigrationSignal::ACCESS, now_ms,
            weight_migration_cooldown_ms_);
        if (target.has_value()) {
            auto started = StartWeightResidencyOperationLocked(
                StartWeightResidencyOperationRequest{
                    .identity = request.identity,
                    .expected_metadata_generation =
                        current->metadata.metadata_generation,
                    .target_residency = target->residency,
                    .mixed_hot_ratio = target->mixed_hot_ratio,
                },
                now_ms);
            if (!started) {
                return tl::make_unexpected(started.error());
            }
            normalized.expected_metadata_generation =
                started->fenced_metadata_generation;
        }
    }
    auto mutation = weight_metadata_.PrepareAcquireLease(normalized, now_ms);
    if (!mutation) {
        return tl::make_unexpected(mutation.error());
    }
    return PersistAndPublishWeightLeaseMutation(*mutation);
}

WeightMetadataStore::Result<WeightRevisionLease>
MasterService::RenewWeightRevisionLease(
    const RenewWeightRevisionLeaseRequest& request) {
    auto current_lease = weight_metadata_.GetLease(request.lease_id);
    if (!current_lease) {
        return tl::make_unexpected(current_lease.error());
    }
    [[maybe_unused]] auto lineage_operation_lock =
        AcquireWeightLineageOperationLock(
            ToWeightLineageIdentity(current_lease->identity));
    const auto canonical_group =
        MakeWeightPayloadGroupId(current_lease->identity);
    if (canonical_group.empty()) {
        return tl::make_unexpected(WeightManagementError::INVALID_ARGUMENT);
    }
    [[maybe_unused]] auto group_operation_lock =
        AcquireWeightGroupOperationLock(
            TenantId(current_lease->identity.tenant_id), canonical_group);
    const auto now_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    auto mutation = weight_metadata_.PrepareRenewLease(request, now_ms);
    if (!mutation) {
        return tl::make_unexpected(mutation.error());
    }
    return PersistAndPublishWeightLeaseMutation(*mutation);
}

WeightMetadataStore::Result<void> MasterService::ReleaseWeightRevisionLease(
    const ReleaseWeightRevisionLeaseRequest& request) {
    auto mutation = weight_metadata_.PrepareReleaseLease(request);
    if (!mutation) {
        return tl::make_unexpected(mutation.error());
    }
    if (!mutation->no_op) {
        const auto canonical_group =
            MakeWeightPayloadGroupId(mutation->previous->identity);
        if (canonical_group.empty()) {
            return tl::make_unexpected(WeightManagementError::INVALID_ARGUMENT);
        }
        [[maybe_unused]] auto group_operation_lock =
            AcquireWeightGroupOperationLock(
                TenantId(mutation->previous->identity.tenant_id),
                canonical_group);
        mutation = weight_metadata_.PrepareReleaseLease(request);
        if (!mutation) {
            return tl::make_unexpected(mutation.error());
        }
    }
    auto released = PersistAndPublishWeightLeaseMutation(*mutation);
    if (!released) {
        return tl::make_unexpected(released.error());
    }
    return {};
}

WeightMetadataStore::Result<WeightResidencyOperation>
MasterService::StartWeightResidencyOperation(
    const StartWeightResidencyOperationRequest& request) {
    const auto canonical_group = MakeWeightPayloadGroupId(request.identity);
    if (canonical_group.empty()) {
        return tl::make_unexpected(WeightManagementError::INVALID_ARGUMENT);
    }
    [[maybe_unused]] auto lineage_operation_lock =
        AcquireWeightLineageOperationLock(
            ToWeightLineageIdentity(request.identity));
    [[maybe_unused]] auto group_operation_lock =
        AcquireWeightGroupOperationLock(TenantId(request.identity.tenant_id),
                                        canonical_group);
    const auto now_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    return StartWeightResidencyOperationLocked(request, now_ms);
}

WeightMetadataStore::Result<WeightResidencyOperation>
MasterService::StartWeightResidencyOperationLocked(
    const StartWeightResidencyOperationRequest& request, uint64_t now_ms) {
    auto mutation = weight_metadata_.PrepareStartOperation(request, now_ms);
    if (!mutation) {
        return tl::make_unexpected(mutation.error());
    }
    if (!mutation->no_op) {
        auto members = SnapshotWeightGroup(
            request.identity,
            mutation->metadata.next->manifest.payload_group_id);
        if (!members ||
            members->size() !=
                mutation->metadata.next->manifest.payload_count + 1) {
            return tl::make_unexpected(WeightManagementError::NOT_READY);
        }
        if (request.target_residency == WeightResidencyState::MIXED) {
            std::map<std::string, uint64_t> affinity_bytes;
            for (const auto& member : *members) {
                if (member.data_type == ObjectDataType::WEIGHT) {
                    affinity_bytes[member.residency_affinity_id] += member.size;
                }
            }
            std::vector<WeightAffinityUnit> units;
            units.reserve(affinity_bytes.size());
            for (const auto& [affinity_id, logical_bytes] : affinity_bytes) {
                units.push_back(WeightAffinityUnit{
                    .affinity_id = affinity_id,
                    .logical_bytes = logical_bytes,
                });
            }
            auto plan =
                PlanMixedWeightResidency(units, *request.mixed_hot_ratio);
            if (!plan) {
                return tl::make_unexpected(plan.error());
            }
        }
    }
    return PersistAndPublishWeightOperationMutation(*mutation);
}

WeightMetadataStore::Result<WeightResidencyOperation>
MasterService::QueryWeightOperation(
    const QueryWeightOperationRequest& request) const {
    if (request.tenant_id.empty() || !TenantId(request.tenant_id).IsValid()) {
        return tl::make_unexpected(WeightManagementError::INVALID_ARGUMENT);
    }
    auto operation = weight_metadata_.QueryOperation(request.operation_id);
    if (!operation || operation->identity.tenant_id == request.tenant_id) {
        return operation;
    }
    return tl::make_unexpected(WeightManagementError::NOT_FOUND);
}

WeightMetadataStore::Result<void> MasterService::ValidateWeightGroupForCommit(
    const CommitWeightImportRequest& request) const {
    const auto expected_manifest_key = MakeWeightManifestKey(request.identity);
    if (request.manifest.manifest_key != expected_manifest_key) {
        return tl::make_unexpected(WeightManagementError::CONFLICT);
    }

    auto members = SnapshotWeightGroup(request.identity,
                                       request.manifest.payload_group_id);
    if (!members) {
        return tl::make_unexpected(members.error());
    }
    if (members->size() != request.manifest.payload_count + 1) {
        return tl::make_unexpected(WeightManagementError::CONFLICT);
    }

    bool found_manifest = false;
    uint64_t logical_bytes = 0;
    std::vector<std::string> payload_keys;
    payload_keys.reserve(request.manifest.payload_count);
    for (const auto& member : *members) {
        if (!member.readable) {
            return tl::make_unexpected(WeightManagementError::NOT_READY);
        }
        if (member.key == request.manifest.manifest_key) {
            if (found_manifest ||
                member.data_type != ObjectDataType::METADATA) {
                return tl::make_unexpected(WeightManagementError::CONFLICT);
            }
            found_manifest = true;
            continue;
        }
        if (member.data_type != ObjectDataType::WEIGHT ||
            member.size >
                std::numeric_limits<uint64_t>::max() - logical_bytes) {
            return tl::make_unexpected(WeightManagementError::CONFLICT);
        }
        logical_bytes += member.size;
        payload_keys.push_back(member.key);
    }
    if (!found_manifest) {
        return tl::make_unexpected(WeightManagementError::NOT_FOUND);
    }
    if (logical_bytes != request.manifest.logical_bytes ||
        payload_keys.size() != request.manifest.payload_count ||
        ComputeWeightPayloadKeysSha256(payload_keys) !=
            request.manifest.payload_keys_sha256) {
        return tl::make_unexpected(WeightManagementError::CONFLICT);
    }
    return {};
}

WeightMetadataStore::Result<WeightRevisionMetadata>
MasterService::PersistAndPublishWeightMutation(
    const WeightMetadataMutation& mutation) {
    if (!mutation.no_op && !weight_management_mutations_enabled_) {
        return tl::make_unexpected(WeightManagementError::DURABILITY_FAILED);
    }
    if (mutation.no_op || !enable_oplog_) {
        return weight_metadata_.Publish(mutation);
    }

    OpType type;
    std::string payload;
    if (mutation.kind == WeightMetadataMutationKind::UPSERT &&
        mutation.next.has_value()) {
        type = OpType::WEIGHT_METADATA_UPSERT;
        const auto encoded = struct_pack::serialize(WeightMetadataUpsertOp{
            .metadata = *mutation.next,
            .operation = mutation.operation,
        });
        payload.assign(encoded.begin(), encoded.end());
    } else if (mutation.previous.has_value()) {
        type = OpType::WEIGHT_METADATA_DELETE;
        WeightMetadataDeleteOp deletion{
            .identity = mutation.identity,
            .metadata_generation = mutation.previous->metadata_generation,
        };
        const auto encoded = struct_pack::serialize(deletion);
        payload.assign(encoded.begin(), encoded.end());
    } else {
        return tl::make_unexpected(WeightManagementError::INVALID_ARGUMENT);
    }

    struct Completion {
        std::mutex mutex;
        std::condition_variable cv;
        std::optional<WeightMetadataStore::Result<WeightRevisionMetadata>>
            result;
    };
    auto completion = std::make_shared<Completion>();
    auto persisted = AppendOpLogWithDurableFinalize(
        type, mutation.identity.tenant_id,
        MakeWeightRevisionMetadataKey(mutation.identity), payload,
        [this, mutation, completion](const OpLogEntry& durable_entry) {
            auto published = weight_metadata_.Publish(mutation);
            if (!published) {
                LOG(ERROR) << "Failed to publish durable weight metadata "
                              "mutation, sequence_id="
                           << durable_entry.sequence_id
                           << ", key=" << durable_entry.object_key
                           << ", error=" << static_cast<int>(published.error());
            }
            {
                std::lock_guard lock(completion->mutex);
                completion->result = std::move(published);
            }
            completion->cv.notify_all();
        });
    if (!persisted) {
        LOG(ERROR) << "Failed to persist weight metadata mutation, key="
                   << MakeWeightRevisionMetadataKey(mutation.identity)
                   << ", error=" << static_cast<int>(persisted.error());
        return tl::make_unexpected(WeightManagementError::DURABILITY_FAILED);
    }

    constexpr auto kPublishTimeout = std::chrono::seconds(30);
    std::unique_lock lock(completion->mutex);
    if (!completion->cv.wait_for(lock, kPublishTimeout, [&] {
            return completion->result.has_value();
        })) {
        LOG(ERROR) << "Timed out waiting for durable weight metadata publish, "
                   << "sequence_id=" << persisted->sequence_id
                   << ", key=" << persisted->object_key;
        return tl::make_unexpected(WeightManagementError::DURABILITY_FAILED);
    }
    return std::move(*completion->result);
}

WeightMetadataStore::Result<WeightLineageMetadata>
MasterService::PersistAndPublishWeightLineageMutation(
    const WeightLineageMutation& mutation) {
    if (!mutation.no_op && !weight_lineage_mutations_enabled_) {
        return tl::make_unexpected(WeightManagementError::DURABILITY_FAILED);
    }
    if (mutation.no_op || !enable_oplog_) {
        return weight_metadata_.Publish(mutation);
    }
    const auto encoded =
        struct_pack::serialize(WeightLineageUpsertOp{.lineage = mutation.next});
    const std::string payload(encoded.begin(), encoded.end());
    struct Completion {
        std::mutex mutex;
        std::condition_variable cv;
        std::optional<WeightMetadataStore::Result<WeightLineageMetadata>>
            result;
    };
    auto completion = std::make_shared<Completion>();
    auto persisted = AppendOpLogWithDurableFinalize(
        OpType::WEIGHT_LINEAGE_UPSERT, mutation.identity.tenant_id,
        MakeWeightLineageMetadataKey(mutation.identity), payload,
        [this, mutation, completion](const OpLogEntry&) {
            auto published = weight_metadata_.Publish(mutation);
            {
                std::lock_guard lock(completion->mutex);
                completion->result = std::move(published);
            }
            completion->cv.notify_all();
        });
    if (!persisted) {
        return tl::make_unexpected(WeightManagementError::DURABILITY_FAILED);
    }
    constexpr auto kPublishTimeout = std::chrono::seconds(30);
    std::unique_lock lock(completion->mutex);
    if (!completion->cv.wait_for(lock, kPublishTimeout, [&] {
            return completion->result.has_value();
        })) {
        return tl::make_unexpected(WeightManagementError::DURABILITY_FAILED);
    }
    return std::move(*completion->result);
}

WeightMetadataStore::Result<WeightResidencyOperation>
MasterService::PersistAndPublishWeightOperationMutation(
    const WeightOperationMutation& mutation) {
    if (!mutation.no_op && !weight_management_mutations_enabled_) {
        return tl::make_unexpected(WeightManagementError::DURABILITY_FAILED);
    }
    if (mutation.no_op || !enable_oplog_) {
        return weight_metadata_.Publish(mutation);
    }
    if (!mutation.metadata.next.has_value() || !mutation.next.has_value()) {
        return tl::make_unexpected(WeightManagementError::INVALID_ARGUMENT);
    }

    const auto encoded = struct_pack::serialize(WeightMetadataUpsertOp{
        .metadata = *mutation.metadata.next,
        .operation = *mutation.next,
    });
    const std::string payload(encoded.begin(), encoded.end());

    struct Completion {
        std::mutex mutex;
        std::condition_variable cv;
        std::optional<WeightMetadataStore::Result<WeightResidencyOperation>>
            result;
    };
    auto completion = std::make_shared<Completion>();
    auto persisted = AppendOpLogWithDurableFinalize(
        OpType::WEIGHT_METADATA_UPSERT, mutation.metadata.identity.tenant_id,
        MakeWeightRevisionMetadataKey(mutation.metadata.identity), payload,
        [this, mutation, completion](const OpLogEntry& durable_entry) {
            auto published = weight_metadata_.Publish(mutation);
            if (!published) {
                LOG(ERROR) << "Failed to publish durable weight operation, "
                              "sequence_id="
                           << durable_entry.sequence_id
                           << ", operation_id=" << mutation.next->operation_id
                           << ", error=" << static_cast<int>(published.error());
            }
            {
                std::lock_guard lock(completion->mutex);
                completion->result = std::move(published);
            }
            completion->cv.notify_all();
        });
    if (!persisted) {
        return tl::make_unexpected(WeightManagementError::DURABILITY_FAILED);
    }

    constexpr auto kPublishTimeout = std::chrono::seconds(30);
    std::unique_lock lock(completion->mutex);
    if (!completion->cv.wait_for(lock, kPublishTimeout, [&] {
            return completion->result.has_value();
        })) {
        LOG(ERROR) << "Timed out waiting for durable weight operation, "
                      "operation_id="
                   << mutation.next->operation_id;
        return tl::make_unexpected(WeightManagementError::DURABILITY_FAILED);
    }
    return std::move(*completion->result);
}

WeightMetadataStore::Result<WeightRevisionLease>
MasterService::PersistAndPublishWeightLeaseMutation(
    const WeightLeaseMutation& mutation) {
    if (!mutation.no_op && !weight_management_mutations_enabled_) {
        return tl::make_unexpected(WeightManagementError::DURABILITY_FAILED);
    }
    if (mutation.no_op || !enable_oplog_) {
        return weight_metadata_.Publish(mutation);
    }

    OpType type;
    std::string tenant_id;
    std::string payload;
    if (mutation.kind == WeightMetadataMutationKind::UPSERT) {
        if (!mutation.next.has_value() ||
            !mutation.last_accessed_at_ms.has_value()) {
            return tl::make_unexpected(WeightManagementError::INVALID_ARGUMENT);
        }
        type = OpType::WEIGHT_LEASE_UPSERT;
        tenant_id = mutation.next->identity.tenant_id;
        const auto encoded = struct_pack::serialize(WeightLeaseUpsertOp{
            .lease = *mutation.next,
            .last_accessed_at_ms = *mutation.last_accessed_at_ms,
        });
        payload.assign(encoded.begin(), encoded.end());
    } else if (mutation.kind == WeightMetadataMutationKind::ERASE &&
               mutation.previous.has_value()) {
        type = OpType::WEIGHT_LEASE_DELETE;
        tenant_id = mutation.previous->identity.tenant_id;
        WeightLeaseDeleteOp deletion{
            .lease_id = mutation.lease_id,
            .identity = mutation.previous->identity,
            .fenced_metadata_generation =
                mutation.previous->fenced_metadata_generation,
        };
        const auto encoded = struct_pack::serialize(deletion);
        payload.assign(encoded.begin(), encoded.end());
    } else {
        return tl::make_unexpected(WeightManagementError::INVALID_ARGUMENT);
    }

    struct Completion {
        std::mutex mutex;
        std::condition_variable cv;
        std::optional<WeightMetadataStore::Result<WeightRevisionLease>> result;
    };
    auto completion = std::make_shared<Completion>();
    auto persisted = AppendOpLogWithDurableFinalize(
        type, tenant_id, MakeWeightLeaseMetadataKey(mutation.lease_id), payload,
        [this, mutation, completion](const OpLogEntry& durable_entry) {
            auto published = weight_metadata_.Publish(mutation);
            if (!published) {
                LOG(ERROR) << "Failed to publish durable weight lease "
                              "mutation, sequence_id="
                           << durable_entry.sequence_id
                           << ", lease_id=" << mutation.lease_id
                           << ", error=" << static_cast<int>(published.error());
            }
            {
                std::lock_guard lock(completion->mutex);
                completion->result = std::move(published);
            }
            completion->cv.notify_all();
        });
    if (!persisted) {
        LOG(ERROR) << "Failed to persist weight lease mutation, lease_id="
                   << mutation.lease_id
                   << ", error=" << static_cast<int>(persisted.error());
        return tl::make_unexpected(WeightManagementError::DURABILITY_FAILED);
    }

    constexpr auto kPublishTimeout = std::chrono::seconds(30);
    std::unique_lock lock(completion->mutex);
    if (!completion->cv.wait_for(lock, kPublishTimeout, [&] {
            return completion->result.has_value();
        })) {
        LOG(ERROR) << "Timed out waiting for durable weight lease publication, "
                      "lease_id="
                   << mutation.lease_id;
        return tl::make_unexpected(WeightManagementError::DURABILITY_FAILED);
    }
    return std::move(*completion->result);
}

}  // namespace mooncake
