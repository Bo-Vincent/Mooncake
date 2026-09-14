#include "master_service.h"

#include <map>
#include <set>

#include "weight_residency_planner.h"

namespace mooncake {

namespace {

constexpr size_t kWeightDeleteBatchSize = 64;

}  // namespace

WeightMetadataStore::Result<WeightRevisionMetadata>
MasterService::ReconcileWeightRevision(
    const ReconcileWeightRevisionRequest& request) {
    [[maybe_unused]] auto lineage_operation_lock =
        AcquireWeightLineageOperationLock(
            ToWeightLineageIdentity(request.identity));
    return ReconcileWeightRevisionInternal(request);
}

WeightMetadataStore::Result<WeightRevisionMetadata>
MasterService::ReconcileWeightRevisionInternal(
    const ReconcileWeightRevisionRequest& request) {
    const auto canonical_group = MakeWeightPayloadGroupId(request.identity);
    if (canonical_group.empty()) {
        return tl::make_unexpected(WeightManagementError::INVALID_ARGUMENT);
    }
    [[maybe_unused]] auto group_operation_lock =
        AcquireWeightGroupOperationLock(TenantId(request.identity.tenant_id),
                                        canonical_group);
    const auto now_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    auto view = weight_metadata_.Get(request.identity, now_ms);
    if (!view) {
        return tl::make_unexpected(view.error());
    }
    const auto current = view->metadata;
    if (current.availability == WeightAvailabilityState::DELETED) {
        return current;
    }
    std::optional<WeightResidencyOperation> active_operation;
    if (current.operation_id.has_value()) {
        auto operation = weight_metadata_.QueryOperation(*current.operation_id);
        if (!operation) {
            return tl::make_unexpected(operation.error());
        }
        active_operation = *operation;
    }

    auto members = SnapshotWeightGroup(request.identity,
                                       current.manifest.payload_group_id);
    const bool absent =
        !members && members.error() == WeightManagementError::NOT_FOUND;
    if (!members && !absent) {
        return tl::make_unexpected(members.error());
    }

    if (current.availability == WeightAvailabilityState::DELETING) {
        if (!absent) {
            return current;
        }
        auto mutation = weight_metadata_.PrepareFinishDelete(
            current.identity, current.metadata_generation, now_ms);
        if (!mutation) {
            return tl::make_unexpected(mutation.error());
        }
        return PersistAndPublishWeightMutation(*mutation);
    }

    struct AffinityObservation {
        std::string affinity_id;
        uint64_t logical_bytes{0};
        bool all_memory{true};
        bool all_have_cold{true};
        bool all_cold{true};
        bool any_readable{false};
        std::vector<std::string> keys;
    };
    auto aggregate_affinities = [](const auto& member_snapshots) {
        std::map<std::string, AffinityObservation> affinities;
        for (const auto& member : member_snapshots) {
            if (member.data_type != ObjectDataType::WEIGHT) {
                continue;
            }
            auto& affinity = affinities[member.residency_affinity_id];
            affinity.affinity_id = member.residency_affinity_id;
            affinity.logical_bytes += member.size;
            affinity.all_memory = affinity.all_memory && member.has_memory;
            affinity.all_have_cold = affinity.all_have_cold && member.has_cold;
            affinity.all_cold =
                affinity.all_cold && member.has_cold && !member.has_memory;
            affinity.any_readable = affinity.any_readable || member.readable;
            affinity.keys.push_back(member.key);
        }
        return affinities;
    };

    std::set<std::string> hot_affinities;
    bool offload_failed = false;
    if (active_operation.has_value() && !absent &&
        active_operation->kind == WeightOperationKind::MIGRATING) {
        auto affinities = aggregate_affinities(*members);
        if (active_operation->target_residency == WeightResidencyState::HOT) {
            for (const auto& [affinity_id, affinity] : affinities) {
                static_cast<void>(affinity);
                hot_affinities.insert(affinity_id);
            }
        } else if (active_operation->target_residency ==
                   WeightResidencyState::MIXED) {
            std::vector<WeightAffinityUnit> units;
            units.reserve(affinities.size());
            for (const auto& [affinity_id, affinity] : affinities) {
                units.push_back(WeightAffinityUnit{
                    .affinity_id = affinity_id,
                    .logical_bytes = affinity.logical_bytes,
                });
            }
            auto plan = PlanMixedWeightResidency(
                units, *active_operation->target_hot_ratio);
            if (!plan) {
                return tl::make_unexpected(plan.error());
            }
            hot_affinities.insert(plan->hot_affinity_ids.begin(),
                                  plan->hot_affinity_ids.end());
        }

        const TenantId tenant_id(current.identity.tenant_id);
        const auto manifest = std::find_if(
            members->begin(), members->end(), [&](const auto& member) {
                return member.key == current.manifest.manifest_key &&
                       member.data_type == ObjectDataType::METADATA;
            });
        if (manifest != members->end() && !manifest->has_memory) {
            const auto result = TryPushPromotionQueue(
                MakeObjectIdentity(manifest->key, tenant_id), false, true);
            VLOG(1) << "weight_manifest_promotion key=" << manifest->key
                    << " result=" << static_cast<int>(result);
        }

        uint64_t scheduled_affinities = 0;
        uint64_t scheduled_members = 0;
        uint64_t scheduled_bytes = 0;
        auto reserve_affinity = [&](const AffinityObservation& affinity) {
            const auto member_count =
                static_cast<uint64_t>(affinity.keys.size());
            if (scheduled_affinities != 0 &&
                (scheduled_members >= weight_migration_max_members_per_round_ ||
                 member_count > weight_migration_max_members_per_round_ -
                                    scheduled_members ||
                 scheduled_bytes >= weight_migration_max_bytes_per_round_ ||
                 affinity.logical_bytes >
                     weight_migration_max_bytes_per_round_ - scheduled_bytes)) {
                return false;
            }
            if (scheduled_affinities == 0) {
                scheduled_members = std::min(
                    weight_migration_max_members_per_round_, member_count);
                scheduled_bytes =
                    std::min(weight_migration_max_bytes_per_round_,
                             affinity.logical_bytes);
            } else {
                scheduled_members += member_count;
                scheduled_bytes += affinity.logical_bytes;
            }
            ++scheduled_affinities;
            return true;
        };

        for (const auto& [affinity_id, affinity] : affinities) {
            const bool should_be_hot = hot_affinities.contains(affinity_id);
            const bool needs_action =
                should_be_hot
                    ? !affinity.all_memory
                    : (!affinity.all_have_cold ||
                       (view->active_lease_count == 0 && !affinity.all_cold));
            if (!needs_action) {
                continue;
            }
            if (!reserve_affinity(affinity)) {
                break;
            }
            if (should_be_hot) {
                for (const auto& key : affinity.keys) {
                    const auto result = TryPushPromotionQueue(
                        MakeObjectIdentity(key, tenant_id), false, true);
                    VLOG(1) << "weight_rehydrate_promotion key=" << key
                            << " result=" << static_cast<int>(result);
                }
                continue;
            }

            bool has_complete_cold_unit = true;
            for (const auto& key : affinity.keys) {
                const auto member = std::find_if(
                    members->begin(), members->end(),
                    [&](const auto& item) { return item.key == key; });
                if (member == members->end() || !member->has_cold) {
                    has_complete_cold_unit = false;
                    offload_failed =
                        !QueueManagedWeightMemberOffload(current, key) ||
                        offload_failed;
                }
            }
            if (has_complete_cold_unit && view->active_lease_count == 0) {
                EvictManagedWeightMembersToCold(current, affinity.keys);
            }
        }

        members = SnapshotWeightGroup(request.identity,
                                      current.manifest.payload_group_id);
        if (!members) {
            return tl::make_unexpected(members.error());
        }
    }

    bool complete =
        !absent && members->size() == current.manifest.payload_count + 1;
    bool manifest_found = false;
    bool all_payload_memory = complete;
    bool all_payload_cold = complete;
    bool any_payload_readable = false;
    uint64_t logical_bytes = 0;
    uint64_t hot_bytes = 0;
    std::vector<std::string> payload_keys;
    if (!absent) {
        for (const auto& member : *members) {
            complete = complete && member.readable;
            if (member.key == current.manifest.manifest_key &&
                member.data_type == ObjectDataType::METADATA) {
                manifest_found = member.readable && member.has_memory;
            } else if (member.data_type == ObjectDataType::WEIGHT &&
                       member.size <= std::numeric_limits<uint64_t>::max() -
                                          logical_bytes) {
                logical_bytes += member.size;
                payload_keys.push_back(member.key);
                any_payload_readable = any_payload_readable || member.readable;
                all_payload_memory = all_payload_memory && member.has_memory;
                all_payload_cold =
                    all_payload_cold && member.has_cold && !member.has_memory;
                if (member.has_memory) {
                    hot_bytes += member.size;
                }
            } else {
                complete = false;
            }
        }
    }
    complete = complete && manifest_found &&
               payload_keys.size() == current.manifest.payload_count &&
               logical_bytes == current.manifest.logical_bytes &&
               ComputeWeightPayloadKeysSha256(payload_keys) ==
                   current.manifest.payload_keys_sha256;
    auto affinities = absent ? std::map<std::string, AffinityObservation>{}
                             : aggregate_affinities(*members);
    complete = complete && !affinities.contains("") &&
               affinities.size() == current.affinity_count;
    const auto observed_residency =
        absent || !any_payload_readable
            ? WeightResidencyState::ABSENT
            : (all_payload_memory
                   ? WeightResidencyState::HOT
                   : (all_payload_cold ? WeightResidencyState::COLD
                                       : WeightResidencyState::MIXED));
    const double observed_hot_ratio =
        current.manifest.logical_bytes == 0
            ? 0.0
            : static_cast<double>(hot_bytes) /
                  static_cast<double>(current.manifest.logical_bytes);

    if (active_operation.has_value()) {
        const auto& operation = *active_operation;
        uint64_t processed_units = 0;
        uint64_t processed_bytes = 0;
        std::string cursor;
        for (const auto& [affinity_id, affinity] : affinities) {
            const bool should_be_hot =
                operation.target_residency == WeightResidencyState::HOT ||
                (operation.target_residency == WeightResidencyState::MIXED &&
                 hot_affinities.contains(affinity_id));
            const bool satisfied =
                should_be_hot ? affinity.all_memory : affinity.all_cold;
            if (satisfied) {
                ++processed_units;
                processed_bytes += affinity.logical_bytes;
                cursor = affinity_id;
            }
        }
        const bool operation_complete =
            complete && processed_units == operation.total_units &&
            processed_bytes == operation.total_bytes &&
            observed_residency == operation.target_residency;
        if (operation_complete) {
            auto mutation = weight_metadata_.PrepareFinishOperation(
                operation.operation_id, observed_residency, observed_hot_ratio,
                now_ms);
            if (!mutation) {
                return tl::make_unexpected(mutation.error());
            }
            auto finished = PersistAndPublishWeightOperationMutation(*mutation);
            if (!finished) {
                return tl::make_unexpected(finished.error());
            }
            auto reconciled = weight_metadata_.Get(request.identity, now_ms);
            if (!reconciled) {
                return tl::make_unexpected(reconciled.error());
            }
            return reconciled->metadata;
        }
        auto progress = weight_metadata_.PrepareUpdateOperationProgress(
            operation.operation_id,
            std::min(processed_units, operation.total_units),
            operation.total_units,
            std::min(processed_bytes, operation.total_bytes),
            operation.total_bytes, std::move(cursor),
            offload_failed ? "cold replica write failed" : "",
            observed_residency, observed_hot_ratio, now_ms);
        if (!progress) {
            return tl::make_unexpected(progress.error());
        }
        auto published = PersistAndPublishWeightOperationMutation(*progress);
        if (!published) {
            return tl::make_unexpected(published.error());
        }
        auto reconciled = weight_metadata_.Get(request.identity, now_ms);
        if (!reconciled) {
            return tl::make_unexpected(reconciled.error());
        }
        return reconciled->metadata;
    }

    const auto availability = complete ? WeightAvailabilityState::READY
                                       : WeightAvailabilityState::DEGRADED;
    auto mutation = weight_metadata_.PrepareReconcile(
        current.identity, current.metadata_generation, availability,
        observed_residency, observed_hot_ratio, now_ms);
    if (!mutation) {
        return tl::make_unexpected(mutation.error());
    }
    return PersistAndPublishWeightMutation(*mutation);
}

WeightMetadataStore::Result<WeightRevisionMetadata>
MasterService::DeleteWeightRevision(
    const DeleteWeightRevisionRequest& request) {
    [[maybe_unused]] auto lineage_operation_lock =
        AcquireWeightLineageOperationLock(
            ToWeightLineageIdentity(request.identity));
    return DeleteWeightRevisionInternal(request, false);
}

WeightMetadataStore::Result<WeightRevisionMetadata>
MasterService::DeleteWeightRevisionInternal(
    const DeleteWeightRevisionRequest& request, bool allow_active_upsert) {
    const auto canonical_group = MakeWeightPayloadGroupId(request.identity);
    if (canonical_group.empty()) {
        return tl::make_unexpected(WeightManagementError::INVALID_ARGUMENT);
    }
    [[maybe_unused]] auto group_operation_lock =
        AcquireWeightGroupOperationLock(TenantId(request.identity.tenant_id),
                                        canonical_group);
    const auto now_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    auto mutation =
        weight_metadata_.PrepareDelete(request, now_ms, allow_active_upsert);
    if (!mutation) {
        return tl::make_unexpected(mutation.error());
    }
    auto deleting = PersistAndPublishWeightMutation(*mutation);
    if (!deleting) {
        return tl::make_unexpected(deleting.error());
    }

    auto keys = GetGroupMemberKeys(TenantId(request.identity.tenant_id),
                                   deleting->manifest.payload_group_id);
    std::stable_sort(keys.begin(), keys.end(),
                     [&](const auto& lhs, const auto& rhs) {
                         return lhs != deleting->manifest.manifest_key &&
                                rhs == deleting->manifest.manifest_key;
                     });
    size_t processed = 0;
    for (const auto& key : keys) {
        if (processed == kWeightDeleteBatchSize) {
            break;
        }
        auto removed =
            RemoveObject(key, TenantId(request.identity.tenant_id), true, true);
        if (!removed) {
            if (removed.error() != ErrorCode::OBJECT_NOT_FOUND) {
                return tl::make_unexpected(WeightManagementError::BUSY);
            }
            UnregisterGroupMember(TenantId(request.identity.tenant_id), key,
                                  deleting->manifest.payload_group_id);
        }
        ++processed;
    }
    group_operation_lock.lock.unlock();
    return ReconcileWeightRevisionInternal(
        ReconcileWeightRevisionRequest{.identity = request.identity});
}

size_t MasterService::RunWeightReconciliationForTesting(uint64_t now_ms,
                                                        size_t limit) {
    return ReconcileWeightMetadataOnce(now_ms, limit);
}

bool MasterService::RestoreWeightMetadataForTesting(
    const WeightMetadataSnapshot& snapshot) {
    return weight_metadata_.RestoreSnapshot(snapshot).has_value();
}

bool MasterService::DropWeightGroupMemberForTesting(
    const WeightRevisionIdentity& identity, const std::string& key) {
    const TenantId tenant_id(identity.tenant_id);
    [[maybe_unused]] auto group_operation_lock =
        AcquireWeightGroupOperationLock(tenant_id,
                                        MakeWeightPayloadGroupId(identity));
    std::shared_lock<std::shared_mutex> shared_lock(snapshot_mutex_);
    MetadataAccessorRW accessor(this, MakeObjectIdentity(key, tenant_id));
    if (!accessor.Exists()) {
        return false;
    }
    bool dropped = false;
    accessor.Get().VisitReplicas(
        [](const Replica& replica) {
            return replica.status() != ReplicaStatus::REMOVED;
        },
        [&dropped](Replica& replica) {
            replica.mark_removed();
            dropped = true;
        });
    return dropped;
}

size_t MasterService::ReconcileWeightMetadataOnce(uint64_t now_ms,
                                                  size_t limit) {
    if (limit == 0) {
        return 0;
    }
    constexpr uint64_t kImportAbandonTimeoutMs = 5 * 60 * 1000;
    size_t actions = 0;

    for (const auto& mutation : weight_metadata_.PrepareExpireLeases(now_ms)) {
        if (actions == limit) {
            break;
        }
        if (PersistAndPublishWeightLeaseMutation(mutation)) {
            ++actions;
        } else {
            MasterMetricManager::instance()
                .inc_weight_reconciliation_failures();
        }
    }

    auto snapshot = weight_metadata_.ExportSnapshot();
    for (const auto& lineage :
         snapshot.lineages.value_or(std::vector<WeightLineageMetadata>{})) {
        if (actions == limit) {
            break;
        }
        if (!lineage.latest_claim.has_value()) {
            continue;
        }
        auto claim = *lineage.latest_claim;
        [[maybe_unused]] auto lineage_lock =
            AcquireWeightLineageOperationLock(lineage.identity);
        auto current_lineage = weight_metadata_.GetLineage(lineage.identity);
        if (!current_lineage ||
            current_lineage->lineage_metadata_generation !=
                lineage.lineage_metadata_generation ||
            !current_lineage->latest_claim.has_value() ||
            current_lineage->latest_claim->request_id != claim.request_id ||
            current_lineage->latest_claim->base_identity !=
                claim.base_identity ||
            current_lineage->latest_claim->target_identity !=
                claim.target_identity ||
            current_lineage->latest_claim->mode != claim.mode ||
            current_lineage->latest_claim->phase != claim.phase) {
            continue;
        }
        claim = *current_lineage->latest_claim;
        if (claim.phase == WeightUpsertPhase::PREPARING_TARGET ||
            claim.phase == WeightUpsertPhase::TARGET_IMPORTING) {
            auto target = weight_metadata_.Get(claim.target_identity, now_ms);
            if (claim.phase == WeightUpsertPhase::PREPARING_TARGET && target &&
                (target->metadata.availability ==
                     WeightAvailabilityState::DELETING ||
                 target->metadata.availability ==
                     WeightAvailabilityState::DELETED)) {
                if (target->metadata.availability ==
                    WeightAvailabilityState::DELETING) {
                    auto cleaned = DeleteWeightRevisionInternal(
                        DeleteWeightRevisionRequest{
                            .identity = claim.target_identity,
                            .expected_metadata_generation =
                                target->metadata.metadata_generation,
                        },
                        true);
                    if (!cleaned || cleaned->availability !=
                                        WeightAvailabilityState::DELETED) {
                        continue;
                    }
                }
                auto aborted = weight_metadata_.PrepareAdvanceUpsert(
                    lineage.identity, claim.request_id,
                    WeightUpsertPhase::ABORTED, now_ms);
                if (aborted &&
                    PersistAndPublishWeightLineageMutation(*aborted)) {
                    ++actions;
                }
                continue;
            }
            if (!target || target->metadata.availability ==
                               WeightAvailabilityState::DELETED) {
                [[maybe_unused]] auto group_operation_lock =
                    AcquireWeightGroupOperationLock(
                        TenantId(claim.target_identity.tenant_id),
                        MakeWeightPayloadGroupId(claim.target_identity));
                auto import = weight_metadata_.PrepareBeginImport(
                    BeginWeightImportRequest{
                        .identity = claim.target_identity,
                        .payload_group_id = claim.import.payload_group_id,
                        .expected_payload_count =
                            claim.import.expected_payload_count,
                        .expected_logical_bytes =
                            claim.import.expected_logical_bytes,
                        .policy = claim.import.policy,
                        .affinity_summary = claim.import.affinity_summary,
                    },
                    now_ms, true);
                if (import && PersistAndPublishWeightMutation(*import)) {
                    ++actions;
                }
                continue;
            }
            if (claim.phase == WeightUpsertPhase::TARGET_IMPORTING && target &&
                target->metadata.availability ==
                    WeightAvailabilityState::DELETING) {
                auto cleaned = DeleteWeightRevisionInternal(
                    DeleteWeightRevisionRequest{
                        .identity = claim.target_identity,
                        .expected_metadata_generation =
                            target->metadata.metadata_generation,
                    },
                    true);
                if (cleaned) {
                    ++actions;
                }
                continue;
            }
            if (!target || target->metadata.availability !=
                               WeightAvailabilityState::READY) {
                continue;
            }
            auto committed = weight_metadata_.PrepareCommitUpsert(
                CommitWeightUpsertRequest{
                    .request_id = claim.request_id,
                    .base_identity = claim.base_identity,
                    .target_identity = claim.target_identity,
                },
                now_ms);
            if (!committed) {
                continue;
            }
            auto published = PersistAndPublishWeightLineageMutation(*committed);
            if (!published || !published->latest_claim.has_value()) {
                continue;
            }
            claim = *published->latest_claim;
            ++actions;
        }
        if (claim.phase != WeightUpsertPhase::DELETING_BASE &&
            claim.phase != WeightUpsertPhase::RETIRING_BASE) {
            continue;
        }
        auto base = weight_metadata_.Get(claim.base_identity, now_ms);
        if (base && base->active_lease_count == 0 &&
            !base->metadata.operation_id.has_value() &&
            base->metadata.availability != WeightAvailabilityState::DELETED) {
            auto deleted = DeleteWeightRevisionInternal(
                DeleteWeightRevisionRequest{
                    .identity = claim.base_identity,
                    .expected_metadata_generation =
                        base->metadata.metadata_generation,
                },
                true);
            if (!deleted) {
                continue;
            }
            base = weight_metadata_.Get(claim.base_identity, now_ms);
            ++actions;
        }
        if (!base ||
            base->metadata.availability != WeightAvailabilityState::DELETED) {
            continue;
        }
        const auto next_phase = claim.phase == WeightUpsertPhase::DELETING_BASE
                                    ? WeightUpsertPhase::TARGET_IMPORTING
                                    : WeightUpsertPhase::COMPLETED;
        auto advanced = weight_metadata_.PrepareAdvanceUpsert(
            lineage.identity, claim.request_id, next_phase, now_ms);
        if (!advanced || !PersistAndPublishWeightLineageMutation(*advanced)) {
            continue;
        }
        if (next_phase == WeightUpsertPhase::TARGET_IMPORTING) {
            const auto canonical_group =
                MakeWeightPayloadGroupId(claim.target_identity);
            if (canonical_group.empty()) {
                continue;
            }
            [[maybe_unused]] auto group_operation_lock =
                AcquireWeightGroupOperationLock(
                    TenantId(claim.target_identity.tenant_id), canonical_group);
            auto imported = weight_metadata_.PrepareBeginImport(
                BeginWeightImportRequest{
                    .identity = claim.target_identity,
                    .payload_group_id = claim.import.payload_group_id,
                    .expected_payload_count =
                        claim.import.expected_payload_count,
                    .expected_logical_bytes =
                        claim.import.expected_logical_bytes,
                    .policy = claim.import.policy,
                    .affinity_summary = claim.import.affinity_summary,
                },
                now_ms, true);
            if (!imported || !PersistAndPublishWeightMutation(*imported)) {
                continue;
            }
        }
        ++actions;
    }
    const double memory_used_ratio =
        segment_manager_.GetMemoryUsage().used_ratio();
    const bool memory_pressure =
        memory_used_ratio > eviction_high_watermark_ratio_ ||
        need_mem_eviction_.load(std::memory_order_relaxed);
    const double memory_low_watermark =
        std::max(0.0, eviction_high_watermark_ratio_ - eviction_ratio_);
    const bool capacity_available =
        !memory_pressure && memory_used_ratio < memory_low_watermark;
    if (memory_pressure) {
        std::sort(snapshot.metadata.begin(), snapshot.metadata.end(),
                  WeightAutoMigrationCandidateLess);
    }
    const size_t revision_count = snapshot.metadata.size();
    const size_t start = revision_count == 0 ? 0
                         : memory_pressure
                             ? 0
                             : weight_reconciliation_offset_.fetch_add(
                                   std::max<size_t>(limit, 1)) %
                                   revision_count;
    for (size_t examined = 0; examined < revision_count && actions < limit;
         ++examined) {
        const auto& metadata =
            snapshot.metadata[(start + examined) % revision_count];
        if (metadata.availability == WeightAvailabilityState::DELETED) {
            continue;
        }

        if (metadata.availability == WeightAvailabilityState::IMPORTING) {
            if (now_ms < metadata.updated_at_ms ||
                now_ms - metadata.updated_at_ms < kImportAbandonTimeoutMs) {
                continue;
            }
            [[maybe_unused]] auto lineage_operation_lock =
                AcquireWeightLineageOperationLock(
                    ToWeightLineageIdentity(metadata.identity));
            [[maybe_unused]] auto group_operation_lock =
                AcquireWeightGroupOperationLock(
                    TenantId(metadata.identity.tenant_id),
                    MakeWeightPayloadGroupId(metadata.identity));
            auto mutation = weight_metadata_.PrepareAbortImport(
                AbortWeightImportRequest{
                    .identity = metadata.identity,
                    .expected_metadata_generation =
                        metadata.metadata_generation,
                },
                now_ms);
            if (!mutation || !PersistAndPublishWeightMutation(*mutation)) {
                MasterMetricManager::instance()
                    .inc_weight_reconciliation_failures();
                continue;
            }
            ++actions;
            continue;
        }

        std::optional<WeightAutoMigrationSignal> auto_signal;
        if (memory_pressure) {
            auto_signal = WeightAutoMigrationSignal::MEMORY_PRESSURE;
        } else if (capacity_available) {
            auto_signal = WeightAutoMigrationSignal::CAPACITY_AVAILABLE;
        }
        if (auto_signal.has_value() &&
            metadata.availability == WeightAvailabilityState::READY) {
            auto current = weight_metadata_.Get(metadata.identity, now_ms);
            if (current) {
                auto target = PlanAutomaticWeightMigration(
                    current->metadata, current->active_lease_count,
                    *auto_signal, now_ms, weight_migration_cooldown_ms_);
                if (target.has_value()) {
                    auto started = StartWeightResidencyOperation(
                        StartWeightResidencyOperationRequest{
                            .identity = metadata.identity,
                            .expected_metadata_generation =
                                current->metadata.metadata_generation,
                            .target_residency = target->residency,
                            .mixed_hot_ratio = target->mixed_hot_ratio,
                        });
                    if (started) {
                        ++actions;
                        continue;
                    }
                    if (started.error() !=
                            WeightManagementError::STALE_GENERATION &&
                        started.error() != WeightManagementError::BUSY) {
                        MasterMetricManager::instance()
                            .inc_weight_reconciliation_failures();
                    }
                }
            }
        }

        if (metadata.availability == WeightAvailabilityState::DELETING &&
            weight_metadata_.HasActiveUpsertClaim(metadata.identity)) {
            continue;
        }
        WeightMetadataStore::Result<WeightRevisionMetadata> reconciled =
            metadata.availability == WeightAvailabilityState::DELETING
                ? DeleteWeightRevision(DeleteWeightRevisionRequest{
                      .identity = metadata.identity,
                      .expected_metadata_generation =
                          metadata.metadata_generation,
                  })
                : ReconcileWeightRevision(ReconcileWeightRevisionRequest{
                      .identity = metadata.identity,
                  });
        if (reconciled) {
            ++actions;
        } else if (reconciled.error() !=
                   WeightManagementError::STALE_GENERATION) {
            MasterMetricManager::instance()
                .inc_weight_reconciliation_failures();
        }
    }

    MasterMetricManager::instance().project_weight_metadata(
        weight_metadata_.ExportSnapshot(), now_ms);
    return actions;
}

WeightMetadataStore::Result<
    std::vector<MasterService::WeightGroupMemberSnapshot>>
MasterService::SnapshotWeightGroup(const WeightRevisionIdentity& identity,
                                   const std::string& payload_group_id) const {
    const TenantId tenant_id(identity.tenant_id);
    auto member_keys = GetGroupMemberKeys(tenant_id, payload_group_id);
    if (member_keys.empty()) {
        return tl::make_unexpected(WeightManagementError::NOT_FOUND);
    }
    std::sort(member_keys.begin(), member_keys.end());

    std::vector<WeightGroupMemberSnapshot> members;
    members.reserve(member_keys.size());
    for (const auto& key : member_keys) {
        MetadataAccessorRO accessor(
            this, MakeObjectIdentity(key, TenantId(identity.tenant_id)));
        if (!accessor.Exists()) {
            return tl::make_unexpected(WeightManagementError::NOT_FOUND);
        }
        const auto& metadata = accessor.Get();
        if (metadata.group_id != payload_group_id) {
            return tl::make_unexpected(WeightManagementError::CONFLICT);
        }
        members.push_back(WeightGroupMemberSnapshot{
            .key = key,
            .residency_affinity_id = metadata.residency_affinity_id,
            .size = metadata.size,
            .data_type = metadata.data_type,
            .readable = HasReadableReplica(metadata),
            .has_memory = metadata.HasReplica([this](const Replica& replica) {
                return replica.is_memory_replica() &&
                       IsReplicaReadable(replica);
            }),
            .has_cold = metadata.HasReplica([this](const Replica& replica) {
                return !replica.is_memory_replica() &&
                       IsReplicaReadable(replica);
            }),
        });
    }
    return members;
}

std::unordered_set<std::string> MasterService::SnapshotManagedWeightGroups()
    const {
    std::unordered_set<std::string> groups;
    const auto snapshot = weight_metadata_.ExportSnapshot();
    groups.reserve(snapshot.metadata.size());
    for (const auto& metadata : snapshot.metadata) {
        groups.insert(TenantId(metadata.identity.tenant_id)
                          .MakeScopedKey(metadata.manifest.payload_group_id));
    }
    return groups;
}

bool MasterService::IsManagedWeightObject(const TenantId& tenant_id,
                                          const std::string& key) const {
    std::string group_id;
    {
        MetadataAccessorRO accessor(this, MakeObjectIdentity(key, tenant_id));
        if (!accessor.Exists()) {
            return false;
        }
        group_id = accessor.Get().group_id;
    }
    return !group_id.empty() && weight_metadata_.IsManagedGroup(group_id);
}

MasterService::GroupEvictionResult MasterService::EvictManagedWeightGroupToCold(
    const WeightRevisionMetadata& revision) {
    auto members = SnapshotWeightGroup(revision.identity,
                                       revision.manifest.payload_group_id);
    if (!members) {
        return {};
    }
    std::vector<std::string> weight_keys;
    for (const auto& member : *members) {
        if (member.data_type == ObjectDataType::WEIGHT) {
            weight_keys.push_back(member.key);
        }
    }
    return EvictManagedWeightMembersToCold(revision, weight_keys);
}

bool MasterService::QueueManagedWeightMemberOffload(
    const WeightRevisionMetadata& revision, const std::string& member_key) {
    const TenantId tenant_id(revision.identity.tenant_id);
    std::shared_lock<std::shared_mutex> shared_lock(snapshot_mutex_);
    MetadataAccessorRW accessor(this,
                                MakeObjectIdentity(member_key, tenant_id));
    if (!accessor.Exists()) {
        return false;
    }
    auto& metadata = accessor.Get();
    if (metadata.group_id != revision.manifest.payload_group_id ||
        metadata.data_type != ObjectDataType::WEIGHT ||
        metadata.HasReplica([this](const Replica& replica) {
            return !replica.is_memory_replica() && IsReplicaReadable(replica);
        })) {
        return metadata.group_id == revision.manifest.payload_group_id &&
               metadata.data_type == ObjectDataType::WEIGHT;
    }

    auto& tenant_state = accessor.GetTenantState();
    auto existing_task = tenant_state.offloading_tasks.find(member_key);
    if (existing_task != tenant_state.offloading_tasks.end()) {
        if (existing_task->second.weight_operation_id.has_value() &&
            existing_task->second.weight_operation_id !=
                revision.operation_id) {
            return false;
        }
        if (!existing_task->second.weight_operation_id.has_value() &&
            revision.operation_id.has_value()) {
            auto task = existing_task->second;
            tenant_state.offloading_tasks.erase(existing_task);
            task.weight_operation_id = revision.operation_id;
            tenant_state.offloading_tasks.emplace(member_key, std::move(task));
        }
        return true;
    }
    std::optional<ReplicaID> source_id;
    std::vector<UUID> mirror_clients;
    metadata.VisitReplicas(
        [this](const Replica& replica) {
            return replica.is_memory_replica() && IsReplicaReadable(replica);
        },
        [&, this](Replica& replica) {
            if (source_id.has_value()) {
                return;
            }
            auto queued =
                PushOffloadingQueue(MakeObjectIdentity(member_key, tenant_id),
                                    replica, &mirror_clients);
            if (queued) {
                replica.inc_refcnt();
                source_id = replica.id();
            }
        });
    if (source_id.has_value()) {
        tenant_state.offloading_tasks.emplace(
            member_key,
            OffloadingTask{*source_id, std::chrono::system_clock::now(),
                           std::move(mirror_clients), revision.operation_id});
    }
    return source_id.has_value();
}

MasterService::GroupEvictionResult
MasterService::EvictManagedWeightMembersToCold(
    const WeightRevisionMetadata& revision,
    const std::vector<std::string>& member_keys) {
    GroupEvictionResult result;
    const auto now_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    if (weight_metadata_.HasActiveLease(revision.identity,
                                        revision.metadata_generation, now_ms)) {
        return result;
    }

    const TenantId tenant_id(revision.identity.tenant_id);
    std::vector<std::vector<Replica>> deferred_replicas;
    std::shared_lock<std::shared_mutex> shared_lock(snapshot_mutex_);
    auto is_evictable_memory = [this](const Replica& replica) {
        return IsEvictableMemoryReplica(replica);
    };
    auto ordered_keys = member_keys;
    std::sort(ordered_keys.begin(), ordered_keys.end());
    for (const auto& member_key : ordered_keys) {
        MetadataAccessorRW accessor(this,
                                    MakeObjectIdentity(member_key, tenant_id));
        if (!accessor.Exists()) {
            continue;
        }
        auto& metadata = accessor.Get();
        auto& tenant_state = accessor.GetTenantState();
        if (metadata.group_id != revision.manifest.payload_group_id ||
            metadata.data_type != ObjectDataType::WEIGHT) {
            continue;
        }
        const bool has_cold =
            metadata.HasReplica([this](const Replica& replica) {
                return !replica.is_memory_replica() &&
                       IsReplicaReadable(replica);
            });
        if (!has_cold) {
            continue;
        }

        if (enable_oplog_) {
            auto reservation = ReserveBatchOpLogSlot();
            if (!reservation) {
                result.stop_scan = true;
                result.error = reservation.error();
                break;
            }
            auto remaining =
                BuildRemainingReplicaDescriptors(metadata, is_evictable_memory);
            std::vector<ReplicaID> removed_ids;
            metadata.VisitReplicas(is_evictable_memory,
                                   [&removed_ids](Replica& replica) {
                                       removed_ids.push_back(replica.id());
                                       replica.mark_removed();
                                   });
            if (removed_ids.empty()) {
                continue;
            }
            auto persisted = AppendReservedOpLogWithDurableFinalize(
                std::move(reservation.value()), OpType::PUT_END,
                tenant_id.value(), member_key,
                SerializeMetadataForOpLogFromReplicaDescriptors(metadata,
                                                                remaining),
                [this, removed_ids](const OpLogEntry& durable_entry) {
                    FinalizeRemovedReplicasAfterDurable(
                        durable_entry, removed_ids, QuotaEraseMode::kFull);
                });
            if (!persisted) {
                for (const auto& id : removed_ids) {
                    if (auto* replica = metadata.GetReplicaByID(id)) {
                        replica->cancel_remove();
                    }
                }
                result.stop_scan = true;
                result.error = persisted.error();
                break;
            }
            PublishKvRemovedAfterEvict(member_key, removed_ids.size(), "cpu",
                                       metadata, tenant_id);
            result.freed_bytes += metadata.size * removed_ids.size();
            ++result.evicted_objects;
            continue;
        }

        const uint64_t before_charge = CompletedMemoryQuotaCharge(metadata);
        auto removed =
            PopReplicasWithCacheTotalAccounting(metadata, is_evictable_memory);
        const uint64_t removed_count = removed.size();
        if (removed_count == 0) {
            continue;
        }
        std::vector<ReplicaID> removed_ids;
        removed_ids.reserve(removed.size());
        for (const auto& replica : removed) {
            removed_ids.push_back(replica.id());
        }
        RecordDynamicReplicaRemoval(metadata, removed_ids);
        deferred_replicas.emplace_back(std::move(removed));
        const uint64_t after_charge = CompletedMemoryQuotaCharge(metadata);
        if (enable_multi_tenants_ && before_charge > after_charge) {
            auto released = metadata.quota_ledger.ReleaseCommitted(
                GetBoundTenantQuotaHandle(tenant_state),
                before_charge - after_charge);
            LogTenantQuotaLedgerError(released, "release_committed", tenant_id,
                                      member_key);
        }
        PublishKvRemovedAfterEvict(member_key, removed_count, "cpu", metadata,
                                   tenant_id);
        result.freed_bytes += metadata.size * removed_count;
        ++result.evicted_objects;
    }
    return result;
}

}  // namespace mooncake
