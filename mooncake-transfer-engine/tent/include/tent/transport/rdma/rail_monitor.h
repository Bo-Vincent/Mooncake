// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENT_RAIL_MONITOR_H
#define TENT_RAIL_MONITOR_H

#include <cstdint>

#include "tent/common/config.h"
#include "tent/common/status.h"
#include "tent/runtime/topology.h"

namespace mooncake {
namespace tent {
// Not thread-safe. Each RDMA worker owns its own RailMonitor via
// WorkerContext::rails, so markFailed / markRecovered / available / load
// all execute on a single worker thread. Do not share across threads
// without adding external synchronization.
class RailMonitor {
    const static size_t kMaxNuma = 16;

    // Upper bound on the exponential-backoff cooldown; prevents the
    // per-rail pause from growing without bound under repeated failure.
    static constexpr std::chrono::seconds kMaxCooldown{300};
    static constexpr size_t kSnapshotCleanupInterval = 64;

   public:
    // Config keys. Exposed as constants so callers (and docs) reference
    // a single source of truth instead of duplicating the string path.
    static constexpr const char *kCfgErrorThreshold =
        "transports/rdma/rail_error_threshold";
    static constexpr const char *kCfgErrorWindowSecs =
        "transports/rdma/rail_error_window_secs";
    static constexpr const char *kCfgCooldownSecs =
        "transports/rdma/rail_cooldown_secs";
    static constexpr const char *kCfgRecoveryProbeEnabled =
        "transports/rdma/rail_recovery_probe_enabled";

   public:
    RailMonitor() = default;

    ~RailMonitor() = default;

    RailMonitor(const RailMonitor &) = delete;
    RailMonitor &operator=(const RailMonitor &) = delete;

   public:
    // The shared topology references keep segment snapshots alive while rail
    // failure handling uses their derived mappings.
    // rail_topo_json: optional JSON string describing rail topology.
    // conf: optional Config pointer; when non-null, overrides the default
    //   error_threshold / error_window_secs / cooldown_secs values via
    //   kCfgErrorThreshold / kCfgErrorWindowSecs / kCfgCooldownSecs.
    // A later call with the same NIC/memory layout only refreshes the pins
    // (segment COW snapshots); it does not rebuild rail_states_ or reprint
    // the config banner.
    Status load(std::shared_ptr<const Topology> local,
                std::shared_ptr<const Topology> remote,
                const std::string &rail_topo_json = "",
                const Config *conf = nullptr,
                uint64_t remote_snapshot_key = 0);

    bool ready() { return ready_; }

    bool available(int local_nic, int remote_nic);

    // A paused rail may admit one slice after a newer remote metadata
    // snapshot arrives. A zero token requests ownership; a non-zero token
    // lets the owning slice retain that ownership across endpoint retries.
    // probe_in_progress distinguishes a sibling's active probe from a rail
    // that has no recovery evidence left.
    bool tryRecoveryProbe(int local_nic, int remote_nic, uint64_t &token,
                          bool *probe_in_progress = nullptr);

    // Return ownership when the admitted slice is cancelled or rejected
    // before post, allowing another slice to use the same fresh snapshot.
    void abandonRecoveryProbe(int local_nic, int remote_nic, uint64_t token);

    void markFailed(int local_nic, int remote_nic,
                    uint64_t probe_token = 0);

    void markRecovered(int local_nic, int remote_nic);

    int findBestRemoteDevice(int local_nic, int remote_numa);

    const Topology *local() const { return local_.get(); }

    const Topology *remote() const { return remote_.get(); }

   private:
    Status loadFromJson(const std::string &rail_topo_json);

    Status loadDefault();

    void updateBestMapping();

   private:
    bool ready_{false};
    std::shared_ptr<const Topology> local_;
    std::shared_ptr<const Topology> remote_;

    struct PairHash {
        std::size_t operator()(const std::pair<int, int> &p) const noexcept {
            return std::hash<int>()(p.first) ^
                   (std::hash<int>()(p.second) << 1);
        }
    };

    struct RailState {
        int error_count = 0;
        std::chrono::seconds cooldown{0};
        std::chrono::steady_clock::time_point last_error{};
        std::chrono::steady_clock::time_point resume_time{};
        uint64_t last_probe_generation = 0;
        uint64_t active_probe_token = 0;

        // Derived: a rail is paused iff a resume_time has been armed.
        bool paused() const {
            return resume_time != std::chrono::steady_clock::time_point{};
        }
    };

    std::unordered_map<std::pair<int, int>, RailState, PairHash> rail_states_;
    std::unordered_map<int, int> direct_rails_;  // keep static after loaded
    std::unordered_map<int, int> best_mapping_[kMaxNuma];
    std::unordered_map<uint64_t, std::weak_ptr<const Topology>>
        remote_snapshots_;
    size_t snapshot_inserts_since_cleanup_ = 0;

    uint64_t metadata_generation_ = 0;
    uint64_t next_probe_token_ = 0;
    bool recovery_probe_enabled_ = true;

    int error_threshold_ = 3;
    std::chrono::seconds error_window_{10};
    std::chrono::seconds cooldown_{30};
};

}  // namespace tent
}  // namespace mooncake

#endif  // TENT_RAIL_MONITOR_H
