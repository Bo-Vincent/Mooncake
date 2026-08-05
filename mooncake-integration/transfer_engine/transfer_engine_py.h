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

#ifndef MOONCAKE_INTEGRATION_TRANSFER_ENGINE_TRANSFER_ENGINE_PY_H_
#define MOONCAKE_INTEGRATION_TRANSFER_ENGINE_TRANSFER_ENGINE_PY_H_

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

enum class BatchTransferCompletionStatus : int {
    COMPLETED = 0,
    FAILED_DRAINED = -1,
    COMPLETION_UNKNOWN = -2,
};

enum class BatchTransferBackendStatus {
    IN_PROGRESS,
    COMPLETED,
    FAILED,
};

constexpr BatchTransferBackendStatus classifyScatterTransferWaitStatus(
    bool ok, bool clock) {
    if (ok) return BatchTransferBackendStatus::COMPLETED;
    return clock ? BatchTransferBackendStatus::IN_PROGRESS
                 : BatchTransferBackendStatus::FAILED;
}

enum class BatchTransferTaskStatus {
    IN_PROGRESS,
    COMPLETED,
    FAILED,
};

template <typename Status>
constexpr BatchTransferTaskStatus classifyBatchTransferTaskStatus(
    Status status, Status completed, Status failed, Status canceled,
    Status timeout, Status invalid) {
    if (status == completed) return BatchTransferTaskStatus::COMPLETED;
    if (status == failed || status == canceled || status == timeout ||
        status == invalid) {
        return BatchTransferTaskStatus::FAILED;
    }
    return BatchTransferTaskStatus::IN_PROGRESS;
}

class BatchTransferTaskTracker {
   public:
    explicit BatchTransferTaskTracker(size_t task_count)
        : terminal_(task_count, false) {}

    bool shouldPoll(size_t task_id) const {
        return task_id < terminal_.size() && !terminal_[task_id];
    }

    void observe(size_t task_id, BatchTransferTaskStatus status) {
        if (!shouldPoll(task_id) ||
            status == BatchTransferTaskStatus::IN_PROGRESS) {
            return;
        }
        terminal_[task_id] = true;
        failed_ = failed_ || status == BatchTransferTaskStatus::FAILED;
    }

    BatchTransferBackendStatus aggregate() const {
        if (std::find(terminal_.begin(), terminal_.end(), false) !=
            terminal_.end()) {
            return BatchTransferBackendStatus::IN_PROGRESS;
        }
        return failed_ ? BatchTransferBackendStatus::FAILED
                       : BatchTransferBackendStatus::COMPLETED;
    }

    size_t size() const { return terminal_.size(); }

   private:
    std::vector<bool> terminal_;
    bool failed_ = false;
};

// Keeps native request, batch, and engine state alive. The caller must retain
// every referenced buffer and memory registration until drained() is true.
class BatchTransferTicket {
   public:
    using Poller = std::function<BatchTransferBackendStatus()>;
    using Releaser = std::function<bool()>;

    BatchTransferTicket(uint64_t batch_id, std::shared_ptr<void> owner,
                        Poller poller, Releaser releaser)
        : batch_id_(batch_id),
          owner_(std::move(owner)),
          poller_(std::move(poller)),
          releaser_(std::move(releaser)) {}

    static std::shared_ptr<BatchTransferTicket> terminal(
        BatchTransferCompletionStatus status) {
        return std::shared_ptr<BatchTransferTicket>(
            new BatchTransferTicket(status));
    }

    BatchTransferCompletionStatus status() const {
        std::lock_guard<std::mutex> guard(state_mutex_);
        return status_;
    }

    uint64_t batchId() const { return batch_id_; }

    bool drained() const {
        std::lock_guard<std::mutex> guard(state_mutex_);
        return drained_;
    }

    BatchTransferCompletionStatus poll() {
        std::lock_guard<std::mutex> poll_guard(poll_mutex_);

        Poller poller;
        Releaser releaser;
        {
            std::lock_guard<std::mutex> state_guard(state_mutex_);
            if (drained_) return status_;
            poller = poller_;
            releaser = releaser_;
        }

        if (!poller || !releaser) {
            return BatchTransferCompletionStatus::COMPLETION_UNKNOWN;
        }

        BatchTransferBackendStatus backend_status;
        try {
            backend_status = poller();
        } catch (...) {
            return BatchTransferCompletionStatus::COMPLETION_UNKNOWN;
        }
        if (backend_status == BatchTransferBackendStatus::IN_PROGRESS) {
            return BatchTransferCompletionStatus::COMPLETION_UNKNOWN;
        }

        bool released = false;
        try {
            released = releaser();
        } catch (...) {
            return BatchTransferCompletionStatus::COMPLETION_UNKNOWN;
        }
        if (!released) {
            return BatchTransferCompletionStatus::COMPLETION_UNKNOWN;
        }

        std::vector<std::function<void()>> callbacks;
        BatchTransferCompletionStatus terminal_status =
            backend_status == BatchTransferBackendStatus::COMPLETED
                ? BatchTransferCompletionStatus::COMPLETED
                : BatchTransferCompletionStatus::FAILED_DRAINED;
        {
            std::lock_guard<std::mutex> state_guard(state_mutex_);
            status_ = terminal_status;
            drained_ = true;
            owner_.reset();
            poller_ = {};
            releaser_ = {};
            callbacks.swap(drained_callbacks_);
        }
        for (auto &callback : callbacks) callback();
        return terminal_status;
    }

    BatchTransferCompletionStatus drain(uint64_t timeout_ms) {
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(timeout_ms);
        while (true) {
            auto completion = poll();
            if (completion !=
                BatchTransferCompletionStatus::COMPLETION_UNKNOWN) {
                return completion;
            }

            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                return BatchTransferCompletionStatus::COMPLETION_UNKNOWN;
            }
            std::this_thread::sleep_for(std::min(
                std::chrono::microseconds(100),
                std::chrono::duration_cast<std::chrono::microseconds>(deadline -
                                                                      now)));
        }
    }

    void addDrainedCallback(std::function<void()> callback) {
        bool call_now = false;
        {
            std::lock_guard<std::mutex> guard(state_mutex_);
            if (drained_) {
                call_now = true;
            } else {
                drained_callbacks_.push_back(callback);
            }
        }
        if (call_now) callback();
    }

   private:
    explicit BatchTransferTicket(BatchTransferCompletionStatus status)
        : status_(status), drained_(true) {}

    uint64_t batch_id_ = 0;
    std::shared_ptr<void> owner_;
    Poller poller_;
    Releaser releaser_;

    mutable std::mutex state_mutex_;
    std::mutex poll_mutex_;
    BatchTransferCompletionStatus status_ =
        BatchTransferCompletionStatus::COMPLETION_UNKNOWN;
    bool drained_ = false;
    std::vector<std::function<void()>> drained_callbacks_;
};

inline bool shouldRetryBatchTransfer(BatchTransferCompletionStatus status) {
    return status == BatchTransferCompletionStatus::FAILED_DRAINED;
}

class BatchTransferPendingRegistry {
   public:
    void retain(const std::shared_ptr<BatchTransferTicket> &ticket) {
        std::lock_guard<std::mutex> guard(mutex_);
        reapDrainedLocked();
        if (!ticket || ticket->drained()) return;
        auto duplicate = std::find(tickets_.begin(), tickets_.end(), ticket);
        if (duplicate == tickets_.end() && !ticket->drained()) {
            tickets_.push_back(ticket);
        }
    }

    size_t size() const {
        std::lock_guard<std::mutex> guard(mutex_);
        reapDrainedLocked();
        return tickets_.size();
    }

    BatchTransferCompletionStatus drain(uint64_t timeout_ms) {
        std::lock_guard<std::mutex> drain_guard(drain_mutex_);
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(timeout_ms);

        while (true) {
            std::vector<std::shared_ptr<BatchTransferTicket>> snapshot;
            {
                std::lock_guard<std::mutex> guard(mutex_);
                if (tickets_.empty()) {
                    auto completion =
                        saw_failed_
                            ? BatchTransferCompletionStatus::FAILED_DRAINED
                            : BatchTransferCompletionStatus::COMPLETED;
                    saw_failed_ = false;
                    return completion;
                }
                snapshot = tickets_;
            }

            bool saw_failed = false;
            for (const auto &ticket : snapshot) {
                auto completion = ticket->poll();
                if (completion ==
                    BatchTransferCompletionStatus::FAILED_DRAINED) {
                    saw_failed = true;
                }
            }

            {
                std::lock_guard<std::mutex> guard(mutex_);
                saw_failed_ = saw_failed_ || saw_failed;
                reapDrainedLocked();
                if (tickets_.empty()) {
                    auto completion =
                        saw_failed_
                            ? BatchTransferCompletionStatus::FAILED_DRAINED
                            : BatchTransferCompletionStatus::COMPLETED;
                    saw_failed_ = false;
                    return completion;
                }
            }

            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                return BatchTransferCompletionStatus::COMPLETION_UNKNOWN;
            }
            std::this_thread::sleep_for(std::min(
                std::chrono::microseconds(100),
                std::chrono::duration_cast<std::chrono::microseconds>(deadline -
                                                                      now)));
        }
    }

    std::vector<std::shared_ptr<BatchTransferTicket>> takeAll() {
        std::lock_guard<std::mutex> drain_guard(drain_mutex_);
        std::lock_guard<std::mutex> guard(mutex_);
        std::vector<std::shared_ptr<BatchTransferTicket>> tickets;
        tickets.swap(tickets_);
        saw_failed_ = false;
        return tickets;
    }

   private:
    void reapDrainedLocked() const {
        for (const auto &ticket : tickets_) {
            if (ticket->drained() &&
                ticket->status() ==
                    BatchTransferCompletionStatus::FAILED_DRAINED) {
                saw_failed_ = true;
            }
        }
        tickets_.erase(std::remove_if(tickets_.begin(), tickets_.end(),
                                      [](const auto &ticket) {
                                          return ticket->drained();
                                      }),
                       tickets_.end());
    }

    std::mutex drain_mutex_;
    mutable std::mutex mutex_;
    mutable std::vector<std::shared_ptr<BatchTransferTicket>> tickets_;
    mutable bool saw_failed_ = false;
};

#ifndef MOONCAKE_TRANSFER_ENGINE_COMPLETION_CONTRACT_ONLY

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pybind11/pybind11.h>
#include <sys/time.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stack>
#include <vector>

#include "common/base/status.h"
#include "transfer_engine.h"
#include "transfer_engine_c.h"
#include "transport/rdma_transport/rdma_transport.h"
#include "transport/transport.h"

using namespace mooncake;

const static size_t kDefaultBufferCapacity = 2ull * 1024 * 1024 * 1024;
const static size_t kSlabSizeKBTabLen = 16;
const static size_t kMaxClassId = kSlabSizeKBTabLen - 1;
const static size_t kSlabSizeKB[] = {
    8,         16,        32,         64,        128,      256,
    512,       1024,      2 * 1024,   4 * 1024,  8 * 1024, 16 * 1024,
    32 * 1024, 64 * 1024, 128 * 1024, 256 * 1024};

class TransferEnginePy {
   public:
    enum class TransferOpcode { READ = 0, WRITE = 1 };
    struct TransferNotify {
        std::string name;
        std::string msg;
    };

   public:
    using BatchDesc = Transport::BatchDesc;

   public:
    TransferEnginePy();

    ~TransferEnginePy();

    int initialize(const char *local_hostname, const char *metadata_server,
                   const char *protocol, const char *device_name);

    int initializeExt(const char *local_hostname, const char *metadata_server,
                      const char *protocol, const char *device_name,
                      const char *metadata_type);

    int getRpcPort();

    uintptr_t allocateManagedBuffer(size_t length);

    int freeManagedBuffer(uintptr_t user_tensor, size_t length);

    int transferSyncWrite(const char *target_hostname, uintptr_t buffer,
                          uintptr_t peer_buffer_address, size_t length,
                          const std::string &transport_hint = "");

    batch_id_t transferSubmitWrite(const char *target_hostname,
                                   uintptr_t buffer,
                                   uintptr_t peer_buffer_address, size_t length,
                                   const std::string &transport_hint = "");

    int transferCheckStatus(batch_id_t batch_id);

    int transferSyncRead(const char *target_hostname, uintptr_t buffer,
                         uintptr_t peer_buffer_address, size_t length,
                         const std::string &transport_hint = "");

    int batchTransferSyncWrite(const char *target_hostname,
                               std::vector<uintptr_t> buffers,
                               std::vector<uintptr_t> peer_buffer_addresses,
                               std::vector<size_t> lengths,
                               const std::string &transport_hint = "");

    std::shared_ptr<BatchTransferTicket> batchTransferSyncWriteWithTicket(
        const char *target_hostname, std::vector<uintptr_t> buffers,
        std::vector<uintptr_t> peer_buffer_addresses,
        std::vector<size_t> lengths, const std::string &transport_hint = "");

    int batchTransferSyncRead(const char *target_hostname,
                              std::vector<uintptr_t> buffers,
                              std::vector<uintptr_t> peer_buffer_addresses,
                              std::vector<size_t> lengths,
                              const std::string &transport_hint = "");

    std::shared_ptr<BatchTransferTicket> batchTransferSyncReadWithTicket(
        const char *target_hostname, std::vector<uintptr_t> buffers,
        std::vector<uintptr_t> peer_buffer_addresses,
        std::vector<size_t> lengths, const std::string &transport_hint = "");

    std::shared_ptr<BatchTransferTicket> scatterTransferSyncWriteWithTicket(
        const std::string &endpoint, const std::vector<uintptr_t> &local_bases,
        const std::vector<size_t> &local_capacities,
        const std::vector<uint64_t> &remote_bases,
        const std::vector<size_t> &remote_sizes,
        const std::vector<std::vector<size_t>> &local_offsets,
        const std::vector<std::vector<size_t>> &remote_offsets,
        const std::vector<std::vector<size_t>> &lengths);

    std::shared_ptr<BatchTransferTicket> scatterTransferSyncReadWithTicket(
        const std::string &endpoint, const std::vector<uintptr_t> &local_bases,
        const std::vector<size_t> &local_capacities,
        const std::vector<uint64_t> &remote_bases,
        const std::vector<size_t> &remote_sizes,
        const std::vector<std::vector<size_t>> &local_offsets,
        const std::vector<std::vector<size_t>> &remote_offsets,
        const std::vector<std::vector<size_t>> &lengths);

    batch_id_t batchTransferAsyncWrite(
        const char *target_hostname, const std::vector<uintptr_t> &buffers,
        const std::vector<uintptr_t> &peer_buffer_addresses,
        const std::vector<size_t> &lengths,
        const std::string &transport_hint = "");

    batch_id_t batchTransferAsyncRead(
        const char *target_hostname, const std::vector<uintptr_t> &buffers,
        const std::vector<uintptr_t> &peer_buffer_addresses,
        const std::vector<size_t> &lengths,
        const std::string &transport_hint = "");

    int transferSync(const char *target_hostname, uintptr_t buffer,
                     uintptr_t peer_buffer_address, size_t length,
                     TransferOpcode opcode, TransferNotify *notify = nullptr,
                     const std::string &transport_hint = "");

    // Known issue: in a few inference engines and benchmarks, accuracy
    // may be affected when using the batchTransferSync API. We currently
    // found this issue only in multi-node NVLink transfers.
    int batchTransferSync(const char *target_hostname,
                          std::vector<uintptr_t> buffers,
                          std::vector<uintptr_t> peer_buffer_addresses,
                          std::vector<size_t> lengths, TransferOpcode opcode,
                          TransferNotify *notify = nullptr,
                          const std::string &transport_hint = "");

    std::shared_ptr<BatchTransferTicket> batchTransferSyncWithTicket(
        const char *target_hostname, std::vector<uintptr_t> buffers,
        std::vector<uintptr_t> peer_buffer_addresses,
        std::vector<size_t> lengths, TransferOpcode opcode,
        TransferNotify *notify = nullptr,
        const std::string &transport_hint = "");

    BatchTransferCompletionStatus drainPendingBatchTransfers(
        uint64_t timeout_ms);

    size_t pendingBatchTransferCount() const {
        return pending_batch_transfers_.size();
    }

    batch_id_t batchTransferAsync(
        const char *target_hostname, const std::vector<uintptr_t> &buffers,
        const std::vector<uintptr_t> &peer_buffer_addresses,
        const std::vector<size_t> &lengths, TransferOpcode opcode,
        const std::string &transport_hint = "");

    int getBatchTransferStatus(const std::vector<batch_id_t> &batch_ids);

#ifdef USE_CUDA
    void batchTransferOnCuda(
        const char *target_hostname, const std::vector<uintptr_t> &buffers,
        const std::vector<uintptr_t> &peer_buffer_addresses,
        const std::vector<size_t> &lengths, TransferOpcode opcode,
        uintptr_t stream_ptr = 0, const std::string &transport_hint = "");

    void transferWriteOnCuda(const char *target_hostname, uintptr_t buffer,
                             uintptr_t peer_buffer_address, size_t length,
                             uintptr_t stream_ptr = 0,
                             const std::string &transport_hint = "");

    void transferReadOnCuda(const char *target_hostname, uintptr_t buffer,
                            uintptr_t peer_buffer_address, size_t length,
                            uintptr_t stream_ptr = 0,
                            const std::string &transport_hint = "");

    void batchTransferWriteOnCuda(
        const char *target_hostname, const std::vector<uintptr_t> &buffers,
        const std::vector<uintptr_t> &peer_buffer_addresses,
        const std::vector<size_t> &lengths, uintptr_t stream_ptr = 0,
        const std::string &transport_hint = "");

    void batchTransferReadOnCuda(
        const char *target_hostname, const std::vector<uintptr_t> &buffers,
        const std::vector<uintptr_t> &peer_buffer_addresses,
        const std::vector<size_t> &lengths, uintptr_t stream_ptr = 0,
        const std::string &transport_hint = "");
#endif

    uintptr_t getFirstBufferAddress(const std::string &segment_name);

    // Pre-connect every (local_ctx, peer_nic) pair for `segment_name` so the
    // first submitTransfer does not stall on handshake RPC + fi_av_insert.
    // No-op on non-EFA builds or when the EFA transport is not installed.
    int warmupEfaSegment(const std::string &segment_name);

    int writeBytesToBuffer(uintptr_t dest_address, char *src_ptr,
                           size_t length) {
        memcpy((void *)dest_address, (void *)src_ptr, length);
        return 0;
    }

    pybind11::bytes readBytesFromBuffer(uintptr_t source_address,
                                        size_t length) {
        return pybind11::bytes(
            static_cast<const char *>(reinterpret_cast<void *>(source_address)),
            length);
    }

    // FOR EXPERIMENT ONLY
    int registerMemory(uintptr_t buffer_addr, size_t capacity,
                       const std::string &location = kWildcardLocation);

    // must be called before TransferEnginePy::~TransferEnginePy()
    int unregisterMemory(uintptr_t buffer_addr);

    int batchRegisterMemory(std::vector<uintptr_t> buffer_addresses,
                            std::vector<size_t> capacities,
                            const std::string &location = kWildcardLocation);

    int batchUnregisterMemory(std::vector<uintptr_t> buffer_addresses);

    std::string getLocalTopology(const char *device_name);

    std::vector<TransferNotify> getNotifies();

    int sendProbe(const std::string &peer_server_name);

    std::shared_ptr<TransferEngine> getEngine() const { return engine_; }

    uintptr_t getEnginePtr() const { return (uintptr_t)engine_.get(); }

   private:
    std::shared_ptr<BatchTransferTicket> scatterTransferSyncWithTicket(
        const std::string &endpoint, const std::vector<uintptr_t> &local_bases,
        const std::vector<size_t> &local_capacities,
        const std::vector<uint64_t> &remote_bases,
        const std::vector<size_t> &remote_sizes,
        const std::vector<std::vector<size_t>> &local_offsets,
        const std::vector<std::vector<size_t>> &remote_offsets,
        const std::vector<std::vector<size_t>> &lengths, TransferOpcode opcode);

    char *allocateRawBuffer(size_t capacity);

    int findClassId(size_t size);

    int doBuddyAllocate(int class_id);

   private:
    std::shared_ptr<TransferEngine> engine_;
    Transport *xport_;

    std::mutex mutex_;
    std::vector<std::stack<char *>> free_list_;
    std::vector<char *> buffer_list_;
    std::unordered_set<char *> large_buffer_list_;
    std::unordered_map<std::string, Transport::SegmentHandle> handle_map_;
    BatchTransferPendingRegistry pending_batch_transfers_;
    bool auto_discovery_;

    uint64_t transfer_timeout_nsec_;
};

#endif  // MOONCAKE_TRANSFER_ENGINE_COMPLETION_CONTRACT_ONLY

#endif  // MOONCAKE_INTEGRATION_TRANSFER_ENGINE_TRANSFER_ENGINE_PY_H_
