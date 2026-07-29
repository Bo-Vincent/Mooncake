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

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>

#define MOONCAKE_TRANSFER_ENGINE_COMPLETION_CONTRACT_ONLY
#include "transfer_engine_py.h"

namespace {

bool require(bool condition, const char* message) {
    if (condition) {
        return true;
    }
    std::cerr << message << '\n';
    return false;
}

struct OwnerProbe {};

}  // namespace

int main() {
    bool ok = true;
    ok &= require(static_cast<int>(BatchTransferCompletionStatus::COMPLETED) == 0,
                  "COMPLETED must preserve the legacy success code");
    ok &= require(
        static_cast<int>(BatchTransferCompletionStatus::FAILED_DRAINED) == -1,
        "terminal failure must remain negative");
    ok &= require(
        static_cast<int>(BatchTransferCompletionStatus::COMPLETION_UNKNOWN) ==
            -2,
        "unknown completion needs a distinct status code");

    auto backend_status =
        std::make_shared<std::atomic<BatchTransferBackendStatus>>(
            BatchTransferBackendStatus::IN_PROGRESS);
    auto release_succeeds = std::make_shared<std::atomic<bool>>(false);
    auto release_calls = std::make_shared<std::atomic<int>>(0);
    auto owner = std::make_shared<OwnerProbe>();
    std::weak_ptr<OwnerProbe> weak_owner = owner;
    auto ticket = std::make_shared<BatchTransferTicket>(
        41, owner,
        [backend_status] { return backend_status->load(); },
        [release_succeeds, release_calls] {
            release_calls->fetch_add(1);
            return release_succeeds->load();
        });
    owner.reset();

    ok &= require(ticket->poll() ==
                      BatchTransferCompletionStatus::COMPLETION_UNKNOWN,
                  "in-progress work must remain completion-unknown");
    ok &= require(release_calls->load() == 0,
                  "in-progress work must not release its batch");
    ok &= require(!weak_owner.expired(),
                  "completion-unknown must retain the owning request state");

    backend_status->store(BatchTransferBackendStatus::COMPLETED);
    ok &= require(ticket->poll() ==
                      BatchTransferCompletionStatus::COMPLETION_UNKNOWN,
                  "BatchBusy must keep a terminal observation unknown");
    ok &= require(!weak_owner.expired(),
                  "BatchBusy must retain the owning request state");

    release_succeeds->store(true);
    ok &= require(ticket->poll() == BatchTransferCompletionStatus::COMPLETED,
                  "completed and released work must become terminal");
    ok &= require(ticket->drained(), "terminal work must report drained");
    ok &= require(weak_owner.expired(),
                  "drained work may release the owning request state");

    ok &= require(
        !shouldRetryBatchTransfer(
            BatchTransferCompletionStatus::COMPLETION_UNKNOWN),
        "completion-unknown must never be retried");
    ok &= require(
        shouldRetryBatchTransfer(
            BatchTransferCompletionStatus::FAILED_DRAINED),
        "only a drained failure may be retried");

    auto pending_status =
        std::make_shared<std::atomic<BatchTransferBackendStatus>>(
            BatchTransferBackendStatus::IN_PROGRESS);
    auto pending_owner = std::make_shared<OwnerProbe>();
    std::weak_ptr<OwnerProbe> weak_pending_owner = pending_owner;
    auto pending_ticket = std::make_shared<BatchTransferTicket>(
        42, pending_owner,
        [pending_status] { return pending_status->load(); },
        [] { return true; });
    std::weak_ptr<BatchTransferTicket> weak_pending_ticket = pending_ticket;

    BatchTransferPendingRegistry registry;
    registry.retain(pending_ticket);
    pending_owner.reset();
    pending_ticket.reset();
    ok &= require(!weak_pending_ticket.expired(),
                  "registry must retain a dropped unknown ticket");
    ok &= require(!weak_pending_owner.expired(),
                  "registry must retain the dropped ticket's owner");

    pending_status->store(BatchTransferBackendStatus::FAILED);
    ok &= require(registry.drain(20) ==
                      BatchTransferCompletionStatus::FAILED_DRAINED,
                  "bounded drain must report a drained failure");
    ok &= require(registry.size() == 0,
                  "drained tickets must leave the pending registry");
    ok &= require(weak_pending_ticket.expired(),
                  "registry may release a drained ticket");
    ok &= require(weak_pending_owner.expired(),
                  "registry may release a drained ticket owner");

    auto sticky_failed_status =
        std::make_shared<std::atomic<BatchTransferBackendStatus>>(
            BatchTransferBackendStatus::FAILED);
    auto sticky_waiting_status =
        std::make_shared<std::atomic<BatchTransferBackendStatus>>(
            BatchTransferBackendStatus::IN_PROGRESS);
    BatchTransferPendingRegistry sticky_registry;
    sticky_registry.retain(std::make_shared<BatchTransferTicket>(
        46, std::make_shared<OwnerProbe>(),
        [sticky_failed_status] { return sticky_failed_status->load(); },
        [] { return true; }));
    sticky_registry.retain(std::make_shared<BatchTransferTicket>(
        47, std::make_shared<OwnerProbe>(),
        [sticky_waiting_status] { return sticky_waiting_status->load(); },
        [] { return true; }));
    ok &= require(sticky_registry.drain(0) ==
                      BatchTransferCompletionStatus::COMPLETION_UNKNOWN,
                  "a remaining unknown ticket must keep the registry unknown");
    sticky_waiting_status->store(BatchTransferBackendStatus::COMPLETED);
    ok &= require(sticky_registry.drain(20) ==
                      BatchTransferCompletionStatus::FAILED_DRAINED,
                  "failure must remain sticky across bounded drain calls");
    ok &= require(sticky_registry.drain(0) ==
                      BatchTransferCompletionStatus::COMPLETED,
                  "a reported sticky failure must be consumed");

    auto externally_drained_status =
        std::make_shared<std::atomic<BatchTransferBackendStatus>>(
            BatchTransferBackendStatus::IN_PROGRESS);
    auto externally_drained_ticket = std::make_shared<BatchTransferTicket>(
        44, std::make_shared<OwnerProbe>(),
        [externally_drained_status] {
            return externally_drained_status->load();
        },
        [] { return true; });
    std::weak_ptr<BatchTransferTicket> weak_externally_drained_ticket =
        externally_drained_ticket;
    BatchTransferPendingRegistry externally_drained_registry;
    externally_drained_registry.retain(externally_drained_ticket);
    externally_drained_status->store(BatchTransferBackendStatus::COMPLETED);
    ok &= require(externally_drained_ticket->drain(0) ==
                      BatchTransferCompletionStatus::COMPLETED,
                  "direct ticket drain must observe terminal completion");
    externally_drained_ticket.reset();
    ok &= require(!weak_externally_drained_ticket.expired(),
                  "registry must retain an externally drained ticket until "
                  "observed");
    ok &= require(externally_drained_registry.size() == 0,
                  "pending count must exclude externally drained tickets");
    ok &= require(
        weak_externally_drained_ticket.expired(),
        "observing the registry must release an externally drained ticket");

    auto externally_failed_status =
        std::make_shared<std::atomic<BatchTransferBackendStatus>>(
            BatchTransferBackendStatus::FAILED);
    auto externally_failed_ticket = std::make_shared<BatchTransferTicket>(
        48, std::make_shared<OwnerProbe>(),
        [externally_failed_status] {
            return externally_failed_status->load();
        },
        [] { return true; });
    std::weak_ptr<BatchTransferTicket> weak_externally_failed_ticket =
        externally_failed_ticket;
    BatchTransferPendingRegistry retain_reaping_registry;
    retain_reaping_registry.retain(externally_failed_ticket);
    ok &= require(externally_failed_ticket->drain(0) ==
                      BatchTransferCompletionStatus::FAILED_DRAINED,
                  "direct ticket drain must observe terminal failure");
    externally_failed_ticket.reset();

    auto replacement_status =
        std::make_shared<std::atomic<BatchTransferBackendStatus>>(
            BatchTransferBackendStatus::IN_PROGRESS);
    retain_reaping_registry.retain(std::make_shared<BatchTransferTicket>(
        49, std::make_shared<OwnerProbe>(),
        [replacement_status] { return replacement_status->load(); },
        [] { return true; }));
    ok &= require(weak_externally_failed_ticket.expired(),
                  "retaining another ticket must release externally drained "
                  "tickets");
    replacement_status->store(BatchTransferBackendStatus::COMPLETED);
    ok &= require(retain_reaping_registry.drain(20) ==
                      BatchTransferCompletionStatus::FAILED_DRAINED,
                  "reaping must preserve an externally drained failure");

    auto callback_status =
        std::make_shared<std::atomic<BatchTransferBackendStatus>>(
            BatchTransferBackendStatus::IN_PROGRESS);
    auto callback_owner = std::make_shared<OwnerProbe>();
    std::weak_ptr<OwnerProbe> weak_callback_owner = callback_owner;
    auto callback_ticket = std::make_shared<BatchTransferTicket>(
        45, callback_owner,
        [callback_status] { return callback_status->load(); },
        [] { return true; });
    auto quarantined_resource = std::make_shared<OwnerProbe>();
    std::weak_ptr<OwnerProbe> weak_quarantined_resource =
        quarantined_resource;
    callback_ticket->addDrainedCallback([quarantined_resource] {});
    callback_owner.reset();
    quarantined_resource.reset();
    callback_ticket->poll();
    ok &= require(!weak_callback_owner.expired(),
                  "unknown ticket must retain its request owner");
    ok &= require(!weak_quarantined_resource.expired(),
                  "unknown ticket must retain quarantined resources");
    callback_status->store(BatchTransferBackendStatus::COMPLETED);
    callback_ticket->poll();
    ok &= require(weak_callback_owner.expired(),
                  "drained ticket may release its request owner");
    ok &= require(weak_quarantined_resource.expired(),
                  "drained callback may release quarantined resources");

    std::atomic<int> bounded_poll_count{0};
    auto bounded_ticket = std::make_shared<BatchTransferTicket>(
        43, std::make_shared<OwnerProbe>(),
        [&bounded_poll_count] {
            bounded_poll_count.fetch_add(1);
            return BatchTransferBackendStatus::IN_PROGRESS;
        },
        [] { return true; });
    const auto drain_start = std::chrono::steady_clock::now();
    ok &= require(bounded_ticket->drain(5) ==
                      BatchTransferCompletionStatus::COMPLETION_UNKNOWN,
                  "bounded drain must return unknown at its deadline");
    const auto drain_elapsed = std::chrono::steady_clock::now() - drain_start;
    ok &= require(drain_elapsed < std::chrono::milliseconds(250),
                  "bounded drain must not block indefinitely");
    ok &= require(bounded_poll_count.load() > 0,
                  "bounded drain must poll at least once");

    BatchTransferTaskTracker task_tracker(2);
    task_tracker.observe(0, BatchTransferTaskStatus::FAILED);
    ok &= require(
        task_tracker.aggregate() == BatchTransferBackendStatus::IN_PROGRESS,
        "one failed task must not stop the remaining task from draining");
    ok &= require(!task_tracker.shouldPoll(0),
                  "a terminal task must not be polled again");
    ok &= require(task_tracker.shouldPoll(1),
                  "a later task must still be polled after an earlier failure");
    task_tracker.observe(1, BatchTransferTaskStatus::COMPLETED);
    ok &= require(task_tracker.aggregate() ==
                      BatchTransferBackendStatus::FAILED,
                  "failure must remain sticky after every task drains");

    ok &= require(
        classifyBatchTransferTaskStatus(4, 1, 2, 3, 4, 5) ==
            BatchTransferTaskStatus::FAILED,
        "backend timeout must be classified as a terminal task failure");
    ok &= require(
        classifyBatchTransferTaskStatus(5, 1, 2, 3, 4, 5) ==
            BatchTransferTaskStatus::FAILED,
        "invalid backend state must be classified as a terminal task failure");

    return ok ? 0 : 1;
}
