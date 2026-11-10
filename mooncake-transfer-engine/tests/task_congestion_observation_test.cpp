// Copyright 2026 KVCache.AI
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

#include <gtest/gtest.h>

#include "task_congestion_status.h"
#include "task_congestion_observation.h"

namespace mooncake::adaptive_congestion_control {
namespace {

TaskCongestionEvidence windowEvidence() {
    TaskCongestionEvidence evidence;
    evidence.reason = {true, TaskCongestionReason::kByteWindow};
    evidence.path = "route-a";
    evidence.observed_at_ns = {true, 100};
    evidence.controller_generation = {true, 7};
    return evidence;
}

TEST(TaskCongestionObservationTest,
     WorstUnresolvedSliceWinsAndRecoveryClearsIt) {
    TaskCongestionObservation task(3, 1);
    EXPECT_EQ(task.state(), TaskCongestionState::kNormal);

    EXPECT_TRUE(task.defer(1, 1, windowEvidence()));
    EXPECT_EQ(task.state(), TaskCongestionState::kCongested);

    TaskCongestionEvidence isolation;
    isolation.reason = {true, TaskCongestionReason::kPathQuarantined};
    isolation.path = "route-b";
    isolation.observed_at_ns = {true, 200};
    isolation.controller_generation = {true, 7};
    EXPECT_TRUE(task.avoid(2, 1, isolation));
    EXPECT_EQ(task.state(), TaskCongestionState::kLongUnavailable);

    EXPECT_TRUE(task.resolve(2, 1));
    EXPECT_EQ(task.state(), TaskCongestionState::kCongested);
    EXPECT_TRUE(task.resolve(1, 1));
    EXPECT_EQ(task.state(), TaskCongestionState::kNormal);

    const auto detail = task.detail();
    EXPECT_TRUE(detail.reason.observed);
    EXPECT_EQ(detail.reason.value, TaskCongestionReason::kPathQuarantined);
    EXPECT_TRUE(detail.resolved);
    EXPECT_TRUE(detail.affected_path.observed);
    EXPECT_EQ(detail.affected_path.value, "route-b");
}

TEST(TaskCongestionObservationTest, FailureWithoutCongestionEvidenceIsUnknown) {
    TaskCongestionObservation task(2, 1);
    task.terminalFailure(1);
    EXPECT_EQ(task.state(), TaskCongestionState::kUnknown);
    const auto detail = task.detail();
    EXPECT_FALSE(detail.reason.observed);
    EXPECT_FALSE(detail.retry_after_ns.observed);
    EXPECT_FALSE(detail.failure_scope.observed);
    EXPECT_FALSE(detail.controller_generation.observed);
}

TEST(TaskCongestionObservationTest, DetailNamesTheWorstActiveSlice) {
    TaskCongestionObservation task(2, 1);
    TaskCongestionEvidence isolation;
    isolation.reason = {true, TaskCongestionReason::kPathQuarantined};
    isolation.path = "isolated-route";
    isolation.observed_at_ns = {true, 100};
    EXPECT_TRUE(task.avoid(0, 1, isolation));

    auto window = windowEvidence();
    window.observed_at_ns = {true, 200};
    EXPECT_TRUE(task.defer(1, 1, window));
    const auto detail = task.detail();
    EXPECT_EQ(detail.state, TaskCongestionState::kLongUnavailable);
    ASSERT_TRUE(detail.reason.observed);
    EXPECT_EQ(detail.reason.value, TaskCongestionReason::kPathQuarantined);
    ASSERT_TRUE(detail.affected_path.observed);
    EXPECT_EQ(detail.affected_path.value, "isolated-route");

    EXPECT_TRUE(task.resolve(0, 1));
    const auto recovered = task.detail();
    EXPECT_EQ(recovered.state, TaskCongestionState::kCongested);
    ASSERT_TRUE(recovered.reason.observed);
    EXPECT_EQ(recovered.reason.value, TaskCongestionReason::kByteWindow);
}

TEST(TaskCongestionObservationTest, OldAttemptCannotOverrideCurrentState) {
    TaskCongestionObservation task(2, 1);
    EXPECT_TRUE(task.defer(0, 1, windowEvidence()));
    EXPECT_TRUE(task.beginAttempt(2, 2));
    EXPECT_EQ(task.state(), TaskCongestionState::kNormal);

    EXPECT_FALSE(task.avoid(0, 1, windowEvidence()));
    EXPECT_EQ(task.state(), TaskCongestionState::kNormal);

    EXPECT_TRUE(task.avoid(1, 2, windowEvidence()));
    EXPECT_EQ(task.state(), TaskCongestionState::kLongUnavailable);
    EXPECT_FALSE(task.resolve(1, 1));
    EXPECT_EQ(task.state(), TaskCongestionState::kLongUnavailable);

    EXPECT_FALSE(task.avoid(2, 2, windowEvidence()));
    EXPECT_EQ(task.state(), TaskCongestionState::kUnknown);
}

TEST(TaskCongestionObservationTest, NewAttemptCanUseMoreSubmittedSlices) {
    TaskCongestionObservation task(1, 1);
    EXPECT_TRUE(task.defer(0, 1, windowEvidence()));
    EXPECT_TRUE(task.beginAttempt(2, 2));
    EXPECT_FALSE(task.defer(0, 1, windowEvidence()));

    TaskCongestionEvidence isolation;
    isolation.reason = {true, TaskCongestionReason::kPathQuarantined};
    isolation.path = "second-slice";
    isolation.controller_generation = {true, 8};
    EXPECT_TRUE(task.avoid(1, 2, isolation));
    EXPECT_EQ(task.state(), TaskCongestionState::kLongUnavailable);
    const auto detail = task.detail();
    ASSERT_TRUE(detail.slice_id.observed);
    EXPECT_EQ(detail.slice_id.value, 1);
}

TEST(TaskCongestionObservationTest, PublishedSlicesExtendCurrentAttempt) {
    TaskCongestionObservation task(1, 1);
    EXPECT_TRUE(task.defer(0, 1, windowEvidence()));
    EXPECT_TRUE(task.extendAttempt(1, 3));
    EXPECT_FALSE(task.extendAttempt(0, 4));
    EXPECT_FALSE(task.extendAttempt(1, 2));

    TaskCongestionEvidence isolation;
    isolation.reason = {true, TaskCongestionReason::kPathQuarantined};
    EXPECT_TRUE(task.avoid(2, 1, isolation));
    EXPECT_EQ(task.state(), TaskCongestionState::kLongUnavailable);
    EXPECT_TRUE(task.resolve(2, 1));
    EXPECT_EQ(task.state(), TaskCongestionState::kCongested);
}

TEST(TaskCongestionObservationTest,
     SlicesOnDifferentControllerGenerationsShareAttempt) {
    TaskCongestionObservation task(2, 1);
    EXPECT_TRUE(task.defer(0, 1, windowEvidence()));

    TaskCongestionEvidence isolation;
    isolation.reason = {true, TaskCongestionReason::kPathQuarantined};
    isolation.path = "other-device";
    isolation.controller_generation = {true, 2};
    EXPECT_TRUE(task.avoid(1, 1, isolation));
    EXPECT_EQ(task.state(), TaskCongestionState::kLongUnavailable);
    const auto detail = task.detail();
    ASSERT_TRUE(detail.controller_generation.observed);
    EXPECT_EQ(detail.controller_generation.value, 2);
    EXPECT_FALSE(task.resolve(1, 0));
    EXPECT_EQ(task.state(), TaskCongestionState::kLongUnavailable);
}

TEST(TaskCongestionObservationTest, RepeatedEventAfterRecoveryIsUnresolved) {
    TaskCongestionObservation task(1, 1);
    const auto evidence = windowEvidence();
    ASSERT_TRUE(task.defer(0, 1, evidence));
    ASSERT_TRUE(task.resolve(0, 1));
    ASSERT_TRUE(task.detail().resolved);

    ASSERT_TRUE(task.defer(0, 1, evidence));
    const auto detail = task.detail();
    EXPECT_EQ(task.state(), TaskCongestionState::kCongested);
    EXPECT_FALSE(detail.resolved);
    ASSERT_TRUE(detail.affected_path.observed);
    EXPECT_EQ(detail.affected_path.value, "route-a");
    EXPECT_EQ(evidence.path, "route-a");
    EXPECT_TRUE(evidence.observed_at_ns.observed);
    EXPECT_EQ(evidence.observed_at_ns.value, 100u);
}

TEST(TaskCongestionObservationTest, NewEvidenceClearsAbsentOptionalFields) {
    TaskCongestionObservation task(1, 1);
    auto previous = windowEvidence();
    previous.failure_scope = {true, TaskCongestionFailureScope::kDevice};
    previous.controller_mode = {true, TaskCongestionControllerMode::kEnforce};
    previous.window_bytes = {true, 128};
    previous.inflight_bytes = {true, 64};
    previous.retry_after_ns = {true, 500};
    ASSERT_TRUE(task.avoid(0, 1, previous));
    ASSERT_TRUE(task.resolve(0, 1));

    TaskCongestionEvidence next;
    next.reason = {true, TaskCongestionReason::kFatal};
    next.path = "x";
    ASSERT_TRUE(task.avoid(0, 1, next));
    const auto detail = task.detail();
    EXPECT_EQ(detail.state, TaskCongestionState::kLongUnavailable);
    EXPECT_FALSE(detail.resolved);
    ASSERT_TRUE(detail.reason.observed);
    EXPECT_EQ(detail.reason.value, TaskCongestionReason::kFatal);
    ASSERT_TRUE(detail.affected_path.observed);
    EXPECT_EQ(detail.affected_path.value, "x");
    EXPECT_FALSE(detail.failure_scope.observed);
    EXPECT_FALSE(detail.controller_mode.observed);
    EXPECT_FALSE(detail.controller_generation.observed);
    EXPECT_FALSE(detail.observed_at_ns.observed);
    EXPECT_FALSE(detail.window_bytes.observed);
    EXPECT_FALSE(detail.inflight_bytes.observed);
    EXPECT_FALSE(detail.retry_after_ns.observed);
    EXPECT_EQ(next.path, "x");
    EXPECT_TRUE(next.reason.observed);
    EXPECT_EQ(next.reason.value, TaskCongestionReason::kFatal);

    ASSERT_TRUE(task.resolve(0, 1));
    ASSERT_TRUE(task.defer(0, 1, TaskCongestionEvidence{}));
    const auto empty = task.detail();
    EXPECT_FALSE(empty.reason.observed);
    EXPECT_FALSE(empty.affected_path.observed);
    EXPECT_FALSE(empty.resolved);
}

TEST(TaskCongestionObservationTest,
     EquivalentPathRepresentationsRecordCurrentEvent) {
    static const std::string path = "route-a";
    auto owned = windowEvidence();
    auto referenced = owned;
    referenced.path.reset();
    referenced.interned_path = &path;
    for (bool reference_first : {false, true}) {
        TaskCongestionObservation task(2, 1);
        ASSERT_TRUE(task.defer(0, 1, reference_first ? referenced : owned));
        ASSERT_TRUE(task.defer(1, 1, windowEvidence()));
        ASSERT_TRUE(task.defer(0, 1, reference_first ? owned : referenced));
        ASSERT_TRUE(task.resolve(0, 1));
        ASSERT_TRUE(task.resolve(1, 1));
        const auto detail = task.detail();
        ASSERT_TRUE(detail.slice_id.observed);
        EXPECT_EQ(detail.slice_id.value, 0u);
        EXPECT_TRUE(detail.resolved);
    }
}

TEST(TaskCongestionObservationTest, EveryEvidenceFieldChangesTheLastEvent) {
    auto base = windowEvidence();
    base.failure_scope = {true, TaskCongestionFailureScope::kDevice};
    base.controller_mode = {true, TaskCongestionControllerMode::kEnforce};
    base.window_bytes = {true, 128};
    base.inflight_bytes = {true, 64};
    base.retry_after_ns = {true, 500};
    std::vector<TaskCongestionEvidence> changed(9, base);
    changed[0].reason = {};
    changed[1].failure_scope = {};
    changed[2].path.reset();
    changed[3].observed_at_ns = {};
    changed[4].controller_mode = {};
    changed[5].controller_generation = {};
    changed[6].window_bytes = {};
    changed[7].inflight_bytes = {};
    changed[8].retry_after_ns = {};
    for (size_t field = 0; field < changed.size(); ++field) {
        SCOPED_TRACE(field);
        TaskCongestionObservation task(2, 1);
        ASSERT_TRUE(task.defer(0, 1, base));
        ASSERT_TRUE(task.defer(1, 1, base));
        ASSERT_TRUE(task.defer(0, 1, changed[field]));
        ASSERT_TRUE(task.resolve(0, 1));
        ASSERT_TRUE(task.resolve(1, 1));
        const auto detail = task.detail();
        ASSERT_TRUE(detail.slice_id.observed);
        EXPECT_EQ(detail.slice_id.value, 0u);
    }
}

TEST(TaskCongestionObservationTest,
     EmptyPathIsObservedAndConstReferenceIsOwned) {
    TaskCongestionObservation task(1, 1);
    ASSERT_TRUE(task.defer(0, 1, TaskCongestionEvidence{}));
    EXPECT_FALSE(task.detail().affected_path.observed);
    {
        TaskCongestionEvidence empty;
        empty.path = std::string{};
        ASSERT_TRUE(task.defer(0, 1, empty));
    }
    ASSERT_TRUE(task.detail().affected_path.observed);
    EXPECT_TRUE(task.detail().affected_path.value.empty());
    {
        TaskCongestionEvidence temporary;
        temporary.path = std::string(256, 'a');
        ASSERT_TRUE(task.defer(0, 1, temporary));
    }
    ASSERT_TRUE(task.resolve(0, 1));
    const auto detail = task.detail();
    ASSERT_TRUE(detail.affected_path.observed);
    EXPECT_EQ(detail.affected_path.value, std::string(256, 'a'));
    EXPECT_TRUE(detail.resolved);
}

TEST(TaskCongestionObservationTest, OwnedEvidenceExtendsOnlyCurrentAttempt) {
    TaskCongestionObservation task(1, 2);
    EXPECT_FALSE(task.defer(2, 1, windowEvidence(), 3));
    EXPECT_FALSE(task.avoid(2, 3, windowEvidence(), 3));
    EXPECT_EQ(task.state(), TaskCongestionState::kNormal);
    EXPECT_FALSE(task.detail().reason.observed);

    ASSERT_TRUE(task.defer(2, 2, windowEvidence(), 3));
    EXPECT_EQ(task.state(), TaskCongestionState::kCongested);
    ASSERT_TRUE(task.resolve(2, 2));
    ASSERT_TRUE(task.avoid(2, 2, windowEvidence(), 1));
    EXPECT_EQ(task.state(), TaskCongestionState::kLongUnavailable);
    const auto detail = task.detail();
    ASSERT_TRUE(detail.slice_id.observed);
    EXPECT_EQ(detail.slice_id.value, 2u);
    EXPECT_FALSE(detail.resolved);
    ASSERT_TRUE(task.resolve(2, 2));
    EXPECT_EQ(task.state(), TaskCongestionState::kNormal);
    EXPECT_TRUE(task.detail().resolved);
}

TEST(TaskCongestionObservationTest, EveryEvidenceFieldIsReturned) {
    TaskCongestionObservation task(1, 1);
    static const std::string path = "interned-route";
    auto evidence = windowEvidence();
    evidence.path.reset();
    evidence.interned_path = &path;
    evidence.failure_scope = {true, TaskCongestionFailureScope::kDevice};
    evidence.controller_mode = {true, TaskCongestionControllerMode::kEnforce};
    evidence.window_bytes = {true, 128};
    evidence.inflight_bytes = {true, 64};
    evidence.retry_after_ns = {true, 500};
    ASSERT_TRUE(task.defer(0, 1, evidence));
    const auto detail = task.detail();
    EXPECT_TRUE(detail.reason.observed);
    EXPECT_TRUE(detail.failure_scope.observed);
    EXPECT_TRUE(detail.affected_path.observed);
    EXPECT_TRUE(detail.observed_at_ns.observed);
    EXPECT_TRUE(detail.controller_mode.observed);
    EXPECT_TRUE(detail.controller_generation.observed);
    EXPECT_TRUE(detail.window_bytes.observed);
    EXPECT_TRUE(detail.inflight_bytes.observed);
    EXPECT_TRUE(detail.retry_after_ns.observed);
    EXPECT_EQ(detail.reason.value, evidence.reason.value);
    EXPECT_EQ(detail.failure_scope.value, evidence.failure_scope.value);
    EXPECT_EQ(detail.affected_path.value, path);
    EXPECT_EQ(detail.observed_at_ns.value, 100u);
    EXPECT_EQ(detail.controller_mode.value, evidence.controller_mode.value);
    EXPECT_EQ(detail.controller_generation.value, 7u);
    EXPECT_EQ(detail.window_bytes.value, 128u);
    EXPECT_EQ(detail.inflight_bytes.value, 64u);
    EXPECT_EQ(detail.retry_after_ns.value, 500u);
}

TEST(TaskCongestionObservationTest, ZeroValuedEvidenceRemainsObserved) {
    TaskCongestionObservation task(2, 1);
    TaskCongestionEvidence evidence;
    evidence.reason = {true, TaskCongestionReason::kUnknown};
    evidence.failure_scope = {true, TaskCongestionFailureScope::kOperation};
    evidence.observed_at_ns = {true, 0};
    evidence.controller_mode = {true, TaskCongestionControllerMode::kOff};
    evidence.controller_generation = {true, 0};
    evidence.window_bytes = {true, 0};
    evidence.inflight_bytes = {true, 0};
    evidence.retry_after_ns = {true, 0};
    ASSERT_TRUE(task.defer(0, 1, evidence));
    ASSERT_TRUE(task.defer(1, 1, TaskCongestionEvidence{}));
    const auto detail = task.detail();
    EXPECT_EQ(detail.slice_id.value, 0u);
    EXPECT_TRUE(detail.reason.observed);
    EXPECT_EQ(detail.reason.value, TaskCongestionReason::kUnknown);
    EXPECT_TRUE(detail.failure_scope.observed);
    EXPECT_EQ(detail.failure_scope.value,
              TaskCongestionFailureScope::kOperation);
    EXPECT_TRUE(detail.observed_at_ns.observed);
    EXPECT_EQ(detail.observed_at_ns.value, 0u);
    EXPECT_TRUE(detail.controller_mode.observed);
    EXPECT_EQ(detail.controller_mode.value, TaskCongestionControllerMode::kOff);
    EXPECT_TRUE(detail.controller_generation.observed);
    EXPECT_EQ(detail.controller_generation.value, 0u);
    EXPECT_TRUE(detail.window_bytes.observed);
    EXPECT_EQ(detail.window_bytes.value, 0u);
    EXPECT_TRUE(detail.inflight_bytes.observed);
    EXPECT_EQ(detail.inflight_bytes.value, 0u);
    EXPECT_TRUE(detail.retry_after_ns.observed);
    EXPECT_EQ(detail.retry_after_ns.value, 0u);
    EXPECT_FALSE(detail.affected_path.observed);

    ASSERT_TRUE(task.resolve(0, 1));
    ASSERT_TRUE(task.defer(0, 1, evidence));
    ASSERT_TRUE(task.resolve(0, 1));
    ASSERT_TRUE(task.beginAttempt(2, 2));
    EXPECT_FALSE(task.avoid(0, 1, windowEvidence()));
    const auto retained = task.detail();
    EXPECT_EQ(retained.state, TaskCongestionState::kNormal);
    EXPECT_EQ(retained.attempt_id.value, 2u);
    EXPECT_TRUE(retained.resolved);
    EXPECT_TRUE(retained.observed_at_ns.observed);
    EXPECT_EQ(retained.observed_at_ns.value, 0u);
    EXPECT_TRUE(retained.retry_after_ns.observed);
    EXPECT_EQ(retained.retry_after_ns.value, 0u);
}

}  // namespace
}  // namespace mooncake::adaptive_congestion_control
