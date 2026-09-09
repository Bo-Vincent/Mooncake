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

#include "adaptive_congestion_control.h"
#include "tent/transport/rdma/endpoint.h"
#include "tent/transport/rdma/slice.h"
#include "tent/transport/rdma/workers.h"

namespace mooncake::tent {
namespace {

adaptive_cc::Config testConfig() {
    adaptive_cc::Config config;
    config.mode = adaptive_cc::Mode::kEnforce;
    config.min_window_bytes = 1024;
    config.max_window_bytes = 1024;
    return config;
}

TEST(TentAdaptiveCongestionControlTest, AttemptReleasesOnce) {
    adaptive_cc::DomainState device(testConfig(), 3);
    auto route = std::make_shared<TentRdmaCcRoute>(testConfig());
    RdmaSlice slice;
    slice.length = 512;

    adaptive_cc::PathHandle path{&device, &route->domain, 3, 1};
    ASSERT_EQ(acquireTentCcAttempt(slice, route.get(), path, 512, 7),
              adaptive_cc::Decision::kAllow);
    EXPECT_TRUE(completeTentCcAttempt(slice,
                                      adaptive_cc::OutcomeClass::kSuccess,
                                      adaptive_cc::FailureScope::kOperation));
    EXPECT_FALSE(completeTentCcAttempt(slice,
                                       adaptive_cc::OutcomeClass::kSuccess,
                                       adaptive_cc::FailureScope::kOperation));
    EXPECT_EQ(route->completed_bytes.load(std::memory_order_relaxed), 512u);
    EXPECT_EQ(adaptive_cc::snapshot(device).inflight_bytes, 0);
    EXPECT_EQ(adaptive_cc::snapshot(route->domain).inflight_bytes, 0);
}

TEST(TentAdaptiveCongestionControlTest, OldEndpointFailureIsIgnored) {
    adaptive_cc::DomainState device(testConfig(), 3);
    auto route = std::make_shared<TentRdmaCcRoute>(testConfig());
    RdmaSlice slice;

    adaptive_cc::PathHandle path{&device, &route->domain, 3, 1};
    ASSERT_EQ(acquireTentCcAttempt(slice, route.get(), path, 512, 7),
              adaptive_cc::Decision::kAllow);
    route->endpoint_generation.store(8, std::memory_order_release);
    EXPECT_TRUE(completeTentCcAttempt(slice, adaptive_cc::OutcomeClass::kFatal,
                                      adaptive_cc::FailureScope::kQp));
    EXPECT_EQ(adaptive_cc::snapshot(route->domain).inflight_bytes, 0);

    adaptive_cc::controlTick(route->domain, 1);
    EXPECT_EQ(adaptive_cc::snapshot(route->domain).state,
              adaptive_cc::PathState::kHealthy);
}

TEST(TentAdaptiveCongestionControlTest,
     CurrentEndpointFailureQuarantinesRoute) {
    auto config = testConfig();
    config.hard_error_threshold = 1;
    adaptive_cc::DomainState device(config, 3);
    auto route = std::make_shared<TentRdmaCcRoute>(config);
    RdmaSlice slice;
    adaptive_cc::PathHandle path{&device, &route->domain, 3, 1};

    ASSERT_EQ(acquireTentCcAttempt(slice, route.get(), path, 512, 7),
              adaptive_cc::Decision::kAllow);
    EXPECT_TRUE(completeTentCcAttempt(slice, adaptive_cc::OutcomeClass::kFatal,
                                      adaptive_cc::FailureScope::kQp));
    adaptive_cc::controlTick(route->domain, 1);

    EXPECT_EQ(adaptive_cc::snapshot(route->domain).state,
              adaptive_cc::PathState::kQuarantined);
}

TEST(TentAdaptiveCongestionControlTest, LargeSliceCanMakeProgressDuringProbe) {
    adaptive_cc::Config config;
    config.mode = adaptive_cc::Mode::kEnforce;
    config.cooldown_ns = 1;
    adaptive_cc::DomainState device(config);
    TentRdmaCcRoute route(config);
    adaptive_cc::Signals failure;
    failure.fatal_failures = 1;
    adaptive_cc::recordSignals(route.domain, 1, failure);
    adaptive_cc::controlTick(route.domain, 1);
    adaptive_cc::controlTick(route.domain, 2);
    ASSERT_EQ(adaptive_cc::snapshot(route.domain).state,
              adaptive_cc::PathState::kProbing);
    RdmaSlice first, second;
    // A default 4 MiB write is capped at 32 slices of 128 KiB each.
    first.length = second.length = 128ULL << 10;
    adaptive_cc::PathHandle path{&device, &route.domain, 1, 1};
    ASSERT_EQ(acquireTentCcAttempt(first, &route, path, first.length, 7),
              adaptive_cc::Decision::kAllow);
    EXPECT_EQ(acquireTentCcAttempt(second, &route, path, second.length, 7),
              adaptive_cc::Decision::kDefer);
    ASSERT_TRUE(completeTentCcAttempt(first,
                                      adaptive_cc::OutcomeClass::kSuccess,
                                      adaptive_cc::FailureScope::kOperation));
    tickTentCcRoute(route, 3, 1);
    EXPECT_EQ(adaptive_cc::snapshot(route.domain).state,
              adaptive_cc::PathState::kHealthy);
    EXPECT_EQ(adaptive_cc::snapshot(device).inflight_bytes, 0u);
}

TEST(TentAdaptiveCongestionControlTest, EndpointUsesOneSharedRouteState) {
    TentRdmaCcRoute first(testConfig());
    TentRdmaCcRoute second(testConfig());
    RdmaEndPoint endpoint;

    EXPECT_EQ(endpoint.bindCongestionRoute(&first), &first);
    EXPECT_EQ(endpoint.bindCongestionRoute(&second), &first);
}

TEST(TentAdaptiveCongestionControlTest, RouteTelemetryGrowsAndShrinksWindow) {
    adaptive_cc::Config config;
    config.mode = adaptive_cc::Mode::kEnforce;
    config.min_window_bytes = 64;
    config.max_window_bytes = 128;
    config.target_drain_time_ns = 1'000;
    config.low_pressure_epochs = 2;
    config.high_pressure_epochs = 2;
    adaptive_cc::DomainState device(config, 1);
    TentRdmaCcRoute route(config);

    for (uint64_t tick = 1; tick <= 2; ++tick) {
        RdmaSlice slice;
        slice.length = 16;
        adaptive_cc::PathHandle path{&device, &route.domain,
                                     adaptive_cc::generation(device),
                                     adaptive_cc::generation(route.domain)};
        ASSERT_EQ(acquireTentCcAttempt(slice, &route, path, slice.length, 7),
                  adaptive_cc::Decision::kAllow);
        ASSERT_TRUE(
            completeTentCcAttempt(slice, adaptive_cc::OutcomeClass::kSuccess,
                                  adaptive_cc::FailureScope::kOperation));
        tickTentCcRoute(route, tick, 1'000'000'000);
    }
    EXPECT_GT(adaptive_cc::snapshot(route.domain).window_bytes, 64u);

    RdmaSlice inflight;
    inflight.length = 64;
    adaptive_cc::PathHandle path{&device, &route.domain,
                                 adaptive_cc::generation(device),
                                 adaptive_cc::generation(route.domain)};
    ASSERT_EQ(acquireTentCcAttempt(inflight, &route, path, inflight.length, 7),
              adaptive_cc::Decision::kAllow);
    route.completed_bytes.store(1, std::memory_order_relaxed);
    tickTentCcRoute(route, 3, 1'000'000'000);
    route.completed_bytes.store(1, std::memory_order_relaxed);
    tickTentCcRoute(route, 4, 1'000'000'000);

    EXPECT_EQ(adaptive_cc::snapshot(route.domain).state,
              adaptive_cc::PathState::kCongested);
    EXPECT_EQ(adaptive_cc::snapshot(route.domain).window_bytes, 64u);
    EXPECT_TRUE(completeTentCcAttempt(inflight,
                                      adaptive_cc::OutcomeClass::kDerivedFlush,
                                      adaptive_cc::FailureScope::kOperation));
}

}  // namespace
}  // namespace mooncake::tent
