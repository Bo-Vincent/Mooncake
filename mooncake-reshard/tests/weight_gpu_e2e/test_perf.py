from __future__ import annotations

import os
import socket
from contextlib import ExitStack
from typing import Mapping

import pytest

from mooncake.reshard.weight import WeightStore

from .buffers import (
    CudaRuntime,
    ManagedBuffer,
    _cuda_rank_buffers,
    _parse_cuda_devices,
    _registered_store_buffers,
)
from .execution import (
    _run_store_iteration,
    _run_te_iteration,
)
from .manifests import _tensor
from .perf import (
    _assert_perf_gate,
    _collect_perf_samples,
    _emit_perf_result,
    _perf_config_from_environ,
    _perf_result_payload,
    _store_perf_result_payload,
    _summarize_perf_samples,
)


@pytest.mark.skipif(
    os.getenv("MOONCAKE_WEIGHT_PERF_STORE_E2E") != "1",
    reason="set MOONCAKE_WEIGHT_PERF_STORE_E2E=1 to run Store performance E2E",
)
def test_gpu_store_heterogeneous_tp_performance() -> None:
    from mooncake.store import MooncakeDistributedStore

    config = _perf_config_from_environ(os.environ)
    source_devices = _parse_cuda_devices(
        os.environ,
        "MOONCAKE_WEIGHT_SOURCE_CUDA_DEVICES",
        default="0",
    )
    target_devices = _parse_cuda_devices(
        os.environ,
        "MOONCAKE_WEIGHT_TARGET_CUDA_DEVICES",
        default="0",
    )
    runtimes = {
        device: CudaRuntime(device)
        for device in sorted(set(source_devices) | set(target_devices))
    }
    protocol = os.getenv("MOONCAKE_WEIGHT_PROTOCOL", "tcp")
    store = MooncakeDistributedStore()
    result = store.setup(
        os.getenv(
            "MOONCAKE_WEIGHT_LOCAL_HOSTNAME", socket.gethostbyname(socket.gethostname())
        ),
        os.getenv(
            "MOONCAKE_WEIGHT_METADATA_SERVER",
            "http://127.0.0.1:8080/metadata",
        ),
        max(128 * 1024 * 1024, config.total_bytes * 2),
        64 * 1024 * 1024,
        protocol,
        os.getenv("MOONCAKE_WEIGHT_DEVICE", "eth0"),
        os.getenv("MOONCAKE_WEIGHT_MASTER", "127.0.0.1:50051"),
    )
    assert result == 0
    try:
        for source_tp, target_tp in config.topologies:
            if config.total_bytes % source_tp or config.total_bytes % target_tp:
                raise ValueError("performance bytes must divide every TP topology")
            with ExitStack() as stack:
                source_buffers, source_rank_devices = _cuda_rank_buffers(
                    stack,
                    runtimes,
                    source_devices,
                    ranks=source_tp,
                    size=config.total_bytes // source_tp,
                )
                target_buffers, target_rank_devices = _cuda_rank_buffers(
                    stack,
                    runtimes,
                    target_devices,
                    ranks=target_tp,
                    size=config.total_bytes // target_tp,
                )
                stack.enter_context(
                    _registered_store_buffers(
                        store,
                        [*source_buffers, *target_buffers],
                    )
                )
                for rank, buffer in enumerate(source_buffers):
                    buffer.fill(rank + 1)

                tensor = _tensor(config.total_bytes)
                samples = _collect_perf_samples(
                    warmups=config.warmups,
                    iterations=config.iterations,
                    run_once=lambda _: _run_store_iteration(
                        store=store,
                        weight_store=WeightStore(store),
                        tensor=tensor,
                        source_buffers=source_buffers,
                        target_buffers=target_buffers,
                        namespace=f"perf-tp{source_tp}-to-tp{target_tp}",
                        pre_registered=True,
                    ),
                )
                logical_bytes = {phase: 0 for phase in samples}
                logical_bytes.update(
                    {
                        "upload": config.total_bytes,
                        "load": config.total_bytes,
                        "e2e": config.total_bytes * 2,
                    }
                )
                payload = _store_perf_result_payload(
                    protocol=protocol,
                    multi_gpu=(
                        len(set(source_rank_devices) | set(target_rank_devices)) > 1
                    ),
                    environ=os.environ,
                    total_bytes=config.total_bytes,
                    source_tp=source_tp,
                    target_tp=target_tp,
                    warmups=config.warmups,
                    iterations=config.iterations,
                    samples=samples,
                    logical_bytes=logical_bytes,
                    placement={
                        "source_cuda_devices": list(source_rank_devices),
                        "target_cuda_devices": list(target_rank_devices),
                    },
                )
                _emit_perf_result(payload)
                for phase in ("upload", "load"):
                    _assert_perf_gate(
                        _summarize_perf_samples(
                            phase=phase,
                            logical_bytes=logical_bytes[phase],
                            samples=samples[phase],
                        ),
                        max_cv=config.max_cv,
                        max_p95_p50=config.max_p95_p50,
                        min_gbps=config.min_store_gbps,
                    )
                _assert_perf_gate(
                    _summarize_perf_samples(
                        phase="e2e",
                        logical_bytes=logical_bytes["e2e"],
                        samples=samples["e2e"],
                    ),
                    max_cv=float("inf"),
                    max_p95_p50=float("inf"),
                    min_gbps=config.min_store_gbps,
                )
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            assert close() == 0


@pytest.mark.skipif(
    os.getenv("MOONCAKE_WEIGHT_PERF_TE_E2E") != "1",
    reason="set MOONCAKE_WEIGHT_PERF_TE_E2E=1 to run TE performance E2E",
)
def test_te_tcp_heterogeneous_tp_performance() -> None:
    from mooncake.engine import TransferEngine

    config = _perf_config_from_environ(os.environ)
    local_hostname = os.getenv(
        "MOONCAKE_WEIGHT_LOCAL_HOSTNAME",
        socket.gethostbyname(socket.gethostname()),
    )
    source_engine = TransferEngine()
    target_engine = TransferEngine()
    assert source_engine.initialize(local_hostname, "P2PHANDSHAKE", "tcp", "") == 0
    assert target_engine.initialize(local_hostname, "P2PHANDSHAKE", "tcp", "") == 0
    assert source_engine.get_rpc_port() != target_engine.get_rpc_port()
    target_endpoint = f"{local_hostname}:{target_engine.get_rpc_port()}"

    for source_tp, target_tp in config.topologies:
        if config.total_bytes % source_tp or config.total_bytes % target_tp:
            raise ValueError("performance bytes must divide every TP topology")
        with ExitStack() as stack:
            source_buffers = [
                stack.enter_context(
                    ManagedBuffer(source_engine, config.total_bytes // source_tp)
                )
                for _ in range(source_tp)
            ]
            target_buffers = [
                stack.enter_context(
                    ManagedBuffer(target_engine, config.total_bytes // target_tp)
                )
                for _ in range(target_tp)
            ]
            for rank, buffer in enumerate(source_buffers):
                buffer.fill(rank + 1)

            tensor = _tensor(config.total_bytes)
            samples = _collect_perf_samples(
                warmups=config.warmups,
                iterations=config.iterations,
                run_once=lambda _: _run_te_iteration(
                    source_engine=source_engine,
                    target_endpoint=target_endpoint,
                    tensor=tensor,
                    source_buffers=source_buffers,
                    target_buffers=target_buffers,
                ),
            )
            logical_bytes = {"plan": 0, "transfer": config.total_bytes}
            logical_bytes["e2e"] = config.total_bytes
            payload = _perf_result_payload(
                backend="te-tcp-managed",
                total_bytes=config.total_bytes,
                source_tp=source_tp,
                target_tp=target_tp,
                warmups=config.warmups,
                iterations=config.iterations,
                samples=samples,
                logical_bytes=logical_bytes,
            )
            _emit_perf_result(payload)
            _assert_perf_gate(
                _summarize_perf_samples(
                    phase="transfer",
                    logical_bytes=config.total_bytes,
                    samples=samples["transfer"],
                ),
                max_cv=config.max_cv,
                max_p95_p50=config.max_p95_p50,
                min_gbps=config.min_te_gbps,
            )


def test_perf_metrics_report_percentiles_bandwidth_and_variation() -> None:
    summary = _summarize_perf_samples(
        phase="load",
        logical_bytes=1_000_000_000,
        samples=(1.0, 2.0, 3.0, 4.0),
    )

    assert summary.phase == "load"
    assert summary.iterations == 4
    assert summary.mean_seconds == pytest.approx(2.5)
    assert summary.p50_seconds == pytest.approx(2.5)
    assert summary.p95_seconds == pytest.approx(3.85)
    assert summary.p50_gbps == pytest.approx(0.4)
    assert summary.cv == pytest.approx(0.4472135955)
    assert summary.p95_p50_ratio == pytest.approx(1.54)


def test_perf_metrics_allow_latency_only_phase() -> None:
    summary = _summarize_perf_samples(
        phase="commit",
        logical_bytes=0,
        samples=(0.001, 0.002),
    )

    assert summary.logical_bytes == 0
    assert summary.p50_gbps is None


def test_perf_result_payload_is_machine_readable() -> None:
    payload = _perf_result_payload(
        backend="te-tcp-managed",
        total_bytes=1_000,
        source_tp=2,
        target_tp=4,
        warmups=1,
        iterations=2,
        samples={"plan": (0.001, 0.002), "transfer": (0.1, 0.2)},
        logical_bytes={"plan": 0, "transfer": 1_000},
        placement={"source_cuda_devices": [0, 1], "target_cuda_devices": [2, 3]},
    )

    assert payload["schema_version"] == 1
    assert payload["topology"] == {"source_tp": 2, "target_tp": 4}
    assert payload["phases"]["plan"]["p50_gbps"] is None
    assert payload["phases"]["transfer"]["p50_gbps"] == pytest.approx(0.0000066666667)
    assert payload["placement"] == {
        "source_cuda_devices": [0, 1],
        "target_cuda_devices": [2, 3],
    }


@pytest.mark.parametrize(
    "protocol,multi_gpu,environ,expected_backend,expected_transport",
    [
        (
            "tcp",
            False,
            {},
            "store-cuda-preregistered",
            {
                "requested_protocol": "tcp",
                "strategy": "runtime-selected",
                "mc_store_memcpy": "auto",
            },
        ),
        (
            "tcp",
            True,
            {"MC_STORE_MEMCPY": "1"},
            "store-cuda-preregistered-multi-gpu",
            {
                "requested_protocol": "tcp",
                "strategy": "runtime-selected",
                "mc_store_memcpy": "1",
            },
        ),
        (
            "rdma",
            True,
            {"MC_STORE_MEMCPY": "0"},
            "store-cuda-preregistered-multi-gpu",
            {
                "requested_protocol": "rdma",
                "strategy": "runtime-selected",
                "mc_store_memcpy": "0",
            },
        ),
    ],
)
def test_store_perf_payload_separates_protocol_from_runtime_strategy(
    protocol: str,
    multi_gpu: bool,
    environ: Mapping[str, str],
    expected_backend: str,
    expected_transport: Mapping[str, object],
) -> None:
    payload = _store_perf_result_payload(
        protocol=protocol,
        multi_gpu=multi_gpu,
        environ=environ,
        total_bytes=1_000,
        source_tp=2,
        target_tp=4,
        warmups=1,
        iterations=2,
        samples={"load": (0.1, 0.2)},
        logical_bytes={"load": 1_000},
        placement={"source_cuda_devices": [0, 1]},
    )

    assert payload["backend"] == expected_backend
    assert payload["transport"] == expected_transport


def test_perf_config_parses_explicit_environment() -> None:
    config = _perf_config_from_environ(
        {
            "MOONCAKE_WEIGHT_PERF_BYTES": "268435456",
            "MOONCAKE_WEIGHT_PERF_WARMUPS": "2",
            "MOONCAKE_WEIGHT_PERF_ITERATIONS": "10",
            "MOONCAKE_WEIGHT_PERF_MAX_CV": "0.10",
            "MOONCAKE_WEIGHT_PERF_MAX_P95_P50": "1.25",
            "MOONCAKE_WEIGHT_PERF_MIN_STORE_GBPS": "3.5",
            "MOONCAKE_WEIGHT_PERF_MIN_TE_GBPS": "5.25",
            "MOONCAKE_WEIGHT_PERF_TOPOLOGIES": "4:2,2:4,4:8",
        }
    )

    assert config.total_bytes == 256 * 1024 * 1024
    assert config.warmups == 2
    assert config.iterations == 10
    assert config.max_cv == pytest.approx(0.10)
    assert config.max_p95_p50 == pytest.approx(1.25)
    assert config.min_store_gbps == pytest.approx(3.5)
    assert config.min_te_gbps == pytest.approx(5.25)
    assert config.topologies == ((4, 2), (2, 4), (4, 8))


def test_perf_config_defaults_match_remote_stability_gate() -> None:
    config = _perf_config_from_environ({})

    assert config.warmups == 3
    assert config.iterations == 30
    assert config.max_cv == pytest.approx(0.20)
    assert config.max_p95_p50 == pytest.approx(1.50)


@pytest.mark.parametrize(
    "name,value",
    [
        ("MOONCAKE_WEIGHT_PERF_BYTES", "0"),
        ("MOONCAKE_WEIGHT_PERF_WARMUPS", "-1"),
        ("MOONCAKE_WEIGHT_PERF_ITERATIONS", "false"),
        ("MOONCAKE_WEIGHT_PERF_MAX_CV", "0"),
        ("MOONCAKE_WEIGHT_PERF_MAX_P95_P50", "nan"),
        ("MOONCAKE_WEIGHT_PERF_MIN_STORE_GBPS", "-0.1"),
        ("MOONCAKE_WEIGHT_PERF_TOPOLOGIES", "4:0"),
    ],
)
def test_perf_config_rejects_invalid_environment(name: str, value: str) -> None:
    with pytest.raises(ValueError, match=name):
        _perf_config_from_environ({name: value})


def test_collect_perf_samples_discards_warmups() -> None:
    calls = []

    def run_once(iteration: int) -> dict[str, float]:
        calls.append(iteration)
        return {"transfer": float(iteration + 1)}

    samples = _collect_perf_samples(warmups=2, iterations=3, run_once=run_once)

    assert calls == [0, 1, 2, 3, 4]
    assert samples == {"transfer": (3.0, 4.0, 5.0)}


def test_perf_gate_accepts_stable_samples_above_minimum_bandwidth() -> None:
    summary = _summarize_perf_samples(
        phase="transfer",
        logical_bytes=1_000_000_000,
        samples=(0.99, 1.0, 1.01),
    )

    _assert_perf_gate(
        summary,
        max_cv=0.02,
        max_p95_p50=1.02,
        min_gbps=0.9,
    )


@pytest.mark.parametrize(
    "samples,max_cv,max_ratio,min_gbps,message",
    [
        ((0.5, 1.0, 1.5), 0.10, 2.0, 0.0, "CV"),
        ((0.5, 0.5, 1.5), 1.0, 1.25, 0.0, "p95/p50"),
        ((1.0, 1.0, 1.0), 0.1, 1.1, 1.1, "GB/s"),
    ],
)
def test_perf_gate_rejects_unstable_or_slow_samples(
    samples: tuple[float, ...],
    max_cv: float,
    max_ratio: float,
    min_gbps: float,
    message: str,
) -> None:
    summary = _summarize_perf_samples(
        phase="transfer",
        logical_bytes=1_000_000_000,
        samples=samples,
    )

    with pytest.raises(AssertionError, match=message):
        _assert_perf_gate(
            summary,
            max_cv=max_cv,
            max_p95_p50=max_ratio,
            min_gbps=min_gbps,
        )
