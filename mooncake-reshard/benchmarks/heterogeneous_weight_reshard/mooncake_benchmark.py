"""Executable source/target roles for the Mooncake heterogeneous benchmark."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from collections.abc import Mapping, Sequence
from uuid import uuid4

from .case_spec import BenchmarkCase, BenchmarkConfig
from .mooncake_runtime import (
    LiveMeshAllocation,
    handle_target_connection,
    run_source_session,
)


class BenchmarkExecutionError(RuntimeError):
    pass


def parse_cuda_devices(value: str) -> tuple[int, ...]:
    if type(value) is not str or not value or value != value.strip():
        raise BenchmarkExecutionError("CUDA devices must be unique non-negative IDs")
    devices = []
    for item in value.split(","):
        try:
            device = int(item)
        except ValueError as error:
            raise BenchmarkExecutionError(
                "CUDA devices must be unique non-negative IDs"
            ) from error
        if str(device) != item or device < 0 or device in devices:
            raise BenchmarkExecutionError(
                "CUDA devices must be unique non-negative IDs"
            )
        devices.append(device)
    if not devices:
        raise BenchmarkExecutionError("CUDA devices must be unique non-negative IDs")
    return tuple(devices)


def ensure_execution_authorized(
    config: BenchmarkConfig, environ: Mapping[str, object]
) -> None:
    if not config.execution_is_authorized(environ):
        raise BenchmarkExecutionError(
            "benchmark execution is not authorized by config and environment guard"
        )


def _positive_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("port must be an integer") from error
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _positive_timeout(value: str) -> float:
    try:
        timeout = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("timeout must be numeric") from error
    if timeout <= 0:
        raise argparse.ArgumentTypeError("timeout must be positive")
    return timeout


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Mooncake placement/binding heterogeneous weight benchmark roles"
    )
    parser.add_argument("--config", required=True)
    subparsers = parser.add_subparsers(dest="role", required=True)

    target = subparsers.add_parser("target")
    target.add_argument("--bind-host", default="0.0.0.0")
    target.add_argument("--control-port", required=True, type=_positive_port)
    target.add_argument("--engine-host", required=True)
    target.add_argument("--protocol", choices=("rdma", "tcp"), default="rdma")
    target.add_argument("--device", default="")
    target.add_argument("--cuda-devices", required=True)
    target.add_argument("--timeout", type=_positive_timeout, default=300.0)

    source = subparsers.add_parser("source")
    source.add_argument("--case", required=True)
    source.add_argument("--target-control-host", required=True)
    source.add_argument("--target-control-port", required=True, type=_positive_port)
    source.add_argument("--engine-host", required=True)
    source.add_argument("--protocol", choices=("rdma", "tcp"), default="rdma")
    source.add_argument("--device", default="")
    source.add_argument("--cuda-devices", required=True)
    source.add_argument("--timeout", type=_positive_timeout, default=300.0)
    source.add_argument("--phase", choices=("cold", "steady"), default="steady")
    return parser


def _initialize_engine(host: str, protocol: str, device: str):
    from mooncake.engine import TransferEngine

    engine = TransferEngine()
    started = time.perf_counter()
    result = engine.initialize(host, "P2PHANDSHAKE", protocol, device)
    elapsed = time.perf_counter() - started
    if result != 0:
        raise BenchmarkExecutionError(
            f"TransferEngine initialize failed for {host}: {result}"
        )
    endpoint = f"{host}:{engine.get_rpc_port()}"
    return engine, endpoint, elapsed


def _physical_case(config: BenchmarkConfig, case_id: str) -> BenchmarkCase:
    matches = [case for case in config.physical_cases if case.id == case_id]
    if len(matches) != 1:
        raise BenchmarkExecutionError(f"unknown physical case: {case_id}")
    return matches[0]


def _connect(host: str, port: int, timeout_s: float) -> socket.socket:
    deadline = time.monotonic() + timeout_s
    last_error: OSError | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BenchmarkExecutionError(
                f"target control connection timed out: {last_error}"
            )
        try:
            return socket.create_connection((host, port), timeout=min(5.0, remaining))
        except OSError as error:
            last_error = error
            time.sleep(min(0.25, remaining))


def _run_target(args: argparse.Namespace) -> None:
    process_started = time.perf_counter()
    engine, endpoint, transport_init_seconds = _initialize_engine(
        args.engine_host, args.protocol, args.device
    )
    devices = parse_cuda_devices(args.cuda_devices)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.bind_host, args.control_port))
        listener.listen(1)
        listener.settimeout(args.timeout)
        print(
            "TARGET_CONTROL_READY="
            + json.dumps(
                {
                    "control_host": args.bind_host,
                    "control_port": args.control_port,
                    "engine_endpoint": endpoint,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        connection, _ = listener.accept()
        with connection:
            handle_target_connection(
                connection,
                engine=engine,
                endpoint=endpoint,
                cuda_devices=devices,
                transport_init_seconds=transport_init_seconds,
                timeout_s=args.timeout,
                process_started=process_started,
            )


def _run_source(args: argparse.Namespace, config: BenchmarkConfig) -> dict[str, object]:
    process_started = time.perf_counter()
    case = _physical_case(config, args.case)
    engine, endpoint, transport_init_seconds = _initialize_engine(
        args.engine_host, args.protocol, args.device
    )
    devices = parse_cuda_devices(args.cuda_devices)
    session_id = uuid4().hex
    generation = time.time_ns()
    revision = f"benchmark:{case.id}:{session_id}"
    with LiveMeshAllocation.open(
        case=case,
        side="source",
        engine=engine,
        endpoint=endpoint,
        cuda_devices=devices,
        revision=revision,
        session_id=session_id,
        generation=generation,
    ) as source_allocation:
        source_allocation.fill_source_pattern()
        with _connect(
            args.target_control_host,
            args.target_control_port,
            args.timeout,
        ) as connection:
            result = run_source_session(
                connection,
                case=case,
                revision=revision,
                session_id=session_id,
                generation=generation,
                engine=engine,
                source_allocation=source_allocation,
                source_transport_init_seconds=transport_init_seconds,
                warmups=1 if args.phase == "cold" else config.warmups,
                iterations=0 if args.phase == "cold" else config.iterations,
                timeout_s=args.timeout,
                one_shot=args.phase == "cold",
                process_started=process_started,
            )
    print("BENCHMARK_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    return result


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    args = _parser().parse_args(argv)
    config = BenchmarkConfig.from_json_file(args.config)
    ensure_execution_authorized(config, os.environ if environ is None else environ)
    if args.role == "target":
        _run_target(args)
        return None
    return _run_source(args, config)


if __name__ == "__main__":
    try:
        main()
    except (BenchmarkExecutionError, ValueError, RuntimeError) as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
