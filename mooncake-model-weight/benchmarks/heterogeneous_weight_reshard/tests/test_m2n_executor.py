from __future__ import annotations

import sys

import pytest

from benchmarks.heterogeneous_weight_reshard.m2n_executor import (
    M2NOutputError,
    execute_m2n,
    parse_m2n_stdout,
)


def _successful_stdout(*, validate: bool = True) -> bytes:
    validation = ""
    if validate:
        validation = """\
[Rank 5] VALIDATION PASSED: 1048576 bytes correct
iZ2-host:123 [0] NCCL INFO Failed to load external plugin libnccl-gin.so
[Rank 4] VALIDATION PASSED: 1048576 bytes correct

*** VALIDATION PASSED ***

"""

    return f"""\
launcher noise before the benchmark
=== Tensor Reshard Benchmark ===
Using: ncclReshardWithWindow (user window API)
Global tensor: [256, 128, 64] (3D)
Source shard dim: 0, Dest shard dim: 0 (same-dim)
Source: 4 ranks = 1 reps x 4 shards, local=[64, 128, 64]
Dest: 2 ranks = 1 reps x 2 shards, local=[128, 128, 64]
Algorithm: RING
Load Balance Mode: UNIFORM
Iterations: 3 (warmup: 1), Validate: {"yes" if validate else "no"}

Running 1 warmup iterations...
Warmup complete.
{validation}Running 3 timed iterations...

=================================
       BENCHMARK RESULTS
=================================
Iterations: 3 (warmup: 1)
Total data: 2097152 bytes (2.00 MB)
Sources: 4 ranks, Destinations: 2 ranks
Sharding: src_dim=0, dst_dim=0 (same-dim)

--- Overall (all ranks) ---
Time per iteration (ms):  Min=1.200  Max=4.500
NCCL INFO harmless interleaved line
Bandwidth (GB/s):         Min=0.40  Max=1.20  Avg=0.80

--- Sources only (4 ranks) ---
Time per iteration (ms):  Min=1.200  Max=4.500
Bandwidth (GB/s):         Min=0.40  Max=0.80  Avg=0.60

--- Destinations only (2 ranks) ---
Time per iteration (ms):  Min=1.300  Max=3.500
Bandwidth (GB/s):         Min=0.90  Max=1.20  Avg=1.05

--- Effective bandwidth ---
Total data throughput: 0.43 GB/s
=================================

Benchmark completed successfully!
""".encode()


def test_parse_validated_success_is_section_aware() -> None:
    result = parse_m2n_stdout(_successful_stdout(), rc=0)

    assert result.hot_latency_ms == 4.5
    assert result.overall.time_max_ms == 4.5
    assert result.sources.time_max_ms == 4.5
    assert result.destinations.time_max_ms == 3.5
    assert result.validation_requested is True
    assert result.validation_passed is True
    assert result.destination_validation_ranks == (4, 5)
    assert result.total_data_bytes == 2_097_152


def test_parse_no_validation_success_needs_no_validation_markers() -> None:
    result = parse_m2n_stdout(_successful_stdout(validate=False), rc=0)

    assert result.hot_latency_ms == 4.5
    assert result.validation_requested is False
    assert result.validation_passed is None
    assert result.destination_validation_ranks == ()


@pytest.mark.parametrize(
    ("stdout", "rc", "message"),
    [
        (
            _successful_stdout().replace(
                b"[Rank 5] VALIDATION PASSED: 1048576 bytes correct\n", b""
            ),
            0,
            "validation passes",
        ),
        (
            _successful_stdout().replace(
                b"[Rank 5] VALIDATION PASSED: 1048576 bytes correct",
                b"[Rank 5] VALIDATION PASSED: 7 bytes correct",
            ),
            0,
            "validation byte count",
        ),
        (
            _successful_stdout().replace(
                b"*** VALIDATION PASSED ***", b"*** VALIDATION FAILED ***"
            ),
            1,
            "exit code",
        ),
        (_successful_stdout(), 9, "exit code"),
        (
            _successful_stdout().replace(b"       BENCHMARK RESULTS\n", b""),
            0,
            "results marker",
        ),
        (
            _successful_stdout().replace(b"Benchmark completed successfully!\n", b""),
            0,
            "completion marker",
        ),
        (
            _successful_stdout().replace(b"--- Overall (all ranks) ---\n", b""),
            0,
            "Overall",
        ),
        (
            _successful_stdout().replace(b"Max=4.500", b"Max=nan", 1),
            0,
            "finite",
        ),
        (
            _successful_stdout().replace(
                b"Total data throughput: 0.43",
                b"Total data throughput: inf",
            ),
            0,
            "finite",
        ),
        (
            b"# Collective test starting: all_reduce_perf\n"
            b"# Out of bounds values : 0 OK\n",
            0,
            "not reshard_bench",
        ),
    ],
)
def test_parse_rejects_failed_or_incomplete_output(
    stdout: bytes, rc: int, message: str
) -> None:
    with pytest.raises(M2NOutputError, match=message):
        parse_m2n_stdout(stdout, rc=rc)


def test_validation_failure_never_publishes_timing() -> None:
    stdout = (
        _successful_stdout()
        .replace(
            b"*** VALIDATION PASSED ***",
            b"*** VALIDATION FAILED ***",
        )
        .replace(
            b"Benchmark completed successfully!",
            b"Benchmark completed with VALIDATION FAILURES.",
        )
    )

    with pytest.raises(M2NOutputError):
        parse_m2n_stdout(stdout, rc=1)


def test_execute_m2n_returns_bytes_rc_and_wall_time_without_a_shell() -> None:
    literal_argument = "$(printf should-not-run); still-one-argument"
    code = (
        "import sys; "
        "sys.stdout.buffer.write(sys.argv[1].encode()); "
        "sys.stderr.buffer.write(b'stderr-bytes'); "
        "raise SystemExit(7)"
    )

    result = execute_m2n([sys.executable, "-c", code, literal_argument])

    assert result.stdout == literal_argument.encode()
    assert result.stderr == b"stderr-bytes"
    assert result.rc == 7
    assert result.returncode == 7
    assert result.timed_out is False
    assert result.process_wall_ms > 0


def test_execute_m2n_timeout_kills_and_reaps_the_process_group() -> None:
    child_code = (
        "import signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(30)"
    )
    parent_code = (
        "import signal,subprocess,sys,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        "sys.stdout.buffer.write(b'ready\\n'); sys.stdout.flush(); "
        "time.sleep(30)"
    )

    result = execute_m2n(
        [sys.executable, "-c", parent_code],
        timeout_s=0.2,
        terminate_grace_s=0.1,
    )

    assert result.timed_out is True
    assert result.rc != 0
    assert result.stdout == b"ready\n"
    assert 150 <= result.process_wall_ms < 5_000


def test_execute_m2n_kills_descendants_after_group_leader_exits() -> None:
    child_code = (
        "import signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(2)"
    )
    parent_code = (
        "import subprocess,sys; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        "sys.stdout.buffer.write(b'leader-exited\\n'); sys.stdout.flush()"
    )

    result = execute_m2n(
        [sys.executable, "-c", parent_code],
        timeout_s=0.2,
        terminate_grace_s=0.1,
    )

    assert result.timed_out is True
    assert result.stdout == b"leader-exited\n"
    assert result.process_wall_ms < 1_000
