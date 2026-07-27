#!/usr/bin/env python3
"""Compare Rust plans byte-for-byte with the standalone Python oracle."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from random import Random

from weight_conversion_plugin import ConversionRequest, ManifestWeightConversionPlugin


def partitions(extent: int, count: int) -> tuple[tuple[int, int], ...]:
    width = extent // count
    return tuple((index * width, width) for index in range(count))


def tensor_record(
    *,
    global_shape: tuple[int, int],
    logical_offset: tuple[int, int],
    logical_shape: tuple[int, int],
    rank: dict[str, int],
    itemsize: int,
    row_stride: int,
    dev: int,
    addr: int,
) -> dict[str, object]:
    rows, columns = logical_shape
    storage_nbytes = itemsize + (rows - 1) * row_stride + (columns - 1) * itemsize
    return {
        "tensor_id": "random-weight",
        "semantic": {
            "model_part": "language",
            "stack": "decoder",
            "layer_id": 3,
            "module_path": ["mlp"],
            "parameter_role": "down_proj_weight",
            "representation_id": f"uint{itemsize * 8}:plain:canonical:v1",
        },
        "global_shape": list(global_shape),
        "logical_offset": list(logical_offset),
        "logical_shape": list(logical_shape),
        "rank": rank,
        "local_shape": list(logical_shape),
        "view_kind": "canonical-affine:v1",
        "dtype": f"uint{itemsize * 8}",
        "itemsize": itemsize,
        "strides_bytes": [row_stride, itemsize],
        "dev": dev,
        "addr": addr,
        "storage_nbytes": storage_nbytes,
    }


def build_case(random: Random, case_index: int) -> tuple[str, int]:
    rows = random.choice((2, 4, 6, 8))
    columns = random.choice((2, 4, 6, 8))
    itemsize = random.choice((1, 2, 4))
    source_ep = random.choice(tuple(v for v in (1, 2, 4) if rows % v == 0))
    source_tp = random.choice(tuple(v for v in (1, 2, 4) if columns % v == 0))
    target_ep = random.choice(tuple(v for v in (1, 2, 4) if rows % v == 0))
    target_tp = random.choice(tuple(v for v in (1, 2, 4) if columns % v == 0))
    source_pp = random.choice((1, 2, 4))
    target_pp = random.choice((1, 2, 4))
    source_pp_rank = random.randrange(source_pp)
    target_pp_rank = random.randrange(target_pp)
    target_dp = random.choice((1, 2))
    source_records: list[dict[str, object]] = []
    target_records: list[dict[str, object]] = []

    for dp in range(2):
        for ep, (row_offset, row_count) in enumerate(partitions(rows, source_ep)):
            for tp, (column_offset, column_count) in enumerate(
                partitions(columns, source_tp)
            ):
                dev = dp * 100 + ep * source_tp + tp
                source_records.append(
                    tensor_record(
                        global_shape=(rows, columns),
                        logical_offset=(row_offset, column_offset),
                        logical_shape=(row_count, column_count),
                        rank={
                            "dp": dp,
                            "pp": source_pp_rank,
                            "tp": tp,
                            "ep": ep,
                        },
                        itemsize=itemsize,
                        row_stride=column_count * itemsize
                        + random.randint(0, 3) * itemsize,
                        dev=dev,
                        addr=0x100000 + dev * 0x10000,
                    )
                )

    for dp in range(target_dp):
        for ep, (row_offset, row_count) in enumerate(partitions(rows, target_ep)):
            for tp, (column_offset, column_count) in enumerate(
                partitions(columns, target_tp)
            ):
                dev = 1000 + dp * 100 + ep * target_tp + tp
                target_records.append(
                    tensor_record(
                        global_shape=(rows, columns),
                        logical_offset=(row_offset, column_offset),
                        logical_shape=(row_count, column_count),
                        rank={
                            "dp": dp,
                            "pp": target_pp_rank,
                            "tp": tp,
                            "ep": ep,
                        },
                        itemsize=itemsize,
                        row_stride=column_count * itemsize
                        + random.randint(0, 3) * itemsize,
                        dev=dev,
                        addr=0x800000 + dev * 0x10000,
                    )
                )

    request = ConversionRequest.from_dict(
        {
            "format_version": 1,
            "plan_id": f"random-{case_index}",
            "source_manifest": {
                "model_id": "random-model-v1",
                "parallel": {
                    "tp_size": source_tp,
                    "pp_size": source_pp,
                    "ep_size": source_ep,
                    "dp_size": 2,
                },
                "tensors": source_records,
            },
            "target_manifest": {
                "model_id": "random-model-v1",
                "parallel": {
                    "tp_size": target_tp,
                    "pp_size": target_pp,
                    "ep_size": target_ep,
                    "dp_size": target_dp,
                },
                "tensors": target_records,
            },
        }
    )
    chunk_bytes = random.randint(1, max(1, columns * itemsize))
    return request.to_json(), chunk_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rust-bin", required=True, type=Path)
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260727)
    arguments = parser.parse_args()

    random = Random(arguments.seed)
    python_plugin = ManifestWeightConversionPlugin()
    for case_index in range(arguments.cases):
        request_json, chunk_bytes = build_case(random, case_index)
        request = ConversionRequest.from_json(request_json)
        expected = python_plugin.plan_scr(
            request, max_chunk_bytes=chunk_bytes
        ).to_json()
        process = subprocess.run(
            [
                str(arguments.rust_bin),
                "--max-chunk-bytes",
                str(chunk_bytes),
            ],
            input=request_json,
            text=True,
            capture_output=True,
            check=False,
        )
        if process.returncode != 0:
            raise SystemExit(
                f"case {case_index}: Rust planner failed:\n{process.stderr}"
            )
        actual = process.stdout.rstrip("\n")
        if actual != expected:
            expected_json = json.loads(expected)
            actual_json = json.loads(actual)
            raise SystemExit(
                f"case {case_index}: plan mismatch\n"
                f"expected transfers={len(expected_json['transfers'])}\n"
                f"actual transfers={len(actual_json['transfers'])}"
            )
    print(
        f"PASS: {arguments.cases} exact Python/Rust plans matched "
        f"(seed={arguments.seed})"
    )


if __name__ == "__main__":
    main()
