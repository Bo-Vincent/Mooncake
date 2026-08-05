import inspect

import pytest

from benchmarks.heterogeneous_weight_reshard import m2n_adapter
from benchmarks.heterogeneous_weight_reshard.case_spec import (
    BenchmarkCase,
    MeshSpec,
    PlacementSpec,
)
from benchmarks.heterogeneous_weight_reshard.m2n_adapter import (
    M2NPlanningError,
    plan_m2n_case,
)


def _case(
    *,
    source_replicas: int = 1,
    source_shards: int,
    source_dim: int,
    target_replicas: int = 1,
    target_shards: int,
    target_dim: int,
    shape: tuple[int, ...],
    case_id: str = "neutral_tensor_case",
    category: str = "physical",
) -> BenchmarkCase:
    source = MeshSpec(source_replicas, source_shards, source_dim)
    target = MeshSpec(target_replicas, target_shards, target_dim)
    return BenchmarkCase(
        id=case_id,
        category=category,
        source=source,
        target=target,
        global_shape=shape,
        required_ranks=source.total_ranks + target.total_ranks,
    )


@pytest.fixture
def placement() -> PlacementSpec:
    return PlacementSpec("source-node", "target-node", 8)


@pytest.mark.parametrize(
    ("case", "source_dims", "target_dims", "source_start", "target_start"),
    [
        (
            _case(
                source_shards=4,
                source_dim=0,
                target_shards=8,
                target_dim=0,
                shape=(2048, 2048, 2048),
            ),
            (1, 4),
            (1, 8),
            0,
            4,
        ),
        (
            _case(
                source_shards=8,
                source_dim=0,
                target_shards=4,
                target_dim=0,
                shape=(2048, 2048, 2048),
            ),
            (1, 8),
            (1, 4),
            0,
            8,
        ),
        (
            _case(
                source_shards=4,
                source_dim=0,
                target_shards=4,
                target_dim=1,
                shape=(2048, 2048, 2048),
            ),
            (1, 4),
            (1, 4),
            0,
            4,
        ),
        (
            _case(
                source_shards=4,
                source_dim=0,
                target_shards=4,
                target_dim=2,
                shape=(128, 4096, 16384),
            ),
            (1, 4),
            (1, 4),
            0,
            4,
        ),
        (
            _case(
                source_replicas=2,
                source_shards=4,
                source_dim=0,
                target_shards=8,
                target_dim=0,
                shape=(2048, 2048, 1024),
            ),
            (2, 4),
            (1, 8),
            0,
            8,
        ),
    ],
)
def test_maps_case_to_exact_m2n_meshes(
    case: BenchmarkCase,
    source_dims: tuple[int, int],
    target_dims: tuple[int, int],
    source_start: int,
    target_start: int,
    placement: PlacementSpec,
) -> None:
    plan = plan_m2n_case(
        case,
        placement,
        binary="/opt/m2n/reshard_bench",
        warmups=3,
        iterations=20,
    )

    assert plan.source_mesh.dims == source_dims
    assert plan.target_mesh.dims == target_dims
    assert plan.source_mesh.start_rank == source_start
    assert plan.target_mesh.start_rank == target_start
    assert plan.source_mesh.placement == (
        "replicate",
        f"shard({case.source.shard_dim})",
    )
    assert plan.target_mesh.placement == (
        "replicate",
        f"shard({case.target.shard_dim})",
    )
    assert plan.world_size == case.required_ranks
    assert plan.global_shape == case.global_shape
    assert plan.physical_execution_eligible is True
    assert plan.physical_execution_refusal_reasons == ()


def test_generates_exact_validated_descriptor_arguments(
    placement: PlacementSpec,
) -> None:
    case = _case(
        source_replicas=2,
        source_shards=4,
        source_dim=0,
        target_shards=8,
        target_dim=1,
        shape=(2048, 4096, 1024),
    )

    plan = plan_m2n_case(
        case,
        placement,
        binary="reshard_bench",
        warmups=3,
        iterations=20,
    )

    assert plan.descriptor_argv == (
        "--src-mesh-dims",
        "2,4",
        "--dst-mesh-dims",
        "1,8",
        "--tensor-dims",
        "2048,4096,1024",
        "--src-shard-dim",
        "0",
        "--dst-shard-dim",
        "1",
        "--iterations",
        "20",
        "--warmup",
        "3",
        "--algorithm",
        "ring",
        "--lb-mode",
        "uniform",
        "--validate",
    )
    assert all(
        context.argv == ("reshard_bench", *plan.descriptor_argv)
        for context in plan.app_contexts
    )


def test_generates_unvalidated_one_shot_descriptor_for_cold_e2e(
    placement: PlacementSpec,
) -> None:
    case = _case(
        source_shards=4,
        source_dim=0,
        target_shards=8,
        target_dim=0,
        shape=(2048, 2048, 2048),
    )

    plan = plan_m2n_case(
        case,
        placement,
        binary="reshard_bench",
        warmups=0,
        iterations=1,
        validate=False,
    )

    assert plan.descriptor_argv[-6:] == (
        "--warmup",
        "0",
        "--algorithm",
        "ring",
        "--lb-mode",
        "uniform",
    )
    assert "--validate" not in plan.descriptor_argv


def test_mpmd_contexts_pin_roles_to_different_hosts(
    placement: PlacementSpec,
) -> None:
    case = _case(
        source_shards=4,
        source_dim=0,
        target_shards=8,
        target_dim=0,
        shape=(2048, 2048, 2048),
    )

    plan = plan_m2n_case(
        case,
        placement,
        binary="/opt/m2n/reshard_bench",
        warmups=3,
        iterations=20,
    )
    source, target = plan.app_contexts

    assert (source.role, source.host, source.rank_begin, source.rank_end) == (
        "source",
        "source-node",
        0,
        4,
    )
    assert (target.role, target.host, target.rank_begin, target.rank_end) == (
        "target",
        "target-node",
        4,
        12,
    )
    assert plan.mpirun_argv == (
        "mpirun",
        "-x",
        "PATH",
        "-x",
        "LD_LIBRARY_PATH",
        "-x",
        "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7",
        "-np",
        "4",
        "--host",
        "source-node:4",
        "/opt/m2n/reshard_bench",
        *plan.descriptor_argv,
        ":",
        "-x",
        "PATH",
        "-x",
        "LD_LIBRARY_PATH",
        "-x",
        "CUDA_VISIBLE_DEVICES=4,5,6,7,0,1,2,3",
        "-np",
        "8",
        "--host",
        "target-node:8",
        "/opt/m2n/reshard_bench",
        *plan.descriptor_argv,
    )
    assert plan.mpirun_argv.count("PATH") == 2
    assert plan.mpirun_argv.count("LD_LIBRARY_PATH") == 2
    assert "--hostfile" not in plan.mpirun_argv
    assert "--oversubscribe" not in plan.mpirun_argv


def test_rendered_command_is_display_only_and_shell_quoted(
    placement: PlacementSpec,
) -> None:
    case = _case(
        source_shards=1,
        source_dim=0,
        target_shards=1,
        target_dim=0,
        shape=(16, 16),
    )

    plan = plan_m2n_case(
        case,
        placement,
        binary="/opt/m2n build/reshard_bench",
        warmups=1,
        iterations=1,
    )

    assert "'/opt/m2n build/reshard_bench'" in plan.rendered_command
    assert "subprocess" not in inspect.getsource(m2n_adapter)


def test_mpirun_repeats_network_options_and_extra_env_for_each_app_context(
    placement: PlacementSpec,
) -> None:
    case = _case(
        source_shards=2,
        source_dim=0,
        target_shards=4,
        target_dim=0,
        shape=(1024, 1024),
    )

    plan = plan_m2n_case(
        case,
        placement,
        binary="reshard_bench",
        warmups=1,
        iterations=2,
        mpi_interface="eth0",
        export_env_names=("NCCL_SOCKET_IFNAME", "NCCL_IB_HCA"),
    )

    context_prefix = (
        "--mca",
        "oob_tcp_if_include",
        "eth0",
        "--mca",
        "btl_tcp_if_include",
        "eth0",
        "-x",
        "PATH",
        "-x",
        "LD_LIBRARY_PATH",
        "-x",
        "NCCL_SOCKET_IFNAME",
        "-x",
        "NCCL_IB_HCA",
    )
    target_context_begin = plan.mpirun_argv.index(":") + 1

    assert plan.mpirun_argv[1 : 1 + len(context_prefix)] == context_prefix
    assert (
        plan.mpirun_argv[
            target_context_begin : target_context_begin + len(context_prefix)
        ]
        == context_prefix
    )
    assert plan.mpirun_argv.count("NCCL_SOCKET_IFNAME") == 2
    assert plan.mpirun_argv.count("NCCL_IB_HCA") == 2


def test_mpirun_maps_global_ranks_to_local_gpu_indices_per_context(
    placement: PlacementSpec,
) -> None:
    case = _case(
        source_shards=4,
        source_dim=0,
        target_shards=4,
        target_dim=1,
        shape=(1024, 1024),
    )

    plan = plan_m2n_case(
        case,
        placement,
        binary="reshard_bench",
        warmups=1,
        iterations=2,
    )
    source_mapping = "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
    target_mapping = "CUDA_VISIBLE_DEVICES=4,5,6,7,0,1,2,3"
    split = plan.mpirun_argv.index(":")

    assert source_mapping in plan.mpirun_argv[:split]
    assert target_mapping not in plan.mpirun_argv[:split]
    assert target_mapping in plan.mpirun_argv[split + 1 :]
    assert source_mapping not in plan.mpirun_argv[split + 1 :]


def test_to_dict_preserves_structured_argv(placement: PlacementSpec) -> None:
    case = _case(
        source_shards=4,
        source_dim=0,
        target_shards=4,
        target_dim=2,
        shape=(128, 4096, 16384),
        case_id="cross_dim0_to_dim2_physical",
    )

    plan = plan_m2n_case(
        case,
        placement,
        binary="reshard_bench",
        warmups=3,
        iterations=20,
    )
    encoded = plan.to_dict()

    assert encoded["case_id"] == "cross_dim0_to_dim2_physical"
    assert encoded["world_size"] == 8
    assert encoded["global_shape"] == [128, 4096, 16384]
    assert encoded["descriptor_argv"] == list(plan.descriptor_argv)
    assert encoded["mpirun_argv"] == list(plan.mpirun_argv)
    assert encoded["physical_execution_eligible"] is True
    assert encoded["physical_execution_refusal_reasons"] == []
    assert encoded["app_contexts"][0]["rank_ids"] == [0, 1, 2, 3]
    assert encoded["app_contexts"][1]["rank_ids"] == [4, 5, 6, 7]


def test_planner_only_geometry_reports_capacity_without_rejecting(
    placement: PlacementSpec,
) -> None:
    case = _case(
        source_replicas=2,
        source_shards=8,
        source_dim=0,
        target_replicas=2,
        target_shards=8,
        target_dim=1,
        shape=(2048, 2048),
        category="planner_only",
        case_id="large_mesh",
    )

    plan = plan_m2n_case(
        case,
        placement,
        binary="reshard_bench",
        warmups=3,
        iterations=20,
    )

    assert plan.world_size == 32
    assert plan.physical_execution_eligible is False
    assert plan.physical_execution_refusal_reasons == (
        "source requires 16 ranks but gpus_per_host is 8",
        "target requires 16 ranks but gpus_per_host is 8",
    )


def test_planner_only_case_without_geometry_fails_clearly(
    placement: PlacementSpec,
) -> None:
    case = BenchmarkCase(
        id="planner_only_without_geometry",
        category="planner_only",
        reason="multi-tensor ownership case",
    )

    with pytest.raises(
        M2NPlanningError,
        match="planner_only_without_geometry: M2N planning requires geometry",
    ):
        plan_m2n_case(
            case,
            placement,
            binary="reshard_bench",
            warmups=3,
            iterations=20,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("binary", ""), ("launcher", ""), ("warmups", -1), ("iterations", True)],
)
def test_rejects_invalid_execution_description_inputs(
    field: str,
    value: object,
    placement: PlacementSpec,
) -> None:
    case = _case(
        source_shards=1,
        source_dim=0,
        target_shards=1,
        target_dim=0,
        shape=(16, 16),
    )
    kwargs: dict[str, object] = {
        "binary": "reshard_bench",
        "launcher": "mpirun",
        "warmups": 3,
        "iterations": 20,
    }
    kwargs[field] = value

    with pytest.raises(M2NPlanningError):
        plan_m2n_case(case, placement, **kwargs)
