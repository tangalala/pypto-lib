# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from golden import TensorSpec


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "models" / "deepseek_v4_flash_mtp"
sys.path.insert(0, str(_MODEL_DIR))

from stats_placement_fixture import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    PLACED_NAMES,
    adapt_mtp_stats_placement_specs,
    load_stats_placement_manifest,
    make_stats_placement_spec,
    stats_layer_spec_factory,
)
from stats_route_fixture import apportion_route_counts  # noqa: E402
from eplb_fixture import CONTIGUOUS_PLACEMENT  # noqa: E402


def _subprocess_check(source: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(_MODEL_DIR), str(_REPO_ROOT), env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=_MODEL_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _logical_expert_tensor(local_experts: int, *tail: int) -> torch.Tensor:
    shape = [8, local_experts, *tail]
    logical_ids = torch.arange(8 * local_experts, dtype=torch.int64).reshape(8, local_experts)
    return logical_ids.reshape(8, local_experts, *([1] * len(tail))).expand(shape).clone()


def _replicated_expert_tensor(*tail: int) -> torch.Tensor:
    shape = [8, 256, *tail]
    logical_ids = torch.arange(256, dtype=torch.int64).reshape(1, 256)
    return logical_ids.reshape(1, 256, *([1] * len(tail))).expand(shape).clone()


def test_checked_in_manifest_is_the_full_decode_profile() -> None:
    manifest = load_stats_placement_manifest(DEFAULT_MANIFEST_PATH)

    assert manifest["source"]["sha256"] == (
        "65d165d9e934f2a273b615b4f6e83b253114a4884da6549d9e2d15a46d2360e3"
    )
    assert manifest["filters"] == {"phases": ["decode_mtp"], "routed_tokens": [384]}
    assert manifest["topology"] == {
        "experts": 256,
        "layers": 44,
        "local_experts": 32,
        "ranks": 8,
    }


def test_checked_in_placement_reduces_every_replayed_layer_peak() -> None:
    manifest = load_stats_placement_manifest()
    contiguous_peaks = []
    stats_peaks = []

    for layer_id in range(44):
        route_counts = apportion_route_counts(
            manifest["expert_loads"][layer_id],
            total_routes=384,
        )
        contiguous_loads = [sum(route_counts[rank * 32 : (rank + 1) * 32]) for rank in range(8)]
        stats_loads = [
            sum(route_counts[expert_id] for expert_id in logical_experts)
            for logical_experts in manifest["layers"][layer_id]["rank_to_logical"]
        ]
        contiguous_peaks.append(max(contiguous_loads))
        stats_peaks.append(max(stats_loads))

    assert all(stats <= contiguous for stats, contiguous in zip(stats_peaks, contiguous_peaks))
    assert sum(stats_peaks) / len(stats_peaks) == pytest.approx(49.79545454545455)
    assert sum(contiguous_peaks) / len(contiguous_peaks) == pytest.approx(85.93181818181819)


def test_manifest_rejects_an_unversioned_algorithm_change(tmp_path: Path) -> None:
    manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["algorithm"]["version"] = 2
    path = tmp_path / "wrong-algorithm.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="placement manifest algorithm"):
        load_stats_placement_manifest(path)


@pytest.mark.parametrize(
    ("name", "tail"),
    [
        ("routed_w1", (2, 3)),
        ("routed_w1_scale", (2,)),
        ("routed_w3", (2, 3)),
        ("routed_w3_scale", (2,)),
        ("routed_w2", (3, 2)),
        ("routed_w2_scale", (3,)),
    ],
)
def test_routed_weights_and_scales_use_the_same_physical_order(name: str, tail: tuple[int, ...]) -> None:
    manifest = load_stats_placement_manifest()
    source = _logical_expert_tensor(32, *tail)
    base_spec = TensorSpec(
        name,
        list(source.shape),
        source.dtype,
        init_value=lambda: source,
        resident="stacked",
    )

    placed = make_stats_placement_spec(name, base_spec, layer_ids=[0, 43]).create_tensor()

    assert list(placed.shape[:2]) == [8, 64]
    for stack_index, layer_id in enumerate((0, 43)):
        physical_to_logical = torch.tensor(
            manifest["layers"][layer_id]["rank_to_logical"], dtype=torch.int64
        ).reshape(-1)
        layer = placed[:, stack_index * 32 : (stack_index + 1) * 32]
        actual = layer[(...,) + (0,) * len(tail)].reshape(-1)
        assert torch.equal(actual, physical_to_logical)


@pytest.mark.parametrize("name", ["gate_w", "gate_bias"])
def test_gate_rows_follow_the_physical_expert_order(name: str) -> None:
    manifest = load_stats_placement_manifest()
    source = _replicated_expert_tensor(1)
    base_spec = TensorSpec(name, list(source.shape), source.dtype, init_value=lambda: source)

    placed = make_stats_placement_spec(name, base_spec, layer_ids=[1, 42]).create_tensor()

    for stack_index, layer_id in enumerate((1, 42)):
        physical_to_logical = torch.tensor(
            manifest["layers"][layer_id]["rank_to_logical"], dtype=torch.int64
        ).reshape(-1)
        layer = placed[:, stack_index * 256 : (stack_index + 1) * 256, 0]
        assert torch.equal(layer[0], physical_to_logical)
        assert torch.equal(layer, layer[0].unsqueeze(0).expand_as(layer))


def test_stats_tid2eid_preserves_histograms_after_physical_mapping() -> None:
    manifest = load_stats_placement_manifest()
    base_spec = TensorSpec("tid2eid", [8, 64, 6], torch.int32, resident="stacked")

    spec = make_stats_placement_spec("tid2eid", base_spec, layer_ids=[0, 43])
    placed = spec.create_tensor()

    assert spec.resident == "stacked"
    assert tuple(placed.shape) == (8, 128, 6)
    for stack_index, layer_id in enumerate((0, 43)):
        physical_routes = placed[0, stack_index * 64 : (stack_index + 1) * 64]
        assert torch.equal(
            placed[:, stack_index * 64 : (stack_index + 1) * 64],
            physical_routes.unsqueeze(0).expand(8, -1, -1),
        )
        assert all(len(set(row.tolist())) == 6 for row in physical_routes)

        physical_to_logical = torch.tensor(
            manifest["layers"][layer_id]["rank_to_logical"], dtype=torch.int64
        ).reshape(-1)
        logical_routes = physical_to_logical[physical_routes.to(torch.int64)]
        actual_counts = torch.bincount(logical_routes.reshape(-1), minlength=256)
        expected_counts = torch.tensor(
            apportion_route_counts(manifest["expert_loads"][layer_id], total_routes=384)
        )
        assert torch.equal(actual_counts, expected_counts)


def test_contiguous_control_preserves_logical_route_ids_and_histograms() -> None:
    manifest = load_stats_placement_manifest()
    base_spec = TensorSpec("tid2eid", [8, 64, 6], torch.int32, resident="stacked")

    spec = make_stats_placement_spec(
        "tid2eid",
        base_spec,
        layer_ids=[0, 43],
        placement=CONTIGUOUS_PLACEMENT,
    )
    routes = spec.create_tensor()

    for stack_index, layer_id in enumerate((0, 43)):
        logical_routes = routes[0, stack_index * 64 : (stack_index + 1) * 64]
        actual_counts = torch.bincount(logical_routes.reshape(-1).to(torch.int64), minlength=256)
        expected_counts = torch.tensor(
            apportion_route_counts(manifest["expert_loads"][layer_id], total_routes=384)
        )
        assert torch.equal(actual_counts, expected_counts)
    assert (
        make_stats_placement_spec(
            "gate_w",
            TensorSpec("gate_w", [8, 256, 1], torch.float32),
            layer_ids=[0],
            placement=CONTIGUOUS_PLACEMENT,
        )
        is None
    )


@pytest.mark.parametrize("layer_id", [0, 42, 43])
def test_route_gate_and_routed_weight_permutations_preserve_logical_experts(
    layer_id: int,
) -> None:
    route_base = TensorSpec("tid2eid", [8, 64, 6], torch.int32, resident="stacked")
    logical_routes = make_stats_placement_spec(
        "tid2eid",
        route_base,
        layer_ids=[layer_id],
        placement=CONTIGUOUS_PLACEMENT,
    ).create_tensor()[0]
    physical_routes = make_stats_placement_spec(
        "tid2eid",
        route_base,
        layer_ids=[layer_id],
    ).create_tensor()[0]

    routed_source = _logical_expert_tensor(32, 1)
    routed_spec = TensorSpec(
        "routed_w1",
        list(routed_source.shape),
        routed_source.dtype,
        init_value=lambda: routed_source,
    )
    routed_physical = make_stats_placement_spec(
        "routed_w1",
        routed_spec,
        layer_ids=[layer_id],
    ).create_tensor()
    routed_values = routed_physical.reshape(256, 1)[physical_routes.to(torch.int64), 0]
    assert torch.equal(routed_values, logical_routes.to(routed_values.dtype))

    gate_source = _replicated_expert_tensor(1)
    gate_spec = TensorSpec(
        "gate_w",
        list(gate_source.shape),
        gate_source.dtype,
        init_value=lambda: gate_source,
    )
    gate_physical = make_stats_placement_spec(
        "gate_w",
        gate_spec,
        layer_ids=[layer_id],
    ).create_tensor()[0, :, 0]
    gate_values = gate_physical[physical_routes.to(torch.int64)]
    assert torch.equal(gate_values, logical_routes.to(gate_values.dtype))


def test_forward_factory_covers_layers_zero_through_42() -> None:
    manifest = load_stats_placement_manifest()
    source = _replicated_expert_tensor(1)
    base_spec = TensorSpec("gate_w", list(source.shape), source.dtype, init_value=lambda: source)

    placed = stats_layer_spec_factory("gate_w", base_spec, 43).create_tensor()

    assert tuple(placed.shape) == (8, 43 * 256, 1)
    for layer_id in (0, 1, 21, 42):
        expected = torch.tensor(manifest["layers"][layer_id]["rank_to_logical"], dtype=torch.int64).reshape(
            -1
        )
        assert torch.equal(placed[0, layer_id * 256 : (layer_id + 1) * 256, 0], expected)
    assert stats_layer_spec_factory("norm_w", base_spec, 43) is None


def test_mtp_adapter_uses_layer_43_and_preserves_spec_order() -> None:
    manifest = load_stats_placement_manifest()
    routed = _logical_expert_tensor(32, 1)
    specs = [
        TensorSpec("before", [1], torch.float32),
        TensorSpec("routed_w1", list(routed.shape), routed.dtype, init_value=lambda: routed),
        TensorSpec("after", [1], torch.float32),
    ]

    adapted = adapt_mtp_stats_placement_specs(specs)
    placed = adapted[1].create_tensor()

    assert [spec.name for spec in adapted] == ["before", "routed_w1", "after"]
    expected = torch.tensor(manifest["layers"][43]["rank_to_logical"], dtype=torch.int64).reshape(-1)
    assert torch.equal(placed[:, :, 0].reshape(-1), expected)


def test_contiguous_mtp_control_only_replaces_the_route_fixture() -> None:
    specs = [
        TensorSpec("gate_w", [8, 256, 1], torch.float32),
        TensorSpec("tid2eid", [8, 64, 6], torch.int32, resident="stacked"),
        TensorSpec("routed_w1", [8, 32, 1], torch.int8),
    ]

    adapted = adapt_mtp_stats_placement_specs(specs, placement=CONTIGUOUS_PLACEMENT)

    assert adapted[0] is specs[0]
    assert adapted[1] is not specs[1]
    assert adapted[2] is specs[2]


def test_only_the_nine_moe_specs_are_placement_owned() -> None:
    assert PLACED_NAMES == {
        "gate_w",
        "gate_bias",
        "tid2eid",
        "routed_w1",
        "routed_w1_scale",
        "routed_w3",
        "routed_w3_scale",
        "routed_w2",
        "routed_w2_scale",
    }


def test_legacy_and_stats_decode_entrypoints_keep_separate_topologies() -> None:
    _subprocess_check(
        """
import sys
sys.argv = ["eplb_decode_logits.py"]
import eplb_decode_logits as decode
assert (decode.N_RANKS, decode.N_LOCAL, decode.N_EXPERTS_GLOBAL) == (8, 16, 128)
"""
    )
    _subprocess_check(
        """
import sys
sys.argv = ["stats_placement_decode_logits.py"]
import stats_placement_decode_logits
import eplb_decode_logits as decode
assert (decode.N_RANKS, decode.N_LOCAL, decode.N_EXPERTS_GLOBAL) == (8, 32, 256)
"""
    )


def test_stats_mtp_entrypoint_uses_the_256_expert_topology() -> None:
    _subprocess_check(
        """
import sys
sys.argv = ["stats_placement_mtp_core.py"]
import stats_placement_mtp_core
import eplb_mtp_core as mtp
assert (mtp.N_RANKS, mtp.N_LOCAL, mtp.N_EXPERTS_GLOBAL) == (8, 32, 256)
"""
    )
