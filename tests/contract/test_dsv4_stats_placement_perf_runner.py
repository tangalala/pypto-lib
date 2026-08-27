# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER = _REPO_ROOT / "tools" / "run_dsv4_stats_placement_perf.sh"
_MANIFEST = _REPO_ROOT / "models" / "deepseek_v4_flash_mtp" / "stats_placement_decode_manifest.json"
_DEVICE_SET = "0,2,4,6,8,10,12,14"
_ALLOCATOR_DEVICE_SET = "1,3,5,7,9,11,13,15"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(_RUNNER), "--device", _DEVICE_SET, *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _command_lines(output: str) -> list[str]:
    return [
        line for line in output.splitlines() if "run_seeded_python.py" in line and "stats_placement_" in line
    ]


def test_dry_run_pairs_identical_ep8x32_workloads_for_both_entrypoints() -> None:
    result = _run("--dry-run")

    assert result.returncode == 0, result.stderr
    assert "Comparison contract: dsv4-stats-placement-compare-v1" in result.stdout
    assert "Metric contract: dsv4-eplb-v2" in result.stdout
    assert "same EP8x32 stats-shaped routes; only expert placement changes" in result.stdout
    assert "decode-logits=runtime_only mtp-core=finite_only golden_replayed=false" in result.stdout
    assert f"Manifest: {_MANIFEST}" in result.stdout

    headings = [
        "decode-logits-contiguous:",
        "decode-logits-stats:",
        "mtp-core-contiguous:",
        "mtp-core-stats:",
    ]
    assert all(heading in result.stdout for heading in headings)
    assert [result.stdout.index(heading) for heading in headings] == sorted(
        result.stdout.index(heading) for heading in headings
    )

    commands = _command_lines(result.stdout)
    assert len(commands) == 4
    for command in commands:
        assert "--ep 8" in command
        assert "--tp 4" in command
        assert "--experts-per-rank 32" in command
        assert "--start-pos 8192" in command
        assert "--num-tokens 8" in command
        assert str(_MANIFEST) in command
        assert "--seed 1807" in command
    assert sum("--expert-placement contiguous" in command for command in commands) == 2
    assert sum("--expert-placement stats" in command for command in commands) == 2
    assert all("--placement-manifest" in command for command in commands)
    assert all("stats_placement_decode_logits.py" in command for command in commands[:2])
    assert all("--finite-only" not in command for command in commands[:2])
    assert all("stats_placement_mtp_core.py" in command for command in commands[2:])
    assert all("--finite-only" in command for command in commands[2:])


def test_dry_run_filters_one_case_and_one_placement() -> None:
    result = _run("--case", "mtp-core", "--placement", "stats", "--dry-run")

    assert result.returncode == 0, result.stderr
    commands = _command_lines(result.stdout)
    assert len(commands) == 1
    assert "mtp-core-stats:" in result.stdout
    assert "--expert-placement stats" in commands[0]
    assert "stats_placement_mtp_core.py" in commands[0]
    assert "decode-logits" not in result.stdout
    assert "contiguous:" not in result.stdout


def test_runner_accepts_the_allocator_device_mapping() -> None:
    result = subprocess.run(
        [str(_RUNNER), "--device", _ALLOCATOR_DEVICE_SET, "--dry-run"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    commands = _command_lines(result.stdout)
    assert len(commands) == 4
    escaped_device_set = _ALLOCATOR_DEVICE_SET.replace(",", r"\,")
    assert all(f"-d {escaped_device_set}" in command for command in commands)


def test_runner_rejects_an_unsupported_device_mapping() -> None:
    result = subprocess.run(
        [str(_RUNNER), "--device", "0,1,2,3,4,5,6,7", "--dry-run"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert (
        f"comparison metric requires --device {_DEVICE_SET} or {_ALLOCATOR_DEVICE_SET}"
        in result.stderr
    )


@pytest.mark.parametrize(
    ("option", "value", "expected"),
    [
        ("--rounds", "1", "requires --rounds 100"),
        ("--warmup", "0", "requires --warmup 5"),
    ],
)
def test_measured_runner_rejects_nonofficial_sample_counts(
    option: str,
    value: str,
    expected: str,
) -> None:
    result = _run(option, value, "--dry-run")

    assert result.returncode == 2
    assert expected in result.stderr


def test_compile_only_execution_records_every_selected_variant(tmp_path: Path) -> None:
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" == "--version" ]]; then\n'
        "  printf 'Python fake\\n'\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    output_dir = tmp_path / "results"

    result = _run(
        "--compile-only",
        "--python",
        str(fake_python),
        "--rounds",
        "1",
        "--warmup",
        "0",
        "--seed",
        "7",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    result_rows = (output_dir / "results.tsv").read_text(encoding="utf-8").splitlines()
    assert len(result_rows) == 5
    assert all(len(row.split("\t")) == 29 for row in result_rows)
    assert [row.split("\t")[0] for row in result_rows[1:]] == [
        "decode-logits-contiguous",
        "decode-logits-stats",
        "mtp-core-contiguous",
        "mtp-core-stats",
    ]
    assert all(row.split("\t")[3] == "pass_compile" for row in result_rows[1:])
    assert [row.split("\t")[4] for row in result_rows[1:]] == [
        "runtime_only",
        "runtime_only",
        "finite_only",
        "finite_only",
    ]
    assert all(row.split("\t")[5] == "false" for row in result_rows[1:])
    comparison_rows = (output_dir / "comparison.tsv").read_text(encoding="utf-8").splitlines()
    assert len(comparison_rows) == 1
    balance_rows = (output_dir / "rank-balance.tsv").read_text(encoding="utf-8").splitlines()
    assert len(balance_rows) == 1
    assert (output_dir / "metadata.tsv").is_file()
    assert (output_dir / "source-status.txt").is_file()
    assert all(
        (output_dir / f"{variant}.log").is_file()
        for variant in (
            "decode-logits-contiguous",
            "decode-logits-stats",
            "mtp-core-contiguous",
            "mtp-core-stats",
        )
    )


def test_comparison_winner_uses_max_rank_median_and_reports_fastest_separately(
    tmp_path: Path,
) -> None:
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" == "--version" ]]; then\n'
        "  printf 'Python fake\\n'\n"
        "  exit 0\n"
        "fi\n"
        'if [[ "${1:-}" == */dsv4_eplb_perf_metrics.py ]]; then\n'
        "  shift\n"
        "  case_name=''\n"
        "  log_file=''\n"
        "  rank_output=''\n"
        '  while [[ "$#" -gt 0 ]]; do\n'
        '    case "$1" in\n'
        '      --case) case_name="$2"; shift 2 ;;\n'
        '      --log) log_file="$2"; shift 2 ;;\n'
        '      --rank-output) rank_output="$2"; shift 2 ;;\n'
        "      *) shift ;;\n"
        "    esac\n"
        "  done\n"
        '  if [[ "$log_file" == *-contiguous.log ]]; then\n'
        "    medians=(10 11 12 13 14 15 16 17)\n"
        "  else\n"
        "    medians=(9 10 11 12 13 14 15 19)\n"
        "  fi\n"
        "  scope='compare3_fastest_rank'\n"
        "  task='eplb_decode_logits'\n"
        "  dispatches=1\n"
        '  if [[ "$case_name" == "mtp-core" ]]; then\n'
        "    scope='compare4_fastest_rank_compute_only'\n"
        "    task='eplb_mtp_core_logits'\n"
        "    dispatches=2\n"
        "  fi\n"
        '  for index in "${!medians[@]}"; do\n'
        "    selected=0\n"
        '    [[ "$index" -eq 0 ]] && selected=1\n'
        '    printf \'%s\\t%s\\t%s\\t%s\\t%s\\t0\\t%s\\t%s\\t100\\t%s\\t%s\\t%s\\t%s\\n\' "$case_name" "$scope" "$index" "$((index * 2))" "$((1000 + index))" "$task" "$selected" "${medians[$index]}" "${medians[$index]}" "${medians[$index]}" "${medians[$index]}" >>"$rank_output"\n'
        '    if [[ "$case_name" == "mtp-core" ]]; then\n'
        '      printf \'%s\\t%s\\t%s\\t%s\\t%s\\t1\\teplb_mtp_core_cleanup\\t0\\t100\\t1000\\t1000\\t1000\\t1000\\n\' "$case_name" "$scope" "$index" "$((index * 2))" "$((1000 + index))" >>"$rank_output"\n'
        "    fi\n"
        "  done\n"
        '  selected_median="${medians[0]}"\n'
        '  printf \'dsv4-eplb-v2\\t%s\\tminimum_rank_median\\t100\\t5\\t8\\t%s\\t0\\t0\\t1000\\t100\\t%s\\t%s\\t%s\\t%s\\t-\\t-\\t-\\t-\\traw_all\\tordered_pid_to_ordered_device_set\\n\' "$scope" "$dispatches" "$selected_median" "$selected_median" "$selected_median" "$selected_median"\n'
        "  exit 0\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    output_dir = tmp_path / "measured-results"

    result = _run(
        "--python",
        str(fake_python),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    balance_rows = (output_dir / "rank-balance.tsv").read_text(encoding="utf-8").splitlines()
    assert balance_rows[1:] == [
        "decode-logits\tcontiguous\t10.000\t17.000\t7.000",
        "decode-logits\tstats\t9.000\t19.000\t10.000",
        "mtp-core\tcontiguous\t10.000\t17.000\t7.000",
        "mtp-core\tstats\t9.000\t19.000\t10.000",
    ]
    comparison_rows = (output_dir / "comparison.tsv").read_text(encoding="utf-8").splitlines()
    assert "fastest_rank_median_us" in comparison_rows[0]
    assert "winner_by_max_rank_median" in comparison_rows[0]
    expected_metrics = [
        "10.000",
        "9.000",
        "-1.000",
        "-10.000",
        "17.000",
        "19.000",
        "2.000",
        "11.765",
        "7.000",
        "10.000",
        "3.000",
        "contiguous",
    ]
    assert comparison_rows[1].split("\t") == [
        "decode-logits",
        *expected_metrics,
        "runtime_only",
        "false",
    ]
    assert comparison_rows[2].split("\t") == [
        "mtp-core",
        *expected_metrics,
        "finite_only",
        "false",
    ]
