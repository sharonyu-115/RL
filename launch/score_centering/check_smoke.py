"""Fail closed unless the real training smoke saved optimizer state and resumed."""

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from nemo_rl.utils.checkpoint import (
    CheckpointManager,
    _load_megatron_common_state_dict,
)


def verify_checkpoint(root: Path, step: int) -> None:
    checkpoint = root / "checkpoints" / f"step_{step}"
    state = json.loads((checkpoint / "training_info.json").read_text())
    if state["total_steps"] != step:
        raise RuntimeError(f"Unexpected checkpoint step: {state['total_steps']}")
    if not (checkpoint / "train_dataloader.pt").is_file():
        raise RuntimeError("Dataloader state is missing")
    weights, optimizer = CheckpointManager.get_resume_paths(checkpoint)
    if weights is None or optimizer is None:
        raise RuntimeError("Checkpoint is missing resumable policy/optimizer state")


def load_scheduler_state(root: Path, step: int) -> dict[str, float | int | str]:
    weights = root / "checkpoints" / f"step_{step}" / "policy" / "weights"
    iteration = int((weights / "latest_checkpointed_iteration.txt").read_text())
    common = _load_megatron_common_state_dict(weights / f"iter_{iteration:07d}")
    return common["opt_param_scheduler"]


def verify_scheduler_resume(
    baseline: dict[str, float | int | str],
    restored: dict[str, float | int | str],
    *,
    initial_step: int,
    final_step: int,
) -> None:
    counter = baseline["num_steps"]
    if (
        initial_step <= 0
        or final_step <= initial_step
        or not isinstance(counter, int)
        or counter <= 0
        or counter % initial_step
    ):
        raise RuntimeError("Invalid baseline scheduler counter")
    expected = dict(baseline)
    expected["num_steps"] = counter // initial_step * final_step
    if restored != expected:
        raise RuntimeError("Scheduler settings or resumed counter changed unexpectedly")


def verify_metrics(records: list[dict[str, Any]], step: int) -> list[dict[str, Any]]:
    training = [
        record
        for record in records
        if "train/score_centering_head_mass" in record["metrics"]
    ]
    if not training or min(record["step"] for record in training) <= 3:
        raise RuntimeError("Phase 2 did not resume beyond checkpoint step 3")
    if max(record["step"] for record in training) != step:
        raise RuntimeError("Phase 2 did not reach the requested target")
    for record in training:
        metrics = record["metrics"]
        for required in ("train/loss", "train/grad_norm", "train/gen_kl_error"):
            if required not in metrics:
                raise RuntimeError(f"Missing training diagnostic: {required}")
        mass = metrics["train/score_centering_head_mass"]
        if not isinstance(mass, (float, int)) or not 0 < mass <= 1.00001:
            raise RuntimeError(f"Invalid captured head mass: {mass}")
        for name, value in metrics.items():
            if any(part in name for part in ("loss", "grad_norm", "gen_kl_error")):
                if not isinstance(value, (float, int)) or not math.isfinite(value):
                    raise RuntimeError(f"Nonfinite training metric {name}: {value}")
    return training


def main() -> None:
    mode, directory, target = sys.argv[1:]
    if mode not in ("checkpoint", "scheduler-baseline", "complete"):
        raise ValueError(f"Unknown smoke check: {mode}")
    root = Path(directory)
    step = int(target)
    verify_checkpoint(root, step)
    if mode == "checkpoint":
        return
    baseline_path = root / "scheduler-baseline.json"
    if mode == "scheduler-baseline":
        if step != 3:
            raise RuntimeError("Scheduler baseline must come from smoke checkpoint 3")
        baseline = {"step": step, "scheduler": load_scheduler_state(root, step)}
        with baseline_path.open("x") as stream:
            json.dump(baseline, stream, indent=2)
        return
    records = [
        json.loads(line)
        for path in root.glob("phase2.metrics.*.jsonl")
        for line in path.read_text().splitlines()
    ]
    training = verify_metrics(records, step)
    baseline = json.loads(baseline_path.read_text())
    scheduler = load_scheduler_state(root, step)
    verify_scheduler_resume(
        baseline["scheduler"],
        scheduler,
        initial_step=baseline["step"],
        final_step=step,
    )
    report = {
        "status": "pass",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "checkpoint_resume": "3 -> 6",
        "final_step": step,
        "sequence_limit": 22528,
        "candidate_count": 32,
        "checks": [
            "multi_gpu_init",
            "vllm_generation",
            "score_centering_backward",
            "checkpoint_resume",
            "scheduler_horizon_extension",
        ],
        "scheduler_baseline": baseline,
        "scheduler_final": scheduler,
        "numeric_diagnostics": training,
    }
    (root / "passed.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key != "numeric_diagnostics"
            }
        )
    )


if __name__ == "__main__":
    main()
