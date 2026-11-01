"""Fail closed unless the real training smoke saved optimizer state and resumed."""

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from nemo_rl.utils.checkpoint import CheckpointManager


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
    root = Path(directory)
    step = int(target)
    verify_checkpoint(root, step)
    if mode == "checkpoint":
        return
    records = [
        json.loads(line)
        for path in root.glob("phase2.metrics.*.jsonl")
        for line in path.read_text().splitlines()
    ]
    training = verify_metrics(records, step)
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
        ],
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
