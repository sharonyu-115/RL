import json
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from launch.score_centering.check_smoke import verify_metrics, verify_scheduler_resume
from launch.score_centering.metrics import ScalarJsonlLogger
from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


@pytest.mark.parametrize("stop_step,scheduler_steps", [(3, 1500), (6, 1800)])
def test_smoke_extension_restores_checkpoint_scheduler(
    stop_step: int, scheduler_steps: int
) -> None:
    register_omegaconf_resolvers()
    root = Path(__file__).resolve().parents[3]
    config = load_config(
        root
        / "examples/configs/recipes/llm/grpo-qwen3-30ba3b-base-8n4g-megatron-te-nvfp4-pertoken.yaml"
    )
    config = parse_hydra_overrides(
        config,
        [
            f"grpo.max_num_steps={stop_step}",
            f"++policy.megatron_cfg.scheduler.max_steps={scheduler_steps}",
            "++policy.megatron_cfg.scheduler.use_checkpoint_opt_param_scheduler=true",
        ],
    )
    resolved = MasterConfig.model_validate(OmegaConf.to_container(config, resolve=True))
    assert resolved.grpo.max_num_steps == stop_step
    assert resolved.policy["megatron_cfg"]["scheduler"]["max_steps"] == scheduler_steps
    assert resolved.policy["megatron_cfg"]["scheduler"][
        "use_checkpoint_opt_param_scheduler"
    ]
    assert resolved.policy["megatron_cfg"]["scheduler"]["lr_warmup_iters"] == 10


def test_production_extension_keeps_original_checkpoint_schedule() -> None:
    register_omegaconf_resolvers()
    root = Path(__file__).resolve().parents[3]
    config = load_config(
        root
        / "examples/configs/recipes/llm/grpo-qwen3-30ba3b-base-8n4g-megatron-te-nvfp4-pertoken.yaml"
    )
    config = parse_hydra_overrides(
        config,
        [
            "grpo.max_num_steps=1800",
            "grpo.max_num_epochs=null",
            "++policy.megatron_cfg.scheduler.use_checkpoint_opt_param_scheduler=true",
        ],
    )
    resolved = MasterConfig.model_validate(OmegaConf.to_container(config, resolve=True))
    assert resolved.grpo.max_num_steps == 1800
    assert resolved.grpo.max_num_epochs is None
    scheduler = resolved.policy["megatron_cfg"]["scheduler"]
    assert scheduler["use_checkpoint_opt_param_scheduler"]
    assert scheduler["lr_decay_iters"] == 1000
    assert scheduler["lr_warmup_iters"] == 10
    assert "max_steps" not in scheduler


def test_scheduler_resume_preserves_settings_and_advances_counter() -> None:
    baseline = {"num_steps": 1536, "wd_incr_steps": 768000, "max_lr": 1e-6}
    restored = {**baseline, "num_steps": 3072}
    verify_scheduler_resume(baseline, restored, initial_step=3, final_step=6)
    assert baseline["num_steps"] == 1536


@pytest.mark.parametrize(
    "field,value", [("num_steps", 1536), ("wd_incr_steps", 921600), ("max_lr", 1e-7)]
)
def test_scheduler_resume_rejects_reset_or_changed_schedule(
    field: str, value: float | int
) -> None:
    baseline = {"num_steps": 1536, "wd_incr_steps": 768000, "max_lr": 1e-6}
    restored = {**baseline, "num_steps": 3072, field: value}
    with pytest.raises(RuntimeError, match="changed unexpectedly"):
        verify_scheduler_resume(baseline, restored, initial_step=3, final_step=6)


def _record(step: int) -> dict[str, Any]:
    return {
        "step": step,
        "metrics": {
            "train/score_centering_head_mass": 0.7,
            "train/loss": -0.1,
            "train/grad_norm": 0.2,
            "train/gen_kl_error": 0.01,
        },
    }


def test_smoke_accepts_resumed_finite_metrics() -> None:
    records = [_record(step) for step in range(4, 7)]
    assert verify_metrics(records, 6) == records


@pytest.mark.parametrize("step", [1, 3])
def test_smoke_rejects_restarted_training(step: int) -> None:
    with pytest.raises(RuntimeError, match="did not resume"):
        verify_metrics([_record(step), _record(6)], 6)


@pytest.mark.parametrize("field", ["loss", "grad_norm", "gen_kl_error"])
def test_smoke_rejects_missing_or_nonfinite_diagnostics(field: str) -> None:
    record = _record(6)
    del record["metrics"][f"train/{field}"]
    with pytest.raises(RuntimeError, match="Missing training diagnostic"):
        verify_metrics([record], 6)
    record["metrics"][f"train/{field}"] = "nan"
    with pytest.raises(RuntimeError, match="Nonfinite training metric"):
        verify_metrics([record], 6)


@pytest.mark.parametrize("mass", [0.0, 1.1, "nan"])
def test_smoke_rejects_invalid_head_mass(mass: float | str) -> None:
    record = _record(6)
    record["metrics"]["train/score_centering_head_mass"] = mass
    with pytest.raises(RuntimeError, match="Invalid captured head mass"):
        verify_metrics([record], 6)


def test_smoke_rejects_incomplete_training() -> None:
    with pytest.raises(RuntimeError, match="requested target"):
        verify_metrics([_record(4)], 6)


def test_local_logger_preserves_nonfinite_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("SC_METRICS_PATH", str(output))
    logger = ScalarJsonlLogger()
    logger.log_metrics(
        {"loss": torch.tensor(0.5), "grad_norm": float("inf"), "array": torch.ones(2)},
        4,
        prefix="train",
    )
    record = json.loads(output.read_text())
    assert record["step"] == 4
    assert record["metrics"] == {"train/loss": 0.5, "train/grad_norm": "inf"}
