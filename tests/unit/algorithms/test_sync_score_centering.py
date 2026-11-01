from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import (
    MasterConfig,
    _preserve_score_centering_data,
    _validate_sync_score_centering,
    add_grpo_token_loss_masks_and_generation_logprobs,
    dynamic_sampling,
)
from nemo_rl.algorithms.loss import ClippedPGLossFn
from nemo_rl.algorithms.loss.loss_input import prepare_loss_input
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.rollouts import generate_responses
from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from nemo_rl.utils.timer import Timer


@pytest.fixture(autouse=True)
def mock_nvtx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda name: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)


@pytest.fixture
def config() -> MasterConfig:
    register_omegaconf_resolvers()
    root = Path(__file__).resolve().parents[3]
    recipe = (
        root
        / "examples/configs/recipes/llm/grpo-qwen3-30ba3b-base-8n4g-megatron-te-nvfp4-pertoken.yaml"
    )
    resolved = OmegaConf.to_container(load_config(recipe), resolve=True)
    resolved["loss_fn"]["score_centering"] = True
    resolved["policy"]["generation"]["top_logprobs"] = 2
    return MasterConfig.model_validate(resolved)


def test_sync_score_centering_preserves_nvfp4_recipe(config: MasterConfig) -> None:
    _validate_sync_score_centering(config)
    assert config.grpo.use_dynamic_sampling
    assert config.grpo.reward_scaling.enabled
    assert config.grpo.reward_shaping.enabled
    assert config.policy["router_replay"]["enabled"]
    assert config.policy["megatron_cfg"]["tensor_model_parallel_size"] == 2
    assert config.policy["megatron_cfg"]["expert_model_parallel_size"] == 8
    loss_fn = ClippedPGLossFn(config.loss_fn)
    assert loss_fn._score_centering_is_bounds == (0.0, 2.0)


@pytest.mark.parametrize("count", [None, 0, -1, True, 1.5])
def test_sync_score_centering_rejects_invalid_count(
    config: MasterConfig, count: int | float | None
) -> None:
    config.policy["generation"]["top_logprobs"] = count
    with pytest.raises(ValueError, match="top_logprobs"):
        _validate_sync_score_centering(config)


@pytest.mark.parametrize(
    "path",
    ["async", "data_plane", "gym", "async_engine", "backend", "fused", "automodel_tp"],
)
def test_sync_score_centering_rejects_unsupported_paths(
    config: MasterConfig, path: str
) -> None:
    if path == "async":
        config.grpo.async_grpo.enabled = True
    elif path == "data_plane":
        config.data_plane["enabled"] = True
    elif path == "gym":
        config.env["should_use_nemo_gym"] = True
    elif path == "async_engine":
        config.policy["generation"]["vllm_cfg"]["async_engine"] = True
    elif path == "backend":
        config.policy["generation"]["backend"] = "sglang"
    elif path == "fused":
        config.policy["megatron_cfg"]["use_fused_linear_logprobs"] = True
    else:
        config.policy["automodel_cfg"]["enabled"] = True
        config.policy["automodel_cfg"]["tensor_parallel_size"] = 2
    with pytest.raises(ValueError):
        _validate_sync_score_centering(config)


@pytest.mark.parametrize("count", [0, 2])
def test_sync_score_centering_off_allows_capture_control(
    config: MasterConfig, count: int
) -> None:
    config.loss_fn.score_centering = False
    config.policy["generation"]["top_logprobs"] = count
    _validate_sync_score_centering(config)


def _worker(config: MasterConfig) -> VllmGenerationWorkerImpl:
    worker = object.__new__(VllmGenerationWorkerImpl)
    worker.cfg = deepcopy(config.policy["generation"])
    worker.cfg["_pad_token_id"] = 0
    worker.tensor_parallel_size = worker.expert_parallel_size = 1
    worker.routed_experts_dtype = torch.int8
    worker._native_lora_request = None
    worker._tokenize_prompt_with_bos = lambda prompt: prompt
    worker.SamplingParams = SimpleNamespace
    worker.llm = MagicMock()
    worker.llm.llm_engine.model_config.max_model_len = 128

    def generate(
        prompts: list[dict[str, Any]], sampling_params: SimpleNamespace, **kwargs: Any
    ) -> list[SimpleNamespace]:
        assert sampling_params.logprobs == (
            (worker.cfg.get("top_logprobs") or 0)
            if sampling_params.temperature != 0
            else 0
        )
        outputs = []
        for row, prompt in enumerate(prompts):
            completion = SimpleNamespace(
                index=0,
                token_ids=[7] * (row + 1),
                logprobs=[
                    {
                        7: SimpleNamespace(rank=3, logprob=-3.0),
                        row + 1: SimpleNamespace(rank=1, logprob=-0.4),
                        row + 4: SimpleNamespace(rank=2, logprob=-1.5),
                    }
                    for _ in range(row + 1)
                ],
                finish_reason="stop",
            )
            outputs.append(
                SimpleNamespace(
                    prompt_token_ids=prompt["prompt_token_ids"], outputs=[completion]
                )
            )
        return outputs

    worker.llm.generate.side_effect = generate
    return worker


def _rollout(config: MasterConfig) -> tuple[VllmGenerationWorkerImpl, BatchedDataDict]:
    worker = _worker(config)
    prompt_tokens = [torch.tensor([1, 2]), torch.tensor([3])]
    batch = BatchedDataDict(
        {
            "message_log": [
                [{"role": "user", "content": "prompt", "token_ids": tokens}]
                for tokens in prompt_tokens
            ]
        }
    )
    inputs = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2], [3, 0]]),
            "input_lengths": torch.tensor([2, 1]),
        }
    )
    tokenizer = SimpleNamespace(
        batch_decode=lambda tokens, **kwargs: ["response"] * len(tokens)
    )
    updated, _, _ = generate_responses(
        worker, inputs, batch, tokenizer, inputs["input_lengths"]
    )
    return worker, updated


def test_sync_candidates_survive_generation_padding_and_loss(
    config: MasterConfig,
) -> None:
    worker, batch = _rollout(config)
    assert worker.llm.generate.call_args.args[1].logprobs == 2
    add_grpo_token_loss_masks_and_generation_logprobs(batch["message_log"])
    flat, lengths = batched_message_log_to_flat_message(
        batch["message_log"], make_sequence_length_divisible_by=8
    )
    assert lengths.tolist() == [3, 3]
    assert flat["sampler_topk_token_ids"].shape == (2, 8, 2)
    assert flat["sampler_topk_token_ids"].dtype == torch.int32
    assert flat["sampler_topk_logprobs"].dtype == torch.float32
    assert flat["sampler_topk_token_ids"][0, 2].tolist() == [1, 4]
    assert flat["sampler_topk_token_ids"][1, 1:3].tolist() == [[2, 5], [2, 5]]
    assert torch.count_nonzero(flat["sampler_topk_token_ids"][:, 3:]) == 0
    data = BatchedDataDict(
        {
            "input_ids": flat["token_ids"],
            "input_lengths": lengths,
            "generation_logprobs": flat["generation_logprobs"],
            "prev_logprobs": flat["generation_logprobs"].clone(),
            "token_mask": flat["token_loss_mask"],
            "sample_mask": torch.ones(2),
            "advantages": torch.ones(2, 8),
        }
    )
    _preserve_score_centering_data(data, flat)
    logits = torch.randn(2, 8, 10, requires_grad=True)
    loss_fn = ClippedPGLossFn(config.loss_fn)
    prepared, _ = prepare_loss_input(logits, data, loss_fn, chunk_size=2)
    expected = (
        logits[:, :-1]
        .log_softmax(-1)
        .gather(-1, data["sampler_topk_token_ids"][:, 1:].long())
    )
    torch.testing.assert_close(prepared["policy_logprobs_at_sampler_topk"], expected)
    loss, metrics = loss_fn(
        **prepared,
        data=data,
        global_valid_seqs=torch.tensor(2),
        global_valid_toks=data["token_mask"].sum(),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert 0 < metrics["score_centering_head_mass"] < 1
    assert torch.count_nonzero(logits.grad[:, 3:]) == 0


def test_sync_candidates_survive_dynamic_sampling_cache(config: MasterConfig) -> None:
    config.grpo.num_prompts_per_step = 2
    config.grpo.num_generations_per_prompt = 1
    _, first = _rollout(config)
    first["total_reward"] = torch.tensor([0.0, 1.0])
    _, complete, cache, _ = dynamic_sampling(
        first,
        torch.ones(2),
        torch.zeros(2),
        1,
        config,
        Timer(),
        is_trivial_prompt_distribution=torch.tensor([True, False]),
    )
    assert not complete
    _, second = _rollout(config)
    second["total_reward"] = torch.tensor([1.0, 0.0])
    selected, complete, _, _ = dynamic_sampling(
        second,
        torch.ones(2),
        torch.zeros(2),
        2,
        config,
        Timer(),
        batch_cache=cache,
        is_trivial_prompt_distribution=torch.tensor([False, True]),
    )
    assert complete
    flat, _ = batched_message_log_to_flat_message(selected["message_log"])
    assert flat["sampler_topk_token_ids"][0, 1:3].tolist() == [[2, 5], [2, 5]]
    assert flat["sampler_topk_token_ids"][1, 2].tolist() == [1, 4]


def test_sync_candidates_preserve_previous_turns_and_fill_tool_tokens(
    config: MasterConfig,
) -> None:
    _, batch = _rollout(config)
    history = batch["message_log"][0]
    history.append(
        {"role": "tool", "content": "observation", "token_ids": torch.tensor([6, 8])}
    )
    history.append(deepcopy(history[1]))
    flat, _ = batched_message_log_to_flat_message([history])
    assert flat["sampler_topk_token_ids"][0].tolist() == [
        [0, 0],
        [0, 0],
        [1, 4],
        [0, 0],
        [0, 0],
        [1, 4],
    ]


def test_sync_candidates_empty_worker_and_greedy(config: MasterConfig) -> None:
    worker = _worker(config)
    empty = worker.generate(
        BatchedDataDict(
            {
                "input_ids": torch.empty(0, 0, dtype=torch.long),
                "input_lengths": torch.empty(0, dtype=torch.long),
            }
        )
    )
    assert empty["sampler_topk_token_ids"].shape == (0, 0, 2)
    assert worker._build_sampling_params(greedy=True, stop_strings=None).logprobs == 0
    populated = worker.generate(
        BatchedDataDict(
            {"input_ids": torch.tensor([[1, 2]]), "input_lengths": torch.tensor([2])}
        )
    )
    combined = BatchedDataDict.from_batches([empty, populated, empty])
    torch.testing.assert_close(
        combined["sampler_topk_token_ids"], populated["sampler_topk_token_ids"]
    )


@pytest.mark.parametrize(("count", "greedy"), [(None, False), (0, False), (2, True)])
def test_sync_candidates_disabled_preserves_ordinary_output(
    config: MasterConfig, count: int | None, greedy: bool
) -> None:
    config.policy["generation"]["top_logprobs"] = count
    worker = _worker(config)
    output = worker.generate(
        BatchedDataDict(
            {"input_ids": torch.tensor([[1, 2]]), "input_lengths": torch.tensor([2])}
        ),
        greedy=greedy,
    )
    assert worker.llm.generate.call_args.args[1].logprobs == 0
    assert "sampler_topk_token_ids" not in output
    assert "sampler_topk_logprobs" not in output
    assert output["output_ids"].tolist() == [[1, 2, 7]]


def test_sync_candidates_zero_fill_assistant_prompt_history(
    config: MasterConfig,
) -> None:
    _, batch = _rollout(config)
    batch["message_log"][0].insert(
        0, {"role": "assistant", "content": "example", "token_ids": torch.tensor([6])}
    )
    add_grpo_token_loss_masks_and_generation_logprobs(batch["message_log"])
    flat, _ = batched_message_log_to_flat_message(batch["message_log"])
    assert flat["sampler_topk_token_ids"][0, 0].tolist() == [0, 0]
    assert flat["token_loss_mask"][0, 0] == 0


@pytest.mark.parametrize("missing", ["sampler_topk_token_ids", "sampler_topk_logprobs"])
def test_sync_candidates_reject_incomplete_turn(
    config: MasterConfig, missing: str
) -> None:
    _, batch = _rollout(config)
    del batch["message_log"][0][1][missing]
    with pytest.raises(ValueError, match="Score centering"):
        batched_message_log_to_flat_message(batch["message_log"])


def test_sync_candidates_reject_incomplete_training_batch() -> None:
    target = BatchedDataDict({"input_ids": torch.zeros(1, 4, dtype=torch.long)})
    with pytest.raises(ValueError, match="missing rollout field"):
        _preserve_score_centering_data(target, BatchedDataDict({}))


def test_sync_candidates_reject_misaligned_training_batch() -> None:
    target = BatchedDataDict({"input_ids": torch.zeros(1, 4, dtype=torch.long)})
    flat = BatchedDataDict(
        {"sampler_topk_token_ids": torch.zeros(1, 3, 2, dtype=torch.int32)}
    )
    with pytest.raises(ValueError, match="align with input_ids"):
        _preserve_score_centering_data(target, flat)
