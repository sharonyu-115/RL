# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contract of score-centering request preparation and sampler extraction."""

from types import SimpleNamespace

import pytest
import torch

from nemo_rl.models.generation.score_centering import (
    build_score_centering_request_overrides,
    configure_score_centering_logprob_cap,
    extract_score_centering_samples,
    validate_score_centering_request,
    validate_score_centering_sampling,
)


@pytest.mark.parametrize(
    ("cap", "expected"), [(None, 32), (16, 32), (64, 64), (-1, -1)]
)
def test_score_centering_engine_logprob_cap(cap: int | None, expected: int) -> None:
    config = {"top_logprobs": 32, "vllm_kwargs": {}}
    if cap is not None:
        config["vllm_kwargs"]["max_logprobs"] = cap
    configure_score_centering_logprob_cap(config)
    assert config["vllm_kwargs"]["max_logprobs"] == expected


@pytest.mark.parametrize("count", [None, 0])
def test_score_centering_disabled_preserves_engine_kwargs(count: int | None) -> None:
    config = {"vllm_kwargs": {"max_logprobs": 8}}
    if count is not None:
        config["top_logprobs"] = count
    configure_score_centering_logprob_cap(config)
    assert config["vllm_kwargs"] == {"max_logprobs": 8}


@pytest.mark.parametrize(
    ("temperature", "top_p", "top_k", "mode", "valid"),
    [
        (1.0, 1.0, None, None, True),
        (0.7, 1.0, None, "processed_logprobs", True),
        (0.7, 1.0, None, "raw_logprobs", False),
        (0.0, 1.0, None, None, False),
        (float("nan"), 1.0, None, None, False),
        (float("inf"), 1.0, None, None, False),
        (1.0, 0.9, None, None, False),
        (1.0, 1.0, 32, None, False),
        (1.0, 1.0, None, "raw_logits", False),
    ],
)
def test_score_centering_sampling_validation(
    temperature: float, top_p: float, top_k: int | None, mode: str | None, valid: bool
) -> None:
    config = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "vllm_cfg": {"logprobs_mode": mode},
    }
    if valid:
        validate_score_centering_sampling(config)
    else:
        with pytest.raises(ValueError):
            validate_score_centering_sampling(config)


def test_score_centering_request_and_sample_contract():
    # Gym merges these fields into existing request overrides. Sampling controls
    # such as temperature must survive, while the requested logprob count wins.
    request = {"temperature": 1.0, "top_logprobs": 0}
    request.update(build_score_centering_request_overrides(num_candidates=2))
    assert request["temperature"] == 1.0
    validate_score_centering_request(SimpleNamespace(**request), num_candidates=2)

    output = SimpleNamespace(
        index=0,
        token_ids=[9, 3],
        logprobs=[
            {
                # The generated token is outside the head and arrives first.
                9: SimpleNamespace(rank=7, logprob=-3.0),
                1: SimpleNamespace(rank=1, logprob=-0.3),
                2: SimpleNamespace(rank=2, logprob=-1.8),
            },
            {
                4: SimpleNamespace(rank=2, logprob=-10000.0),
                3: SimpleNamespace(rank=1, logprob=-0.1),
            },
        ],
    )
    # Staging receives CPU tensors even if the serving caller has selected a
    # different default device.
    with torch.device("meta"):
        sampler_ids, sampler_logprobs = extract_score_centering_samples(
            output, num_candidates=request["top_logprobs"]
        )

    assert sampler_ids.dtype == torch.int32
    assert sampler_logprobs.dtype == torch.float32
    assert sampler_ids.device.type == sampler_logprobs.device.type == "cpu"
    assert sampler_ids.tolist() == [[1, 2], [3, 4]]
    torch.testing.assert_close(
        sampler_logprobs, torch.tensor([[-0.3, -1.8], [-0.1, -10000.0]])
    )
    # The retained logprobs describe the full sampler distribution, not a
    # distribution renormalized over the chosen candidates.
    assert torch.all(sampler_logprobs.exp().sum(-1) < 1)


@pytest.mark.parametrize("logprobs", [None, []])
def test_score_centering_empty_completion(logprobs):
    sampler_ids, sampler_logprobs = extract_score_centering_samples(
        SimpleNamespace(index=0, token_ids=[], logprobs=logprobs), num_candidates=2
    )
    assert sampler_ids.shape == sampler_logprobs.shape == (0, 2)
    assert sampler_ids.dtype == torch.int32
    assert sampler_logprobs.dtype == torch.float32


@pytest.mark.parametrize("token_ids", [[1], [1, 2]])
def test_score_centering_rejects_specific_logprob_ids(token_ids):
    request = SimpleNamespace(
        logprobs=True, top_logprobs=2, logprob_token_ids=token_ids
    )
    with pytest.raises(ValueError, match="omit logprob_token_ids"):
        validate_score_centering_request(request, num_candidates=2)


@pytest.mark.parametrize("token_ids", [None, []])
def test_score_centering_allows_unset_specific_logprob_ids(token_ids):
    request = SimpleNamespace(
        logprobs=True, top_logprobs=2, logprob_token_ids=token_ids
    )
    validate_score_centering_request(request, num_candidates=2)


def test_score_centering_rejects_incomplete_candidates():
    output = SimpleNamespace(
        index=0, token_ids=[1], logprobs=[{1: SimpleNamespace(rank=1, logprob=-0.1)}]
    )
    with pytest.raises(RuntimeError, match="fewer than top_k"):
        extract_score_centering_samples(output, num_candidates=2)


@pytest.mark.parametrize(
    "request_fields",
    [{"logprobs": False, "top_logprobs": 2}, {"logprobs": True, "top_logprobs": 0}],
)
def test_score_centering_rejects_inconsistent_request(request_fields):
    with pytest.raises(ValueError, match="top_logprobs=2 over HTTP"):
        validate_score_centering_request(
            SimpleNamespace(**request_fields), num_candidates=2
        )


def test_score_centering_rejects_zero_candidates():
    with pytest.raises(ValueError, match="candidate count must be positive"):
        build_score_centering_request_overrides(num_candidates=0)


@pytest.mark.parametrize(
    ("token_id", "logprob"),
    [
        (-1, -1.0),
        (True, -1.0),
        (2**31, -1.0),
        (1, float("nan")),
        (1, float("inf")),
        (1, -float("inf")),
        (1, 0.1),
        (1, True),
        (1, -1e300),
    ],
)
def test_score_centering_rejects_invalid_probabilities(token_id, logprob):
    output = SimpleNamespace(
        index=0,
        token_ids=[1],
        logprobs=[{token_id: SimpleNamespace(rank=1, logprob=logprob)}],
    )
    with pytest.raises(ValueError, match="Invalid sampler"):
        extract_score_centering_samples(output, num_candidates=1)


def test_score_centering_rejects_excess_probability_mass():
    output = SimpleNamespace(
        index=0,
        token_ids=[1],
        logprobs=[
            {
                1: SimpleNamespace(rank=1, logprob=-0.1),
                2: SimpleNamespace(rank=2, logprob=-0.1),
            }
        ],
    )
    with pytest.raises(ValueError, match="exceed unit mass"):
        extract_score_centering_samples(output, num_candidates=2)


def test_score_centering_rejects_missing_logprob_rows():
    with pytest.raises(RuntimeError, match="generated token count"):
        extract_score_centering_samples(
            SimpleNamespace(index=0, token_ids=[1], logprobs=None), num_candidates=1
        )
