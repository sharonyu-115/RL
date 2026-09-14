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

"""Guards on pins we declare directly even though a submodule already declares them.

`megatron-energon` reaches us transitively as
nemo-rl[mcore] -> megatron-bridge[te,ssm] -> megatron-core[dev,mlm] -> megatron-energon,
and the `mcore` extra floors it higher than Megatron-LM does. Bumping the
Megatron-Bridge submodule moves the Megatron-LM pointer underneath us, so these tests
fail the moment upstream's own pin changes and the floor needs a second look.
"""

import subprocess
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.markers import default_environment
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

REPO_ROOT = Path(__file__).parents[2]
MEGATRON_LM_PYPROJECT = (
    REPO_ROOT
    / "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/pyproject.toml"
)

# What Megatron-LM's `dev` extra pins today. If a Megatron-Bridge bump moves this,
# re-check the megatron-energon floor in our `mcore` extra and update this constant in
# the same commit.
MEGATRON_LM_ENERGON_SPEC = "~=7.0"


def _requirement(pyproject: Path, extra: str, name: str) -> Requirement:
    """The lone requirement for `name` in `pyproject`'s `extra` optional-dependency list."""
    project = tomllib.loads(pyproject.read_text())["project"]
    deps = project["optional-dependencies"][extra]
    matches = [
        req
        for req in map(Requirement, deps)
        if canonicalize_name(req.name) == canonicalize_name(name)
    ]
    assert len(matches) == 1, (
        f"expected exactly one {name} requirement in [{extra}] of {pyproject}, got {matches}"
    )
    return matches[0]


def _bounds(spec: SpecifierSet) -> tuple[Version, Version]:
    """Inclusive lower and exclusive upper bound of a lone `~=X.Y[.Z]` specifier."""
    specifiers = list(spec)
    assert len(specifiers) == 1 and specifiers[0].operator == "~=", (
        f"expected a single compatible-release pin, got '{spec}'"
    )
    release = Version(specifiers[0].version).release
    prefix = release[:-1]  # `~=X.Y` implies `<X+1`; `~=X.Y.Z` implies `<X.Y+1`
    upper = prefix[:-1] + (prefix[-1] + 1,)
    return Version(specifiers[0].version), Version(
        ".".join(str(part) for part in upper)
    )


@pytest.fixture(scope="module")
def megatron_lm_energon() -> Requirement:
    if not MEGATRON_LM_PYPROJECT.exists():
        pytest.skip(
            f"{MEGATRON_LM_PYPROJECT} missing; run `git submodule update --init --recursive`"
        )
    return _requirement(MEGATRON_LM_PYPROJECT, "dev", "megatron-energon")


def test_megatron_lm_energon_pin_is_unchanged(megatron_lm_energon: Requirement) -> None:
    assert str(megatron_lm_energon.specifier) == MEGATRON_LM_ENERGON_SPEC, (
        f"Megatron-LM now pins megatron-energon '{megatron_lm_energon.specifier}', not "
        f"'{MEGATRON_LM_ENERGON_SPEC}'. Re-check the megatron-energon floor in the `mcore` "
        f"extra of pyproject.toml against the new range, then update "
        f"MEGATRON_LM_ENERGON_SPEC here in the same commit."
    )


def test_mcore_energon_floor_is_within_megatron_lm_range(
    megatron_lm_energon: Requirement,
) -> None:
    ours = _requirement(REPO_ROOT / "pyproject.toml", "mcore", "megatron-energon")
    our_low, our_high = _bounds(ours.specifier)
    their_low, their_high = _bounds(megatron_lm_energon.specifier)
    assert our_low >= their_low and our_high <= their_high, (
        f"the `mcore` extra pins megatron-energon '{ours.specifier}', which is not a subset "
        f"of Megatron-LM's '{megatron_lm_energon.specifier}'. Our pin only exists to raise "
        f"the floor; widening it past upstream either makes `uv lock` unsatisfiable or "
        f"silently has no effect."
    )


@pytest.fixture(
    scope="module",
    params=[
        ("vllm",),
        ("vllm", "nemo_gym"),
        ("vllm", "modelopt"),
        ("mcore",),
        ("mcore", "modelopt"),
        ("fsdp",),
        ("automodel",),
        ("sglang",),
        ("trtllm",),
    ],
)
def backend_lock_requirements(
    request: pytest.FixtureRequest,
) -> tuple[str, list[Requirement]]:
    """Export the actual selected lock graph without resolving or installing it."""
    extras = request.param
    result = subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--offline",
            "--no-hashes",
            "--no-header",
            "--no-annotate",
            "--no-emit-local",
            *(flag for extra in extras for flag in ("--extra", extra)),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return extras[0], [
        Requirement(line) for line in result.stdout.splitlines() if line.strip()
    ]


@pytest.mark.parametrize("architecture", ["x86_64", "aarch64"])
def test_backend_lock_preserves_torch_and_compiler_versions(
    backend_lock_requirements: tuple[str, list[Requirement]], architecture: str
) -> None:
    """Catch a global override or lock fork silently downgrading a worker stack."""
    backend, requirements = backend_lock_requirements
    environment = default_environment() | {
        "platform_machine": architecture,
        "sys_platform": "linux",
        "platform_system": "Linux",
        "python_version": "3.13",
        "python_full_version": "3.13.14",
    }
    selected = {
        canonicalize_name(req.name): req
        for req in requirements
        if req.marker is None or req.marker.evaluate(environment)
    }
    expected = {
        "torch": "2.13.0+cu130" if backend == "vllm" else "2.11.0+cu130",
        "torchvision": "0.28.0+cu130" if backend == "vllm" else "0.26.0+cu130",
        "triton": "3.7.1" if backend == "vllm" else "3.6.0",
    }
    if backend in ("vllm", "mcore"):
        expected["tilelang"] = "0.1.12"
    if backend == "vllm":
        expected.update(
            {
                "flashinfer-python": "0.6.18",
                "flashinfer-cubin": "0.6.18",
                "flashinfer-jit-cache": "0.6.18+cu130",
                "nvidia-cutlass-dsl": "4.6.2",
                "quack-kernels": "0.6.4",
            }
        )
        assert selected["vllm"].url == (
            "https://github.com/vllm-project/vllm/releases/download/v0.29.0/"
            f"vllm-0.29.0-cp38-abi3-manylinux_2_28_{architecture}.whl"
        )
    for package, version in expected.items():
        assert str(selected[package].specifier) == f"=={version}", (
            f"{backend} on {architecture}: unexpected {selected[package]}"
        )
