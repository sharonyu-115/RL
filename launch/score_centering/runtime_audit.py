"""Audit the pinned driver and worker runtimes before saving the smoke image."""

import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import torch


def main() -> None:
    role = sys.argv[1]
    campaign = Path(os.environ["CAMPAIGN"])
    if role == "finalize":
        reports = {
            name: json.loads((campaign / f"runtime-{name}.json").read_text())
            for name in ("driver", "policy", "generation")
        }
        report = {
            "status": "pass",
            "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "submodules": subprocess.check_output(
                ["git", "submodule", "status", "--recursive"], text=True
            ),
            "fingerprint": json.loads(
                (campaign / "runtime-fingerprint.json").read_text()
            ),
            "roles": reports,
        }
        (campaign / "runtime-audit.json").write_text(json.dumps(report, indent=2))
        return
    expected = {"torch": "2.13.0", "ray": "2.58.0", "nvidia-cudnn-cu13": "9.26.0.51"}
    expected["transformers"] = "5.12.1" if role == "generation" else "5.16.1"
    if role == "policy":
        expected["transformer-engine"] = "2.18.0"
    if role == "generation":
        expected["vllm"] = "0.29.0"
    installed = {name: importlib.metadata.version(name) for name in expected}
    for name, version in expected.items():
        if installed[name].split("+")[0] != version:
            raise RuntimeError(
                f"{role}: expected {name}={version}, got {installed[name]}"
            )
    if not torch.cuda.is_available():
        raise RuntimeError(f"{role}: CUDA is unavailable")
    if torch.cuda.device_count() != 4:
        raise RuntimeError(f"{role}: expected all four GPUs of the exclusive node")
    if torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError(f"{role}: expected Blackwell SM 10.x")
    if torch.backends.cudnn.version() < 92600:
        raise RuntimeError(
            f"{role}: loaded cuDNN is older than the pinned 9.26 runtime"
        )
    probe = torch.ones((4, 4), device="cuda")
    if (probe @ probe).sum().item() != 64:
        raise RuntimeError(f"{role}: CUDA matrix multiplication probe failed")
    if role == "policy":
        import megatron.bridge
        import megatron.core
        import transformer_engine.pytorch

        if not transformer_engine.pytorch.__file__:
            raise RuntimeError("Transformer Engine failed to import")
        for module in (megatron.bridge, megatron.core):
            if "/opt/nemo-rl/3rdparty/" not in str(module.__file__):
                raise RuntimeError(
                    f"Unexpected source for {module.__name__}: {module.__file__}"
                )
    if role == "generation":
        import flashinfer
        import vllm

        if not flashinfer.__file__ or not vllm.__file__:
            raise RuntimeError("Generation libraries failed to import")
    report = {"role": role, "versions": installed, "gpu": torch.cuda.get_device_name()}
    (campaign / f"runtime-{role}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
