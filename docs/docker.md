# Build Docker Images

This guide explains how to build the NeMo RL Docker image.

The **release** image is our recommended option as it provides the most complete environment. It includes the base image with pre-fetched NeMo RL python packages in the `uv` cache, plus the nemo-rl source code and pre-fetched virtual environments for isolated workers. This is the ideal choice for production deployments.

## Building the Release Image

```sh
# Self-contained build (default: builds from main):
docker buildx build -f docker/Dockerfile \
    --tag <registry>/nemo-rl:latest \
    --push .

# Self-contained build (specific git ref):
docker buildx build -f docker/Dockerfile \
    --build-arg NRL_GIT_REF=r0.6.0 \
    --tag <registry>/nemo-rl:r0.6.0 \
    --push .

# Self-contained build (remote NeMo RL source; no need for a local clone of NeMo RL):
docker buildx build -f docker/Dockerfile \
    --build-arg NRL_GIT_REF=r0.6.0 \
    --tag <registry>/nemo-rl:r0.6.0 \
    --push https://github.com/NVIDIA-NeMo/RL.git

# Local NeMo RL source override:
docker buildx build --build-context nemo-rl=. -f docker/Dockerfile \
    --tag <registry>/nemo-rl:latest \
    --push .
```

> [!NOTE]
> The `--tag <registry>/nemo-rl:latest --push` flags are not necessary if you just want to build locally.

## Backend dependency versions

The universal lockfile selects different CUDA 13.0 Torch builds for separate
worker environments:

| Environment | Torch | torchvision | TileLang |
| --- | --- | --- | --- |
| Default environment (no backend extra) | 2.13.0 | 0.28.0 | 0.1.12 |
| `vllm` | 2.13.0 | 0.28.0 | 0.1.12 |
| `mcore` | 2.11.0 | 0.26.0 | 0.1.12 |
| `automodel`, `sglang` | 2.11.0 | 0.26.0 | 0.1.11 |
| `fsdp`, `trtllm` | 2.11.0 | 0.26.0 | 0.1.9 on x86_64; 0.1.11 on aarch64 |

The vLLM 0.29.0 wheel requires Torch 2.13.0, FlashInfer 0.6.18, TileLang 0.1.12,
CUTLASS DSL 4.6.2, and Quack 0.6.4. FlashInfer includes the
[small-batch BF16 MoE intermediate-allocation fix](https://github.com/flashinfer-ai/flashinfer/pull/4319).
Megatron independently needs TileLang 0.1.12 for Qwen3.5 Gated DeltaNet backward:
0.1.9 produced a misaligned-address failure, and 0.1.11 produced incorrect
gradients in the recorded GB200 reproducer. See the related
[Blackwell TMA swizzle-alignment issue](https://github.com/tile-ai/tilelang/issues/2379).

Select one backend per environment. In particular, `fsdp` cannot be combined
with `mcore` or `vllm`. The Docker build and Ray actor registry select backend
extras separately. A plain `uv sync` selects the default environment; always
include the backend extra when syncing a worker environment. The default
environment now uses Torch 2.13; training workers explicitly retain Torch 2.11.

Rebuild compiled CUDA dependencies and worker environments when changing this
stack; copying Torch into an existing worker environment can leave extensions
linked against the previous version. Lockfile resolution does not validate CUDA
imports, optional expert-parallel backends, or end-to-end training. The Qwen3.5
campaign used a separately prepared image; a fresh image built from these pins
still needs runtime validation.

## Skipping vLLM or SGLang Dependencies

If you don't need vLLM, SGLang, or TRT-LLM support, you can skip building those dependencies to reduce build time and image size. Use the `SKIP_VLLM_BUILD`, `SKIP_SGLANG_BUILD`, and/or `SKIP_TRTLLM_BUILD` build arguments:

```sh
# Skip vLLM dependencies:
docker buildx build -f docker/Dockerfile \
    --build-arg SKIP_VLLM_BUILD=1 \
    --tag <registry>/nemo-rl:latest \
    .

# Skip SGLang dependencies:
docker buildx build -f docker/Dockerfile \
    --build-arg SKIP_SGLANG_BUILD=1 \
    --tag <registry>/nemo-rl:latest \
    .

# Skip TRT-LLM dependencies:
docker buildx build -f docker/Dockerfile \
    --build-arg SKIP_TRTLLM_BUILD=1 \
    --tag <registry>/nemo-rl:latest \
    .

# Skip all three:
docker buildx build -f docker/Dockerfile \
    --build-arg SKIP_VLLM_BUILD=1 \
    --build-arg SKIP_SGLANG_BUILD=1 \
    --build-arg SKIP_TRTLLM_BUILD=1 \
    --tag <registry>/nemo-rl:latest \
    .
```

When these build arguments are set, the corresponding `uv sync --extra` commands are skipped, and the virtual environment prefetching will exclude actors that depend on those packages.

> [!NOTE]
> If you skip vLLM, SGLang, or TRT-LLM during the build but later try to use those backends at runtime, the dependencies will be fetched and built on-demand. This may add significant setup time on first use.

## Custom Setup Commands

By default, the Docker image installs [apptainer](https://apptainer.org/) (with a `singularity` symlink) via a pluggable `custom-setup` build stage. The default script is `docker/install_apptainer.sh`. You can override or skip this step at build time.

### Override with a custom script

Create a directory containing your setup script(s), then pass it as a build context along with the script filename:

```sh
# my-setup-dir/my_script.sh
#!/bin/bash
set -euo pipefail
apt-get update && apt-get install -y my-custom-package
apt-get clean && rm -rf /var/lib/apt/lists/*
```

```sh
docker buildx build \
  --build-context custom-setup=my-setup-dir/ \
  --build-arg CUSTOM_SETUP_FNAME=my_script.sh \
  -f docker/Dockerfile --tag <registry>/nemo-rl:latest .
```

### Skip custom setup entirely

To build without any custom setup commands, set `CUSTOM_SETUP_FNAME` to empty:

```sh
docker buildx build --build-arg CUSTOM_SETUP_FNAME= -f docker/Dockerfile --tag <registry>/nemo-rl:latest .
```
