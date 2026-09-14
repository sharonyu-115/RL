# PR3566 follow-up branch notes

This is a development note for the Qwen3.5 NVFP4 follow-up to NVIDIA-NeMo/RL
PR3566. Keep it on the development branch so later sessions can recover the
stack relationship. Remove this note from the final upstream publication branch
after transferring useful dependency and validation details into product docs
and the PR description.

## Branch provenance

- Development branch: `feat/qwen35-nvfp4-pertoken`.
- Original parent base: `0df0aeee07f7b40e92eb5df232ffba1b2f1a7e53`.
- Annotated local tag: `followup/pr3566-original-base`.
- Worktree: `/lustre/fsw/general_sa/shuangy/src/NeMo-RL/nemo-rl-qwen35-nvfp4-followup`.
- At setup, the user reported PR3566 was unmerged and had not yet been rebased
  onto upstream main. This is historical context, not a live GitHub status check.
- Never move the original-base tag when PR3566 rebases. Tags are shared by local
  worktrees but are not automatically present in another clone. The full SHA
  above is the fallback when the tag is absent.
- Setup creates a local branch and tag only; remote publication is separate.
  `origin` currently points to NVIDIA-NeMo/RL; `shuangy` points to
  sharonyu-115/RL. Verify destinations before pushing. Do not overwrite PR3566.

The source patch was reconstructed from:

```text
/lustre/fsw/general_sa/shuangy/src/NeMo-RL/nemo-rl-pr3566-qwen35-nvfp4-20260911/session/20260912_051742/code-commit-20260914.patch
SHA256 b35fcb909369ed3dfbd96909539e5339acf79ccf8891b4ce312caccf9f2b3f8e
```

The adjacent `code-commit-handoff.md`, `code-commit-20260914.sha256`, and
`code-commit-20260914-submodules.txt` describe the exact reconstruction. All 11
resulting file hashes passed at setup. The older `validated-code.patch` is an
incomplete, five-file BF16 handoff; do not substitute it.

## Development scope

Keep follow-up changes in reviewable, signed-off commits:

1. Grouped MoE checkpoint quantization/refit support, vLLM RPC compatibility and
   capture rebinding, with generation tests (five files).
2. Megatron self-packing router replay alignment and its tests (two files).
3. Verified dependency updates, with `pyproject.toml` and `uv.lock` together.
4. Portable Qwen3.5 NVFP4 replay OFF/ON recipes and drivers, appropriate nightly
   registration, and supporting documentation.

The seven source/test files are committed, unchanged from the verified handoff:

- `a6ba9ed80`: grouped MoE checkpoint refits and generation tests (five files).
- `5c374a318`: self-packing router replay alignment and tests (two files).

The four campaign recipes/scripts remain untracked. They contain personal
defaults and depend on an untracked operational launcher; replace those in this
worktree before committing. Nightly lists have not yet been updated.

The dependency update includes `pyproject.toml`, `uv.lock`, `docs/docker.md`,
and `tests/unit/test_dependency_pins.py` in one signed-off commit.
The lock was generated with Docker's uv 0.11.28.
Independent local clones of the four recorded dependency checkouts below were
initialized without changing their revisions or the source worktrees.

The prepared lock selects vLLM 0.29 / Torch 2.13 / FlashInfer 0.6.18 /
TileLang 0.1.12 / CUTLASS DSL 4.6.2 / Quack 0.6.4. Megatron's exported graph
changes only TileLang 0.1.9 -> 0.1.12, retaining Torch 2.11. Export comparisons
preserve the FSDP, Automodel, SGLang and TRT-LLM package selections on both Linux
architectures. The default driver environment now selects Torch 2.13,
torchvision 0.28, Triton 3.7.1 and TileLang 0.1.12; that is an additional change
from the historical campaign's base environment. FSDP is declared incompatible
with vLLM and Megatron extras, matching their differing runtime/compiler pins.

Fresh dependency checks: 20 tests passed using `--noconftest` (these tests only
inspect package metadata and export the lock, so they do not need Ray). Coverage
includes both Linux architectures and vLLM/Gym and ModelOpt worker combinations.
`uv lock --check`, locked dry-run syncs for vLLM/Gym and Megatron, Ruff and Taplo
checks passed. No packages were installed into worker environments; no image
was rebuilt or GPU validation launched. Fresh-container CUDA validation remains
outstanding. Local evidence is in `session/20260913_194317/dependency-*`.

Before these commits, all seven files passed Ruff lint, Ruff format checks,
Python syntax parsing, and handoff SHA256 verification. The local generation
pytest attempt failed during Ray setup before running a test (report under
`session/20260913_194317/source-generation-local.xml`). The host also lacks
the vLLM and Megatron worker environments; no fresh worker-suite pass is claimed.
Historical `packing-unit-2808430.log` records 103 Megatron tests passed with
three distributed cases deselected, and 74 generation tests passed. That is
historical evidence, not a substitute for a fresh run in the worker environments.

The recorded dependency checkouts are:

```text
Automodel       1814c6c93a66b9d59d254960ef6a99a64249b671
Gym             fd5e84d6b1c485c80e7ae61553bbd485611c03b4
Megatron-Bridge 5ed97996cc2b422904d18179375b6d7366915097
Megatron-LM     1e7598cbfae888cdd3d741a351aae588d56f66c0
```

Initialize independent dependency checkouts at these revisions before testing;
do not reuse writable paths from live sessions.

## Dependency evidence to preserve

- The preserved vLLM 0.29.0 ARM wheel declares `torch==2.13.0`,
  `flashinfer-python==0.6.18`, and `tilelang==0.1.12`. Torch 2.13 was installed
  transitively when upgrading generation, not to address a separately diagnosed
  Torch failure. Training remained on Torch 2.11 / Transformer Engine 2.18.
- Generation upgrades addressed small-batch BF16/NVFP4 MoE crashes. The BF16
  investigation identified FlashInfer PR4319; the exact NVFP4 fixing commit was
  not established. Preserved evidence and installer:
  `/lustre/fsw/general_sa/shuangy/qwen35-v029-validation-20260912/`.
- Training independently required TileLang 0.1.9 -> 0.1.12 for FLA Gated
  DeltaNet backward. 0.1.9 generated a misaligned TMA shared-memory load;
  0.1.11 avoided the crash but produced incorrect gradients. 0.1.12 passed the
  recorded 24 output/gradient/repeatability checks and device-memory checking.
  Only TileLang changed in the Megatron worker; FLA remained 0.5.1. Diagnosis:
  `/lustre/fsw/general_sa/shuangy/src/NeMo-RL/nemo-rl-pr3566-qwen35-bf16-20260912/bf16_baseline/diagnosis/ROOT_CAUSE.md`.
- The image bypassed source-lock fingerprint checks. Image-based success does
  not prove a clean source install. Verify dependency resolution and optional
  compiled backend compatibility; the generation image recorded ABI warnings
  for optional EP/DeepGEMM backends outside its TP4 validation scope.
- Convergence was incomplete at handoff, with unresolved logprob tails. Recheck
  evidence before publication; do not claim convergence passed.

## After PR3566 is rebased and merged

Leave the development branch on its original base while PR3566 changes. If
testing against the rebased parent is needed, use a separate integration branch.
Do not merge or rebase the parent into this preserved development branch without
explicitly revising this procedure.

Once the follow-up changes are committed and the worktree is clean, verify that
PR3566 is merged, verify `origin` is NVIDIA-NeMo/RL, and use a new publication
branch (verify its name is unused):

```bash
git fetch origin
git switch -c feat/qwen35-nvfp4-pertoken-upstream feat/qwen35-nvfp4-pertoken
git rebase --onto origin/main followup/pr3566-original-base
```

If the local tag is missing, use its exact commit instead:

```bash
git rebase --onto origin/main 0df0aeee07f7b40e92eb5df232ffba1b2f1a7e53
```

These commands replay only commits after the original parent base, including
the development note. They work with a squashed, rebased, or merge-commit parent.
Use one rebase command, not both. A plain `git rebase origin/main` can attempt to
replay rewritten parent commits. Resolve conflicts against the final merged
parent, dropping changes already implemented upstream when appropriate.

Compare the old and new follow-up series and inspect the final upstream diff:

```bash
git range-diff followup/pr3566-original-base..feat/qwen35-nvfp4-pertoken origin/main..HEAD
git diff --stat origin/main...HEAD
git diff --check
```

Rerun relevant tests after integration. Remove this development note on the
publication branch, then push that new branch normally to the verified fork and
open the follow-up PR against upstream `main`. The original branch remains a
backup; no force push is needed. Do not trigger CI comments implicitly.

## Live campaign isolation

The shared `nemo-rl-pr3566-live-review`, original NVFP4 development worktree,
BF16 worktree, and all running snapshots remain read-only for this work. Leave
watchers, launchers, operational state, checkpoints, images, and live recipes
under their existing session's control. Do not launch GPU jobs or rebuild images
as part of branch setup.
