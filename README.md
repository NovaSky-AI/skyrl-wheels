# SkyRL prebuilt CUDA extension wheels — torch 2.13 / CUDA 13.0

Stopgap wheels so [SkyRL](https://github.com/NovaSky-AI/SkyRL) can move to **torch 2.13**.

## Why

SkyRL pins several CUDA extensions from [Astral's GPU wheel index](https://wheels.astral.sh/simple/cu130/),
which tags each build with the exact torch it was compiled against
(`+cu.13.0.torch.2.11`). As of 2026-09-03 that index has **no torch 2.13 builds** —
`flash-attn`, `causal-conv1d` and `transformer-engine-torch` stop at `torch.2.12`, and
`mamba-ssm` at `torch.2.11`. So `uv lock` cannot resolve a torch 2.13 environment.

torch 2.13 is wanted because vLLM 0.28's vendored DeepGEMM extension is **not**
stable-ABI and is linked against torch 2.13's `c10`; under torch 2.11 it fails with
`undefined symbol: c10::ValueError(c10::SourceLocation, std::string)`, disabling the
DeepGEMM fused-MoE and sparse-attention-indexer paths.

**These wheels are temporary. Delete them and go back to the Astral index as soon as it
publishes `torch.2.13` builds.**

## Build environment

| | |
|---|---|
| torch | 2.13.0+cu130 |
| CUDA | 13.0.88 (`cuda-toolkit-13-0`, 13.0.3-1) |
| Python | 3.12 (cp312 only) |
| `TORCH_CUDA_ARCH_LIST` | `9.0;10.0` — **H100 and B200 only** |
| OS | Ubuntu 24.04, x86_64, glibc 2.39 |

### Note on architecture coverage

Astral's wheels cover many more architectures. These cover **sm_90 (H100) and sm_100
(B200) only**, to keep build times sane. On any other GPU they will fail at kernel
launch. That is the main reason not to treat these as a general replacement.

`flash-attn` is built at exactly **2.8.3**, not `2.8.3.post1`: Transformer Engine gates
flash-attn on `max_version = 2.8.3`, and the `.post1` compares greater.

## Verification

Each wheel was installed into a clean torch 2.13.0+cu130 venv and exercised on a B200 —
import of the compiled extension plus a forward pass returning finite values.

## Reproducing

See `build_all.sh` in this repo.
