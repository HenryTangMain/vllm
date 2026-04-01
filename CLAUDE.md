# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

## Architecture Overview

vLLM is a high-throughput LLM inference and serving engine. The system follows a layered architecture:

**Entrypoints → Engine → Scheduler → Worker/Executor → Model Runner → CUDA Kernels**

- **Entrypoints** (`vllm/entrypoints/`): User-facing APIs. `LLM` class (`llm.py`) for offline inference, OpenAI-compatible server (`openai/`), gRPC, and specialized endpoints (Anthropic, SageMaker, MCP).
- **Engine** (`vllm/v1/engine/`): `EngineCore` (`core.py`) runs the main loop with scheduling. `LLMEngine` wraps it. `core_client.py` enables multi-process/async access. Input/output processors convert between user formats and internal representations. (`vllm/engine/` is a legacy shim that aliases to `vllm/v1/`.)
- **Scheduler** (`vllm/v1/core/sched/`): Manages request batching, KV cache allocation, prefix caching, chunked prefill, and speculative decoding scheduling.
- **Executor** (`vllm/v1/executor/`): Abstraction over execution backends — `uniproc_executor.py` (single process), `multiproc_executor.py` (multi-process), `ray_executor.py` (Ray distributed).
- **Worker/Model Runner** (`vllm/v1/worker/`): `gpu_worker.py` and `gpu_model_runner.py` handle GPU execution. `gpu_input_batch.py` manages request batching. `block_table.py` manages KV cache blocks.
- **Model Implementations** (`vllm/model_executor/models/`): 260+ model architectures. `registry.py` maps HuggingFace model class names to vLLM implementations. Weight loaders in `model_loader/` support various formats (safetensors, GGUF, etc.).
- **Layers** (`vllm/model_executor/layers/`): Reusable building blocks — attention backends, linear layers, quantization methods (AWQ, GPTQ, FP8, INT4/INT8), normalization, and activation functions.
- **C++/CUDA Kernels** (`csrc/`): Custom high-performance kernels compiled via CMake. `setup.py` handles platform detection (CUDA/ROCm/XPU/CPU) and compilation.

### Parallelism

Supports Tensor Parallel (TP), Pipeline Parallel (PP), Data Parallel (DP), Expert Parallel (EP), and Context Parallel (CP). Distributed communication is in `vllm/distributed/` with `parallel_state.py` managing multi-rank coordination.

### Configuration

`vllm/config/` has per-concern config classes (`ModelConfig`, `CacheConfig`, `ParallelConfig`, `SchedulerConfig`, `SpeculativeConfig`, etc.). `VllmConfig` (`vllm.py`) is the master config aggregating all others. `vllm/envs.py` manages all environment variables.

## Testing

- **CI**: Buildkite-based. Config in `.buildkite/ci_config.yaml` with test area definitions in `.buildkite/test_areas/`.
- **Pytest markers**: `slow_test`, `core_model`, `hybrid_model`, `cpu_model`, `cpu_test`, `distributed`, `optional`
- **Key fixtures** in `tests/conftest.py`: `hf_runner`, `vllm_runner` (model comparison), `image_assets`/`video_assets`/`audio_assets` (multimodal), `example_prompts`, `dist_init` (distributed)
- **Custom pre-commit validators** in `tools/pre_commit/`: SPDX headers, forbidden imports (no direct `torch.cuda` calls), config field validation, lazy import checks
- **Trigger patterns**: Changes to `csrc/`, `cmake/`, `requirements/`, `setup.py`, or Dockerfiles trigger full CI rebuilds

## PR Guidelines

PR titles must be prefixed with one of: `[Bugfix]`, `[CI/Build]`, `[Doc]`, `[Model]`, `[Frontend]`, `[Kernel]`, `[Core]`, `[Hardware][Vendor]`, or `[Misc]`. Use multiple prefixes if the PR spans categories.

All commits require a DCO sign-off: use `git commit -s` (adds `Signed-off-by:` trailer automatically).

Major architectural changes (>500 LOC excluding kernels/data/config/tests) require a GitHub RFC issue before a PR is opened.

## Kernel Development (`csrc/`)

- Custom ops must be registered following PyTorch's [Custom C++ and CUDA Operators](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html) guidelines.
- Ops that return `Tensors` require meta-functions registered in Python for dynamic dim handling.
- Use `torch.library.opcheck()` to validate op registration and meta-functions. See `tests/kernels/` for examples.
- When changing a C++ op signature, update the schema to match.
- For iterating on kernels, use the [Incremental Compilation Workflow](docs/contributing/incremental_build.md).
