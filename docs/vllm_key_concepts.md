# vLLM Key Concepts

A guide to the core ideas behind vLLM's high-throughput LLM serving.

---

## Architecture Overview

```
User Request
    ↓
┌─────────────────────────────────────────────┐
│  Entrypoints                                │
│  (OpenAI API server, LLM class, gRPC)       │
│  vllm/entrypoints/                          │
└────────────────────┬────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│  Engine                                     │
│  Input processor → EngineCore → Output proc │
│  vllm/v1/engine/                            │
└────────────────────┬────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│  Scheduler                                  │
│  FCFS/Priority queue, preemption,           │
│  chunked prefill, KV block allocation       │
│  vllm/v1/core/sched/                        │
└────────────────────┬────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│  Executor                                   │
│  (uniproc / multiproc / Ray)                │
│  vllm/v1/executor/                          │
└────────────────────┬────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│  Worker + Model Runner                      │
│  GPU memory mgmt, CUDA graphs,              │
│  batch construction, KV connector           │
│  vllm/v1/worker/                            │
└────────────────────┬────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│  Model Implementation                       │
│  260+ architectures (Llama, Qwen, etc.)     │
│  Registry maps HF names → vLLM classes      │
│  vllm/model_executor/models/                │
└────────────────────┬────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│  Layers                                     │
│  Attention backends (FlashInfer, FlashAttn)  │
│  TP-aware linear, quantized layers          │
│  Sampler, structured output                 │
│  vllm/model_executor/layers/                │
└────────────────────┬────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│  CUDA/C++ Kernels                           │
│  PagedAttention, fused ops                  │
│  csrc/                                      │
└─────────────────────────────────────────────┘
```

**Cross-cutting concerns:**

```
Config           vllm/config/                    VllmConfig aggregates all configs
Distributed      vllm/distributed/               TP, PP, EP communication
KV Transfer      vllm/distributed/kv_transfer/   PD disaggregation (NIXL/UCX)
Multi-Modal      vllm/multimodal/                Vision/audio processing
LoRA             vllm/lora/                      Multi-adapter serving
Compilation      vllm/compilation/               torch.compile + CUDA graphs
```

**Where each concept lives in the architecture:**

| Concept | Architecture Layer |
|---|---|
| KV Cache, PagedAttention | Worker (block table) + Kernels (paged attention) |
| Continuous Batching, Chunked Prefill, Preemption | Scheduler |
| Speculative Decoding | Scheduler + Worker + Sampler |
| Quantization | Layers (quantized linear) |
| TP/PP/EP | Distributed + Executor |
| CUDA Graphs, torch.compile | Worker (model runner) + Compilation |
| PD Disaggregation | KV Transfer (NIXL → UCX) |
| Sampling, Structured Output | Layers (sampler) |
| Model Registry | Model Implementation |
| Attention Backends | Layers (attention) |
| Multi-Modal | Multi-Modal + Model Implementation |
| LoRA | LoRA + Layers |

**Why vLLM reimplements each model:**

vLLM does NOT use HuggingFace model code directly. Each of the 260+ models has its own reimplementation in `vllm/model_executor/models/`. They share the same **weights** (loaded from the same HuggingFace checkpoint), but the **code** is different — vLLM replaces standard PyTorch layers with optimized equivalents:

```
HuggingFace (original)              vLLM replacement
──────────────────────              ──────────────────
nn.Linear                    →      ColumnParallelLinear / RowParallelLinear (TP)
F.scaled_dot_product_attention →    PagedAttention via FlashInfer/FlashAttn backend
nn.RMSNorm                   →      Fused RMSNorm kernel
nn.Embedding                 →      VocabParallelEmbedding (TP)
```

This is why adding a new model to vLLM requires writing a new file — someone must translate the HuggingFace architecture into vLLM's optimized layer primitives. The layers are reusable building blocks, so most of the work is wiring them together to match the model's architecture.

---

Each concept below follows the same structure: **The Problem → The Solution → How It Works → The Constraint → The Tradeoff → Connection to Everything Else**.

---

## 1. KV Cache

### The Problem

In autoregressive generation, each new token must attend to all previous tokens. Naively, this means recomputing the Key (K) and Value (V) projections for every previous token at every step — O(n²) redundant work.

### The Solution

**KV cache** stores the K and V vectors from all previous tokens so each new step only computes K/V for the new token and appends it to the cache.

### How It Works

For a sequence "The cat sat on":

```
Step 1: compute K,V for "The"   → cache: [The]
Step 2: compute K,V for "cat"   → cache: [The, cat]         → attend to full cache
Step 3: compute K,V for "sat"   → cache: [The, cat, sat]    → attend to full cache
Step 4: compute K,V for "on"    → cache: [The, cat, sat, on] → attend to full cache
```

For a single layer and attention head, each token produces:
- **K vector**: `[head_dim]` (typically 128 floats)
- **V vector**: `[head_dim]` (typically 128 floats)

Full model cache shape:
```
K cache: [num_layers, num_kv_heads, seq_len, head_dim]
V cache: [num_layers, num_kv_heads, seq_len, head_dim]
```

The cache covers the **entire context** — both input tokens (prefill) and generated tokens (decode). For a request with 1000 input tokens generating 200 output tokens, the KV cache holds 1200 entries per layer per head.

### The Constraint

KV cache grows linearly with sequence length. Long-context models (128K+ tokens) are extremely memory-hungry. A single Llama 3 70B sequence at 1200 tokens costs ~314 MB in fp16. The cache must be held in GPU memory for the entire lifetime of the request — it can't be offloaded mid-generation without stalling.

### The Tradeoff

- **With KV cache**: O(n) compute per step (only new token), but O(n) memory that grows with context
- **Without KV cache**: O(1) memory, but O(n) redundant compute per step (recompute all previous tokens)

KV cache trades memory for compute. This is always worth it — the compute savings are massive — but the memory cost becomes the central problem that everything else in vLLM is designed to solve.

### Connection to Everything Else

KV cache is the foundation. Every other concept in vLLM exists because of the memory pressure it creates:
- **PagedAttention** solves how to manage KV cache memory efficiently
- **Prefix Caching** shares KV cache blocks across requests
- **Continuous Batching** dynamically allocates/frees KV cache as requests come and go
- **Quantization** can reduce KV cache size (FP8 KV cache)
- **Chunked Prefill** controls how quickly KV cache is populated

**vLLM code**: KV cache tensors are allocated and managed in `vllm/v1/worker/gpu_model_runner.py`.

---

## 2. PagedAttention

### The Problem

Traditional KV cache management pre-allocates a **contiguous** memory chunk per sequence for its maximum possible length. A request with max length 2048 reserves 2048 slots even if it only uses 100. This causes:
- **Internal fragmentation**: allocated but unused slots
- **External fragmentation**: free memory scattered in unusable small chunks
- **No sharing**: identical prefixes are stored redundantly

### The Solution

PagedAttention borrows the concept of **virtual memory paging** from operating systems. Instead of contiguous per-sequence allocation, KV cache is divided into fixed-size **blocks** (e.g., 16 tokens each), and a **block table** maps each sequence's logical blocks to physical blocks in GPU memory.

### How It Works

```
Sequence: "The cat sat on the mat" (6 tokens, block_size=4)

Block table (per sequence):
  logical block 0 → physical block 7   [The, cat, sat, on]
  logical block 1 → physical block 3   [the, mat, _, _]

GPU memory (physical blocks):
  block 0: [used by other seq]
  block 1: [free]
  block 2: [used by other seq]
  block 3: [the, mat, _, _]         ← this seq's logical block 1
  ...
  block 7: [The, cat, sat, on]      ← this seq's logical block 0
```

The block table uses a **dual-copy** (`CpuGpuBuffer`) pattern:

```
Scheduler (CPU)                      Attention Kernel (GPU)
  updates block_table.cpu   →copy→     reads block_table.gpu
  (numpy/torch pinned memory)          (torch tensor on device)
```

- **CPU side**: the scheduler updates mappings as requests arrive/finish
- **GPU side**: the attention kernel reads during execution
- Pinned memory enables fast CPU→GPU transfer (DMA)

The KV cache data itself (the actual K,V vectors) lives entirely on GPU. Only the small integer mapping table gets this dual-copy treatment. The attention kernel is modified to read K,V through block table indirection rather than from a contiguous tensor.

### The Constraint

- Block size is fixed — there's always some internal fragmentation in the last block of each sequence (up to `block_size - 1` wasted slots)
- The attention kernel must support indirect memory access through the block table, requiring custom CUDA kernels instead of standard implementations
- Block table indirection adds a small overhead per attention operation

### The Tradeoff

- **Pro**: Near-zero memory waste, on-demand allocation, enables sharing and copy-on-write
- **Con**: Slightly more complex attention kernel, small overhead from indirection, last-block fragmentation

The overhead is negligible compared to the memory savings. In practice, PagedAttention wastes <4% of KV cache memory versus up to 60-80% with contiguous allocation.

### Connection to Everything Else

PagedAttention is the enabling foundation for most other vLLM features:
- **Continuous Batching**: adding/removing requests is just updating block tables — no memory reshuffling
- **Prefix Caching**: shared prefixes point block tables to the same physical blocks — zero-copy sharing
- **Chunked Prefill**: partial prefills allocate blocks incrementally
- **CUDA Graphs**: block table data can be updated between graph replays without invalidating the graph

**vLLM code**: `vllm/v1/worker/block_table.py` (block table management), `vllm/model_executor/layers/attention/` (attention backends), `csrc/` (CUDA kernels).

**Paper**: [Kwon et al., 2023 — "Efficient Memory Management for Large Language Model Serving with PagedAttention"](https://arxiv.org/abs/2309.06180)

---

## 3. Continuous Batching

Also known as: **in-flight batching** (NVIDIA), **iteration-level batching** (Orca paper), **interleaved batching**.

### The Problem

Traditional **static batching** collects N requests, runs them as a batch, and waits until ALL finish before starting the next batch. If one request generates 10 tokens and another generates 500, the short one sits idle waiting.

```
Static batching:

Time →
Req A: [===]....................   ← done early, GPU idle
Req B: [========================]  ← everyone waits for this
Req C: [========]................   ← done early, GPU idle
         batch 1                     batch 2 can't start yet
```

### The Solution

After **each decode step**, the scheduler checks: did any request finish? Are new requests waiting? It immediately evicts completed requests and inserts new ones.

```
Continuous batching:

Time →
Req A: [===]
Req D:      [=======]            ← slotted in right after A finished
Req B: [========================]
Req C: [========]
Req E:           [===========]   ← slotted in right after C finished
```

### How It Works

Every iteration, the scheduler (`vllm/v1/core/sched/`) runs:

1. **Check finished requests** — hit EOS or max tokens → remove them, free their KV cache blocks
2. **Check waiting requests** — new requests in the queue → allocate KV blocks, add to batch
3. **Build the batch** — mix of prefill tokens (new requests) and decode tokens (ongoing requests)
4. **Execute** — one forward pass processes everything together

### The Constraint

- Scheduler runs on CPU every iteration — must be fast enough not to become a bottleneck
- Batch composition changes every step, so the model must handle variable batch sizes efficiently
- New requests require prefill (many tokens), while ongoing requests are decode (one token) — mixing these in one batch creates asymmetric compute

### The Tradeoff

- **Pro**: GPU is never idle waiting for a batch to drain. Much higher throughput and lower average latency.
- **Con**: Per-iteration scheduling overhead. Mixed prefill+decode batches can be less compute-efficient than uniform batches (addressed by Chunked Prefill).

### Connection to Everything Else

- **PagedAttention**: makes continuous batching practical — adding/removing requests is just a block table update, no memory reshuffling
- **Chunked Prefill**: prevents long prefills from blocking the batch, making continuous batching work well with long inputs
- **CUDA Graphs**: requires pre-captured graphs for different batch sizes since the batch changes every iteration
- **Speculative Decoding**: draft-then-verify loop integrates into the per-iteration scheduling

**vLLM code**: `vllm/v1/core/sched/` (scheduler), `vllm/v1/engine/core.py` (engine main loop).

---

## 4. Prefix Caching

### The Problem

Many requests share the same prefix. A chatbot with a 1000-token system prompt receiving 100 requests/second would recompute those same 1000 tokens 100 times per second — pure waste.

### The Solution

When multiple requests share the same prefix, vLLM **reuses the KV cache blocks** from the first request instead of recomputing them.

### How It Works

**Step 1 — First request arrives:**

```
Req A: "You are a helpful assistant." + "What is 2+2?"
        ↑ system prompt (1000 tokens)    ↑ user question (10 tokens)
```

vLLM processes all 1010 tokens. KV cache stored in blocks (block_size=16):

```
Physical GPU memory:
  block 5:  KV for tokens [0:16]
  block 12: KV for tokens [16:32]
  block 8:  KV for tokens [32:48]
  ...
  block 20: KV for tokens [992:1008]
  block 3:  KV for tokens [1008:1018]  ← "What is 2+2?"
```

vLLM **hashes** each block's token content and stores the mapping:

```
Hash table:
  hash(tokens[0:16])     = 0xAB3 → physical block 5
  hash(tokens[16:32])    = 0xF21 → physical block 12
  hash(tokens[32:48])    = 0x9C4 → physical block 8
  ...
  hash(tokens[992:1008]) = 0xD17 → physical block 20
```

**Step 2 — Second request arrives with same system prompt:**

```
Req B: "You are a helpful assistant." + "Tell me a joke"
        ↑ same system prompt              ↑ different question
```

Before computing, vLLM hashes Req B's blocks and looks them up:

```
tokens[0:16]      → hash 0xAB3 → FOUND! reuse block 5
tokens[16:32]     → hash 0xF21 → FOUND! reuse block 12
...
tokens[992:1008]  → hash 0xD17 → FOUND! reuse block 20
tokens[1008:1022] → hash 0xE55 → NOT FOUND → allocate new block, compute
```

**Step 3 — Block tables share physical blocks:**

```
Req A block table: logical 0→phys 5,  logical 1→phys 12, ..., logical 63→phys 3
Req B block table: logical 0→phys 5,  logical 1→phys 12, ..., logical 63→phys 9
                            ↑                    ↑
                      same physical blocks — no copy, no recompute
```

Req B skips computing KV for the entire 1000-token system prompt. Only the new question tokens are computed.

### The Constraint

- Prefix must align on **block boundaries**. If the shared prefix is 1000 tokens and block_size is 16, only the first 992 tokens (62 full blocks) can be cached. The remaining 8 tokens fall in a partial block that may differ between requests.
- Hash collisions are theoretically possible (though extremely unlikely). vLLM uses content-based hashing to ensure correctness.
- Cached blocks consume GPU memory even after the original request finishes. There's an eviction policy to free stale cached blocks under memory pressure.

### The Tradeoff

- **Pro**: Massive prefill savings for workloads with shared prefixes. A 1000-token system prompt computed once instead of 100× is a 99% compute reduction for that portion.
- **Con**: Cached blocks use GPU memory that could hold KV cache for active requests. The hash table adds bookkeeping overhead. Ineffective when requests don't share prefixes.

### Connection to Everything Else

- **PagedAttention**: enables prefix caching — two block tables point to the same physical blocks, zero-copy sharing (like shared memory pages in an OS)
- **Continuous Batching**: new requests with cached prefixes start faster, reducing their time in the queue
- **Chunked Prefill**: only the non-cached portion of the prefix needs chunked prefill, further reducing work
- **KV Cache**: prefix caching is fundamentally about avoiding redundant KV cache computation

**vLLM code**: `vllm/v1/core/` (KV cache manager and block allocator handle hash-based block lookup and sharing).

---

## 5. Chunked Prefill

### The Problem

When a new request arrives with a long prompt (say 10,000 tokens), the **prefill** step must compute KV for all those tokens in one forward pass. This is a big, expensive operation that **blocks** all ongoing decode requests.

```
Without chunked prefill:

Time →
Decode reqs: [=][=][=]                    [=][=][=][=]
New long req:          [======PREFILL======]
                       ↑                   ↑
                       decode is blocked for entire prefill
```

All the short decode steps (one token each) are stuck waiting for the long prefill to finish. This creates a **latency spike** for every in-flight request.

### The Solution

Split the long prefill into smaller **chunks** and interleave them with decode steps from other requests.

```
With chunked prefill (chunk_size = 2048):

Time →
Decode reqs: [=][=][=][=][=][=][=][=][=][=][=][=][=]
New long req:    [P1]  [P2]  [P3]  [P4]  [P5]
                 ↑ chunk1  ↑ chunk3  ↑ chunk5
                    ↑ chunk2  ↑ chunk4

Decode never stops — chunks are interleaved
```

### How It Works

1. Request with 10,000 tokens arrives
2. Scheduler says: "I'll process 2048 tokens of this prefill per iteration"
3. Iteration 1: compute KV for tokens [0:2048] + run decode for existing requests
4. Iteration 2: compute KV for tokens [2048:4096] + run decode for existing requests
5. ... repeat until all 10,000 tokens are prefilled
6. Request transitions to decode phase like normal

Each iteration's batch is a mix of:
- **Prefill tokens** from new/partially-prefilled requests
- **Decode tokens** (one per request) from ongoing requests

### The Constraint

- The partially-prefilled request must track how far it's gotten — resuming from token 2048 requires knowing that tokens [0:2048] are already in the KV cache
- Chunk size is a tuning parameter. Too small = more iterations to complete prefill, slower time-to-first-token for the new request. Too large = longer decode stalls for existing requests.
- Attention within a chunk can only attend to tokens in the current and previous chunks — the attention mask must handle this correctly

### The Tradeoff

- **Without chunking**: prefill is fastest for the new request, but all other requests stall
- **With chunking**: prefill takes slightly longer for the new request (spread across multiple iterations), but existing requests maintain consistent latency

This is especially important for **time-to-first-token (TTFT)** SLA guarantees — you don't want one user's long prompt to spike latency for everyone else.

### Connection to Everything Else

- **Continuous Batching**: chunked prefill is what makes continuous batching work well with long inputs. Without it, a single long prefill would monopolize an entire iteration.
- **PagedAttention**: KV blocks are allocated incrementally as each chunk is processed, not all at once
- **Prefix Caching**: only the non-cached suffix needs chunked prefill, further reducing work
- **CUDA Graphs**: prefill chunks are large enough that kernel launch overhead is negligible, so CUDA graphs aren't used for prefill

**vLLM code**: The scheduler in `vllm/v1/core/sched/` controls chunk sizes and decides how many prefill tokens to include in each iteration alongside decode tokens.

---

## 6. Speculative Decoding

Also known as: **speculative sampling**, **assisted generation**.

### The Problem

LLM decoding is **memory-bandwidth bound**, not compute bound. Each decode step generates just one token but must load the entire model's weights from GPU memory. A 70B model loads ~140GB of weights to produce a single token. The GPU's compute units are mostly idle — they're waiting on memory reads.

### The Solution

Use a small **draft model** (e.g., 1B params) to guess several tokens ahead, then have the large **target model** verify them all in one forward pass. This works because one forward pass costs roughly the same whether it's verifying 1 token or 5 — the bottleneck is loading the weights, which happens once regardless.

### How It Works

```
Traditional decoding (1 token per forward pass of large model):

Large model: [load weights → "The"] [load weights → "cat"] [load weights → "sat"] [load weights → "on"]
              70B forward pass        70B forward pass        70B forward pass        70B forward pass
Total: 4 forward passes of 70B model → 4 tokens

Speculative decoding:

Draft model: [guess "The cat sat on the"]     ← 5 tokens, cheap (1B model)
Large model: [verify all 5 at once]            ← 1 forward pass, checks all 5
Result: "The cat sat on" ✓✓✓✓ "the" ✗ → accept 4 tokens

Total: 1 forward pass of 70B + 5 of 1B → 4 tokens (but much faster)
```

Step by step:

1. **Draft phase**: Small model generates K candidate tokens autoregressively (fast, small model)
2. **Verify phase**: Large model runs one forward pass on the original context + K candidates
3. **Accept/reject**: Compare draft and target probability distributions at each position:
   - If they agree → **accept** the token
   - If they disagree → **reject** and resample from an adjusted distribution
   - All tokens after the first rejection are discarded
4. **Repeat** from the first rejected position

```
Draft generates: "The" "cat" "sat" "by" "the"
Target verifies:  ✓     ✓     ✓     ✗
                                    ↑ "by" rejected, target says "on"
Accept: "The cat sat on" (4 tokens from 1 large forward pass)
Discard: "the" (was after rejection point)
```

This is mathematically exact — the acceptance/rejection scheme guarantees the final output distribution is **identical** to what you'd get from the large model alone. No quality loss.

vLLM supports three draft methods:

1. **`draft_model`** — A separate smaller model (e.g., Llama-3-8B drafting for Llama-3-70B). Both loaded in GPU memory simultaneously.
2. **`ngram`** — No neural network. Looks at patterns in the prompt text to guess next tokens. Zero extra memory cost.
3. **`ngram_gpu`** — GPU-accelerated version of the ngram approach.

Example usage:
```bash
vllm serve meta-llama/Llama-3-70B \
    --speculative-model meta-llama/Llama-3-8B \
    --num-speculative-tokens 5
```

### The Constraint

- Draft model must be loaded alongside the target model — both consume GPU memory simultaneously (except ngram method)
- Accept rate depends heavily on how well the draft model approximates the target. Unpredictable text (creative writing) has low accept rates.
- All tokens after the first rejection are wasted computation
- The verify step must process K+1 tokens in one forward pass, which requires KV cache space for the speculative tokens

### The Tradeoff

| Scenario | Benefit |
|---|---|
| Interactive use, user waiting for streaming output | High — latency matters |
| Predictable text (code, structured output, translations) | High — draft model guesses well |
| Low concurrency, GPU compute underutilized | High — draft compute is "free" |
| High throughput serving, many concurrent requests | Low — memory better used for KV cache |
| Creative/unpredictable text | Low — draft model guesses poorly |
| Memory constrained | Low — two models eat VRAM |

**The fundamental tradeoff**: speculative decoding trades **throughput** (max requests/sec) for **latency** (faster response per request).

### Connection to Everything Else

- **KV Cache**: draft tokens temporarily consume KV cache slots; rejected tokens waste those slots
- **Continuous Batching**: the draft-then-verify loop integrates into per-iteration scheduling. The scheduler must account for speculative tokens when managing batch composition.
- **CUDA Graphs**: the verify step processes multiple tokens, requiring different graph captures than standard single-token decode
- **Quantization**: a quantized draft model uses even less memory, making the memory tradeoff more favorable

**vLLM code**: Configuration in `vllm/config/speculative.py`, implementation in `vllm/v1/spec_decode/`.

---

## 7. Quantization

### The Problem

Large models use massive amounts of GPU memory. A 70B parameter model in fp16 (2 bytes per param) needs ~140GB just for weights — more than a single GPU can hold. Even if it fits, loading all those bytes from GPU memory is the bottleneck during decoding (memory-bandwidth bound).

### The Solution

Store weights in **lower precision** formats. Fewer bits per weight = less memory = faster memory reads.

```
Format comparison for 70B model:

FP16  (16-bit): 70B × 2 bytes = 140 GB
INT8  (8-bit):  70B × 1 byte  =  70 GB
INT4  (4-bit):  70B × 0.5 byte = 35 GB
```

A 70B model that needs 2× A100-80GB in fp16 fits on a **single GPU** in INT4.

### How It Works

Map a range of float values to a smaller set of integer values:

```
FP16 weights:  [0.23, -0.87, 0.45, -0.12, 0.91, -0.33, 0.67, -0.55]

Quantize to INT4 (16 possible values):
  scale = (max - min) / 15 = (0.91 - (-0.87)) / 15 = 0.1187
  zero_point = -0.87

  0.23  → round((0.23 - (-0.87)) / 0.1187) = round(9.27) = 9
  -0.87 → 0
  0.45  → 11
  ...

INT4 weights:  [9, 0, 11, 6, 15, 5, 13, 3]
+ scale: 0.1187, zero_point: -0.87   ← stored per group of weights

Dequantize at inference:
  9 × 0.1187 + (-0.87) = 0.198  (was 0.23, close enough)
```

The `scale` and `zero_point` are stored alongside the quantized weights. During inference, weights are dequantized on-the-fly before matrix multiplication.

**Quantization granularity** — how many weights share one scale/zero_point:
- **Per-tensor**: one scale for the whole weight matrix — fastest, least accurate
- **Per-channel**: one scale per output channel — good balance
- **Per-group** (e.g., 128 weights): one scale per group — most accurate, more overhead

**Methods supported in vLLM:**

| Method | Bits | How it works |
|---|---|---|
| **FP8** | 8-bit float | Uses FP8 format (E4M3/E5M2), hardware-native on H100+ |
| **INT8** | 8-bit int | SmoothQuant-style, per-channel scaling |
| **GPTQ** | 4/3/2-bit | Calibration-based, minimizes layer-wise quantization error |
| **AWQ** | 4-bit | Identifies "salient" weights and protects them during quantization |
| **AutoRound** | 4-bit | Sign gradient descent-based quantization |
| **BitsAndBytes** | 4/8-bit | NF4/INT8 with double quantization |

**Weight-only vs Weight+Activation**:
- **Weight-only** (GPTQ, AWQ): only weights are quantized; activations stay in fp16. Saves memory, less speedup.
- **Weight+Activation** (FP8, INT8): both quantized. Saves memory AND speeds up matrix multiply (GPU has specialized int8/fp8 tensor cores).

Quantized checkpoints are typically **pre-quantized offline** (not by vLLM) and saved in quantized form. vLLM loads the INT4/FP8 weights directly and dequantizes on-the-fly during inference. Exception: FP8 can also be quantized online at startup.

### The Constraint

- Quantization is lossy — lower bits = less precise weights = some quality degradation
- Not all quantization methods work on all hardware (FP8 requires H100+, some INT4 kernels need specific GPU architectures)
- Quantized models must be pre-quantized using external tools (GPTQ, AWQ, AutoRound) before serving. vLLM loads them, it doesn't create them (except FP8 online quantization).
- Mixed precision interactions: activations, KV cache, and weights may each use different precisions, requiring careful dtype management

### The Tradeoff

```
                    Quality
FP16    ████████████████████  (baseline)
FP8     ███████████████████   (nearly identical)
INT8    ██████████████████    (very close)
AWQ-4   ████████████████      (slight degradation)
GPTQ-4  ███████████████       (slight degradation)
INT4    █████████████         (noticeable on hard tasks)

                    Speed / Memory savings →→→
```

FP8 on H100 is the sweet spot for most production use — nearly zero quality loss with significant speedup from hardware tensor core support.

### Connection to Everything Else

- **KV Cache**: KV cache can also be quantized (FP8 KV cache) to fit more concurrent sequences in memory
- **Tensor Parallelism**: quantization can reduce the need for TP — a 70B model in INT4 fits on one GPU instead of requiring TP=2
- **Speculative Decoding**: quantized draft models use even less memory, making the memory tradeoff more favorable
- **PagedAttention**: quantized KV cache means each block holds the same number of tokens but uses fewer bytes, allowing more blocks in the same GPU memory

**vLLM code**: `vllm/model_executor/layers/quantization/` (one file per method — `awq.py`, `gptq.py`, `fp8.py`, etc.). Weight loaders in `vllm/model_executor/model_loader/` handle detecting checkpoint format and loading quantized tensors.

---

## 8. Tensor/Pipeline/Expert Parallelism

### The Problem

Large models don't fit on a single GPU. A 70B model in fp16 needs ~140GB — more than any single GPU's memory. Even with quantization, the largest models (400B+) require multiple GPUs.

### The Solution

Split the model across multiple GPUs. There are four ways to split, each with different tradeoffs.

### How It Works

**Tensor Parallelism (TP)** — split each layer's weight matrices **horizontally** across GPUs. Every GPU runs every layer, but each handles a slice of the computation.

```
Single GPU:
  Input → [====== Linear (4096 × 4096) ======] → Output

TP=2 (split columns):
  GPU 0: Input → [=== Linear (4096 × 2048) ===] → partial output ─┐
                                                                     ├→ AllReduce → Output
  GPU 1: Input → [=== Linear (4096 × 2048) ===] → partial output ─┘
```

Each GPU holds half the weights, does half the math, then they **AllReduce** to combine results. This happens at every layer.

**Pipeline Parallelism (PP)** — split layers **sequentially** across GPUs. Each GPU runs a different subset of layers.

```
PP=4 (80 layers split across 4 GPUs):

GPU 0: layers  0-19  →─┐
GPU 1: layers 20-39  ←─┘ →─┐
GPU 2: layers 40-59  ←─────┘ →─┐
GPU 3: layers 60-79  ←────────┘ → Output
```

**Expert Parallelism (EP)** — specific to **Mixture-of-Experts (MoE)** models (Mixtral, DeepSeek-V2/V3). These models have a router that sends each token to only a few experts out of many.

```
MoE model:
  Token → Router → picks Expert 2 and Expert 7 out of 64
                    [Expert 2] ─┐
                    [Expert 7] ─┤→ combine → output
                    (other 62 experts not used for this token)

EP=4 (64 experts across 4 GPUs):
  GPU 0: experts  0-15
  GPU 1: experts 16-31
  GPU 2: experts 32-47
  GPU 3: experts 48-63

Token needs Expert 2 (GPU 0) and Expert 35 (GPU 2):
  → All-to-All: send token to GPU 0 and GPU 2
  → Each GPU runs its expert
  → All-to-All: send results back
```

**Data Parallelism (DP)** — not splitting a model — **replicate** the entire model on multiple GPUs, each handling different requests.

```
DP=2:
  GPU 0: [full model copy] ← handles requests A, B, C
  GPU 1: [full model copy] ← handles requests D, E, F
```

**TP + PP combined** for very large models across many GPUs:

```
8 GPUs, TP=4, PP=2:

         TP (fast NVLink within node)
         ──────────────────────────→
  PP     GPU0  GPU1  GPU2  GPU3        ← node 1
  │      GPU4  GPU5  GPU6  GPU7        ← node 2
  ↓      (slower cross-node network)
```

Rule of thumb: TP within a node (needs fast interconnect), PP across nodes (tolerates slower network).

### The Constraint

| Method | Communication | Requirement |
|---|---|---|
| **TP** | AllReduce every layer | Fast GPU-GPU interconnect (NVLink). Doesn't scale beyond 8 GPUs efficiently. |
| **PP** | Activations between stages only | Pipeline bubbles — downstream GPUs idle while waiting. Lower GPU utilization. |
| **EP** | All-to-All per MoE layer | Load imbalance if some experts are "hotter" than others. |
| **DP** | None during inference | Each GPU needs enough memory for the full model. |

### The Tradeoff

| Method | What's split | Pro | Con |
|---|---|---|---|
| **TP** | Each layer's weights | Low latency (all GPUs work simultaneously) | AllReduce at every layer, needs fast interconnect |
| **PP** | Layers sequentially | Minimal communication, works across nodes | Pipeline bubbles, lower utilization |
| **EP** | MoE experts | Only way to handle huge MoE models | All-to-All overhead, load imbalance |
| **DP** | Nothing (replicated) | Linear throughput scaling, zero communication | No memory saving per GPU |

### Connection to Everything Else

- **KV Cache**: in TP, each GPU holds a shard of the KV cache (fewer KV heads per GPU). In PP, each GPU only caches its own layers.
- **PagedAttention**: block tables are per-GPU in TP (each GPU manages its shard). In PP, each stage has independent block management.
- **Quantization**: reduces per-GPU memory, potentially allowing fewer GPUs (less TP/PP) or fitting larger models
- **Continuous Batching**: the scheduler coordinates across all GPUs — same batch composition on every device
- **CUDA Graphs**: each GPU captures its own graphs for its portion of the model

**vLLM code**: `vllm/distributed/` (parallel state, communication ops), `vllm/config/parallel.py` (configuration).

---

## 9. CUDA Graphs

### The Problem

During decode, each step generates just **one token per request**. The actual matrix math is tiny — but launching GPU kernels has overhead.

```
One decode step:
  launch kernel: layer norm           ~5μs launch overhead
  launch kernel: Q projection         ~5μs launch overhead
  launch kernel: K projection         ~5μs launch overhead
  launch kernel: V projection         ~5μs launch overhead
  launch kernel: attention            ~5μs launch overhead
  launch kernel: output projection    ~5μs launch overhead
  launch kernel: FFN gate             ~5μs launch overhead
  launch kernel: FFN up               ~5μs launch overhead
  launch kernel: FFN down             ~5μs launch overhead
  ... × 80 layers

  Total launch overhead: ~3600μs (3.6ms)
  Actual GPU compute:    ~2ms

  Launch overhead > actual work!
```

The CPU tells the GPU what to do one kernel at a time. Each launch has a fixed overhead (CPU-GPU communication, driver work). When the compute per kernel is small (decode = one token), this overhead dominates.

### The Solution

**Record** the entire sequence of kernel launches once, then **replay** it as a single unit — no per-kernel launch overhead.

```
Without CUDA Graph:
  CPU: launch K1 → wait → launch K2 → wait → launch K3 → wait → ...
  GPU:         [K1]              [K2]              [K3]
                    ↑ idle            ↑ idle

With CUDA Graph:
  CPU: replay graph (one call)
  GPU: [K1][K2][K3][K4][K5]...  ← back-to-back, no gaps
```

### How It Works

**Step 1 — Capture** (happens once at startup):

```python
# Allocate fixed-size input/output buffers
static_input = torch.empty(max_batch_size, hidden_dim, device="cuda")
static_output = torch.empty(max_batch_size, vocab_size, device="cuda")

# Record the graph
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    static_output = model.forward(static_input)  # all kernels recorded
```

**Step 2 — Replay** (every decode step):

```python
# Copy real data into the static buffer
static_input.copy_(real_input)

# Replay — executes all recorded kernels with zero launch overhead
graph.replay()

# Read results from static buffer
result = static_output
```

vLLM captures graphs for **multiple batch sizes** since the batch changes every iteration:

```
At startup, capture graphs for:
  batch_size=1   → graph_1
  batch_size=2   → graph_2
  batch_size=4   → graph_4
  batch_size=8   → graph_8
  ...
  batch_size=256 → graph_256

At runtime:
  Current batch has 5 requests
  → pad to 8, use graph_8
```

### The Constraint

CUDA graphs are **static** — the exact same kernels must execute in the exact same order:

- **Fixed tensor shapes**: can't change batch size or sequence length between captures
- **No control flow**: no if/else based on data, no dynamic shapes
- **Fixed memory addresses**: input/output tensors must stay at the same GPU addresses

This is why graphs are only used for decode (fixed: one token per request) and not prefill (variable: different prompt lengths).

### The Tradeoff

- **Pro**: Eliminates kernel launch overhead, can be 2-3× faster for small batch decode
- **Con**: Extra GPU memory for each captured graph (stores copies of all intermediate tensors). Capturing multiple batch sizes multiplies this cost. Startup time increases.

```
Memory cost:

  graph_1:   ~200MB (all intermediate buffers for batch=1)
  graph_4:   ~250MB
  graph_16:  ~400MB
  graph_64:  ~800MB
  graph_256: ~2GB

  Total: several GB of GPU memory reserved for graphs
```

### Connection to Everything Else

- **Continuous Batching**: batch size changes every iteration, so vLLM picks the right pre-captured graph each time
- **PagedAttention**: the attention kernel inside the graph reads through the block table, which can be updated between replays (data changes are fine, kernel sequence is fixed)
- **Chunked Prefill**: prefill chunks are large enough that kernel launch overhead is negligible — CUDA graphs aren't needed for prefill
- **Speculative Decoding**: verify step processes multiple tokens, needs different graph captures
- **Tensor Parallelism**: each GPU captures its own graphs for its portion of the model. AllReduce operations are part of the captured graph.

**vLLM code**: Graph capture and replay logic in `vllm/v1/worker/gpu_model_runner.py`.

---

## 10. PD Disaggregation

Also known as: **prefill-decode disaggregation**, **disaggregated serving**.

### The Problem

Prefill and decode have fundamentally different compute profiles:

```
Prefill: process 1000 tokens at once
  → compute-bound, high GPU utilization, wants large batch of tokens
  → benefits from high tensor parallelism

Decode: generate 1 token per request per step
  → memory-bandwidth-bound, low GPU utilization, wants many concurrent requests
  → benefits from large KV cache capacity
```

When both run on the same GPU, they interfere with each other:
- A long prefill **blocks** decode steps for all in-flight requests (even with chunked prefill, there's still interference)
- Prefill wants high TP for fast processing; decode wants more memory for KV cache
- You can't independently tune hardware/config for each phase

### The Solution

Run **two separate vLLM instances**: one dedicated to prefill (the "producer"), one dedicated to decode (the "consumer"). After the prefill instance computes the KV cache, it **transfers** the KV cache to the decode instance over the network.

```
                    KV cache transfer
                    (RDMA/network)
Prefill Instance ─────────────────────→ Decode Instance
(GPU pool A)                            (GPU pool B)
  ├─ optimized for throughput            ├─ optimized for latency
  ├─ high TP (e.g., TP=8)               ├─ lower TP (e.g., TP=2)
  ├─ processes prompts                   ├─ generates tokens
  └─ frees KV after transfer             └─ serves streaming output
```

### How It Works

1. **User sends request** to a proxy/router
2. **Router sends to prefill instance** — processes the full prompt, computes KV cache
3. **Prefill instance transfers KV cache** to decode instance over network (via NIXL/UCX)
4. **Decode instance receives KV cache** — starts generating tokens without recomputing prefill
5. **Decode instance streams tokens** back to the user

```
User → Proxy → Prefill Instance: "Summarize this 10K token document"
                    ↓
              [compute KV for all 10K tokens]
                    ↓
              [transfer KV blocks over RDMA]
                    ↓
         Decode Instance: receives KV cache
                    ↓
              [generate tokens using received KV]
                    ↓
         User ← streaming response
```

Configuration:
```bash
# Prefill instance (producer)
vllm serve <MODEL> --kv-transfer-config \
  '{"kv_connector":"NixlConnector", "kv_role":"kv_producer"}'

# Decode instance (consumer)
vllm serve <MODEL> --kv-transfer-config \
  '{"kv_connector":"NixlConnector", "kv_role":"kv_consumer"}'

# Or both roles in one instance:
  '{"kv_connector":"NixlConnector", "kv_role":"kv_both"}'
```

### The Constraint

- Both instances must load the **same model** — the KV cache is only compatible if model architecture, dtype, and attention config match
- KV cache transfer adds **network latency** — the decode instance can't start until the transfer completes
- Requires a **proxy/router** to coordinate requests between prefill and decode instances
- Both instances consume full GPU memory for their model weights — you need 2× the GPUs compared to a single instance
- The prefill and decode instances can have different TP sizes, but the connector must handle the mapping

### The Tradeoff

- **Pro**: Independent tuning — optimize TTFT (prefill) and ITL (inter-token latency, decode) separately. Eliminates prefill-decode interference. Scale each pool independently based on workload.
- **Con**: Doesn't improve overall throughput (same total compute). Adds network transfer latency. Doubles GPU requirement. More operational complexity (two services + proxy).

Best for: **latency-sensitive production deployments** where tail latency SLAs matter (e.g., chatbots where users notice decode stalls during other users' prefills).

### Connection to Everything Else

- **PagedAttention**: KV cache blocks are the unit of transfer — block IDs from the prefill instance map to block IDs on the decode instance
- **KV Cache**: the entire point is transferring KV cache between instances without recomputing it
- **Prefix Caching**: the decode instance can cache received KV blocks — subsequent requests with the same prefix skip both prefill AND transfer
- **Tensor Parallelism**: prefill and decode can use different TP sizes (e.g., TP=8 for prefill, TP=2 for decode), handled by the connector
- **Continuous Batching**: each instance runs its own scheduler independently

**vLLM code**: `vllm/config/kv_transfer.py` (configuration), `vllm/distributed/kv_transfer/` (implementation).

---

## 11. KV Connector

### The Problem

Transferring KV cache between disaggregated prefill and decode instances isn't just "send bytes over network." Several complications arise:
- Prefill and decode process requests in **different orders** — prefill might finish requests A, B, C in that order, but the decode instance needs C first
- Different transport backends exist (RDMA, NCCL, shared filesystem, CPU offload) — the engine shouldn't care which one is used
- KV transfer must integrate with vLLM's scheduler (to know when KV is available) AND the model runner (to transfer layer-by-layer during forward pass)

### The Solution

vLLM uses a **three-layer abstraction** for KV cache transfer:

```
Layer 3: KV Connector     ← integrates with vLLM engine (scheduler + worker)
Layer 2: KV Lookup Buffer  ← handles out-of-order requests (key-value store)
Layer 1: KV Pipe            ← raw tensor transport (FIFO)
```

### How It Works

**Layer 1 — KV Pipe** (raw transport):
```
send_tensor(tensor) ──→ network/RDMA/disk ──→ recv_tensor() → tensor
```
Simple FIFO: send a torch tensor, receive it on the other side. This is the transport layer.

**Layer 2 — KV Lookup Buffer** (reordering):
```
Prefill finishes:  A, B, C  (in this order)
Decode needs:      C first

Lookup buffer:
  insert(key=req_A, value=kv_cache_A)
  insert(key=req_B, value=kv_cache_B)
  insert(key=req_C, value=kv_cache_C)

  drop_select(key=req_C) → returns kv_cache_C  ← decode gets C first
  drop_select(key=req_A) → returns kv_cache_A
```
Translates FIFO into key-value lookup, solving the ordering mismatch.

**Layer 3 — KV Connector** (engine integration):

The connector has two halves — a **scheduler-side** and a **worker-side**:

```
Scheduler-side:                          Worker-side:
  get_num_new_matched_tokens()            start_load_kv()
    → "how many tokens available           → begin async transfer
       from remote?"
                                          wait_for_layer_load(layer_name)
  update_state_after_alloc()                → block until layer ready
    → "blocks allocated, update state"
                                          save_kv_layer(layer_name, kv_cache)
  build_connector_meta()                    → save KV after layer forward
    → "build metadata for this step"
                                          wait_for_save()
  request_finished()                        → block until saves complete
    → "request done, trigger transfer"
                                          get_finished()
                                            → which requests done transferring
```

The worker-side integrates with the model forward pass via a decorator:
```python
@maybe_transfer_kv_layer   # wraps each attention layer
def forward(self, ...):
    # 1. wait_for_layer_load() — block until this layer's KV arrives
    # 2. run attention
    # 3. save_kv_layer() — save this layer's KV for transfer
```

**Available connector implementations:**

| Connector | Transport | Use case |
|---|---|---|
| **NixlConnector** | UCX/RDMA | Production disaggregated serving |
| **P2pNcclConnector** | NCCL | Single-node P-D pairs |
| **LMCacheConnector** | LMCache service | External cache service |
| **OffloadingConnector** | CPU memory | KV cache offload to host RAM |
| **MultiConnector** | Multiple backends | Fallback chains |

### The Constraint

- The connector must handle **asynchronous** transfer — KV for different layers may arrive at different times, and the forward pass must wait per-layer
- Scheduler and worker run in **different processes** — the connector is split into two halves that communicate via metadata in the SchedulerOutput
- Compatibility checking is critical — if prefill and decode instances have different model configs, transferred KV cache is garbage. The connector validates this via compatibility hashes.

### The Tradeoff

- **Pro**: Clean abstraction — swap transport backends without changing engine code. Handles all the messy details (reordering, async, per-layer transfer).
- **Con**: Abstraction overhead. The three-layer design adds complexity. Each connector implementation must correctly implement both scheduler-side and worker-side logic.

### Connection to Everything Else

- **PD Disaggregation**: KV connector is the mechanism that makes disaggregation possible
- **PagedAttention**: transfers happen at the block level — block IDs are the transfer unit
- **Tensor Parallelism**: connectors handle heterogeneous TP (different TP sizes on P and D) by mapping blocks across TP ranks
- **Continuous Batching**: the scheduler queries the connector each step to know which requests have KV available from remote
- **CUDA Graphs**: KV transfer happens outside the CUDA graph — data is written to KV cache buffers between graph replays

**vLLM code**: `vllm/distributed/kv_transfer/kv_connector/v1/base.py` (base class), `vllm/distributed/kv_transfer/kv_connector/factory.py` (registry), `vllm/v1/worker/kv_connector_model_runner_mixin.py` (model runner integration).

---

## 12. NIXL

### The Problem

Transferring KV cache blocks between GPUs on different machines needs to be **fast**. A single request's KV cache can be hundreds of megabytes. If the transfer takes longer than the prefill computation itself, disaggregation is pointless. Standard TCP/IP adds too much latency (kernel copies, protocol overhead).

### The Solution

**NIXL (Network Interchange eXtension Library)** is a high-performance communication library that uses **RDMA (Remote Direct Memory Access)** to transfer data directly between GPU memories — bypassing the CPU and OS kernel entirely.

```
Traditional TCP transfer:
  GPU A → CPU A → kernel → NIC A → network → NIC B → kernel → CPU B → GPU B
  (6 copies, high latency)

NIXL/RDMA transfer:
  GPU A → NIC A → network → NIC B → GPU B
  (zero CPU involvement, low latency)
```

### How It Works

NIXL is the **primary production connector** in vLLM for disaggregated serving. It wraps UCX (the underlying transport) and provides a KV-cache-aware transfer API.

**Setup — Handshake phase** (happens once per P-D pair):

```
Prefill Instance                         Decode Instance
      ↓                                       ↓
  start ZMQ listener                    connect via ZMQ
      ↓                                       ↓
  [wait for handshake]  ←──── ZMQ ────→  [request metadata]
      ↓                                       ↓
  send compatibility hash               verify hash matches
      ↓                                       ↓
  send NIXL agent metadata               register remote agent
      ↓                                       ↓
  [NIXL agents connected]               [NIXL agents connected]
```

The compatibility hash ensures both instances have matching: vLLM version, model architecture, dtype, KV cache format, attention backend.

**Transfer — per request**:

```
Prefill Instance                         Decode Instance
      ↓                                       ↓
  [prefill complete]                     scheduler: get_num_new_matched_tokens()
      ↓                                       → "yes, KV available from remote"
  request_finished()                           ↓
      → returns kv_transfer_params       start_load_kv()
        (block IDs, engine ID, host)           ↓
                                         _read_blocks_for_req()
                                               ↓
  ┌─────────── RDMA read ────────────────┘
  │  [NIXL reads directly from prefill GPU memory]
  │  [block by block, async]
  └──→ KV data lands in decode GPU memory
                                               ↓
                                         wait_for_layer_load(layer_N)
                                               → blocks until layer N transferred
                                               ↓
                                         [forward pass uses received KV]
                                               ↓
                                         send_notif() → prefill can free blocks
```

**Key detail**: NIXL performs **RDMA reads** — the decode instance pulls data from the prefill instance's GPU memory. The prefill instance doesn't actively send; it just exposes its memory and waits for notification that the read is complete so it can free the blocks.

**Configuration:**
```bash
vllm serve <MODEL> --kv-transfer-config '{
  "kv_connector": "NixlConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "backends": ["UCX"],
    "num_threads": 4
  }
}'
```

### The Constraint

- Requires **RDMA-capable NICs** (InfiniBand, RoCE) for best performance. Falls back to TCP but loses the zero-copy benefit.
- NIXL must be installed separately (`pip install nixl` or build from source)
- Memory must be **registered** with NIXL before transfer — this is done at startup when KV cache is allocated. Registration is expensive but one-time.
- UCX has a known memory leak with its registration cache (`rcache`) — vLLM mitigates this by setting `UCX_RCACHE_MAX_UNRELEASED=1024`
- ZMQ side-channel port must be accessible between instances (default: 5600)

### The Tradeoff

- **Pro**: Near-hardware-speed transfers — RDMA can saturate 200Gbps+ InfiniBand links. Async transfers overlap with computation. Supports heterogeneous TP sizes between prefill and decode.
- **Con**: Requires RDMA hardware for full benefit. NIXL is an additional dependency. Complex setup (ZMQ handshake, memory registration, compatibility checking). The 2600+ line implementation reflects the real-world complexity of high-performance networking.

### Connection to Everything Else

- **KV Connector**: NIXL is the primary implementation of the KV Connector interface — it implements both scheduler-side and worker-side logic
- **PD Disaggregation**: NIXL is the transport that makes disaggregation practical at production scale
- **PagedAttention**: NIXL transfers individual KV cache blocks using block IDs — the paged block structure maps naturally to network transfers
- **Tensor Parallelism**: NIXL handles heterogeneous TP — if prefill uses TP=8 and decode uses TP=2, it correctly maps KV heads across ranks
- **UCX**: NIXL uses UCX as its underlying transport library (see next section)

**vLLM code**: `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py` (2600+ lines — NixlConnector, NixlConnectorScheduler, NixlConnectorWorker).

---

## 13. UCX

### The Problem

High-performance computing needs to transfer data between machines using whatever network hardware is available — InfiniBand, RoCE, shared memory, TCP. Each has different APIs, capabilities, and performance characteristics. Application code shouldn't have to handle all these backends.

### The Solution

**UCX (Unified Communication X)** is a communication framework that provides a **single API** for multiple transport backends. It automatically selects the best available transport.

```
Application (NIXL)
       ↓
      UCX API
       ↓
  ┌────┴─────────────────────┐
  │  Transport selection     │
  ├──────────────────────────┤
  │ InfiniBand │ RoCE │ TCP  │ Shared Memory │ ...
  └──────────────────────────┘
       ↓
  Network Hardware
```

### How It Works

UCX sits below NIXL in the stack. NIXL calls UCX; UCX talks to the hardware.

```
vLLM Engine
    ↓
KV Connector (scheduler + worker interface)
    ↓
NIXL (KV-cache-aware transfer, handshake, async ops)
    ↓
UCX (transport abstraction, memory registration, RDMA verbs)
    ↓
Hardware (InfiniBand NIC, Ethernet NIC, shared memory)
```

**What UCX provides to NIXL:**

1. **Memory registration** — registers GPU memory regions with the NIC so RDMA can access them directly
2. **Point-to-point RDMA** — zero-copy reads/writes between registered memory regions on different machines
3. **Transport negotiation** — automatically picks the fastest available transport (InfiniBand > RoCE > TCP)
4. **Multi-threading** — UCX handles concurrent transfers across multiple threads

**How vLLM configures UCX:**

```python
# NIXL initializes UCX with these defaults:
os.environ["UCX_RCACHE_MAX_UNRELEASED"] = "1024"  # prevent memory leak
nixl_agent_config(num_threads=4)                    # 4 transfer threads

# UCX environment variables (user-configurable):
UCX_TLS=all              # transport types (rc, ud, tcp, shm, ...)
UCX_NET_DEVICES=all       # which NICs to use
```

**Data flow for a single KV block transfer:**

```
Decode GPU                               Prefill GPU
    ↓                                         ↑
1. NIXL: prep_xfer_dlist()              [KV data in registered memory]
    ↓                                         ↑
2. UCX: post RDMA read                       ↑
    ↓                                         ↑
3. NIC: DMA read from remote NIC  ────→  NIC: DMA from GPU memory
    ↓
4. NIC: DMA write to local GPU memory
    ↓
5. NIXL: check completion
    ↓
[KV block now in decode GPU memory]
```

The CPU is not involved in step 3-4 — the NICs and GPU DMA engines handle the transfer autonomously.

### The Constraint

- **RDMA requires special hardware** — InfiniBand or RoCE-capable NICs. Without them, UCX falls back to TCP which is significantly slower.
- **Memory registration is expensive** — GPU memory must be registered with UCX before RDMA. This is a one-time cost at startup but can take seconds for large KV caches.
- **UCX rcache memory leak** — UCX's registration cache can exhaust NIC User Access Regions (UARs). vLLM mitigates with `UCX_RCACHE_MAX_UNRELEASED=1024`.
- **Thread limits** — too many UCX threads can also exhaust NIC UARs on Mellanox NICs. Default is 4 threads.

### The Tradeoff

- **Pro**: Hardware-speed transfers (200Gbps+ InfiniBand). Zero-copy — no CPU involvement. Automatic transport selection. Battle-tested in HPC (used by MPI implementations like OpenMPI).
- **Con**: Complex setup — RDMA hardware, drivers, firmware, network config all must be correct. Debugging transport issues is hard. Falls back to TCP gracefully but with major performance loss.

### Connection to Everything Else

- **NIXL**: UCX is the transport backend that NIXL uses. NIXL adds KV-cache-awareness, handshake, and async management on top of UCX's raw transfer capability.
- **PD Disaggregation**: UCX/RDMA is what makes the KV transfer fast enough for disaggregation to be practical. Without it, network latency would negate the benefits.
- **Tensor Parallelism**: vLLM also uses NCCL (which can use InfiniBand) for TP AllReduce. UCX is specifically for KV transfer, not model parallelism communication. They can share the same physical network.
- **CUDA Graphs**: UCX transfers happen outside CUDA graphs — they write to pre-registered GPU memory buffers that the attention kernel reads from.

**vLLM code**: UCX is configured inside `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py` (lines 102-128 for env vars, line 1022 for backend selection).

---

## 14. Sampling Strategies

### The Problem

After the model produces logits (a score for every token in the vocabulary), you need to pick which token to output. Different use cases need different selection strategies — deterministic code generation vs creative writing vs constrained output.

### The Solution

A multi-stage pipeline that transforms raw logits into a sampled token, with configurable controls at each stage.

### How It Works

The sampler runs a 9-phase pipeline on every decode step:

```
Raw logits [batch_size, vocab_size]
    ↓
1. Compute raw logprobs (if requested)
2. Convert to float32 (numerical stability)
3. Apply allowed_token_ids filter (whitelist)
4. Apply bad_words filter (blacklist)
5. Apply logit_bias and min_tokens (non-argmax-invariant processors)
6. Apply penalties:
   - repetition_penalty (multiplicative, discourages repeating prompt/output tokens)
   - frequency_penalty (subtractive, discourages frequent tokens)
   - presence_penalty (subtractive, discourages any seen tokens)
7. Core sampling:
   - If temperature ≈ 0 → argmax (greedy, deterministic)
   - Otherwise → temperature scaling → min_p → top_k → top_p → random sample
8. Gather logprobs (if requested)
9. Return sampled token IDs
```

**Key parameters explained:**

```
temperature = 0.0:  always pick highest-probability token (deterministic)
temperature = 0.7:  moderate randomness (good default for chat)
temperature = 1.5:  high randomness (creative writing)

top_k = 50:     only consider top 50 tokens, zero out the rest
top_p = 0.9:    only consider tokens whose cumulative probability reaches 90%
min_p = 0.05:   only consider tokens with probability ≥ 5% of the top token

These filters stack: top_k first, then top_p, then sample from what remains.
```

**Greedy shortcut:** If ALL requests in the batch use temperature=0, the sampler skips the entire random sampling path and does a single argmax — much faster.

### The Constraint

- Top-k/top-p filtering requires sorting the entire vocabulary (32K-128K tokens) — this is expensive. vLLM uses optimized FlashInfer or Triton kernels for this.
- Penalties require tracking all previously generated tokens per request — memory and compute overhead that scales with output length.
- Custom logits processors (user-provided Python functions) break CUDA graph capture since they involve arbitrary Python code.

### The Tradeoff

- **Greedy (temp=0)**: fastest, deterministic, but repetitive and boring for open-ended generation
- **Low temperature (0.3-0.7)**: good balance of quality and diversity
- **High temperature (>1.0)**: creative but can produce nonsense
- **Top-k/top-p**: prevents sampling rare garbage tokens, but adds compute for filtering

### Connection to Everything Else

- **OpenAI API**: parameters like `temperature`, `top_p`, `max_tokens` map directly to vLLM's `SamplingParams`
- **Speculative Decoding**: the acceptance/rejection step compares draft and target probability distributions — this is part of the sampling system (`rejection_sampler.py`)
- **Structured Output**: bitmask filtering happens before the sampling step — disallowed tokens get logits set to -inf
- **CUDA Graphs**: greedy sampling can be part of captured graphs; custom processors cannot

**vLLM code**: `vllm/v1/sample/sampler.py` (main pipeline), `vllm/v1/sample/ops/` (top-k/top-p kernels, penalties), `vllm/sampling_params.py` (parameter definitions).

---

## 15. Structured Output / Guided Decoding

### The Problem

You want the LLM to output valid JSON matching a specific schema, or follow a regex pattern, or conform to a grammar. Without constraints, the model might produce almost-valid output with a missing bracket or wrong field name — unusable by downstream code.

### The Solution

At each decode step, **before sampling**, restrict which tokens are allowed based on the current state of a **finite state machine (FSM)** that tracks the grammar/schema.

### How It Works

```
User request: "Output JSON matching this schema: {name: string, age: int}"

Step 1: Compile JSON schema → FSM (grammar)
Step 2: For each decode step:

  Model outputs logits: [0.5, -1.2, 0.8, ...]  (one per vocab token)
                              ↓
  Grammar FSM says: current state expects '{' or whitespace
  Bitmask: [0, 0, 0, ..., 1, ..., 0, 1, ..., 0]
           only '{' and ' ' are allowed (1), everything else forbidden (0)
                              ↓
  Apply bitmask: logits[forbidden] = -inf
                              ↓
  Normal sampling on remaining tokens → picks '{'
                              ↓
  Advance FSM: '{' consumed → new state expects '"' (for field name)
                              ↓
  Next step: only '"' is allowed → picks '"'
  ... continues until FSM reaches terminal state
```

**Supported constraint types:**

| Type | Example | Use case |
|---|---|---|
| `json_schema` | `{"type": "object", "properties": {...}}` | API responses |
| `json_object` | (any valid JSON) | Generic JSON |
| `regex` | `[0-9]{3}-[0-9]{4}` | Phone numbers, IDs |
| `grammar` | EBNF/Lark grammar | Custom languages |
| `choice` | `["yes", "no", "maybe"]` | Multiple choice |

**Backend implementations:**

| Backend | Library | Strengths |
|---|---|---|
| **XGrammar** | `xgrammar` | Fast FSM, GPU bitmask filling, rollback support |
| **Guidance** | `llguidance` | JSON schema handling, `additionalProperties` support |
| **Outlines** | `outlines` | Regex and grammar support |
| **LM-Format-Enforcer** | `lm-format-enforcer` | Lightweight alternative |

Selection:
```bash
vllm serve <MODEL> --guided-decoding-backend xgrammar  # default
```

### The Constraint

- Grammar compilation is expensive — compiling a JSON schema into an FSM can take 10-100ms. vLLM caches compiled grammars.
- The bitmask must be filled every decode step for every constrained request — for large batches, this is parallelized across threads.
- Speculative decoding + structured output requires **rollback** — if a speculative token is rejected, the FSM must undo that step. Only XGrammar supports this.
- The FSM is per-request state that runs on CPU — it can't be part of CUDA graphs.

### The Tradeoff

- **Pro**: Guarantees valid output format. Eliminates parsing failures and retry loops. Works with any model without fine-tuning.
- **Con**: Constraining the token space can degrade output quality — the model might want to say something that doesn't fit the schema. Adds per-token overhead for bitmask computation.

### Connection to Everything Else

- **Sampling**: bitmask is applied to logits before the sampling step — structured output is a logits filter
- **OpenAI API**: `response_format: {"type": "json_schema", "json_schema": {...}}` triggers guided decoding
- **Speculative Decoding**: needs FSM rollback support when draft tokens are rejected
- **CUDA Graphs**: FSM state machine runs on CPU, outside the graph

**vLLM code**: `vllm/v1/structured_output/` (manager, backend implementations), `vllm/config/structured_outputs.py` (configuration).

---

## 16. Model Registry

### The Problem

vLLM supports 260+ model architectures (Llama, Qwen, Mistral, DeepSeek, CLIP, Whisper, ...). When a user loads `meta-llama/Llama-3-70B` from HuggingFace, vLLM needs to find the right implementation class that knows how to run that model efficiently with PagedAttention, TP, quantization, etc.

### The Solution

A **registry** that maps HuggingFace model class names (from the model's `config.json`) to vLLM implementation classes.

### How It Works

```
User: LLM("meta-llama/Llama-3-70B")
          ↓
1. Download config.json from HuggingFace
   → architectures: ["LlamaForCausalLM"]
          ↓
2. Lookup in registry:
   "LlamaForCausalLM" → ("llama", "LlamaForCausalLM")
                          ↑ module    ↑ class name
          ↓
3. Import: from vllm.model_executor.models.llama import LlamaForCausalLM
          ↓
4. Instantiate with vLLM-optimized layers (PagedAttention, TP-aware linear, etc.)
```

The registry is a dictionary in `registry.py`:

```python
_TEXT_GENERATION_MODELS = {
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "MistralForCausalLM": ("llama", "LlamaForCausalLM"),  # shares impl!
    "DeepseekV3ForCausalLM": ("deepseek_v2", "DeepseekV3ForCausalLM"),
    # ... 160+ entries
}

_MULTIMODAL_MODELS = {
    "LlavaForConditionalGeneration": ("llava", "LlavaForConditionalGeneration"),
    # ... vision/audio models
}
```

**Lazy loading**: model info is cached to avoid importing heavy model classes until needed. A subprocess inspects model capabilities without initializing CUDA.

**Model capabilities** are detected via interfaces:
- `SupportsLoRA` — can serve multiple LoRA adapters
- `SupportsMultiModal` — can process vision/audio inputs
- `SupportsPP` — supports pipeline parallelism

### The Constraint

- The HuggingFace architecture name must match an entry in the registry — typos or unknown architectures fail
- vLLM model implementations are NOT the same as HuggingFace implementations — they use vLLM-specific layers (paged attention, TP-sharded linear). Adding a new model means writing a vLLM-specific implementation.
- Some models share implementations (e.g., Mistral uses the Llama implementation) — this can cause confusion when debugging

### The Tradeoff

- **Pro**: Clean mapping from any HuggingFace model to optimized vLLM execution. New models can be added by implementing one file + one registry entry.
- **Con**: Every new model architecture needs manual implementation in vLLM. Can lag behind HuggingFace releases.

### Connection to Everything Else

- **Quantization**: the model implementation must support quantized layers — the registry records which quantization methods each model supports
- **Tensor Parallelism**: model implementations use TP-aware layers (column/row parallel linear). The registry records `supports_pp` for pipeline parallelism.
- **LoRA**: model must implement `SupportsLoRA` interface and declare which layers support LoRA adapters
- **Multi-modal**: model must implement `SupportsMultiModal` and register a multi-modal processor

**vLLM code**: `vllm/model_executor/models/registry.py` (registry), `vllm/model_executor/models/` (260+ model implementations).

---

## 17. Multi-Modal Models

### The Problem

Modern LLMs handle not just text but also images, video, and audio. These non-text inputs can't be tokenized the same way as text — they need separate encoders (like CLIP for images) that produce embeddings, which then get inserted into the text token stream.

### The Solution

A multi-modal processing pipeline that converts raw media (images, video frames, audio waveforms) into embeddings, then substitutes them into the token sequence at placeholder positions.

### How It Works

```
User input: "Describe this image: <image>" + [PIL Image]

Processing pipeline:

1. Text tokenizer:
   "Describe this image: <image>"
   → [15496, 420, 2217, 25, <img_placeholder>, <img_placeholder>, ...]
                                ↑ placeholder tokens (e.g., 576 of them)

2. Image processor (HuggingFace):
   PIL Image → resize, normalize → [1, 3, 336, 336] pixel tensor

3. Vision encoder (CLIP/SigLIP):
   pixel tensor → vision transformer → [1, 576, 1024] image embeddings

4. Projection layer:
   [1, 576, 1024] → [1, 576, 4096]  (project to LLM hidden dim)

5. Embedding substitution:
   text embeddings: [E_Describe, E_this, E_image, E_:, placeholder, placeholder, ...]
                                                       ↓         ↓
   final embeddings: [E_Describe, E_this, E_image, E_:, img_emb_0, img_emb_1, ...]

6. LLM forward pass on combined embeddings → generate text
```

**Supported modalities:**
- **Image**: CLIP, SigLIP, InternViT encoders
- **Video**: extract frames → process as multiple images
- **Audio**: Whisper, audio encoders → audio embeddings

### The Constraint

- Vision encoders are expensive — encoding a single image can take 10-50ms, multiple images or video frames multiply this
- Image tokens consume context length — a single image may use 576+ tokens of the model's context window
- The scheduler must budget for encoder compute (`encoder_compute_budget`) to prevent vision encoding from starving text generation
- Different models use different placeholder token conventions and different numbers of image tokens per image

### The Tradeoff

- **Pro**: Single serving infrastructure handles text, image, video, audio. No separate vision service needed.
- **Con**: Multi-modal requests use more GPU compute (vision encoder) and more context length (image tokens). Mixed batches of text-only and multi-modal requests create scheduling asymmetry.

### Connection to Everything Else

- **KV Cache**: image tokens generate KV cache entries just like text tokens — 576 image tokens = 576 KV entries per layer
- **Prefix Caching**: shared image prefixes can be cached (same image across requests)
- **Chunked Prefill**: long multi-modal inputs benefit from chunking
- **Model Registry**: multi-modal models register via `_MULTIMODAL_MODELS` dict and implement `SupportsMultiModal`
- **LoRA**: multi-modal LoRA can fine-tune the vision encoder, projection layer, or LLM independently

**vLLM code**: `vllm/multimodal/` (registry, inputs, processing), model-specific implementations in `vllm/model_executor/models/` (e.g., `llava.py`, `qwen2_vl.py`).

---

## 18. LoRA Serving

### The Problem

Fine-tuning creates specialized model variants (customer support tone, medical terminology, code style). Each fine-tuned model is a full copy of the weights — serving 10 variants means 10× the GPU memory. Most fine-tuning only changes a small fraction of the model.

### The Solution

**LoRA (Low-Rank Adaptation)** stores only the small fine-tuning delta, not the full weights. vLLM serves multiple LoRA adapters simultaneously on a single base model.

### How It Works

LoRA decomposes the weight update into two small matrices:

```
Original weight:  W         [4096 × 4096]  = 67M params
LoRA update:      ΔW = A × B
                  A         [4096 × 16]    = 65K params
                  B         [16 × 4096]    = 65K params
                  Total:                     130K params (0.2% of original)

Forward pass:
  output = input × W  +  input × A × B
           ↑ base model    ↑ LoRA adapter (tiny)
```

**Multi-LoRA serving** — different requests use different adapters in the same batch:

```
Batch:
  Req A (adapter: "customer-support")  → uses LoRA weights A₁, B₁
  Req B (adapter: "medical")           → uses LoRA weights A₂, B₂
  Req C (no adapter)                   → base model only
  Req D (adapter: "customer-support")  → reuses LoRA weights A₁, B₁

All share the same base weights W. Only the small A,B matrices differ.
```

vLLM uses **Punica kernels** to batch-compute multiple LoRA adapters efficiently in a single kernel launch — no separate forward pass per adapter.

**Adapter management:**
```
Register:  load LoRA checkpoint → store A,B weights in GPU memory
Activate:  LRU cache of active adapters (limited by max_loras config)
Evict:     least-recently-used adapter removed when cache is full
```

### The Constraint

- Only certain layers support LoRA (linear projections: Q, K, V, O, gate, up, down). Not all layers in all models.
- All LoRA adapters must have the same rank to be batched together efficiently
- Max number of concurrent active adapters is limited (`max_loras` config) — LRU eviction when exceeded
- Adapter switching has latency — loading a cold adapter requires GPU memory copy

### The Tradeoff

- **Pro**: Serve hundreds of specialized models with memory cost of one base model + tiny per-adapter overhead. Hot-swap adapters per request.
- **Con**: Punica kernel overhead compared to native linear. Adapter quality limited by LoRA rank. Not all model layers can be adapted.

### Connection to Everything Else

- **Model Registry**: models must implement `SupportsLoRA` interface and declare which layers support LoRA
- **Tensor Parallelism**: LoRA A,B matrices are sliced across TP ranks, just like base weights
- **Quantization**: base model can be quantized (INT4/FP8) while LoRA weights stay in fp16 — best of both worlds
- **Continuous Batching**: different requests in the same batch can use different adapters — the Punica kernel handles this
- **Multi-Modal**: LoRA can be applied to vision encoder, projection layer, or LLM independently

**vLLM code**: `vllm/lora/` (model manager, LoRA model loading, layer implementations, Punica kernels).

---

## 19. Scheduler Internals

### The Problem

Every decode iteration, the scheduler must answer: which requests get to run? How many tokens does each get? What happens when GPU memory is full? This decision directly impacts throughput, latency, and fairness.

### The Solution

A **token-level, request-centric scheduler** that makes per-iteration decisions about which requests to prefill, decode, or preempt.

### How It Works

Each `schedule()` call processes two queues:

```
RUNNING queue                        WAITING queue
(requests already generating)        (new requests not yet started)
          ↓                                    ↓
1. For each running request:          2. For each waiting request:
   - How many new tokens needed?         - Any prefix cache hits?
   - Allocate KV cache blocks            - Any remote KV available? (PD disagg)
   - If allocation fails → preempt       - Allocate KV cache blocks
                                         - Apply chunked prefill limits
                                         - Move to RUNNING
          ↓                                    ↓
3. Build SchedulerOutput:
   - scheduled_new_reqs (from WAITING)
   - scheduled_cached_reqs (from RUNNING)
   - num_scheduled_tokens per request
   - preempted_req_ids
```

**Key insight**: there are no separate "prefill phase" and "decode phase." Each request simply tracks `num_computed_tokens` vs `num_total_tokens`. The scheduler assigns however many tokens are needed to catch up — whether that's 1 (decode), 2048 (chunked prefill), or 0 (waiting for KV transfer).

**Preemption** — when KV cache is full:

```
Running requests: A(priority=1), B(priority=2), C(priority=3)
New request D(priority=0) arrives, no free blocks

Scheduler preempts C (lowest priority):
  1. Free C's KV cache blocks
  2. Reset C.num_computed_tokens = 0  (will re-prefill later)
  3. Move C back to WAITING queue
  4. Allocate freed blocks for D
```

**Scheduling policies:**
- **FCFS** (First-Come-First-Served): fair, simple, default
- **Priority**: min-heap ordered by `(priority, arrival_time)`, supports priority-based preemption

**Budget limits enforced per iteration:**
- `max_num_running_reqs` — max concurrent requests
- `max_num_scheduled_tokens` — total token budget per step
- `long_prefill_token_threshold` — chunked prefill limit
- `encoder_compute_budget` — multimodal encoder tokens

### The Constraint

- Scheduler runs on CPU, single-threaded, every iteration — must be fast (microseconds)
- Preemption is expensive: the preempted request loses all computed KV cache and must re-prefill from scratch
- With PD disaggregation, the scheduler must coordinate with the KV connector to know when remote KV is available — async and potentially delayed

### The Tradeoff

- **FCFS**: fair but can starve short requests behind long ones
- **Priority**: better latency control but unfair — low-priority requests may never run under load
- **Aggressive preemption**: frees memory for new requests but wastes compute on re-prefill
- **Conservative preemption**: avoids waste but new requests wait longer

### Connection to Everything Else

- **Continuous Batching**: the scheduler IS the continuous batching mechanism — it adds/removes requests every iteration
- **PagedAttention**: the scheduler calls the KV cache manager to allocate/free blocks
- **Chunked Prefill**: the scheduler decides chunk size per iteration
- **Prefix Caching**: the scheduler checks for cache hits before allocating blocks
- **KV Connector**: the scheduler queries the connector for remote KV availability
- **Speculative Decoding**: the scheduler accounts for speculative tokens in the token budget

**vLLM code**: `vllm/v1/core/sched/scheduler.py` (main scheduler), `vllm/v1/core/sched/request_queue.py` (FCFS and priority queues).

---

## 20. torch.compile Integration

### The Problem

Beyond kernel launch overhead (solved by CUDA graphs), individual kernels themselves can be suboptimal. For example, a RMSNorm followed by a quantization step launches two separate kernels — but they could be fused into one kernel that reads/writes GPU memory once instead of twice.

### The Solution

Use **torch.compile** with a custom vLLM backend to fuse operations into optimized Triton kernels, then wrap the result in CUDA graphs for launch overhead elimination.

### How It Works

vLLM uses a **piecewise** compilation strategy:

```
Model forward pass:

  [RMSNorm + Quantize]  ← compiled & fused by torch.compile
  [Attention]            ← kept in eager mode (already optimized by FlashAttention)
  [RMSNorm + Quantize]  ← compiled & fused
  [FFN gate + activate]  ← compiled & fused
  [FFN down]             ← compiled & fused

Then wrapped in CUDA graph:
  CUDA Graph captures: [fused_kernel_1][attention][fused_kernel_2][fused_kernel_3]
```

**Why piecewise**: attention already uses highly optimized kernels (FlashAttention/FlashInfer). Compiling them through torch.compile would produce worse code. So vLLM excludes attention from compilation and only compiles the surrounding ops.

**Compilation modes:**
```python
NONE = 0                 # No compilation (eager PyTorch)
STOCK_TORCH_COMPILE = 1  # Standard torch.compile
DYNAMO_TRACE_ONCE = 2    # Single trace, guards dropped
VLLM_COMPILE = 3         # Custom vLLM Inductor backend (default)
```

**Fusion passes enabled by default:**
- `fuse_norm_quant` — RMSNorm + quantization → one kernel
- `fuse_act_quant` — SiLU activation + quantization → one kernel
- `fuse_attn_quant` — attention output + quantization → one kernel
- `fuse_gemm_comms` — overlap GEMM with TP AllReduce communication
- `eliminate_noops` — remove redundant operations

**Critical trick**: vLLM drops all Dynamo guards so compilation happens exactly once, regardless of dynamic batch sizes. This avoids recompilation overhead.

### The Constraint

- Compilation happens at startup — adds 30-120 seconds to model loading time
- Not all operations are compilable — custom CUDA extensions, certain control flow patterns cause "graph breaks"
- Compiled code is less debuggable — errors show Triton IR instead of Python stack traces
- Guards are dropped for performance, meaning the compiled code must handle all possible tensor shapes — this requires careful implementation

### The Tradeoff

- **Pro**: 10-30% speedup from kernel fusion. Free once compiled — no per-request overhead. Stacks with CUDA graphs for maximum performance.
- **Con**: Long startup time. Debugging difficulty. Some operations can't be compiled. Complexity in the compilation pipeline.

### Connection to Everything Else

- **CUDA Graphs**: torch.compile and CUDA graphs work together — compile fuses kernels, graphs eliminate launch overhead. The piecewise strategy means attention is outside both.
- **Quantization**: fusion passes specifically target quantization operations (norm+quant, act+quant) — this is where most of the speedup comes from for quantized models
- **Tensor Parallelism**: `fuse_gemm_comms` overlaps matrix multiply with AllReduce communication — hiding TP overhead
- **Attention Backends**: attention is explicitly excluded from compilation — it uses its own optimized kernels

**vLLM code**: `vllm/config/compilation.py` (configuration, fusion passes), `vllm/compilation/wrapper.py` (torch.compile wrapper).

---

## 21. Attention Backends

### The Problem

The attention operation (`softmax(Q × K^T / √d) × V`) with PagedAttention (block table indirection) is the most performance-critical kernel in the system. Different GPU architectures and model configurations benefit from different kernel implementations.

### The Solution

A pluggable **attention backend** system where different optimized implementations can be selected based on hardware, model type, and feature requirements.

### How It Works

Available backends:

| Backend | Hardware | Key Feature |
|---|---|---|
| **FlashInfer** | NVIDIA (SM 7.5 - 12.1) | Default on CUDA. FP8 KV cache, cascade attention. |
| **FlashAttention** | NVIDIA (SM 8.0+) | Widely used, stable. Good for standard attention. |
| **Triton Attention** | NVIDIA | Pure Triton, no CUDA dependency. |
| **ROCm Attention** | AMD GPUs | AMD-optimized kernels. |
| **CPU Attention** | x86/ARM CPUs | CPU fallback. |
| **FlashInfer MLA** | NVIDIA | Multi-head Latent Attention (DeepSeek). |
| **FlashAttention MLA** | NVIDIA | MLA via FlashAttention kernels. |
| **Triton MLA** | NVIDIA | MLA via Triton kernels. |

**Backend selection** happens automatically based on:
1. Hardware (CUDA capability, AMD, CPU)
2. Model features (MLA, sparse attention, sink tokens)
3. Data types (fp16, bf16, fp8 KV cache)
4. User override (`--attention-backend` flag)

```python
# Automatic selection (simplified):
if use_mla:
    return FlashInferMLABackend  # or FlashAttnMLABackend
elif platform.is_cuda():
    return FlashInferBackend     # default for NVIDIA
elif platform.is_rocm():
    return RocmAttentionBackend
elif platform.is_cpu():
    return CPUAttentionBackend
```

**KV cache layout differs per backend:**

```
FlashAttention: (2, num_blocks, block_size, num_kv_heads, head_size)
FlashInfer:     (num_blocks, 2, block_size, num_kv_heads, head_size)
                 ↑ different memory layout!
```

This matters for NIXL KV transfer — the connector must know which layout each side uses.

### The Constraint

- Not all backends support all features. FP8 KV cache is only supported by FlashInfer (not FlashAttention). Sink tokens only work with certain backends.
- Backend selection is cached for the lifetime of the process — can't switch mid-inference
- Different backends have different supported block sizes (e.g., FlashInfer: 16, 32, 64)
- KV cache layout is determined by the backend — all components (scheduler, model runner, KV transfer) must agree

### The Tradeoff

- **FlashInfer**: most features (FP8, cascade, sink), good performance, wider GPU support (SM 7.5+)
- **FlashAttention**: very stable, well-tested, but fewer features (no FP8 KV cache)
- **Triton**: no external dependency, but generally slower than specialized CUDA kernels
- **MLA backends**: required for DeepSeek-style models, not applicable to standard attention

### Connection to Everything Else

- **PagedAttention**: all backends implement paged attention — they read KV through the block table
- **KV Cache**: the backend determines KV cache memory layout
- **CUDA Graphs**: attention is typically excluded from CUDA graph capture (piecewise mode) — it runs as its own optimized kernel
- **torch.compile**: attention is excluded from compilation — already optimized
- **KV Transfer (NIXL)**: the compatibility hash includes attention backend — P and D must use the same layout
- **Quantization**: FP8 KV cache requires a backend that supports it (FlashInfer)

**vLLM code**: `vllm/v1/attention/backends/` (implementations), `vllm/v1/attention/selector.py` (selection logic), `vllm/v1/attention/backend.py` (abstract interface).
