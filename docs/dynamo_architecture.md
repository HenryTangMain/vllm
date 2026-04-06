# NVIDIA Dynamo + vLLM Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DYNAMO PLATFORM                                       │
│              etcd (service discovery) · NATS (event/request plane)              │
│              Dynamo Operator (K8s CRD) · Planner (autoscaler)                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Request Flow (Aggregated Mode — your current deployment)

```
  USER / APPLICATION
  POST /v1/chat/completions
  {"model":"Qwen3-Coder-Next", "messages":[...]}
         │
         │ HTTP (NodePort 30866)
         ▼
╔══════════════════════════════════════════════════════════╗
║  FRONTEND  (dynamo.frontend)                             ║
║  ┌────────────────────┐  ┌─────────────────────────┐    ║
║  │   Preprocessing    │  │     KV Smart Router      │    ║
║  │  - Tokenize prompt │  │  - Radix tree indexer    │    ║
║  │  - Jinja template  │  │  - Score each worker:    │    ║
║  │  - Sampling params │  │    cost = w × prefill_blks│   ║
║  └────────────────────┘  │          + decode_blks   │    ║
║                          │  - Route to min-cost     │    ║
║                          │  - DYN_ROUTER_MODE=kv    │    ║
║                          └─────────────────────────┘    ║
╚══════════════════════════════════════════════════════════╝
         │
         │ Request Plane (TCP)  — DYN_REQUEST_PLANE=tcp
         ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  WORKER POOL  (4× VllmDecodeWorker, 1 B200 GPU each)                        ║
║                                                                              ║
║  ┌──────────────────────────┐  ┌──────────────────────────┐                 ║
║  │  Worker 0  (GPU 0)       │  │  Worker 1  (GPU 1)       │  · · ·          ║
║  │ ┌──────────────────────┐ │  │ ┌──────────────────────┐ │                 ║
║  │ │   dynamo.vllm        │ │  │ │   dynamo.vllm        │ │                 ║
║  │ │  ┌────────────────┐  │ │  │ │  ┌────────────────┐  │ │                 ║
║  │ │  │  vLLM Engine   │  │ │  │ │  │  vLLM Engine   │  │ │                 ║
║  │ │  │ (v1 AsyncLLM)  │  │ │  │ │  │ (v1 AsyncLLM)  │  │ │                 ║
║  │ │  │                │  │ │  │ │  │                │  │ │                 ║
║  │ │  │ • Scheduler    │  │ │  │ │  │ • Scheduler    │  │ │                 ║
║  │ │  │ • KV Cache Mgr │  │ │  │ │  │ • KV Cache Mgr │  │ │                 ║
║  │ │  │ • GPU Runner   │  │ │  │ │  │ • GPU Runner   │  │ │                 ║
║  │ │  │ • Block Pool   │  │ │  │ │  │ • Block Pool   │  │ │                 ║
║  │ │  └──────┬─────────┘  │ │  │ │  └──────┬─────────┘  │ │                 ║
║  │ │         │ ZMQ events │ │  │ │         │ ZMQ events │ │                 ║
║  │ │  KvEventPublisher    │ │  │ │  KvEventPublisher    │ │                 ║
║  │ └──────────────────────┘ │  │ └──────────────────────┘ │                 ║
║  └──────────────────────────┘  └──────────────────────────┘                 ║
║                                                                              ║
║  model: Qwen3-Coder-Next  ·  TP=1  ·  gpu-memory-utilization=0.95          ║
╚══════════════════════════════════════════════════════════════════════════════╝
         │                          │
         │ Token Stream (SSE)       │ KV Cache Events (ZMQ → NATS)
         ▼                          ▼
  USER gets response        ╔══════════════════════════╗
  (streamed token by token) ║   EVENT PLANE (async)    ║
                            ║  ZMQ tcp://*:20080        ║
                            ║       → NATS              ║
                            ║       → Router Indexer    ║
                            ║                           ║
                            ║  BlockStored(hash, tokens)║
                            ║  BlockRemoved(hash)       ║
                            ║  AllBlocksCleared()       ║
                            ╚══════════════════════════╝
                                        │
                                        │ updates radix tree
                                        ▼
                                  KV Router Indexer
                                  (next request routing)
```

---

## KV Routing Decision (detail)

```
New request arrives with tokens [t1, t2, t3 ... tN]
         │
         ▼
  Compute block hashes
  hash_A = hash([t1..t16])
  hash_B = hash([t17..t32])
  ...
         │
         ▼
  Query radix tree for each worker:

  Worker 0: cached = {hash_A ✓, hash_B ✓, hash_C ✗}  → overlap = 2 blocks
  Worker 1: cached = {hash_A ✓, hash_B ✗}             → overlap = 1 block
  Worker 2: cached = {}                                → overlap = 0 blocks
  Worker 3: cached = {hash_A ✓, hash_B ✓, hash_C ✓}  → overlap = 3 blocks ← WINNER
         │
         ▼
  Score each worker:
    cost = overlap_score_weight × prefill_blocks + decode_blocks
    (lower = better; default overlap_score_weight = 1.0)
         │
         ▼
  Route request to Worker 3
  → tokens hash_A, hash_B, hash_C already cached → skip recomputation
  → only new tokens need prefill
```

---

## KV Cache Memory (per B200 GPU)

```
  B200 HBM: 192 GB
  ├── gpu-memory-utilization × total  =  0.95 × 192  =  182 GB  (vLLM budget)
  │
  ├── [current: BF16 weights]
  │     weights:               149 GB
  │     activations + overhead:  ~3 GB
  │     ─────────────────────────────
  │     KV cache available:    ~30 GB   ← ~1.4M tokens
  │
  └── [recommended: FP8 weights + FP8 KV]
        weights (FP8):          75 GB
        activations + overhead:  ~3 GB
        ─────────────────────────────
        KV cache available:    ~104 GB  ← ~9M tokens  (6.5× improvement)
```

---

## Aggregated vs Disaggregated Mode

```
  AGGREGATED (current: exp_1_agg)          DISAGGREGATED (future: exp_2_disagg)
  ─────────────────────────────────        ────────────────────────────────────
  Client                                   Client
    │                                        │
    ▼                                        ▼
  Frontend                                 Frontend
    │                                        │
    ▼                                        ├──────────────────┐
  Worker (prefill + decode)                  ▼                  ▼
    │                                   Prefill Worker    Decode Worker
    │  same GPU handles both              (1–2 GPUs)        (2–3 GPUs)
    │                                        │                  │
    ▼                                        │ KV blocks        │
  Token stream                               └──── NIXL ───────▶│
                                                                 │
                                                                 ▼
                                                           Token stream

  Pros: simple, no KV transfer            Pros: TTFT ↓↓, scales prefill
        overhead                                independently
  Cons: prefill blocks decode             Cons: NIXL transfer overhead,
                                                more complex
```

---

## Event Plane: What Flows Between Components

```
  vLLM (inside worker)                  Dynamo Router
  ────────────────────                  ─────────────
  prefill completes
    → ZMQ: BlockStored {
        block_hashes: [0xABCD, 0xEF01]
        parent_hash:  0x1234
        token_ids:    [128, 256, 512...]
        block_size:   16
        medium:       "GPU"
      }                   ──── NATS ────▶  radix_tree.insert(worker_id, hash)

  LRU eviction fires
    → ZMQ: BlockRemoved {
        block_hashes: [0xABCD]
        medium:       "GPU"
      }                   ──── NATS ────▶  radix_tree.remove(worker_id, hash)

  Worker restart
    → ZMQ: AllBlocksCleared {}  ── NATS ──▶  radix_tree.clear(worker_id)
```

---

## Your Deployment Config (exp_1_agg_qwen_coder_next_b200.yaml)

```
  DynamoGraphDeployment: vllm-agg-coder-next
  │
  ├── Frontend  (1 replica)
  │     image:  dynamo-nvidia:f2c30388_vllm-0.16
  │     DYN_ROUTER_MODE=kv
  │     DYN_STORE_KV=mem
  │     DYN_REQUEST_PLANE=tcp
  │     OTEL → Tempo (traces)
  │
  └── VllmDecodeWorker  (4 replicas, node: sc09super22-b200)
        image:  dynamo-nvidia:f2c30388_vllm-0.16
        GPU:    1× B200 per pod  (nvidia.com/gpu: "1")
        cmd:    python -m dynamo.vllm
                  --model Qwen/Qwen3-Coder-Next
                  --tensor-parallel-size=1
                  --gpu-memory-utilization 0.95
                  --no-enable-prefix-caching        ← ⚠ breaks KV routing
                  --kv-events-config '{"publisher":"zmq",
                                       "endpoint":"tcp://*:20080",
                                       "enable_kv_cache_events":true}'
```

---

## Issues & Fix Plan

```
  Issue                          Root Cause                  Fix
  ──────────────────────────────────────────────────────────────────────────
  KV routing ineffective         --no-enable-prefix-caching  remove the flag
  Low KV cache capacity (~30GB)  BF16 weights use 149GB      use FP8 checkpoint
  Slow TTFT on first query       cold prefill every request  enable prefix caching
  ZMQ events published but wasted blocks evicted immediately  enable prefix caching

  Recommended command (worker):
    python -m dynamo.vllm \
      --model Qwen/Qwen3-Coder-Next-FP8 \          # 75GB vs 149GB
      --tensor-parallel-size=1 \
      --gpu-memory-utilization 0.95 \
      --kv-cache-dtype fp8 \                        # halves KV footprint
      --kv-events-config '{"publisher":"zmq",
                           "endpoint":"tcp://*:20080",
                           "enable_kv_cache_events":true}'
                                                    # prefix caching on by default
```

---

## Metrics to Watch

```
  Signal                              Healthy         Action if not
  ──────────────────────────────────────────────────────────────────
  prefix_cache_hits/queries           > 0.5           check routing + caching
  kv_cache_usage_perc                 0.7 – 0.9       FP8 if > 0.9
  num_preemptions                     = 0             reduce --max-num-seqs
  time_to_first_token p99             < SLA           chunked prefill + DBO
  inter_token_latency p99             < SLA           speculative decoding
  num_requests_waiting                ≈ 0             increase --max-num-seqs
  kv_block_idle_before_evict p50      > request gap   cache large enough
```
