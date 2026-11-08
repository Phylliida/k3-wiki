# MoE expert placement: GPU share, VRAM expert cache, prefetch

[Back to index](../README.md) · Related: [CPU kernels](04-cpu-expert-kernels.md), [prompt processing](12-prompt-processing.md) (range split, fused MoE op), [where the time goes](03-where-the-time-goes.md), [memory bandwidth](14-memory-bandwidth-thermals.md)

The routed experts (~765 GiB at 2 bits) cannot fit in 48 GB of VRAM, so the question is which experts each step reads from where. All changes on this page are **lossless** (the same experts are computed, only on a different device; CPU vs GPU arithmetic differs in rounding, which K3 amplifies chaotically, see [expert skipping and quality](06-expert-skipping-and-quality.md)).

## Results at a glance

| Change | Measured | Date / commit |
|---|---|---|
| Deferred shared expert (runs on the GPU while the CPU does routed experts) | 4.82 -> 4.99 tok/s (llama-bench), real generation 4.86 -> 5.01 | 10-11, `604bafadb` |
| GPU share of the routed experts, read from the model mmap via HMM (FRAC 0.4, K 3) | 4.99 -> **5.58 tok/s** | 10-11, `604bafadb` |
| Host-source nodes on a second CUDA stream | 5.65 -> 5.70 | 10-11 |
| GPU share for verify batches (batch take rule, expert-major kernel) | DFlash lossless 460 -> 403 ms/step (5.50 -> 6.18 tok/s greedy) | 10-12, `4f867184b` |
| Second GPU (layer split) + VRAM expert cache (18 slots, KB 3) | 306 -> **243 ms/step** (7.95 -> 10.32 tok/s greedy) | 10-12, `367f377c6` |
| KB 2 + cache 18 at PCIe x16 | 233 ms/step, 10.10 tok/s sampled | 10-26 |
| Cross-layer expert prefetch into VRAM (P 4, admit, slim async copy) | -1.3% ms/step (256.5 vs 260) | 11-03, `12f3d199e`, `56801b5b3`, `7d3132042` |
| Second-GPU decode share (`LLAMA_MOE_GPU_KB2`) | no gain (exact, but no P2P) | 11-02, `8a6c6d4cf` |

## Placement basics (HANDOFF §5, 18.3)

- `-ngl 99` with everything on the GPU fails (`cudaMalloc` of 66292 MiB). `--n-cpu-moe 93` (`-ncmoe`) keeps the experts on the CPU (`LLM_FFN_EXPS_REGEX` matches `blk.N.ffn_{up,down,gate}_exps.weight`).
- Plain GPU offload of experts does nothing at batch 1 (`-ngl 16` vs `-ngl 0`: 0.30 vs 0.33 tok/s on the old build): the CUDA backend needs device-resident weights, and op-offload only moves host weights for batches >= 32 rows. Streaming experts over PCIe needed new code.
- With dense-v1 fully on the GPU, `-nopo 1` (no op offload) is **required**: otherwise the context reserves a 3.6 GB staging buffer (one IQ3_XXS expert tensor) and does not fit. Tools without the flag: `-ub` < 32.

## Deferred shared expert (HANDOFF 20.4)

New tensor flag `GGML_TENSOR_FLAG_DEFER = 32`. The scheduler starts a new split when the flag changes; a deferred split followed by a split on another backend is launched after that split's input copies, so it runs while the CPU computes. In `build_latent_moe` the shared expert is built first and its 4 nodes are flagged; `routed_in` must be expanded before it (otherwise the latent projection lands between shexp and the experts: 277 instead of 185 CUDA splits per token). Result 4.82 -> 4.99 tok/s; DEFER on/off byte-identical greedy output. `GGML_SCHED_DEFER=0` disables.

**Bug found later (commit `4f33592bc`):** a deferred split copied its inputs after the next split's, but the allocator can reuse the deferred split's host-side inputs (the MLA `k_idxs` / `kq_mask`) for the next split's tensors -> illegal memory access in flash attention when attention was deferred. Inputs are now copied in graph order; only the launch is deferred.

## GPU share over PCIe: from pinning to HMM (HANDOFF 20.6-20.7)

Feasibility:

- Zero-copy kernel reads of `cudaHostRegister`ed memory: **25.3 GB/s** (same as DMA). memlock 8 MB is not enforced by the driver. Works for anonymous memory and for a MAP_SHARED PROT_READ|PROT_WRITE mapping of a tmpfs file, **not** for the model's mmap on `/data2`: a read-only mapping needs `cudaHostRegisterReadOnly` (`HostRegisterReadOnlySupported = 0` here) and writable file-backed long-term pins are refused by Linux >= 6.5 on disk filesystems. `/dev/shm` is 498 GB, the model 766 GB.
- Read width matters: 16 B/lane coalesced 25.3 GB/s, u16 coalesced 9.2 GB/s, an mmvq-like pattern 18.2 GB/s.
- **DMA does not hurt the CPU kernel** (corrected measurement): with a spinning host thread the CPU IQ2_XS kernel lost 25-30% even at 6 GB/s of DMA (the spinner shares a core with a worker; the same artifact made llama-bench + DMA show 3.66 tok/s). With `cudaDeviceScheduleBlockingSync`: CPU kernel 104.5 GB/s solo -> 101.3 GB/s under 24.9 GB/s of DMA (-3%). Rule: no spinning host thread in any GPU-expert design.

**HMM (no pinning):** with the open kernel module + Linux 6.12 (`PageableMemoryAccess = 1`), GPU kernels read the read-only file mmap directly: warm 25.2 GB/s (uint4 loads), 16.2 GB/s (mmvq pattern); first touch 1.3-1.9 GB/s; page tables cost 2 MiB of VRAM per GiB mapped (`hmm_vram.cu`; later measured ~1.9 MiB/GiB in the server); `cudaMemPrefetchAsync` only maps (file pages cannot migrate).

Code: `ggml_mul_mat_id_set_split(t, mode, k, e0, e1)` (per token the first k slots with expert id in [e0, e1) are "taken": mode 1 computes only those, mode 2 all others, skipped slots written as zeros), `GGML_TENSOR_FLAG_HOST_SRC = 64` (read host weights in place, no copy), CUDA `mul_mat_id_split_ids`, a GPU chain (mode 1, HOST_SRC, DEFER) built before the CPU chain (mode 2), results added. Env `LLAMA_MOE_GPU_FRAC` (eligible ids [0, frac x 896)), `LLAMA_MOE_GPU_K` (3). Test `MUL_MAT_ID_SPLIT` 8/8.

| Config (FRAC 0.4, 297 GiB mapped in 168 s) | tok/s |
|---|---:|
| no share | 4.99 |
| k=2 | 5.22 |
| **k=3** | **5.58** |
| k=4 | 4.84 |
| gather-into-VRAM path (`GGML_CUDA_MMID_GATHER=1`), k=3 / k=4 | 5.51 / 4.68 |
| `GGML_CUDA_GRAPH_OPT=1` | 4.96 (hurts) |

The GPU-share MUL_MAT_ID took 390 us per 9.55 MB call = **24.5 GB/s, already at the PCIe maximum**, balanced against the CPU's 13 experts at k=3. Then: host-source nodes on stream 1 (overlap PCIe-bound expert reads with VRAM-bound shexp): 5.65 -> 5.70 (`GGML_CUDA_HOST_SRC_STREAM=0` disables). Best clean single-GPU lossless result: **5.79-5.84 tok/s** (FRAC 0.4, K 3, 2026-10-11).

Startup mapping (`prefetch_bench.cu`, 31): `cudaMemPrefetchAsync` with 1/4/8 threads 2.14 / 2.13 / 2.10 GiB/s; GPU touch per 4 KiB 1.77 GiB/s; touch per 2 MiB maps in 0.41 s but the first read then takes 9.3 s: HMM maps 4 KiB pages even though the page cache holds 2 MiB folios. ~2 GiB/s is the driver limit, hence the ~10 min startup with ~1.1 TiB mapped over two GPUs.

## Routing locality: is an expert cache worth it? (HANDOFF §21)

TEMP routing dump (`GGML_MMID_DUMP`), analysis `routing_overlap.py`, `routing_lru.py`:

| Text (PPL) | Shared with next token /16 | Union 4 tok | Union 8 tok | Hottest 3.1% of slices carry |
|---|---:|---:|---:|---:|
| wiki (2.00, memorized) | 5.98 | 42.8 | 70.8 | 33.0% |
| unseen-2026 (8.95) | 5.02 | 45.8 | 77.6 | 28.0% |
| code (1.65) | 5.47 | 44.2 | 73.5 | 38.5% |
| novel code (2.58) | 5.91 | 42.6 | 69.5 | 41.8% |
| own generations | 4.51 | 46.4 | 77.4 | 31.2% |
| random routing | 0.29 | 62.3 | 120.3 | 3.1% |

- Locality is real but moderate: verifying 4 tokens reads ~0.7x the expert bytes per token, 8 tokens ~0.6x.
- Hot sets are **document-specific**: the hot 3.1% picked on one text covers only 3-14% of another's routing, so a static VRAM cache gains ~nothing. LRU needs >= 22 GB for 32-37% hits; small caches (<= 7 GB) get ~0% (one token touches 1472 slices). Naive fill-on-miss would need ~10 GB/token of PCIe, so only admit slices the GPU streams anyway. (This is the design of the cache below.)
- 4-bit what-if (31): the most-used 25% of (layer, expert) slices carry 72-83% of a document's routing; an LRU over 40% / 60% of all slices hits 85-92% / 94-97%.

## GPU share for verify batches (HANDOFF §29, commit `4f867184b`)

**Batch take rule:** over the whole batch, eligible experts are ranked by how many tokens selected them, then by first occurrence; the slots of the first k are taken. Shared experts go to the GPU because its cost is per expert (PCIe bytes) while the CPU's cost grows with tokens per expert. CUDA: `mul_mat_id_split_take` (one block, shared-memory ranking) and an expert-major mat-vec `mul_mat_vec_q_moe_xm` (a warp computes 2 rows of one expert for all its tokens, so host weights cross PCIe once per expert). Env `LLAMA_MOE_GPU_KB`. Tests `MUL_MAT_ID_SPLIT` 68/68. Op profile (KB 6, 3 tokens): 805 us per 18.2 MB call = 22.6 GB/s.

Real-text batches (batch 3, 6 x 252 code tokens, FRAC 0.4): no share 33.86 s/pass; KB 4 31.19, **KB 6 30.19 (-11%)**, KB 8 32.31. KLD 0.024-0.026 (floor run 0.044).

DFlash server (router on CPU, FRAC 0.4, IQ4_XS draft, n_max 2, greedy):

| Config | code | code2 | math | chat | rust | Mean tok/s |
|---|---|---|---|---|---|---|
| lossless, no share | 5.45 | 5.94 | 5.75 | 4.87 | 5.47 | 5.50 |
| lossless, share KB 6 | 6.46 | 6.51 | 6.28 | 5.60 | 6.06 | **6.18** |
| lossless, share KB 5 | 6.32 | 6.63 | 6.55 | 5.74 | 6.26 | 6.30 |
| union skip 0.04, no share | 6.03 | 6.80 | 6.45 | 5.30 | 5.94 | 6.11 |
| union 0.04, share KB 4 | 6.65 | 7.35 | 7.25 | 6.22 | 7.62 | 7.02 |
| union 0.04, share KB 5 | 6.80 | 7.50 | 7.18 | 6.94 | 6.93 | **7.07** |
| union 0.05, no share | 7.13 | 7.02 | 7.75 | 6.31 | 7.02 | 7.05 |
| union 0.05, share KB 3 | 7.88 | 8.25 | 8.28 | 8.01 | 8.91 | **8.27** |
| union 0.05, share KB 4 | 7.43 | 8.01 | 8.19 | 7.81 | 8.32 | 7.95 |
| union 0.05, share KB 5 | 7.33 | 7.71 | 8.07 | 6.90 | 7.74 | 7.55 |

tok/s moves +-5% with acceptance alone, so compare ms per verify step: lossless 460 -> 403 (KB 6), 404 (KB 5); union 0.04 400 -> 352 (KB 4), 354 (KB 5); union 0.05 354 -> 310 (KB 3), 306 (KB 4), 323 (KB 5): -12 to -14% per step.

## Second GPU: layer split + VRAM expert cache (HANDOFF §30, commit `367f377c6`)

Two readers at once (`dual_bw.cu`): each GPU alone 26.2 GB/s, both 52.5 GB/s; CPU kernel alone 115.5 GB/s, + GPU0 111.6 + 26.1 (~138 total), + both GPUs 95.7 + ~46-50 (~142-146). So a second concurrent expert stream adds only ~4-10% to the expert phase: **the second card's value is VRAM**.

| Run | Change | Mean tok/s | Accept | ms/step |
|---|---|---|---|---|
| 1 GPU (§29) | router on CPU, no graphs, KB 4 | 7.95 | 0.71 | 306 |
| A | `-sm layer -ts 1,0 -devd CUDA1 -ot ^output\.weight=CUDA1` (draft + head on GPU1), router back on GPU0 | 8.69 | 0.77 | 293 |
| B | A + CUDA graphs | 9.04 | 0.77 | 281 |
| C | `-sm layer -ts 1,1 -devd CUDA1` + graphs (share runs on the layer's GPU) | 9.01 | 0.77 | 283 |
| D | C + Q8_0 draft | 8.90 | 0.76 | 286 |
| E | C, lossless (no skip), KB 6 | 6.92 | 0.77 | 367 (1 GPU: 403) |
| G | C + expert cache 16 slots/layer, FRAC 1.0, KB 4 | 9.33 | 0.74 | 268 |
| **H** | C + expert cache 18 slots/layer, FRAC 1.0, KB 3 | **10.32** | 0.75 | **243** |

(All greedy; union skip 0.05 unless noted. The draft uses the target's lm head, so with the draft on GPU1 the head must live there too.)

**VRAM expert cache** (`ggml_mul_mat_id_set_cache(t, state, cache)`): fed only by experts the share streams anyway. Per layer a state (expert per slot, last use, clock); GPU and CPU keep separate copies that stay equal because both run the same deterministic update: cached experts of the batch are taken, plus the k most shared uncached eligible ones, which are admitted (empty slot, else LRU among slots not used by this batch). CUDA: `mul_mat_id_cache_plan` (one block, serial walk), `mul_mat_id_cache_fill` (uint4 copy host -> slot), then the expert-major kernel on the cache. `LLAMA_MOE_GPU_CACHE=<slots>`; 18 slots = 17.4 GiB over both GPUs. Tests `MUL_MAT_ID_CACHE` 40/40 (chained evaluations on one state, so a state divergence changes later outputs).

Cache simulation (`cache_sim.py`, 3-token steps, lossless routing, union 35.7/layer): FRAC 1.0, k 4: 10 GB -> 5.7 hits/layer/step, 20 GB -> 9.0 (CPU 31.7 -> 22.7 experts); FRAC 0.4 only 3.8 / 5.6. Hits barely depend on k (k 2/3/4 at 15 GB: 7.74 / 7.74 / 7.70), so stream as few experts as the CPU/PCIe balance allows. With skip 0.05 (union ~22) KB 3 balances (~1.3 ms/layer each); run G at KB 4 was PCIe-bound (4 x 0.42 ms).

### Tuning at PCIe x16 (HANDOFF §31, sweeps 3-5, 2026-10-25/26)

Method: `night_sweep.sh`, recommended sampling, 3 seeds, COOLDOWN 240 s, MAX_MS=300 drops throttled requests (the DRAM throttle was not yet fixed). `-ts 49,44`, UB 512, RS tau 0.7, union 0.05.

| Config | ms/step | tok/s |
|---|---|---|
| KB 3, cache 17 | 245 | 9.31 |
| **KB 2**, cache 17 | 242 | 9.69 (+3.5% vs KB 3 on the same prompts) |
| KB 4, cache 17 | 268 | worse (GPU-bound) |
| KB 1 | 246 | (only 1/15 requests throttled: less memory traffic per second?) |
| **KB 2 + cache 18** | **233** | **10.10** |
| KB 2 + LFU, 17 slots | 236 (paired with LRU: 234.7 vs 241.3, -2.7%) | |
| KB 2 + cache 18 + LFU | 233 (no gain on top of 18 slots) | |
| KB 2 + cache 19 + LFU | 240 | |
| n_max 3 (3 drafted tokens) | 296 | 2.80 tok/step: no gain (9.51 vs 9.70 tok/s on the same prompts), hits the throttle sooner |

LFU (`LLAMA_MOE_GPU_CACHE_LFU=1`, commit `fa0afd606`): pick by decayed usage score (s -= s >> 3, +256 per token), victim lowest score. -2.7% at 17 slots, nothing at 18: off. (An earlier comparison without cooldowns on the first fast requests: LFU 245 / 242 / 248 vs LRU 242 / 238 / 245 ms.)

Value of a slot (31.z): ~0.89 GB over both GPUs; 16 -> 18 slots had been worth ~10 ms/step (~2% per slot at 16-18, more per slot lower down), which set a budget of ~4-6 GB of dense upgrades to stay >= 9 tok/s. In practice the dense-v2 move showed slots matter less than estimated (below).

Later values of a cache slot: dense-v2 needed VRAM, and cutting cache 16 -> 11 cost only ~3 ms/step (31.z c2r); one more slot with q8 KV (cache 10) gave 249 vs 251 ms/step (31.z10). KB 3 + CACHE 8 (same VRAM, one slot traded for a third streamed expert): 276 / 276 ms/step vs 274 / 275: no gain (Overnight 2). KB 3 + CACHE 9 without prefetch: CPU split -3% but GPU1 shares overran (larger IQ3_XXS experts in some layers): neutral (31.zzzz).

## Expert prefetch into L3 during attention: negative (HANDOFF §26, commit `6c1d05947`)

Idea: during each layer's GPU phase (~0.6 ms of a ~1.9 ms layer) the CPU/DRAM idle; predict the layer's experts from its pre-attention state and prefetch them into the 128 MB L3.

L3 microbench (`l3bench.c`): re-reading a per-core 1-4 MB slice from L3 390-750 GB/s aggregate (DRAM cold 117); same-CCX sibling slice 170-320; other-CCX slice 74-100 (slower than DRAM).

Router prediction (TEMP `LLAMA_K3_PRED_DUMP`, 2048 tokens code / prose, 92 layers):

| Predictor | recall@16 | recall@32 | Used among top-2/4/6/8/12 predicted |
|---|---|---|---|
| ffn-side mix, code | 0.751 | 0.885 | 1.95 / 3.83 / 5.62 / 7.28 / 10.08 |
| ffn-side mix, prose | 0.731 | 0.871 | 1.95 / 3.81 / 5.57 / 7.17 / 9.85 |
| attention input, code | 0.635 | 0.789 | 1.89 / 3.63 / 5.19 / 6.56 / 8.71 |

Weak layers: banked layers (il % 12 == 0: 12: 0.19, 24: 0.3, 36: 0.47) where the attention output restarts the stream.

Results (`LLAMA_MOE_PREFETCH=<n>`, llama-bench tg32; greedy output identical):

| Config | P=0 | P=2 | P=3 |
|---|---|---|---|
| lossless FRAC 0.4 K_gpu 3 | **5.79** | 5.57 | 5.46 |
| lossy tau 0.05 K_gpu 2 | **7.56** | - | 7.22 |
| no GPU share (tg16) | 5.10 | 4.97 | (P=4 4.82, P=6 4.50; P=6 without reads 4.92) |

Why: the prediction chain + extra GPU->CPU sync cost ~85 us/layer (~7 ms/token); MUL_MAT_ID per layer only 1485 -> 1455 us with 4 experts prefetched (hot kernel 120 vs cold 107 GB/s); the attention window is only ~0.2-0.25 ms; with the GPU share the PCIe side is the critical path anyway. Lesson: the CPU expert phase is bounded by IQ2_XS decode compute at about the DRAM rate; moving DRAM traffic elsewhere cannot help.

## Second-GPU decode share `LLAMA_MOE_GPU_KB2`: exact, no gain (HANDOFF 31.zz, commit `8a6c6d4cf`)

During a layer the other GPU's PCIe link is idle; it already maps the top half of every layer's experts. Split mode 3 replays the layer GPU's choice on a third copy of the cache state and takes the first k2 remaining experts in [e2, e3). Tests: SPLIT 264/264, CACHE 132/132.

- Correctness: PPL of one 2k chunk at 3-token batches 8.9019 (KB2=0) vs 8.9547 (KB2=2, +0.6%, within chaotic noise); tensor check of the first MoE layer: sums 1.6058 vs 1.6054 while the second GPU's own contribution sums to 0.271: its experts are counted exactly once.
- Speed (summary ms/step = time / (512 - accepted drafts)): base 269 / 267 bench, ~287 / ~289 summary; KB2=2 276 (274 profiled) / ~294; KB2=3 293 / ~285; KB2=4 279 / ~289.
- Profile: CPU compute 186 -> 172 ms/step (-14 ms, ~7 ms per expert moved), but the wait moved into the GPU splits' copies (CUDA0 33 + CUDA1 41 ms vs ~7): the second GPU's split copies its inputs from the layer GPU through host memory (no P2P). From 3-4 experts its PCIe streaming (~0.5 ms per expert per layer) is the critical path. Off by default; would likely pay with P2P/NVLink.

## Cross-layer expert prefetch into VRAM (HANDOFF 31.zzz-31.z7)

Idea: the PCIe links carry ~1.8 GB of ~11-12 GB possible per step because routing is known only right before a layer; predict layer L+k's experts early and copy them into VRAM.

**Prediction (no training)**, TEMP `LLAMA_K3_PRED_AHEAD` (commit `630e6f4a0`): layer L+k's own ffn-side mix + norm + router run on layer L's stream. Used experts among the top-P predicted (of 16), mixed prose+code / technical docs+source: post-attention 1 ahead h@6 5.4 / 5.5, h@8 6.8 / 7.0, h@12 9.1 / 9.6; post-attention 2 ahead h@8 5.9 / 6.3; pre-attention 2 ahead h@8 5.5 / 6.0; 4 ahead h@8 4.7 / 5.5 (random 0.14). (The GLM-era study on this box had found cross-layer prediction no better than random for GLM-5.3-Flash; K3's router predicts well. See [prior art](19-prior-art.md).)

**Policy simulation** (`prefetch_sim.py`, calibrated: today's policy 19.6 CPU experts/layer, 281 ms/step vs measured 17-22, 267-287): post-attention 2 ahead + KB 2: +9.1% / +7.1% at 2.0 experts/ms (~19.5 GB/s HMM), +13.4% / +11.6% at 2.6 (~25 GB/s); oracle +10-12% / +16-18%. The predictor gets ~70-75% of the oracle: **PCIe bandwidth, not prediction, is the ceiling** (~5-6 experts per layer time).

**Copy rate** (`stage_bw.cu`): gathering P experts (3 x 3,182,592 B each) from the page-cache-warm mmap into VRAM over HMM with 16 B loads: **26.1-26.3 GB/s = 2.73-2.75 experts/ms** with 3-6 blocks; `cudaMemcpyAsync` from the pageable mmap ~9 GB/s (useless). Sim at 2.75 experts/ms: staging + admission +14.2 / +12.7%; staging without admission +11.9 / +10.3%; straight into the LRU +6.1 / +3.8%; copy confined to the predicting layer's CPU phase +8.0 / +5.9%.

**Design** (commit `12f3d199e`): new op `GGML_OP_MOE_PREFETCH` (`ggml_moe_prefetch_plan`, `ggml_moe_prefetch_copy`), `ggml_mul_mat_id_set_staged` (staged experts taken after cache hits, optional admission to the LRU), ring of 3 staging slots per GPU, copy on its own high-priority stream, prediction at layer L for L+2, skipped across the GPU boundary. Tests `MUL_MAT_ID_CACHE` 204/204 -> 228/228.

**First in-situ result: slower.** Async P=4: 6.3-7.4 vs 8.2-8.8 tok/s on the same prompt. `pf_diag.sh` per layer (CPU split wait / compute, us):

| Config | CPU split wait / compute (us/layer) | tok/s |
|---|---|---|
| base | 630 / 2015 | 8.2-8.8 |
| plan only (prediction + plan, no copy) | 625 / 2035 | 8.6-8.8 |
| plan + async copy, staged experts NOT used | 1460 / 2010 | 6.4-7.1 |
| full async P=4 | 1423 / 1692 | 6.3-7.4 |

Prediction was free and staged experts cut CPU work (-15%), but the copy's HMM reads slowed whatever GPU work they overlapped: +830 us per layer of critical path for a 1.6 ms copy.

**In-order and abortable copies** (commit `56801b5b3`): copy at the end of the planning layer's deferred split, stopped when the next split needs the GPU (epoch in pinned memory, checked before each expert and every ~64 KiB; unfinished experts removed from the list so CPU and GPU mirrors stay equal). Four fixes were needed (in-order placement, abortable copy, stop within an expert, launch-time epoch). Also found an **allocator hazard**: main-stream nodes placed after the GPU share inside the deferred split run concurrently with it while ggml-alloc assumes graph order; `LLAMA_MOE_DEFER_LATE` garbled output (draft acceptance 13/174).

| Config (B-tree chat x3) | CPU split wait / compute (us) | GPU0 / GPU1 split wait (us) | tok/s |
|---|---|---|---|
| base | 632 / 2015 | 32 / 51 | 8.16 8.43 8.84 |
| base2 (repeat) | 638 / 2022 | 32 / 47 | 8.68 8.44 8.55 |
| P=4 (abortable, launch epoch) | 645 / 2002 | 47 / 66 | 8.96 8.12 8.50 |
| **P=4 + admission** | 633 / **1907** | 42 / 63 | 8.06 9.55 8.88 |
| P=6 + admission | 633 / 1965 | 44 / 67 | 8.34 8.62 8.48 |

Interleaved `speed_test` A/B (P=4 + admission vs default): pfab_base1 9.48 tok/s 266 ms/step, pfab_pf4a1 9.25 / 263, pfab_base2 9.32 / 267, pfab_pf4a2 9.21 / 262: **-1.5% per step**, technical text +3%, peak VRAM +130 / +110 MB. PPL check (one 2k chunk, 3-token batches): off 8.9019, on 8.9498 (+0.54%, same size as the proven-exact KB2 change). Tensor check: layers 1-2 bit-identical, layer 3 differs only in slots moved to the GPU (q8_1 vs q8_K activations): exact up to CPU/GPU numerics.

**Root cause of the async cost** (31.z5, after `iommu=pt` changed nothing): reads in flight. Split timing (TEMP `GGML_CUDA_SPLIT_TIMING`): attention/router split 634 / 690 us (GPU0 / GPU1) without copy, **1578 / 1317 us** with the default async copy (24 blocks x 256 threads, ~400 KB of PCIe reads in flight); **12 blocks x 64 threads 708 / 724 us**, 12 x 32 711 / 712 us, at ~15-16 GB/s. The flood of outstanding reads inflates every small latency-bound kernel's memory latency.

Slim async copy runs (per step):

| Run | Step ms | CPU wait / compute (us) | GPU-split waits (ms) | CPU experts | Staged used |
|---|---|---|---|---|---|
| st_base | 256.7 | 637 / 1981 | 3.4 / 5.0 | 18.96 | - |
| asf4 (P4 admit, 64 threads) | 254.4 | 696 / 1731 | 7.4 / 14.7 | 16.42 | 2.69 |
| asf4t32 (32 threads) | 254.0 | 679 / 1693 | 7.6 / 12.9 | 16.31 | 2.65 |
| asf4t32na (no admit) | 256.3 | 678 / 1754 | 5.4 / 11.8 | 16.85 | 2.95 |
| asf6 (P6 admit) | 294.1 | 786 / 1592 | 20.1 / 42.7 | 15.12 | 3.74 |
| sl4a (single launch) | 256.2 | 698 / 1723 | 6.5 / 10.7 | 16.39 | 2.68 |
| kb1p4 (KB 1) | 255.0 | 698 / 1832 | 3.3 / 4.3 | 17.62 | |
| kb1p5 | 253.8 | 730 / 1766 | 4.2 / 8.1 | 16.68 | |
| **aa4** (KB 1, P4, copy after attention) | **250.1** | 656 / 1847 | 4.2 / 5.6 | 17.43 | |
| aa4t128 (128 threads) | 253.1 | 675 / 1811 | 4.2 / 6.3 | 17.14 | |
| is4 (copy inside the attention split) | 251.4 | | | | |
| is5 (P5) | 255.7 | | | | |

P 4 is the sweet spot. **Final A/B without instrumentation** (31.z7, commit `7d3132042`): ab2_base1 9.62 tok/s 261 ms/step, ab2_pf1 9.54 / 257, ab2_base2 9.42 / 259, ab2_pf2 9.89 / 256: **-1.3% per step** (256.5 vs 260 ms), tok/s within acceptance noise, +126 / +166 MB peak VRAM. Left off in the code defaults; **on in `run-kimi.sh` at the user's request** (KB 1, P 4, admit, async, 1 block x 64 threads). If a very long prompt ever OOMs on GPU0 (~0.23 GB headroom at 96k), drop the prefetch lines and set KB back to 2.

Ideas not tried: copy-engine DMA from a pinned (tmpfs, huge-page) mirror of hot experts (isolated DMA interference ~2% vs ~7% for HMM gathers).
