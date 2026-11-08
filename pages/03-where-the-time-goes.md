# Where the time goes: budgets, profiles, phase timing

[Back to index](../README.md) · Related: [CPU kernels](04-cpu-expert-kernels.md), [GPU kernels](08-gpu-kernels.md), [MoE placement](05-moe-placement-cache-prefetch.md), [tools](15-tools-and-methodology.md)

Every optimization round started from a profile. This page collects them in order, so you can see how the bottleneck moved. "Step" = one speculative verify step (3-token batch) once DFlash was in use; before that, numbers are per token.

## Summary: how the budget moved

| Date | Config | Time per token / step | Dominant cost | Source |
|---|---|---|---|---|
| 10-10 | CPU-only, native build | 1024 ms/token (712 predicted) | ~300 ms unexplained (later: OpenMP barriers) | HANDOFF §6 |
| 10-10 | patched threading, `-ngl 36` | 1.600 tok/s (= 625 ms/token) | dense Q8_0 on CPU ~52% of CPU samples, barriers ~16% | HANDOFF 17.3 |
| 10-10 | dense-v1, all dense on GPU | 238 ms/token | expert kernels 49%, workers idle during GPU phases 28% | HANDOFF 18.3 |
| 10-11 | + GPU kernel, deferred shexp | ~200 ms/token | CPU expert splits 145 ms (1.56 ms/layer) | HANDOFF 20.5 |
| 10-11 | + HMM GPU share | 5.79-5.84 tok/s (= ~172 ms/token) | expert phase at the RAM wall: CPU ~94 + PCIe ~25 GB/s | HANDOFF 20.9 |
| 11-02 | 2 GPUs, DFlash, dense-v2, skip 0.045 | 273 ms/step | CPU experts ~186, serial GPU ~60, GPU-split host overhead ~13, between steps ~10 | HANDOFF 31.z "Overnight 2" |
| 11-04 | + kernel/iommu/CPU-setup fixes, q8 KV, ub 1024 | ~251 ms/step | CPU experts ~165, serial GPU ~60, overhead ~25 | HANDOFF "Plan for the next sessions" |
| 11-06 | session 3 phase timing | 251 ms/step: target decode 242-246, sampling 6.1 -> 2.2 (fast top-p), draft 3.2, rest 0.3 | target decode 96-97% of a step | SESSION3 Result 4, 11 |

## 2026-10-10: the unexplained 300 ms (HANDOFF §6, §17)

Prediction from measured kernel rates vs measurement, CPU-only:

```
dense   61.0 GB / 118  GB/s = 517 ms
experts 14.27 GB / 73.3 GB/s = 195 ms
                             = 712 ms      measured 1024 ms  ->  ~310 ms missing
config B (-ngl 28): dense 43.0/118 + experts 14.27/73.3 = 559 ms   measured 850 ms  ->  ~290 ms missing
```

perf profile (`-ngl 0`, unpatched, 1.05 tok/s under perf):

| Symbol | Share | ~ms/token |
|---|---:|---:|
| `ggml_vec_dot_q8_0_q8_0` (dense) | 49.3% | 469 (predicted 517: dense is fine) |
| gomp barriers (3 symbols) | **23.8%** | **~227: the gap** |
| `vec_dot_iq2_xs` + `iq3_xxs` | 17.6% | 167 (85 GB/s, matches the microbench) |
| memmove | 3.5% | 33 |
| `vec_dot_f32` (F32 router) | 2.3% | 22 |

Root cause: ggml CPU ops partition by rows, and at batch 1 a tensor has one row, so ops like `get_rows` (the KDA state gather: 6.3 MB x 69 layers = 434 MB/token copied by thread 0), `concat`, `tanhf`/`expm1f`, `sum_rows`, `sigmoid` ran on thread 0 while 31 threads spun. Fixed by column-splitting those ops ([CPU kernels](04-cpu-expert-kernels.md)). After the patches decode was **core-bound** (near-linear 16 -> 32 cores, SMT hurts).

## 2026-10-10: dense-v1 decode profile (HANDOFF 18.3, 238 ms/token)

| Part | Share | ~ms |
|---|---:|---:|
| expert kernels (iq2_xs 44.9% + iq3_xxs 4.1%) | 49% | ~117 (~120 GB/s wall) |
| `gomp_barrier_wait_end`: workers idle during GPU phases | 28% | ~67 |
| `gomp_team_barrier_wait_end`: barriers inside CPU splits | 17% | ~40 |
| other | 6% | ~14 |

- The main thread spent ~29% in libcuda + vdso `clock_gettime` (sync spin), ~65 ms.
- `GGML_SCHED_DEBUG=2`: 186 splits per token (93 CPU, 93 CUDA). Each CPU split was 12 nodes: 3 x MUL_MAT_ID + 9 element-wise SiTU nodes, each with a 32-thread barrier and scalar glibc `tanhf`/`expf`.
- GPU dense ~55 G params / 1.35 G/ms = ~41 ms; ~24 ms of launch/sync/copy over 186 splits.
- GPU mat-vec rates (`test-backend-ops perf`, 4096x14336, n=1, RTX 3090): q8_0 79 us (786 GB/s), q4_K 45 us (739 GB/s), iq2_s 43 us (433 GB/s), iq3_xxs 42 us (540 GB/s), iq2_xs 39 us: low-bit IQ types are decode-bound on CUDA too.

### Expert-count sweep (speed only, quality not measured; HANDOFF 18.4)

`llama-batched-bench --override-kv kimi-k3.expert_used_count=int:K`: K=16 4.63, K=12 5.62, K=8 7.20, K=1 12.73 tok/s -> **~9.6 ms per expert slot, ~62-70 ms fixed**. `GGML_CUDA_DISABLE_GRAPHS=1` at K=1: 12.65 -> 11.46 (graphs help ~8 ms).

Scheduler timing at K=1 (TEMP `GGML_SCHED_TIMING`), per token: CPU-split input copy (= waiting for the GPU stream + D2H) 54.2 ms (582 us/layer), CPU-split compute 18.5 ms (200 us/layer; one expert should be ~90 us -> ~10 ms of fixed CPU overhead), CUDA H2D copy 1.8 ms, CUDA graph launch 1.4 ms. So hand-offs were cheap; the fixed part was **GPU execution ~54 ms**: ~42 ms dense matmuls (IQ2_S 24.5 + IQ3_XXS 13.0 + ~4 rest) + ~12 ms of ~60 small kernels per layer.

## 2026-10-11: ~200 ms per token (HANDOFF 20.5, 20.8, 20.9)

- CPU expert splits 145 ms (1.56 ms/layer; kernels ~117 GB/s in-kernel, barriers + misc ~20 ms); GPU non-overlapped ~45 ms (matmuls ~35); H2D + launch ~6 ms.
- CPU op profile (TEMP `GGML_CPU_OP_PROFILE`) per layer: MUL_MAT_ID 3 x 533 us (~70 us over bandwidth each), SITU 93 us (scalar `tanhf`/`expf`), fork/join ~24 us. (Vectorizing SITU took it to 24.5 us.)
- With the GPU share, total DRAM traffic ~120 GB/s (CPU ~94 + PCIe ~25): the expert phase sits at the RAM wall (~1.27 ms/layer ideal, ~1.33 measured). CPU-side cuts (fused gate+up+SITU) gave a lot without the GPU share (4.68 -> 5.18) but little with it.
- **Do not use `-v` for speed numbers**: it logs "CUDA Graph id reused" per launch on the critical path (56 us per CUDA split vs 14 without).

## 2026-10-11: what multi-token verify batches cost (HANDOFF §23, §28)

- llama-bench, target only, no share, graphs on: tg 223 ms, pp2 421, pp3 588, pp4 775 ms (random tokens).
- CPU MoE 1588 us/layer at 1 token (16 slices, 99 us/slice) vs 5283 us/layer at pp3 (~46 slices with random tokens, ~115 us/slice).
- Real-text expert union per layer: 2 tokens 26.5-27.5, 3 tokens 35.6-37.3, 4 tokens 44-46, 8 tokens 74-77. Only ~5 of 16 experts are shared with the next token.
- The finding that drove the multi-token kernel: a k-token verify cost ~sum of (expert, token) slots, not the expert union, because the CPU re-decoded IQ2_XS per token. After the kernel: 3 tokens 489.5 -> 421.2 ms, 4 tokens 610.6 -> 509.2 ms (`llama-perplexity -b 3/4`, PPL identical).

## 2026-11-02: the 273 ms step (HANDOFF 31.z "Overnight 2")

Defaults then: dense-v2, skip 0.045, KB 2, cache 9, 2 GPUs.

| Part of a 273 ms step | ms | Notes |
|---|---|---|
| CPU expert reads | ~186 | `GGML_CPU_OP_PROFILE`: CPU graph time 173-187 ms per step (MUL_MAT_ID 169-184), ~65% of the step. ~24 experts/layer x 9.55 MB x 92 layers ~ 21 GB/step at ~120 GB/s |
| GPU serial path | ~60 | ~0.6 ms per layer of attention, norms, router and latent projections before the CPU can start; the CPU split's input "copy" waits ~0.66 ms/layer (93 x ~665 us = 62 ms) |
| GPU splits, host side | ~13 | input copies and launches (CUDA0 101 splits, CUDA1 85 per step) |
| Between steps | ~10 | draft (1-2 splits, ~0.1 ms of GPU work), sampling |

GPU time per decode layer (TEMP `GGML_CUDA_OP_PROFILE`, graphs off, 393 ms/step under the profiler):

| Op | us per layer | Effective rate |
|---|---:|---|
| expert share MUL_MAT_IDs (2 streamed + cached) | ~1000 | runs beside the CPU's ~2 ms: hidden |
| IQ2_S [7168,12288] x3 (KDA projections) | 71/call x 2.6 = 184 | ~390 GB/s (~40% of bandwidth) |
| `attn_output` Q5_K | 78 | ~780 GB/s |
| gated delta net | 70 (isolated: 33 us; the rest was launch prep with graphs off) | ~770 GB/s isolated |
| MLA q IQ3_XXS [7168,12288] | 52 | ~500 GB/s |
| latent projections Q6_K | 48 + 34 | ~440 GB/s |
| flash attention (short context) | 40-50 | |
| router Q8_0 | 15 | |
| shared expert Q4_K + IQ3_XXS | 84 + 40 | deferred, parallel |
| tiny ops (cpy, norms, adds, concat) | ~280 | inflated by graphs-off launches |

Isolated serial-path kernels at n=3 (31.zz): iq2_s 53 us x2.6/layer, q5_K 77, q6_K 31 + 32, iq3_xxs 55 x0.75, GDN 33 x0.75, router q8_0 10: ~350-375 us/layer of matmuls vs the ~665 us/layer the CPU waits, so ~290 us/layer (~27 ms/step) was small ops, launches and host<->device copies.

### Content matters more than context length (31.zz)

TEMP decode counters per gate/up call (per layer per step, 48 slots): technical summary: 9.1-9.7 skipped, 11.7-13.6 on the GPU, **20.5-22.6 read on the CPU**; bench prompts: 10.1-11.5 skipped, 13.2-14.8 on the GPU, **17.3-19.6 on the CPU** (~19% fewer). Technical text has flatter router weights (fewer skips) and less expert reuse (fewer cache hits): ~300-340 vs ~265 ms/step. `speed_test.sh` therefore also reports a 2k-token technical summary.

## 2026-11-03/04: CPU split efficiency (HANDOFF 31.z7, 31.z9)

- First claim: the server's CPU expert work ran at ~92 GB/s effective (~19 experts x 9.55 MB = 181 MB per layer in ~1980 us) vs ~116 GB/s for the kernel in isolation. **This comparison was unfair**: 116 GB/s is the single-token kernel.
- Multi-token kernel rate (`cpu_load_k3 kernN`, cold DRAM, 32 threads): **x1 114.9, x2 91.8, x3 73.6, x4 59.6 GB/s** (plain read 123.9): at 2+ tokens per row the kernel is compute-bound.
- In-server mix per layer of a 3-token verify batch: ~17.5 CPU experts = 14-14.7 with 1 token, 2.7-3.0 with 2, 0.15-0.2 with 3 (the GPU takes the shared ones). Predicted pure kernel time ~1520 us/layer; measured MUL_MAT_ID ~1810 -> 84% efficiency.
- Per-thread phases per layer: fused gate/up setup 60 / busy 1077 / tail 42 us; down setup 48 / busy 602 / tail 20 us (sum 1849 = the sched-timing CPU split).
- Two setup fixes (parallel zeroing of skipped rows, even src1 quantization) took setup 60 -> 8.6 us (gate/up) and 48 -> 16 us (down), efficiency ~88%, and the server step 256/257 -> 252/253 ms ([CPU kernels](04-cpu-expert-kernels.md)). Remaining gap ~210 us/layer: tails ~60 us, down kernel short rows (1536-wide rows run ~8% slower by nature: x1 108.1 / x2 87.9 GB/s at NE=1536 vs 117.8 / 93.9 at NE=7168).

## 2026-11-05/06: step-phase timing (SESSION3 Result 4)

Commit `8092e89fb` adds `| pre draft decode sample` (us) to every step-log line.

| Run | Transition (drafts prev -> cur) | n | Step | Pre | Draft | Decode | Sample |
|---|---|---|---|---|---|---|---|
| pmin04 (unpadded) | 1->1 | 117 | 199 | 0.2 | 3.2 | 192.1 | 4.2 |
| | 1->2 | 122 | 257 | 0.2 | 3.2 | 247.6 | 6.0 |
| | 2->1 | 123 | 217 | 0.2 | 3.2 | 209.8 | 4.2 |
| | 2->2 | 409 | 253 | 0.2 | 3.2 | 243.4 | 6.1 |
| pmin04_pad | 1->1 | 121 | 206 | 0.2 | 3.2 | 198.9 | 4.3 |
| | 1->2 | 167 | 249 | 0.2 | 3.2 | 240.2 | 6.0 |
| | 2->1 | 156 | 211 | 0.2 | 3.2 | 203.7 | 4.3 |
| | 2->2 | 533 | 261 | 0.2 | 3.2 | 251.5 | 6.1 |
| nocut | 2->2 | 730 | 251 | 0.2 | 3.2 | 242.5 | 6.1 |

- The target decode is 96-97% of a step; drafting 3.2 ms; pre/other ~0.3 ms.
- Verification sampling was 4.3 ms (2 positions) / 6.1 ms (3 positions): ~2 ms per position, nearly all top-p. Fixed by the fast exact top-p (6.05 -> 2.2 ms; [sampling](11-sampling.md)).
- **CUDA graphs are worth ~23 ms/step** (~9%): causal draft without graphs 9.08 tok/s at 274 ms/step vs 251 with graphs (SESSION3 progress log). In the transition table graphs saved ~16 ms in every case.
- A step with no draft costs ~209 ms (vs ~250 ms for a 3-token step) (`dab_full`, where every draft failed).

## Long context (SESSION3 Result 6, HANDOFF 31.zz)

Target decode per 2-draft step on the same technical-summary task: 265 ms (2-8k), 268 (16k), 275 (32k), 296 (64k): **+31 ms at 64k**, from MLA flash attention (34 us/call at 512 KV, 127 at 8k, 434 at 32k, 844 at 64k, 1257 at 96k; x23 layers ~19 ms/step at 64k, ~29 at 96k; compute-bound at ~49 TFLOPS). Draft time does not grow (3.3-3.4 ms; sliding window). See [long context](13-long-context.md).

## Small GPU ops: big in a graphs-off profile, ~free with graphs (SESSION3)

The Nov 2 graphs-off CUDA op profile counted ~7,090 small-op calls per step, ~50 ms/step event-timed: CPY 666 calls at 11.4 us (7.6 ms), RMS_NORM(+mul) 785 at 8.9 (7.0), CONCAT 480 at 12.7 (6.1), ADD 665 at 7.8 (5.2), GET_ROWS 349 at 7.3 (2.5), SUM_ROWS 495 at 4.7 (2.3), MUL 272 at 8.1 (2.2), TOP_K 102 at 16 (1.6), SSM_CONV 222 (1.4), SCALE 346 (1.4), DSV4_HC_PRE 199 at 6.6 (1.3), CONT 272 (1.3). The residual-mix chain alone was ~37 us per call, 199 calls/step (~7 ms graphs-off). Fusing it into one op (`GGML_OP_RES_MIX`) removed ~800 kernels/step and **changed the step time by nothing** (decode 245.3-245.5 vs 245.4-245.8 ms): with CUDA graphs on, tiny kernels on the serial path cost ~nothing, and the graphs-off profile overstated them ([GPU kernels](08-gpu-kernels.md)).
