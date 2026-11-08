# Dead ends and negative results

[Back to index](../README.md) · Related: [lessons and pitfalls](17-lessons-and-pitfalls.md), [timeline](18-timeline.md)

Everything tried that did not pay, with the numbers and the reason. Kept here so nobody re-runs them without a new idea. "Lossless" = output unchanged up to float order; lossy items give KLD.

## CPU side

| Idea | Measured | Why it failed | Source |
|---|---|---|---|
| Port to ik_llama.cpp / repack IQ2_XS (`_R4`) | ik native build 22.6-48.8 GB/s vs mainline 64.1-82.5 on the same GEMV harness; iq2_xs_r4 22.5 GB/s at L2 | its AVX2 IQ2_XS dequantizer is the same algorithm; its advantage (multi-row GEMM) is irrelevant at batch 1 | HANDOFF §3, 18.1 |
| SMT (64 threads) | 0.768 vs 1.218 tok/s (Oct 10); 0.45 vs 5.62 (Oct 11); 0.45-0.84 for any GOMP_SPINCOUNT | barriers wait on preempted workers; spinning workers starve the CUDA thread | HANDOFF 17.3, 26.2, §27 |
| `OMP_PROC_BIND=spread` | 1.221 vs 1.225 tok/s | same as close | HANDOFF 17.3 |
| Non-OpenMP ggml threadpool | poll 0/50/100: 5.24 / 5.80 / 5.48 tok/s vs ~5.8 | no gain | HANDOFF 20.8 |
| `OMP_WAIT_POLICY=ACTIVE` | no gain | libgomp ignores ggml's KMP_BLOCKTIME | HANDOFF 20.5 |
| Software prefetch (raw read and IQ2_XS kernel) | raw read ~120 GB/s at pf 0-8 KB; kernel 106.7 -> 101.2 and 108 -> 101-105 GB/s | the wall is real (GMI/LRDIMM); the kernel is near compute-bound | HANDOFF 18.3, 20.6 |
| Huge pages | `MADV_HUGEPAGE` read 111.9 vs 119.6 GB/s; page walks on the 766 GB mapping <= 4% | | HANDOFF §13, §19 |
| IQ2_XS kernel round 2 (gather, 64-bit index loads, blend grid, computed masks, 2 rows per call) | 0.49-1.01x of the current kernel | the kernel is instruction-throughput bound (IPC 4.23); trading loads for ALU adds instructions | HANDOFF §27 |
| Expert prefetch into L3 during GPU attention (`LLAMA_MOE_PREFETCH`) | lossless 5.79 -> 5.57 (P=2) / 5.46 (P=3); lossy 7.56 -> 7.22 | prediction chain + sync ~85 us/layer; hot vs cold kernel only 120 vs 107 GB/s; window only ~0.2-0.25 ms | HANDOFF 26.3 |
| `GGML_MMID_CHUNK=64` | tails -30 us but busy +2.5% | net loss; keep 128 | HANDOFF 31.z9 |
| Warm-up, page faults, disk as the cause of the early 300 ms gap | faults flat (162, 0 major), NVMe counters static | the gap was OpenMP barriers | HANDOFF §5.4 |
| Lossless entropy coding of expert bytes | <= 4.0% compressible (12.3% needed) | IQ2_XS index+sign codes are near-uniform | K3-IDEAL-BOUND §5 |

## GPU side

| Idea | Measured | Why | Source |
|---|---|---|---|
| Plain `-ngl` offload of experts at batch 1 | 0.30 vs 0.33 tok/s (`-ngl 16` vs 0, old build) | op-offload only moves host weights for >= 32-row batches | HANDOFF 5.2 |
| CUDA mat-vec variants: warp-per-row, cp.async weight staging, nibble-coded IQ2_S grid in smem, more warps per block | no gain | decode (ALU) is the limit, not loads | HANDOFF 20.1 |
| Gather selected experts into VRAM before the mat-vec (single token) | k=3 5.51 vs 5.58; k=4 4.68 vs 4.84 | in-place HMM reads already at PCIe max | HANDOFF 20.7 |
| `GGML_CUDA_GRAPH_OPT=1` | 4.96 vs ~5.58 tok/s; no gain earlier | | HANDOFF 20.5, 20.7 |
| Gated delta net 4-columns-per-warp | 33.7 vs 33 us | already ~770 GB/s | HANDOFF 31.zz |
| Rows per CUDA block 4 / 8 for 2-8 columns | +-1 us / worse (q4_K 35.6 -> 40.3 us) | small mat-vecs at ~5-8 us fixed cost | HANDOFF 31.zz |
| Fused residual-mix op (`GGML_OP_RES_MIX`) | decode 245.4-245.8 vs 245.3-245.5 ms/step; KLD 0.0065 vs bit-identical (unexplained) | with CUDA graphs tiny kernels cost ~nothing | SESSION3 Result 12 |
| KDA short-conv fusion | not built (est. 2-3 ms/step) | same cost model as res_mix: dropped | SESSION3 |
| MLA attention: int8 Q*K^T / FP16 accumulation / head split over GPUs | not built: ~-17% attention (~3-5 ms/step at 64-96k) / too lossy / VRAM | near the practical limit | SESSION3 |
| Double-buffered range copies (prompt processing) | ub 512: 31.56 / 30.15 vs 31.87 / 31.06; ub 2048: 69.90 vs 70.76 serial | no gain at any ub | HANDOFF 31.y |

## Expert placement and caching

| Idea | Measured | Why | Source |
|---|---|---|---|
| Pin the model mmap for zero-copy (`cudaHostRegister`) | not possible: read-only needs `cudaHostRegisterReadOnly` (unsupported); writable file-backed long-term pins refused by Linux >= 6.5 | solved with HMM instead | HANDOFF 20.6-20.7 |
| Static VRAM hot-expert cache | hot 3.1% from one text covers 3-14% of another's routing | hot sets are document-specific (also true for GLM earlier) | HANDOFF §21 |
| LFU eviction | -2.7% at 17 slots, nothing at 18 | | HANDOFF §31 |
| 19 cache slots (LFU) | 240 vs 233 ms/step at 18 | | HANDOFF §31 |
| KB 4 / KB 3 / KB 1 at x16 (vs KB 2) | 268 / 245 / 246 vs 242 ms/step | GPU-bound at KB 4; KB 2 balances | HANDOFF §31 |
| KB 3 + CACHE 8 (one slot traded for a streamed expert) | 276 / 276 vs 274 / 275 ms/step | | HANDOFF 31.z |
| KB 3 + CACHE 9 (no prefetch) | CPU split -3% but GPU1 overruns: neutral (8.55-8.62 tok/s) | a fixed extra expert cannot adapt to each layer's slack | HANDOFF 31.zzzz |
| KB 3 + prefetch 4 | CUDA OOM on GPU0 | extra scratch slot + staging ring | HANDOFF 31.zzzz |
| Second-GPU decode share (`LLAMA_MOE_GPU_KB2` 2-4) | 274-293 vs 267-269 ms/step | no P2P: staged cross-GPU copies and syncs cost what the CPU saves | HANDOFF 31.zz |
| Async cross-layer prefetch with the default copy grid | 6.3-7.4 vs 8.2-8.8 tok/s | ~400 KB of PCIe reads in flight made attention 2-2.5x slower (fixed with a slim copy: final -1.3% ms/step, kept) | HANDOFF 31.zzzz, 31.z5 |
| `LLAMA_MOE_DEFER_LATE` (shexp after the GPU share) | garbled output (draft acceptance 13/174) | allocator assumes graph order; nodes after the side-stream fork run concurrently | HANDOFF 31.zzzz |
| `iommu=pt` to cut the HMM copy interference | wait 1438 vs 1460 us: no change (but the baseline got ~2-2.5% faster) | | HANDOFF 31.z5, 31.z7 |

## Expert reduction (lossy)

| Idea | Measured | Why | Source |
|---|---|---|---|
| Fixed top-12 | +0.055 KLD (5.62 vs 4.63 tok/s at the time) | worse than adaptive skip at low skip rates | HANDOFF §24, 18.4 |
| Shared input basis / input projection (MoBE-style), r2560 | KLD 0.351 (PPL +27%) for 19% fewer expert reads | weights look random; input subspace content-dependent | HANDOFF §25 |
| Plain (non-union) skip on verify batches | plain 0.04 +0.051 vs union 0.04 +0.005 dKLD | union rule reads the shared experts anyway | HANDOFF §28 |
| Per-layer "no skip" groups | no group dominates; same KLD as a lower global tau at equal cost | | HANDOFF 31.z |
| Native-precision expert layers (50-59) | -0.0146 KLD for -3 tok/s (~9 ms/step per layer) | CPU bandwidth: native experts are ~2x the bytes | HANDOFF 31.z |
| Draft-position skip | at 12.2 experts KLD 0.0455 vs ~0.042 uniform | accepted drafts keep KV computed with their skip | HANDOFF 31.z10 |
| Importance-aware skip (static norms) | 1-5% lower KLD at equal experts, within noise | inconclusive; opt-in | HANDOFF 31.z10 |
| Trained output-norm predictor | oracle ceiling: ~3.5% fewer experts at equal KLD (~1-2% speed) | router weight is nearly as good as the real contribution | HANDOFF 31.z16 |

## Speculative decoding and sampling

| Idea | Measured | Why | Source |
|---|---|---|---|
| n_max 3 (3 drafted tokens) | Oct 12: 5.98 vs 6.11 / 5.40 vs 5.50; Oct 26: 296 ms at 2.80 tok/step (9.51 vs 9.70 tok/s); Nov 2: bench -8%, technical break-even; Nov 5: OOM | a 4-token verify reads ~25% more experts | HANDOFF §28, §31, 31.zz; SESSION3 |
| Q8_0 draft instead of IQ4_XS | 8.90 vs 9.01 tok/s, accept 0.76 vs 0.77 | no acceptance gain | HANDOFF §30 |
| Greedy-when-confident drafting (`LLAMA_SPEC_RS_GREEDY_ABOVE`) | 2.385 vs 2.384 tok/step | | HANDOFF §31 |
| Top-k truncation of sampling | accept 0.66 vs 0.53 | changes the output distribution: not allowed | HANDOFF §31 |
| `"backend_sampling": true` per request | identical output, no effect | | HANDOFF §31 |
| Confidence cutoff for draft 2 (`LLAMA_SPEC_P_MIN_MODE=2`, p_min 0.4) | mean 9.81 vs 9.97 tok/s (-1.6%) | ~10 ms per batch-size switch; real saving ~40 ms on cheap steps only | SESSION3 Result 4 |
| Padded verify batches (`LLAMA_SPEC_PAD_TOKEN`) | 9.80 vs 9.88 unpadded; removes the switch penalty but the pad costs ~7 ms per 1-draft step | nets out | SESSION3 Result 4 |
| RadixArk DSpark draft | 9.67 / 9.73 vs 9.97 tok/s; +1.9 GB VRAM at 96k | | SESSION3 |
| Full-block decoding of the bidirectional DFlash draft | accept 0.20, 5.50 tok/s | the draft was trained causal | SESSION3 |
| Distilled drafts on generic data (distill1, distill2, distill2b) | +0.009 / -0.027 / 0 per draft on the bench; -0.029 / -0.040 / -0.016 pos1 on held-out sessions | off-policy or generic data; distill2 also had the wrong loss | SESSION3 Results 2-3, 8, 10, 13 |

## Ideas analysed but not pursued

| Idea | Why not | Source |
|---|---|---|
| Second or third GPU for bandwidth (K3-IDEAL-BOUND: 3x 3090 -> 13.22 tok/s model) | DRAM total caps at ~142-146 GB/s with two PCIe readers; no budget | K3-IDEAL-BOUND §11, HANDOFF §30 |
| 8-CCD CPU (EPYC 7532/7543/7713) | +10-15% bandwidth expected; no budget | HANDOFF 31.z6-31.z7 |
| NPS4 + NUMA-aware expert placement | BIOS change + large work, at the wall anyway | SESSION3 |
| Better 2-bit expert format, per-layer bit allocation, 3-bit layers 50-59 | user: no requants | memory note, HANDOFF 31.z10 |
| A 1.5 TB 4-bit model with ~60% in RAM | estimated 3.5-4.5 tok/s (LRU over 60% of slices hits 94-97%; ~3-6% of expert bytes from NVMe at 6.5 GB/s) | HANDOFF §31 |
| Tree verification, n-gram + DFlash | planned, not run | HANDOFF "Plan for the next sessions" |
| Merge down into the fused CPU op; IQ2_XS sign trick | small (1-2%, 3-5% of kernel time) | HANDOFF 31.z9 |
