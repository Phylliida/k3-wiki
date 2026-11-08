# CPU expert kernels and threading

[Back to index](../README.md) · Related: [where the time goes](03-where-the-time-goes.md), [MoE placement](05-moe-placement-cache-prefetch.md), [memory bandwidth](14-memory-bandwidth-thermals.md)

All routed experts live in RAM, and the CPU computes most of them (~165-186 ms of a ~250-273 ms step), so the CPU MUL_MAT_ID path got the most kernel work. Everything on this page is **lossless** (bit-identical or verified with `test-backend-ops` against CUDA) unless noted.

## Results at a glance

| Change | Measured | Date / commit | Source |
|---|---|---|---|
| Native build (AVX2/FMA) instead of a baseline-SSE build | 0.32-0.35 -> **0.977 tok/s** CPU-only (~3x) | 10-10, build flag | HANDOFF §2 |
| Thread pinning `OMP_PLACES=cores OMP_PROC_BIND=close` | IQ2_XS kernel 73.3 -> 82.5 GB/s (+12.6%) | 10-10 | HANDOFF §3 |
| Batch-1 threading patches (get_rows, unary, concat column splits) | 0.977 -> 1.183 tok/s (`-ngl 0`); 1.176 -> 1.498 at `-ngl 28` (+27%) | 10-10, in snapshot `604bafadb` | HANDOFF 17.2-17.3 |
| IQ2_XS AVX2 dot product (index loads from memory, scale shuffles) | kernel 82 -> 110 GB/s (1.33x); model 1.600 -> 1.692 tok/s | 10-10, `604bafadb` | HANDOFF 18.1 |
| Pooled (work-stealing) decode MUL_MAT_ID, 128-row chunks | 4.17 -> 4.60 tok/s (dense-v1) | 10-10, `604bafadb` | HANDOFF 18.4 |
| Fused SiTU GLU op (`GGML_GLU_OP_SITU`) | 4.60 -> 4.68 tok/s; CPU split 12 nodes -> 4 | 10-10, `604bafadb` | HANDOFF 18.4 |
| Vectorized SiTU (AVX2 exp/tanh) | SITU 93 -> 24.5 us/layer; 5.58 -> 5.65 tok/s | 10-11, `604bafadb` | HANDOFF 20.8 |
| Fused gate+up+SiTU MUL_MAT_ID | CPU split 1662 -> 1489 us/layer; 4.68 -> 5.18 tok/s without GPU share (little with it) | 10-11, `604bafadb` | HANDOFF 20.8 |
| Multi-token IQ2_XS kernel (decode a row once for up to 4 tokens) | 1.60 / 1.95 / 1.99x per row for 2/3/4 tokens; 3-token batch 489.5 -> 421.2 ms; DFlash 4.9 -> 5.50 tok/s | 10-12, `dbe3f5f18` | HANDOFF §28 |
| MUL_MAT_ID setup off thread 0 (parallel zeroing, even src1 quantization) | 256/257 -> 252/253 ms/step; prompts 29.6-30.0 -> 31.3-31.5 tok/s | 11-04, `297e37d9f` | HANDOFF 31.z9 |

## The build: no AVX2 (HANDOFF §2)

The original `llama.cpp/build` had `GGML_NATIVE=OFF` and every SIMD flag off: `objdump -d libggml-cpu.so | grep -c ymm` gave **0** vs **19050** for a native build. A separate native build (`-DGGML_NATIVE=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_FA=ON -DGGML_OPENMP=ON`, ~4 min at 48 jobs) went from 0.32-0.35 to 0.977 tok/s CPU-only (reps 0.958 / 0.953 / 0.991 / 1.006). Lesson: check build flags before believing any number.

## Kernel microbenchmarks (HANDOFF §3)

`iq2xs_bench.cpp` drives ggml's batch-1 GEMV for one expert-shaped matrix (3584 columns), mainline, `-march=native`, 32 threads, ~2 GB DRAM-resident:

| Type | GB/s | G weights/s |
|---|---:|---:|
| iq2_xs | 73.3 (unpinned) -> **82.5** (pinned) | 254 -> 285 |
| iq3_xxs | 74.7 -> **82.6** | 195 -> 216 |
| q2_K | 115.9 | 353 |
| q3_K | 117.4 | 273 |
| q4_K | 117.8 | 209 |
| q6_K | 117.8 | 144 |
| q8_0 | 118.4 | 111 |
| f16 | 119.2 | 60 |

- K-quants and Q8_0 hit the DRAM wall (~118 GB/s); **IQ2_XS / IQ3_XXS were compute-bound at ~78 GB/s**: the 2-bit expert format was the slowest to decode, the opposite of the usual "lower bpw is cheaper" intuition.
- 64 SMT threads added a little in the microbench (iq2_xs 83.4, iq3_xxs 88.4) but are unusable in the model (below).
- **ik_llama.cpp was slower here**: its native build reached 22.6-48.8 GB/s on the same harness (mainline 64.1 vs ik 48.8 GB/s on the L2-resident case); its `_R4` row-interleaved layout was slower still (iq2_xs_r4 22.5 GB/s at L2). Porting K3 to ik_llama.cpp and repacking IQ2_XS were dropped.
- Trap hit: linking the bench against the non-AVX build first gave "IQ2_XS is 7.9 GB/s" (pure artifact). Verify with `ldd` and `objdump | grep -c ymm`.

## Batch-1 threading patches (HANDOFF 17.2-17.3)

The perf profile showed 23.8% in OpenMP barriers: ops that partition by rows ran on thread 0 at batch 1 ([where the time goes](03-where-the-time-goes.md)). Patches in `ggml-cpu`:

1. `get_rows_f32`: when rows < threads, every thread takes all rows but a 16-float-aligned column slice (the KDA state gather is 434 MB/token).
2. `apply_unary_op`: same column split (tanh, sigmoid, silu, exp; K3's SiTU).
3. `concat_f32`: partition over all (i1, i2, i3) rows, not only i2 (the residual stack is `[n_embd, n_ckpt, 1]`).

Verified with `test-backend-ops -b CUDA0`: GET_ROWS 217 OK, CONCAT 178 OK, unary ops 5 OK each.

| Config (`-p 0 -n 16 -r 3`) | tok/s |
|---|---:|
| get_rows patch only, `-ngl 0`, unpinned | 1.161 (1.157 / 1.172 / 1.154) |
| all three patches, `-ngl 0`, unpinned | 1.183 |
| + `OMP_PLACES=cores OMP_PROC_BIND=close`, 32 threads | **1.225** |
| `OMP_PROC_BIND=spread`, 32 threads | 1.221 (same) |
| `OMP_PLACES=threads`, 64 threads (SMT) | 0.768 (much worse) |
| pinned, 16 / 24 / 28 / 32 threads | 0.776 / 1.038 / 1.183 / 1.218 |
| pinned, `-ngl 28 -ncmoe 93` | 1.498 (was 1.176: +27%) |
| pinned, `-ngl 31 / 34 / 36` | 1.532 / 1.585 / **1.600** |
| `-ngl 38`, `-ngl 40` | do not fit |

Decode was core-bound: near-linear 16 -> 32 physical cores. After the patches, each layer moved to the GPU paid ~3x more than before.

## SMT is unusable (HANDOFF 17.3, 26.2, §27, 31.z7)

- `-t 64` = 0.768 tok/s vs 1.218 at `-t 32` (Oct 10); `-t 64` = **0.45 tok/s** vs 5.62 at `-t 32` (Oct 11); `-t 64/62` = 0.45-0.84 tok/s for any `GOMP_SPINCOUNT` (2000 / 20000).
- Cause: with every hardware thread busy, outside processes preempt a worker and each OpenMP barrier waits a scheduler timeslice (MUL_MAT_ID 8.3 ms vs 1.25 ms per call); spinning workers also starve the CUDA/main thread. At `-t 32` outside load runs on the idle siblings.
- Isolated, SMT does help the raw kernel (`cpu_load kern`: 32 threads 115.7 GB/s, 64 SMT 120.3; hot 137-140 vs 120-124 GB/s), which is why the microbenchmarks mislead.

## IQ2_XS AVX2 kernel, round 1 (HANDOFF 18.1)

Harness `iq2xs_kernel_bench.c`: mainline kernel copied as `ref`, candidate `opt` with macro toggles, 4096 random rows checked bit-identical. 1 core, L2-resident (mainline = 9.89 G weights/s):

| Change | G w/s | x |
|---|---:|---:|
| scalar shift/mask index extraction (`OPT_IDX=1`) | 13.17 | 1.33 |
| **movzx each index from memory (`OPT_IDX=2`)** | 14.71 | 1.49 |
| vbroadcastsd + vpblendd grid build (`OPT_GRID=1`) | 10.34 | 1.05 alone, hurts combined |
| one cvt per block + one shuffle per chunk for scales (`OPT_SCALE=1`) | 10.78 | 1.09 |
| **`OPT_IDX=2` + `OPT_SCALE=1`** | **15.62** | **1.58** |

Root cause: mainline stores the masked index vector to the stack and reloads 16 x u16, which stalls on store-to-load forwarding on Zen 2. Reading `q2[k] & 511` directly is faster. IQ3_XXS already reads indices from memory. 32 threads pinned, 2-4 GB from DRAM: **82 -> 110 GB/s (1.33x)**, near the ~118 GB/s wall. Ported into `ggml_vec_dot_iq2_xs_q8_K` (AVX2); MUL_MAT 14 OK, MUL_MAT_ID 15 OK. Model (`-ngl 36 -ncmoe 93`, pinned): 1.600 -> 1.692 tok/s (expert term ~168 -> ~127 ms), small because dense dominated then.

## IQ2_XS kernel, round 2: at the practical limit (HANDOFF §27)

Harness `iq2xs_kern2.c` (`perf_event_open` cycles/instructions). The current kernel: **55.6 cycles per 256-weight block, IPC 4.23 (~235 instructions per block)**, 15.5 G w/s on 1 core, 121-124 GB/s all-core hot. Zen 2 sustains ~4.5-5 macro-ops/cycle: instruction-throughput bound. Per block: grid lookup ~104 instructions, signs ~56, dot/scale ~40, overhead ~25.

| Variant | x current (1 core / all cores) |
|---|---|
| AVX2 gather for the grid | 0.49 / 0.55 (microcoded on Zen 2) |
| 64-bit index loads + shifts | 0.88 / 0.93 |
| vpbroadcastq + vpblendd grid (no shuffle uops) | 0.91 / 0.98 |
| scale masks computed by vpaddb instead of loaded | 0.93-0.98 / 1.00 |
| two rows per call sharing q8 loads | 0.98 / 1.01 (IPC 4.48, same cycles) |

Everything that trades loads for ALU or shuffles for blends adds instructions and loses. Left on the table (~3-5% of kernel time, needs an API change): sign via `vpblendvb(q8, -q8, mask)` with `-q8` precomputed once per MUL_MAT_ID.

Other measurements: IQ2_XS vec_dot is nearly compute-bound: hot (L2/L3) 120-124 GB/s vs cold DRAM 107 GB/s (steady clocks; schedutil needs ~0.5 s of load to ramp, short runs mislead). Software prefetch in the kernel (`OPT_PF` 1-4 KB): no gain (108 -> 101-105 GB/s); an earlier test: 106.7 -> 101.2 GB/s with prefetch (HANDOFF 18.3, 20.6).

## Pooled MUL_MAT_ID and fused SiTU (HANDOFF 18.4)

| Step | tok/s (`llama-bench -ngl 99 -ncmoe 93 -nopo 1 -n 32`, dense-v1) |
|---|---:|
| baseline | 4.17 |
| + pooled MUL_MAT_ID decode path (64-row chunks) | 4.56 |
| chunk 128 (32: 4.43, 256: 4.56) | 4.60 |
| + fused SiTU op | **4.68** (llama-completion real generation 4.70) |

1. **Pooled MUL_MAT_ID** (`ggml-cpu.c`): for <= 4 tokens, all (expert, 128-row chunk) pairs go into one pool handed out by an atomic counter. The old path gave each thread a fixed 96-row slice per expert, so the slowest thread paced every op (a few cores ran at ~2.2-2.9 GB/s vs 3.8-4.4 because of outside load or lower clocks). `GGML_MMID_DYN=0` disables, `GGML_MMID_CHUNK=N` tunes. MUL_MAT_ID 930/930 OK.
2. **`GGML_GLU_OP_SITU`** (`ggml_situ_split(ctx, gate, up, beta, linear_beta)`): new GLU op (CPU + CUDA + test, 17/17 OK) replacing a 9-node element-wise chain (scale, tanh, scale, sigmoid, mul, scale, tanh, scale, mul) that had a 32-thread barrier per node. CPU split 12 nodes -> 4. Profile after: expert kernels 57.5%, intra-split barriers 16.6% -> 4.8%.
3. **Vectorized SiTU** (HANDOFF 20.8): `ggml_v_expf`, tanh = 1 - 2/(exp(2x)+1), clamp |x| <= 10: 93 -> 24.5 us per layer; 5.58 -> 5.65 tok/s; SITU 16/16 vs CUDA.
4. **Fused gate+up+SiTU** (`ggml_compute_forward_mul_mat_id_situ_fused`, matched in `ggml_cpu_try_fuse_ops`): one pooled pass computes the gate and up rows of an (expert, chunk) and writes the activation. CPU split 1662 -> 1489 us/layer at 16 experts (4.68 -> 5.18 tok/s in the profiling config without GPU share). `GGML_CPU_FUSE_SITU=0` disables; `MUL_MAT_ID_SITU` 8/8.

## Multi-token kernel for verify batches (HANDOFF §28, commit `dbe3f5f18`)

Finding: a k-token verify step cost ~the sum of (expert, token) slots, not the expert union, because the CPU called vec_dot per (row, token) and re-decoded IQ2_XS each time. `ggml_vec_dot_iq2_xs_q8_K_multi` decodes a row once for up to 4 q8_K vectors (grid lookup, sign mask, scale shuffles shared; each extra vector costs psignb + maddubs + madd + add), bit-identical.

- Harness: 1.60 / 1.95 / 1.99x per row for 2 / 3 / 4 vectors.
- Real text (`llama-perplexity -b 3/-b 4 -c 192`): 3 tokens 489.5 -> 421.2 ms, 4 tokens 610.6 -> 509.2 ms, PPL identical.
- DFlash server (greedy): ~4.9 -> **5.50 tok/s**.
- Multi-token kernel rates later measured (31.z9): x1 114.9, x2 91.8, x3 73.6, x4 59.6 GB/s: with 2+ tokens per row the kernel is compute-bound, which is why the GPU share takes the experts shared by several tokens.
- MXFP4 (for native-expert layers): CPU MUL_MAT_ID at 1 token ~ IQ2_XS per weight (~108 GB/s of MXFP4 data, DRAM-bound); no multi-token MXFP4 kernel was written (HANDOFF 31.z).

## MUL_MAT_ID setup off thread 0 (HANDOFF 31.z9, commit `297e37d9f`)

Per-thread timers (TEMP) showed ~110 us of serial setup per layer while 31 threads waited:

1. Thread 0 memset the dst rows of skipped / GPU-taken slots (~30 rows x 28 KB for down). Now it lists them and all threads zero them after the barrier. Gate/up setup 60 -> 8.6 us, down 48 -> 38.6 us; MUL_MAT_ID 1809 -> 1694 us/layer at a slightly smaller mix (efficiency 84.1 -> 87.5%).
2. src1 quantization was split per row; the down projection's rows are only 6 blocks wide, so only 6 of 32 threads quantized and arrived ~33 us late. `ggml_mmid_quantize_src1` splits (row, block) pairs evenly: down setup 39 -> 16 us. Efficiency ~88% of the bench-predicted kernel time.

A/B in the server (`cpu_ab.sh`, old library via `LD_LIBRARY_PATH` prepend vs new, daily config, 2 runs each): **256/257 -> 252/253 ms/step** (-1.6%; the profiler predicted -8 to -10 ms, part is hidden behind GPU work); prompts 2k 29.6-30.0 -> 31.3-31.5 tok/s, 7.2k 29.5-29.7 -> 31.1 (+5.5%: prompt batches give most slots to the GPUs, so thread 0 used to zero many rows). Tests: MUL_MAT_ID 932/932, SITU 24/24, SPLIT 264/264, MOE_FFN_RANGE 51/51, CACHE 228/228.

`GGML_MMID_CHUNK=64` halves the tails (-30 us) but costs ~2.5% busy time: net loss, keep 128.

## Tried without gain

| Idea | Result | Source |
|---|---|---|
| Non-OpenMP build (ggml threadpool, `-C 0xFFFFFFFF --cpu-strict 1`) | poll 0 / 50 / 100 = 5.24 / 5.80 / 5.48 tok/s vs ~5.8 with OpenMP | HANDOFF 20.8 |
| `OMP_WAIT_POLICY=ACTIVE` | no gain (libgomp ignores ggml's KMP_BLOCKTIME) | HANDOFF 20.5 |
| Software prefetch in the IQ2_XS kernel | 108 -> 101-105 GB/s (slower) | HANDOFF 20.6 |
| Huge pages / page-walk cost on the 766 GB mapping | <= 4% | HANDOFF §19 |
| Expert prefetch into L3 during GPU attention | slower (5.79 -> 5.57 / 5.46) | HANDOFF 26.3, [MoE page](05-moe-placement-cache-prefetch.md) |
| Kernel round-2 variants | all <= 1.01x | HANDOFF §27 |
| Repacking IQ2_XS / ik_llama.cpp kernels | slower | HANDOFF §3 |

## Remaining small ideas (not done)

Merge the down projection into the fused gate/up op so its tail overlaps work (~1-2%); the IQ2_XS sign trick (~3-5% of kernel time); keep per-CCD balance in mind (each CCD's GMI link caps at ~39 GB/s, so uneven work across CCDs wastes bandwidth; HANDOFF 31.z6/z7).
