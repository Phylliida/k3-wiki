# Timeline and changelog

[Back to index](../README.md) · Related: [current setup](02-current-setup.md), [dead ends](16-dead-ends.md)

Commits are on branch `k3-decode-speed` of the `llama.cpp-k3` worktree (forked from the user's `j-lens` branch at `74f368d88`, 2026-09-28, which already carried upstream K3 support from `ad1de39e0`, 2026-08-15). The first K3 work (2026-10-10/11) was uncommitted on `j-lens` and then saved as the snapshot `604bafadb`.

## Decode speed milestones

Greedy numbers (before 2026-10-25) overstate real speed; from 2026-10-25 on, numbers are sampled (temp 1.0, top-p 0.95).

| Date | Decode | What changed | Lossy? |
|---|---|---|---|
| 10-10 | 0.32-0.35 tok/s | start: CPU-only, build without AVX2 | |
| 10-10 | 0.977 | native AVX2 build | no |
| 10-10 | 1.176 | dense of 28 layers on the GPU | no |
| 10-10 | 1.225 (`-ngl 0`) / 1.600 (`-ngl 36`) | batch-1 threading patches + pinning | no |
| 10-10 | 1.692 | IQ2_XS AVX2 kernel 1.33x | no |
| 10-10 | 4.19 | dense-v1 requant, all dense on the GPU | yes (KLD 0.361 vs UD-Q2_K_XL) |
| 10-10 | 4.68 | pooled MUL_MAT_ID + fused SiTU | no |
| 10-11 | 4.78 -> 4.99 | CUDA IQ2_S/IQ3_XXS kernel; deferred shared expert | no |
| 10-11 | 5.58 -> 5.79-5.84 | HMM GPU expert share; SiTU AVX2; side stream; 5-op res_mix | no |
| 10-11 | 7.20 (7.56 later) | adaptive skip tau 0.05, K_gpu 2 | yes (+0.10 KLD) |
| 10-11 | 4.7-5.2 | first DFlash server (greedy) | no |
| 10-12 | 5.50 / 6.11 / 7.05 | DFlash + multi-token kernel / + union skip 0.04 / 0.05 (greedy) | union skip yes |
| 10-12 | 6.18 / 8.27 | + GPU share for verify batches (lossless / union 0.05, greedy) | |
| 10-12 | **10.32** | second GPU + VRAM expert cache (greedy, union 0.05) | yes |
| 10-25 | 7.72 | sampled re-baseline (GPU1 at x8) | yes |
| 10-25 | 8.10-8.14 | stochastic verification | no |
| 10-25 | 9.46 (240 ms/step) | GPU1 reseated to x16 | |
| 10-26 | 10.10 (233 ms/step) | KB 2 + cache 18 | |
| 10-26 | 10.21 sustained, 9.81 over a 1 h soak | fans on Full (thermal throttle fixed) | |
| 10-30 | 10.22 | deployed fast2 with two-GPU prompt processing | |
| 10-31 | 9.83 | 96k-context defaults (cache 16) | |
| 11-01 | 10.17 | dense-v2 overlay (KLD vs native 0.243 -> 0.095 at the same speed) | quality up |
| 11-02 | 9.23 (274 ms/step) | skip tau 0.045 chosen for quality (KLD 0.113 as served) | yes (less lossy) |
| 11-02 | 266 ms/step | multi-column IQ2_S/IQ3_XXS mat-vec | no |
| 11-03 | 259-261 ms/step | `iommu=pt` | no |
| 11-03 | 256.5 ms/step | cross-layer prefetch on | no |
| 11-04 | 252/253 ms/step | CPU MUL_MAT_ID setup fix | no |
| 11-04 | 251 ms/step, prompts 47 tok/s | 8-bit KV + ub 1024 | no |
| 11-05 | 9.80 tok/s (250 ms/step) | session-3 baseline (`dab_base`) | |
| 11-05 | 9.97 (mean of 4) | causal in-block draft attention (env knob) | no |
| 11-07 | ~10.1 (9.93 -> 10.08) | fast exact top-p | no |
| 11-07 | (~+2-5% tok/step in agentic sessions) | distilled draft distill3 recommended (offline +0.046 / +0.055 per position; online 24-probe paired +0.054 +- 0.028) | no |
| 11-07 | real use ~5.6 tok/s (~6.9 unthrottled) | first real session with distill3 + fast top-p at 13-79k context: 0.494 accepted per draft, DRAM throttle about half the session (SESSION3 Result 16) | measurement |

## Changelog by day

**Before K3 (2026-08/09).** The same box ran GLM-5.3-Flash with ik_llama.cpp (then 125 GB of RAM and one 3090): expert caching, hot-expert studies, SSD offload. See [prior art](19-prior-art.md).

**2026-10-10, session 1 (research + baseline).** Web research on local K3 (REPORT.md, REPORT-GGUF.md, LOCAL-K3.md, KIMI-K3-LOCAL-REPORT.md), first-principles bound (K3-IDEAL-BOUND.md), software plan (SOFTWARE-PLAN-9TPS.md, later disproven). Measured: 0.32-0.35 tok/s; the build had no AVX2 -> 0.977; `-ngl 28 -ncmoe 93` 1.176; per-type GEMV rates (IQ2_XS compute-bound at ~78 GB/s); byte accounting (dense 81% of traffic); DMA additivity (25 GB/s per GPU); entropy coding dead (4.0%); ik_llama.cpp slower.

**2026-10-10, session 2 (evening).** perf profile: 23.8% OpenMP barriers -> batch-1 threading patches (1.225 / 1.600); IQ2_XS kernel 1.33x (1.692); dense-v1 requant with a new imatrix (4.19, KLD 0.361); pooled MUL_MAT_ID (4.60); fused SiTU op (4.68); expert-count sweep (~9.6 ms per expert slot).

**2026-10-11.** CUDA persistent IQ2_S/IQ3_XXS mat-vec (4.78); KLD noise floor ~0.016; deferred shared expert (4.99); HMM GPU expert share (5.58); vectorized SiTU (5.65), host-source stream (5.70), 5-op res_mix (5.79), warp top-k; fused gate+up+SiTU; best clean 5.79-5.84. Machine degraded by outside load at 15:10. Routing-locality study; DFlash setup (4.7-5.2 tok/s at first); adaptive expert skip; shared-basis study (negative); L3 prefetch study (negative); scheduler deferred-copy bug fixed. Commits `604bafadb` (snapshot), `f3f84b015` (skip), `4d85dba96` (TEMP xproj), `4f33592bc` (deferred split input order), `6c1d05947` (TEMP L3 prefetch).

**2026-10-12.** IQ2_XS kernel round 2 (no gain; SMT unusable); multi-token IQ2_XS kernel `dbe3f5f18`; union-aware skip `a3ea61f0e`; GPU share for verify batches `4f867184b`; second GPU (borrowed) with layer split + VRAM expert cache `367f377c6`: **10.32 tok/s greedy**. `serve_k3.sh` chat server; user rule: benchmark sampled.

**2026-10-25.** Both 3090s permanent; K3 work moves to the `llama.cpp-k3` worktree. GPU1 at x8 after the reboot. Sampled re-baseline 7.72 tok/s; sampler cost study; stochastic verification + top-p bound `42d85672b` (8.10); startup mapping study; range split for prompt batches `c747ccd50` (22.52 tok/s on GPU0; pp2048 36.45 with two GPUs at ub 2048). Evening: GPU1 reseated, x16: 240 ms/step, 9.46 tok/s.

**2026-10-26.** Long runs degrade to ~400 ms/step: CPU memory bandwidth halves after ~3 min (31.x). Overnight sweeps: KB 2 + cache 18 = 233 ms/step, 10.10 tok/s; LFU cache `fa0afd606` (no gain at 18 slots), several deferred splits per backend, second-GPU prompt share (experimental). ~13:00 throttle fixed with BMC fans on Full + a third fan; 10.21 tok/s sustained; 1 h soak 0 throttled. Evening: op_params fix `c1cfdfd7f`; PP2 both GPUs on prompt layers: 31.87 tok/s (+38%).

**2026-10-30.** PP2 KLD verified (0.0260 vs 0.0272); ub 1024 48.62 tok/s, ub 2048 70.76; double-buffered range copies reverted; first-touch mapping fix (other GPU takes the top experts); server deployed at ub 512: prompts 29.45 / 29.40, generation 10.22 tok/s.

**2026-10-31.** Fused MoE range op `eb1bf7e81` (ub 2048 69.53 tok/s) and on the main stream `dd993055e` (server ub 1024: prompts 41.5 / 44.1, 10.12 tok/s); HMM page tables ~1.9 MiB/GiB; `LLAMA_MOE_GPU_K=2`. Native 4-bit UD-Q4_K_XL downloaded as the quality reference; UD-Q2_K_XL deleted after verifying its experts. Loader tensor sources `a8405846e` / `aa0525637`, MXFP4 expert-major kernel `299c3ecbc`, overlay writer `8a1e0e5a5`; precision study started (dense-v1 vs native KLD 0.243; dense_all 0.071). 96k-context defaults (9.83 tok/s).

**2026-11-01.** Group and overlay KLD results; 8-bit KV crashes the server (dropped then); c2r 10.21 tok/s, c3r 9.93; c3r KLD 0.0953 -> installed as **dense-v2** (`b4f56553a`, `6bf6407e6`); verification 10.17 tok/s. Queue-quoting bug: 5 runs never ran. Native-expert layer ranges (40-59 best); x4_5059 7.19 / x4_5054 8.50 tok/s: rejected. Skip tau sweep on dense-v2.

**2026-11-02.** Decision: skip tau 0.045 (9.23 tok/s, KLD 0.113); per-layer tau `c7fb4f4cd`; step budget profiled (CPU ~186, serial GPU ~60 ms of 273), TEMP profilers `08335ee2f`; multi-column mat-vec `cf1f96157` (266 ms/step); long-context and content profiles; second-GPU decode share `8a6c6d4cf` (no gain); DRAFT_N 3 (no gain); cross-layer prediction study `630e6f4a0`.

**2026-11-03.** Cross-layer expert prefetch built and debugged (`12f3d199e`, `56801b5b3`, `7d3132042`): -1.3% ms/step, turned on in `run-kimi.sh` at the user's request. `iommu=pt` reboot: baseline ~2-2.5% faster. Memory-wall mechanism (4 CCD GMI links, 39.1 GB/s each); user: no hardware budget. Agreed software plan. Disk prompt cache `78c934053`, `867aeb6b9`. CPU split efficiency study.

**2026-11-04.** CPU MUL_MAT_ID setup fix `297e37d9f` (252/253 ms/step, prompts +5.5%). User: no requants, lossy only at equal or better KLD. 8-bit KV works (K3 flash-attention tests `10b684e22`); daily config gets q8 KV + ub 1024 (prompts 47 tok/s; 90k prompt 42.1 tok/s). Draft-position skip (dropped), importance-aware skip `5c0c99c2a` (opt-in). Disk cache 2048+ tokens `8a44b3ad9`. Confidence-cutoff knobs and step log `ec5c26a52`; distillation research; causal-mask knob `934620b52`; DSpark downloaded and converted; full-block knob `505338506`.

**2026-11-05 (session 3).** Draft A/B: causal mask +1.7% (adopt); DSpark OOM, then not better; full block broken then fixed; cutoff -1.5% with a batch-size switch penalty; CUDA graph key fix `c52898b5f`; graph stats `0099fa178`; oracle skip bound `d3f8c075c` (dropped); DSpark stochastic verification `015994ebc`; `3adf5d263`; `llama-dflash-dump` `bc665ba0b`; padded verify batches `88a810398`; distillation chain validated; night6 started (dump1 602k tokens).

**2026-11-06.** distill1 rejected (+0.009 per draft); night7 phase timing `8092e89fb`: padding works but nets zero -> cutoff closed; fast exact top-p `0c587ea0d`; acceptance flat vs context (night8); DRAM throttle returns under rebuild + decode -> `thermostat.sh`, fan layouts (layout 3 best); on-policy generation (293 answers, 301k tokens); fused res_mix op `c9666450c`; distill2 val shows the loss mismatch -> `--loss acc`; the user's own sessions prepared as distillation data; distill2 A/B rejected (-0.027).

**2026-11-07.** night10: fast top-p +1.5% (confirmed); night11: fused res_mix no gain (off); distill3 trained (val +0.04 per position); session bench (ref 0.603, distill3 0.612, distill2b 0.556, distill2 0.542); held-out offline eval: distill3 +0.046 / +0.055 / +0.060 (+5.4% expected tok/step); distill4 (2.3x data) same offline, online 0.643 vs ref 0.583-0.603 (+4.6% tok/step); night14 24-probe session benches: ref 0.572, distill3 0.594, distill4 0.571 pooled; paired per probe distill3 +0.054 +- 0.028, distill4 +0.051 +- 0.029: distill3 recommended.

## Commit list (oldest first)

| Commit | Date | What |
|---|---|---|
| `604bafadb` | 10-11 | WIP snapshot: CPU kernel work (threading patches, IQ2_XS kernel, pooled MUL_MAT_ID, fused SiTU / gate+up+SiTU, vectorized SiTU), CUDA persistent IQ2_S/IQ3_XXS mat-vec, warp top-k, DEFER/HOST_SRC flags, split MUL_MAT_ID + HMM GPU share, deferred splits, 5-op res_mix, DFlash taps |
| `f3f84b015` | 10-11 | adaptive MoE expert skip (`LLAMA_MOE_SKIP`) |
| `4d85dba96` | 10-11 | TEMP: expert input projection hook, expert input dump |
| `4f33592bc` | 10-11 | copy a deferred split's inputs in graph order |
| `6c1d05947` | 10-11 | TEMP: expert prefetch into L3 during attention, router prediction dump |
| `dbe3f5f18` | 10-12 | multi-token IQ2_XS dot for MUL_MAT_ID |
| `a3ea61f0e` | 10-12 | union-aware expert skip |
| `4f867184b` | 10-12 | GPU expert share for verify batches (batch take rule, expert-major mat-vec) |
| `367f377c6` | 10-12 | per-layer GPU share for layer splits, VRAM expert cache |
| `42d85672b` | 10-25 | stochastic verification with sampled DFlash drafts; faster exact top-p (bucket bound) |
| `c747ccd50` | 10-25 | range split of the experts for prompt batches |
| `fa0afd606` | 10-26 | cache usage scores (LFU); deferred splits per backend; second-GPU prompt share |
| `c1cfdfd7f` | 10-26 | split params in op_params 4-7 |
| `eb1bf7e81` | 10-31 | fused MoE range op; second GPU takes the top experts, prefetched at startup |
| `dd993055e` | 10-31 | fused MoE range op on the main stream; TEMP pool log |
| `a8405846e` | 10-31 | TEMP: tensor source (`LLAMA_TENSOR_SOURCE`) |
| `299c3ecbc` | 10-31 | MXFP4 in the expert-major MoE mat-vec |
| `8a1e0e5a5` | 10-31 | TEMP: `LLAMA_QUANT_ONLY` for llama-quantize |
| `aa0525637` | 10-31 | map tensor-source files without prefetch |
| `b4f56553a` | 11-01 | tensor source only for models of the source's architecture |
| `6bf6407e6` | 11-01 | second tensor source |
| `c7fb4f4cd` | 11-02 | per-layer-range skip threshold |
| `08335ee2f` | 11-02 | TEMP: per-scheduler sched timing; CUDA op profiler on two GPUs |
| `cf1f96157` | 11-02 | persistent IQ2_S/IQ3_XXS mat-vec for 1-4 columns |
| `8a6c6d4cf` | 11-02 | second-GPU MoE share at 1-4 tokens (off); K3 tests; TEMP decode counters |
| `630e6f4a0` | 11-02 | TEMP: cross-layer routing prediction dump |
| `12f3d199e` | 11-03 | WIP: cross-layer expert prefetch (plan/copy ops, staged experts) |
| `56801b5b3` | 11-03 | prefetch: in-order abortable copy, launch-time stop epoch |
| `7d3132042` | 11-03 | prefetch: slim async copy after the attention split, single launch, split timing |
| `78c934053` | 11-03 | disk prompt cache (`--cache-disk`) |
| `867aeb6b9` | 11-03 | disk cache: write after 2 s idle, no draft copy per checkpoint |
| `297e37d9f` | 11-04 | MUL_MAT_ID setup off thread 0 |
| `10b684e22` | 11-04 | per-position skip, KLD by ubatch position, K3 MLA flash-attention tests, expert norm dump |
| `5c0c99c2a` | 11-04 | importance-aware expert skip (opt-in) |
| `8a44b3ad9` | 11-04 | disk cache stores 2048+ token conversations only |
| `ec5c26a52` | 11-04 | TEMP: draft-length knobs, step log |
| `934620b52` | 11-04 | TEMP: `LLAMA_DFLASH_CAUSAL` |
| `505338506` | 11-04 | TEMP: `LLAMA_DFLASH_FULL_BLOCK` |
| `015994ebc` | 11-05 | DSpark: stochastic verification, confidence logging, p_min modes |
| `3adf5d263` | 11-05 | full-block outputs fix; `LLAMA_SPEC_DRAFT_CTX` |
| `d3f8c075c` | 11-05 | TEMP: oracle expert skip |
| `c52898b5f` | 11-05 | CUDA graph key includes node count and first/last node shapes |
| `0099fa178` | 11-05 | TEMP: CUDA graph stats |
| `bc665ba0b` | 11-05 | TEMP: `llama-dflash-dump` |
| `88a810398` | 11-05 | TEMP: padded verify batches |
| `8092e89fb` | 11-05 | TEMP: step-log phase timing, visible pad / graph-stats logs |
| `0c587ea0d` | 11-06 | fast exact top-p |
| `c9666450c` | 11-06 | fused residual mix op (off by default) |
