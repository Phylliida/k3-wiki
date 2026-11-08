# Tools, scripts and measurement methodology

[Back to index](../README.md) · Related: [lessons and pitfalls](17-lessons-and-pitfalls.md), [memory bandwidth and thermals](14-memory-bandwidth-thermals.md), [expert skipping and quality](06-expert-skipping-and-quality.md) (KLD harnesses)

Scripts live in `k3-research/` (and `k3-research/distill/`); results and logs in `/data2/k3eval/`; code on branch `k3-decode-speed` of the `llama.cpp-k3` worktree.

## Benchmark rules

1. **Sampled decoding with Moonshot's settings** (temperature 1.0, top-p 0.95, no top-k / min-p), never greedy (user rule since 2026-10-12). Report acceptance with every tok/s number.
2. **Compare ms per verify step** (or step-log statistics) when acceptance differs between runs: tok/s moves +-5% with acceptance alone.
3. **Interleave A/B runs** (base, new, base, new) to cancel drift.
4. **Check machine state first:** fans Full, both GPUs at x16, page cache warm, no leftover server, no outside load (`uptime`), no heavy builds during measurements; `speed_test.sh` waits for DIMM A-D <= 62 C + 3 idle minutes and flags throttled runs.
5. **Lossy changes need a KLD number** from a harness with a floor run; lossless changes need `test-backend-ops` agreement (CUDA vs CPU) or a bit-identity check.
6. Do not use `-v` for speed numbers (per-launch logging on the critical path: 56 vs 14 us per CUDA split); do not judge CPU/GPU overlap with the TEMP op profilers on (they serialize).

## Noise levels (what differences mean something)

| Quantity | Noise | Source |
|---|---|---|
| `speed_test.sh` tok/s, same config | 1-2% between two 10-request runs; acceptance +-0.03; one 512-token technical generation +-7% | SESSION3 lessons |
| ms/step, same config | repeats within ~1 ms in one session (274 / 275); 264-275 ms across runs of identical configs on another day | HANDOFF 31.z, 31.zz |
| Session bench (8 probes) | two identical ref runs: 0.603 vs 0.583 per draft | SESSION3 Result 14 |
| Step-log acceptance (~900-1000 steps) | +-0.011 (pos1), +-0.014 (pos2) binomial s.e. | SESSION3 Result 3 |
| KLD, summation-order changes | ~0.016 (batch-1 decode); last-bit perturbation 0.0706 in the dense-v1 skip study | HANDOFF 20.3, §24 |
| KLD, deterministic `kld_pos` harness | runs reproduce to all digits; a numerics change (q8 KV) moved KLD by ~0.002 | HANDOFF 31.z10 |
| PPL of one 2k chunk under CPU/GPU reshuffles | +0.5-0.6% (8.9019 -> 8.9498 / 8.9547) for exact changes | HANDOFF 31.zz, 31.zzzz |
| Greedy text | diverges after ~60 tokens on any float-order change | HANDOFF 20.2 |
| Machine load | 5-15% drift with load average; bandwidth 119.6 vs 92.6 GB/s at load 3.0 vs 7.4 | HANDOFF §9, K3-IDEAL-BOUND §13 |

## Main benchmark scripts

| Script | What it does |
|---|---|
| `speed_test.sh <name> [VAR=..]` | start `serve_k3.sh fast2` with overrides; decode bench (5 prompts x 2 seeds, `dflash_bench.py`), 2k-token technical summary (`long_ctx.py`, 512 tokens), 7.2k-token prompt (`pp_server.py`, skip with `SKIP_PP=1`), VRAM trace (`speed-<name>.vram`), throttle line from the step log; appends to `/data2/k3eval/speed-tests.txt` |
| `dflash_bench.py` / `dflash_summary.py` | 5 fixed chat prompts (code, code2, math, chat, rust), recommended sampling, `--seed`, `--greedy`; summary prints per-prompt tok/s, acceptance, tok/step, ms/step (= predicted_ms / steps) |
| `run_dflash_bench.sh`, `run_dflash_gpu_sweep.sh`, `night_sweep.sh name:KB:CACHE:NMAX`, `seq_baseline2.sh`, `sampler_cost.sh` | earlier server sweeps (single GPU, two GPUs, sampler cost) |
| `long_run.sh [port]` | one-hour thermal soak with BMC temperature log |
| `long_ctx.py` | decode at long context (llama.cpp docs/sources as context, "summarize") |
| `pp_server.py`, `pp_bench.sh` | prompt speed of a running server (unique prefix per request) / via llama-bench |
| `smoke_chat.py` | one chat request with Moonshot's sampling, prints timings |
| `pf_diag.sh`, `pf_summary.sh`, `cpu_prof.sh`, `cpu_ab.sh`, `kb2_prof.sh` | one server per config with sched timing + CPU counters, per-layer summaries; library A/B via `LD_LIBRARY_PATH` prepend |
| `pcache_test.py`, `pcache_k3.py` | disk prompt cache tests (small hybrid model on CPU; K3 first/resume) |
| `thermostat.sh` | DRAM-throttle guard (SIGSTOP/SIGCONT the heavy process) |
| `distill/session_bench.py` | real-use acceptance benchmark: replays held-out session windows (`session-bench.txt`) |
| `/data2/k3eval/night*.sh`, `draft_ab.sh` | unattended queues (chained with `while ps ...; sleep`) |

## Step log (`LLAMA_SPEC_STEP_LOG=<file>`, commits `ec5c26a52`, `8092e89fb`)

One line per verify step: `t_us n_tokens n_draft n_acc replay conf... | pre draft decode sample` (phase times in us). Analysis: `draft_len.py` (acceptance by confidence, 1- vs 2-draft step time per context bucket, simulated cutoff gain + bootstrap), `step_phases.py` (median step and phases by transition), scratch `trans.py` / `sel.py`. With thousands of steps it is far less noisy than tok/s, and it found the batch-size-switch penalty that a short A/B hid.

## KLD harnesses

`run_kld_skip.sh`, `run_kld_batch.sh` + `kld_paired.py` (per-chunk paired differences vs a floor run), `kld_dense.sh` / `kld_overlay.sh` / `kld_study.sh` / `kld_runner.sh` (queue in `/data2/k3eval/kld-queue.txt`) / `kld_table.py`, `kld_pp.sh` (prompt-batch paths), `kld_pos.sh` (KLD by ubatch position; `perplexity.cpp` prints it for 2-8 token batches), `kld_oracle.sh`. Details and references on the [expert skipping and quality](06-expert-skipping-and-quality.md) page. `make_overlay.sh` builds requantized overlays; `verify_experts.py` hashes matching tensors of two split sets; `warm_tensors.py` reads tensor ranges into the page cache. Note: `--chunks` is ignored in llama-perplexity's KLD mode.

## Profiling and TEMP instrumentation (in the fork; to remove before upstreaming)

| Knob / tool | What it shows |
|---|---|
| `GGML_CPU_OP_PROFILE=<period>` | thread-0 wall time per CPU op incl. barrier; later decode counters (slots, skipped, GPU-taken, CPU-read experts per gate/up call, token histogram, per-thread setup/busy/tail, staged experts) |
| `GGML_SCHED_TIMING=1` | per-scheduler split input-copy / compute times, wall time inside and between compute calls (to stderr since `08335ee2f`) |
| `GGML_CUDA_OP_PROFILE=1` (+ `GGML_CUDA_DISABLE_GRAPHS=1`) | per-op GPU times (events per device since `08335ee2f`); syncs every graph, so absolute step times are inflated (393 ms/step) |
| `GGML_CUDA_SPLIT_TIMING=1` | timestamp kernels around each split's graph (found the 2-2.5x slower attention under the async prefetch copy) |
| `GGML_CUDA_PF_TIMING=1`, `GGML_CUDA_PF_AFTER_ATTN`, `GGML_CUDA_PF_PRIO_LOW`, `LLAMA_MOE_GPU_PREFETCH_MODE` | prefetch copy diagnostics |
| `GGML_CUDA_POOL_LOG=1` | VMM pool growth |
| `GGML_CUDA_GRAPH_STATS=1` | CUDA graph replays / captures / direct runs |
| `GGML_SCHED_DEBUG=2` | split listing (186 splits/token in dense-v1) |
| `GGML_MMID_DUMP=<file>`, `GGML_MMID_DUMP_X=<file>` | routing dump (layer, ids) / expert input dump |
| `GGML_MOE_NORM_DUMP=<file>` | per (layer, expert) output norms |
| `LLAMA_K3_PRED_DUMP`, `LLAMA_K3_PRED_AHEAD` | router predictions (same layer / k layers ahead) |
| `LLAMA_SPEC_RS_STATS=<file>` | per-position acceptance estimates for stochastic verification |
| `LLAMA_SPEC_CUT_RAND=<p>` | drop drafts after the first at random (lossless) to measure step time |
| `LLAMA_K3_XPROJ`, `LLAMA_MOE_PREFETCH`, `LLAMA_MOE_DEFER_LATE` | negative experiments kept for reproducibility |
| `perf` via `nix-shell -p perf` | not on the host; `perf_event_paranoid=2` allows per-process user-space sampling; perf 7.1 needs `--all-user` (not `-e cpu-clock:u`) and `/sys` bound into the sandbox; `perf report --pid/--sort tid` did not work, per-thread splits via `perf script -F tid,ip,sym` + awk. Wrapper `profile_k3.sh` |
| `stack_sample.sh` + `stack_summary.py` | gdb-based sampler (ptrace_scope=1: the tracer must be the parent); validated on stories15M |
| `llama-eval-callback` + `tensor_diff.py` | tensor-level comparisons (KB2 and prefetch exactness checks, the chaos trace) |

## Microbenchmarks and simulators

| File | Purpose |
|---|---|
| `iq2xs_bench.cpp` | per-type batch-1 GEMV bandwidth at three working-set sizes (mainline or ik_llama.cpp) |
| `iq2xs_kernel_bench.c`, `iq2xs_kern2.c` + `iq2xs_kern2_variants.h` | IQ2_XS AVX2 kernel variants, bit-identity check, cycles/IPC via `perf_event_open` |
| `cpu_load.c` (`cpu_load`, `cpu_load_k3`) | CPU kernel / plain read for a fixed time (bandwidth and throttle checks; `kernN` = multi-token kernel) |
| `readbw*.c`, `l3bench.c` | streaming read and L3 re-read benchmarks |
| `mmvq_bench.cu` | CUDA IQ2_S/IQ3_XXS mat-vec harness |
| `mapped_bw.cu`, `hmm_vram.cu`, `dma_bg.cu`, `dual_bw.cu`, `prefetch_bench.cu`, `stage_bw.cu`, `devattr.cu` | zero-copy / HMM read rates, HMM page-table VRAM, DMA vs CPU interference, two-GPU DRAM wall, HMM mapping speed, expert staging copy rate and interference, device attributes |
| `dma_additivity.py`, `cpu_read_under_dma.py` | CPU vs GPU DMA contention |
| `routing_overlap.py`, `routing_lru.py`, `cache_sim.py`, `cache_policy_sim.py` | routing locality, LRU/cache simulations |
| `pred_recall.py`, `prefetch_sim.py`, `union_skip.py` | router prediction recall, cross-layer prefetch policy simulation, union-skip offline study |
| `expert_basis.py`, `expert_input_spectrum.py`, `fit_xproj.py` | shared-basis study |
| `rs_stats.py`, `rs_math_check.py`, `rs_cut_prelim.py` | stochastic verification statistics and checks |
| `norm_stats.py` | expert output-norm spread |
| `k3q2_tensors.json`, `gguf_tensors.py`, `gguf_probe.py` | tensor table of the 19 shards; GGUF dumper (has a KV-skipper bug: FLOAT32 vs BOOL sizes); HTTP-range header prober |
| `iq2xs_entropy.py` | entropy of IQ2_XS fields (its printed total double-counts; per-field numbers are right) |
| `tokps_model.py` | early throughput model: **superseded** (inputs predate the measured kernel rates) |

## Tests added to `test-backend-ops` (CUDA vs CPU)

| Test | Count (latest) | Covers |
|---|---|---|
| `MUL_MAT_ID_SPLIT` | 264 | split modes 1/2/3, skip, ranges (incl. offset ranges, chunk 3/5), 1-4 and 16-129 tokens, MXFP4 |
| `MUL_MAT_ID_CACHE` | 228 | cache plan over chained steps, LRU/LFU, scratch slots, mode-3 replay, staged/admit prefetch (`GGML_CUDA_PF_ABORT=2`) |
| `MUL_MAT_ID_SITU` | 24 | fused gate+up+SiTU, skip, 3-4 tokens |
| `MOE_FFN_RANGE` | 51 | fused range op, chunk/row caps, empty ranges, MXFP4 |
| `MUL_MAT_ID` | 932 | upstream + multi-token IQ2_XS cases |
| `RES_MIX` | 20 | fused residual mix |
| `FLASH_ATTN_EXT` K3 shapes | all q8_0 pass | 576/512, 96:1 GQA, 512-98304 KV, 1-4 tokens |
| `GATED_DELTA_NET` K3 shapes | 45/45 | KDA decode shape |
| MUL_MAT iq2_s / iq3_xxs 1-4 columns | 40/40 | multi-column persistent kernel |
| `SITU`, `TOP_K` | 16-17 / 525 | fused SiTU GLU, warp top-k |

A mutation test (routing weight dropped in the fused op's scatter) fails `MOE_FFN_RANGE` as it should.

## Data and logs (`/data2/k3eval/`)

`speed-tests.txt` (every server A/B line since 2026-11-01), `kld-dense.txt` (precision study), `kld-pos.txt`, `kld-oracle.txt`, `session-bench.txt`, `heldout-eval.txt`, `pp-bench.txt`, `kld-pp.txt`, `prefetch-sweep.txt`, `skip-speed-sweep.txt`, `cpu_load_multi.txt`, `cpu-prof.txt`, `pf-diag.txt`, `steplog/` (real use `steps.txt`, per-A/B `ab*-*.txt`), `dflash/` (October bench jsonl and `rs-stats*.txt`), `routing/`, `pred/`, `basis/`, reference logits (`base-logits-*.bin`, `kldpos-ref-*.bin`), `imatrix-k3-calv3.gguf`, `thermostat.log`, `thermo-trace.txt`, per-run server logs and VRAM traces. Distillation data on `/mnt/items/distill/`.

## Plain decoding reference

There is no sampled no-draft baseline on the two-GPU setup in the notes. The closest data point: `dab_full` (2026-11-05), where every draft decode failed, ran as plain decoding at **4.77 tok/s, 209 ms/step** on the daily config, against ~9.8-10 tok/s with the draft. Single-GPU non-speculative numbers (Oct 11): 5.79 lossless, 7.56 at plain skip tau 0.05.
