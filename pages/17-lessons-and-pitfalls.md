# Lessons and pitfalls

[Back to index](../README.md) · Related: [dead ends](16-dead-ends.md), [tools and methodology](15-tools-and-methodology.md)

## Broad lessons

1. **Measure the machine before believing a number.** The first 3x came from a build flag (no AVX2), not an algorithm (HANDOFF §2). Later "results" turned out to be outside load, a PCIe link at x8, or a hidden DRAM thermal throttle.
2. **This is a streaming-read box, not a compute box.** The most compressed weights (IQ2_XS) were the slowest to decode on AVX2; per-token time is bytes x seconds-per-byte, and both terms move (HANDOFF §9).
3. **Dense first, then experts.** Dense weights were 81% of per-token traffic; moving them to the GPUs (lossy requant) took 1.69 -> 4.19 tok/s. Then every lever was about reading fewer expert bytes or overlapping the reads.
4. **Once at the DRAM wall, moving traffic does not help; only fewer bytes or fewer decode ops do** (L3 prefetch, KB2, prefetch tuning all ~neutral; skip, cache, batch sharing worked) (HANDOFF 26.3).
5. **K3 is chaotic in KLD terms:** compare against a floor run, look at median and p99, and accept ~0.5% PPL wobble from exact CPU/GPU reshuffles (HANDOFF §24, 31.zzzz).
6. **Quality per speed is a frontier:** skip tau, per-layer skip and native-expert layers all cost ~0.008 KLD per tok/s, so quality at fixed speed needs a faster step (HANDOFF 31.z).
7. **A better target improves the draft:** dense-v2 (closer to native) raised acceptance 0.69 -> 0.76 and paid for its VRAM (HANDOFF 31.z).
8. **Train what you score:** the distilled draft only improved once the loss was the expected acceptance of the actual top-10 / tau 0.7 proposal, and only on real-use data (SESSION3).
9. **Systems effects hide in averages:** the confidence cutoff lost because each batch-size change cost ~10 ms, visible only in step-to-step transition times (SESSION3).
10. **Estimates were usually optimistic** (table below).

## Estimates vs measurements

| Lever | Estimate before | Measured | Source |
|---|---|---|---|
| Software plan's ceiling | 8.82 tok/s at 125.8 GB/s ingest (1 GPU) | IQ2_XS decoded at 73-88 GB/s (compute-bound): premise false; yet 10+ tok/s reached later with 2 GPUs, speculation and lossy skip | SOFTWARE-PLAN-9TPS, HANDOFF §14 |
| Expert cache on K3 (pre-measurement research) | "buys nothing at all" (Quantile Balancing) | 18 slots/layer: 306 -> 243 ms/step | LOCAL-K3 §6, HANDOFF §30 |
| Speculative decoding (pre-measurement research) | "a measured loss on K3, don't build it" | ~2x: a run where every draft failed (effectively plain decoding) gave 4.77 tok/s vs ~9.8 with the draft | LOCAL-K3 §6, SESSION3 |
| CUDA IQ2_S/IQ3_XXS kernels | ~5.0 tok/s (+~14 ms) | 4.65 -> 4.78 | HANDOFF §19, 20.2 |
| GPU expert share | ~5.4 tok/s | 5.58 | HANDOFF 20.6-20.7 |
| L3 prefetch | +4-8% | slower | HANDOFF 26.2-26.3 |
| Second-GPU decode share | -15 ms/step | +7 ms/step | HANDOFF 31.zz |
| Cross-layer prefetch | +6-11% | -1.3% to -1.5% ms/step | HANDOFF 31.zzz, 31.z7 |
| CPU split efficiency | up to +15% (-40 ms/step) | -4 ms/step (-1.6%) (the 92 vs 116 GB/s premise was unfair) | HANDOFF 31.z7, 31.z9 |
| Draft-position skip | +5-10% | worse than uniform tau | HANDOFF 31.z10 |
| Importance-aware skip | +3-8% | 1-5% lower KLD, inconclusive | HANDOFF 31.z10 |
| Output-norm skip predictor | +5-10% | oracle ceiling ~1-2% | HANDOFF 31.z16 |
| Confidence-based draft length | +3-8%, then +3.4% (sim) | -1.6% | HANDOFF 31.z11-z17, SESSION3 |
| Draft distillation | +10-25%, then +4-13% | generic data 0 or negative; user sessions +5.4% tok/step offline, +4.6% online (8 probes) | SESSION3 |
| Fused res_mix | -3..-5 ms/step | 0 | SESSION3 Result 12 |
| 8-CCD CPU | ~+15% tok/s | per-CCD test: +10-15% bandwidth at most | HANDOFF 31.z6-31.z7 |

## Bugs found and fixed

| Bug | Symptom | Fix |
|---|---|---|
| `llama.cpp/build` compiled with `GGML_NATIVE=OFF` and every SIMD flag off | 0.33 tok/s; 0 ymm instructions | native build (`build-avx2`) |
| Microbenchmark linked against the non-AVX library | "IQ2_XS is 7.9 GB/s" | verify with `ldd` and `objdump \| grep -c ymm` |
| Row-partitioned ggml CPU ops at batch 1 (get_rows, unary, concat) | 23.8% of CPU time in barriers | column splits (HANDOFF 17.2) |
| Mainline IQ2_XS AVX2 index extraction via stack store/reload | store-to-load forwarding stalls on Zen 2 | read indices from memory (1.33x) |
| Deferred split copied its inputs after the next split's | illegal memory access in flash attention (allocator reused the host inputs) | `4f33592bc`: copy in graph order, defer only the launch |
| Split params in `op_params[3]` shared with MUL_MAT_ID precision | e1 == 10 read as `GGML_PREC_F32`: CUDA refused the op | `c1cfdfd7f`: op_params 4-7 |
| Tensor-source files mapped with MAP_POPULATE | each override run read the 1.4 TB native model and evicted dense-v1 (17 min load + 34 min first pass) | `aa0525637` |
| Tensor source applied to the draft too | same-named draft tensors with other shapes | `b4f56553a` |
| CUDA op profiler's global event pool | "invalid resource handle" on device 1 | `08335ee2f`: events per device |
| Main-stream nodes after the GPU-share fork in a deferred split | garbled output (allocator assumes graph order) | reverted the placement |
| Prefetch copy took its stop baseline at GPU start | ran all experts when the share overran (big GPU1 overruns) | launch-time epoch (`56801b5b3`) |
| Async prefetch with 24 x 256 threads of loads in flight | attention splits 2-2.5x slower | slim copy, 1 block x 64 threads (`7d3132042`) |
| Thread 0 zeroed skipped rows and quantization split per row | ~110 us serial setup per layer | `297e37d9f` |
| DSpark drafter without stochastic verification | exact-match verify, much lower acceptance under sampling | `015994ebc` |
| `LLAMA_DFLASH_FULL_BLOCK` requested outputs at every position | every draft decode failed ("no draft") | `3adf5d263` |
| CUDA graph key = first node's address | re-capture on every 2 <-> 3-token batch switch (~20 ms) | `c52898b5f` |
| Full-attention draft KV sized for the 96k target context | cudaMalloc OOM 1920 MiB on GPU1 | `LLAMA_SPEC_DRAFT_CTX` (`3adf5d263`) |
| `GGML_CUDA_GRAPH_STATS` and pad logs at INFO | invisible (the server log drops ggml/llama INFO) | WARN (`8092e89fb`) |
| Fork defaulted the DFlash draft to bidirectional in-block attention | ~3% lower tok/step; full block wrecked (0.20 acceptance) | `LLAMA_DFLASH_CAUSAL=1` (`934620b52`); GGUF key pending |
| 8-bit KV crashed the server (2026-11-01) | CUDA illegal memory access after ~2 requests | gone by 2026-11-04; cause never identified (the note's guess, the deferred-split fix, predates the crash) |
| Disk prompt cache stored the whole draft KV per checkpoint | ~15 GB per 90k conversation | slim checkpoints (`867aeb6b9`) |
| Disk cache filled by short chats | ~1.4 GB each, budget full | store 2048+ token conversations only (`8a44b3ad9`) |
| `/completion` `content` drops special tokens | training text lost the channel markers | `return_tokens` + `/detokenize` |
| `gguf_tensors.py` KV skipper; `iq2xs_entropy.py` totals | wrong sizes / double-counted entropy | noted (field numbers right) |

## Process mistakes (do not repeat)

- **Queue quoting:** queue lines like `'LLAMA_TENSOR_SOURCE2_RE=...' ./kld_overlay.sh ...` make bash run the quoted word as a command ("command not found"): 5 study runs never ran, the runner idled 7.5 h, and the status report said "still running". Quote only the value: `VAR='value' cmd`. Every wait loop must also exit on "job gone / runner idle / error in log", and job status must be checked with `ps` and logs before reporting (memory note `verify-background-jobs`; HANDOFF 31.z).
- **`pgrep -f` / `pkill -f` match their own command line:** a wait loop hung ~20 min, a `pkill` killed the calling shell, and `while pgrep -f llama-dflash-dump` never ended. Use PIDs, pid files or `pgrep -x <comm>` (HANDOFF §30; SESSION3).
- **`LD_LIBRARY_PATH=<dir>` replaces the path:** CUDA fell back to the stub driver and the baseline server died, while the check looked at the other server. Prepend instead, and verify the run that should differ via `/proc/<pid>/maps` / `environ` (memory note).
- **Stale binaries:** after changing server code, build the `llama-server` target explicitly; the night6 "padded" A/B ran an unpadded server because only another target was rebuilt (it relinked `libllama.so` but not the server library). An old `llama-completion` segfaulted against a new `libllama`. Check binary mtimes before an A/B (SESSION3).
- **Plain `find` / `grep` were broken wrapper functions in the agent shell** and returned nothing: three empty `find`s led to telling the user the October draft logs were "gone" (they were in `/data2/k3eval/dflash`). Use `command find` / `command grep`; `/` is a tmpfs there, so `find / -xdev` never reaches real disks (memory note `shell-find-grep-wrappers`).
- **Greedy benchmarks** overstated speed for two weeks (acceptance 0.71-0.78 vs 0.51-0.64 in real chats) (HANDOFF §30).
- **Throttled runs:** a long run's later requests were throttled and briefly blamed on prompt processing; the 4th back-to-back A/B of a night throttled although each run started with a cool DIMM sensor; a heavy rebuild during decode triggered a throttle. Gate runs on temperature + idle minutes and check step-log medians (HANDOFF 31.x; SESSION3).
- **Spinning host thread in a DMA test** made llama-bench + DMA show 3.66 tok/s and the CPU kernel lose 25-30%; use blocking sync (HANDOFF 20.6).
- **Short runs mislead:** schedutil needs ~0.5 s of load to ramp clocks; one 512-token technical generation is ±7%; two 10-request runs differ 1-2% (HANDOFF 26.2; SESSION3).
- **Unfair comparisons:** the "92 vs 116 GB/s" CPU efficiency gap compared a 3-token batch against the single-token kernel (HANDOFF 31.z9). The "12 native layers fit RAM" estimate ignored ~86 GB of process memory (HANDOFF 31.z).
- **Assumed ideals:** `SOFTWARE-PLAN-9TPS.md` assumed expert bytes stream at DRAM speed; the plan itself flagged it as the least-measured assumption, and the benchmark killed it. Label outputs of unverified models (`tokps_model.py`) as such (HANDOFF §10).
- **Tools in the wrong tree:** `--expert-trace` existed in the ik_llama.cpp checkouts, not in the K3-capable tree (HANDOFF §10).
- **Validation sets from the training sessions** overstated the distilled draft (+0.04 val vs +0.009 online at first); a random holdout picked unsuitable sessions twice (SESSION3).
- **Running queue scripts cannot be edited:** night2 still ran an n_max 3 config that had just OOMed (~10 min lost). `speed_test.sh` started the next server immediately and overwrote a crashed run's log; per-run log copies since night2. Syntax-check (`-fsyntax-only` with `compile_commands.json` flags) before arming an overnight queue that rebuilds (SESSION3).

## Environment gotchas

- `llama-cli -c 256` segfaults on K3 (cross-layer residual attention needs a real context); `llama-bench` has no `-c` (HANDOFF §5.1).
- `-ngl 36` + `-p 512` failed to create a context (CUDA compute buffer at ub 512): configs that fill VRAM need small ubatches (HANDOFF 18.2).
- llama-server can ignore SIGTERM and keep ~21 GB of VRAM; teardown with ~800 GB HMM-mapped takes ~2 min, a crashed server up to ~16 min (zombie in `exit_mmap`). Check `nvidia-smi --query-compute-apps` before restarting (HANDOFF §28, §29, 31.y; SESSION3).
- The agent sandbox (bwrap) must bind `/data2` and `/sys`, or perf cannot resolve events and libgomp has no topology for `OMP_PLACES=cores` (HANDOFF 17.4).
- The default `python3` on PATH had a broken numpy (`libz.so.1`); use the system interpreter (HANDOFF §13).
- `ulimit -l` is 8 MB (not enforced by the NVIDIA driver for `cudaHostRegister`) (HANDOFF 20.6).
- After a reboot the page cache is cold: read the model once (`xargs -P 4 dd`, ~2-3 min for 766 GB) (HANDOFF §30).
- GPU1 drives the display (~0.5-1 GB of VRAM) and hosts the draft and output head, so it is the tighter GPU.
