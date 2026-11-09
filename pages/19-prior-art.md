# Prior art, other projects and early research

[Back to index](../README.md) · Related: [hardware and model](01-hardware-model-constraints.md), [speculative decoding](09-speculative-decoding.md), [lessons](17-lessons-and-pitfalls.md)

Copies of the projects' READMEs and reports were saved in `k3-research/` on 2026-10-10 (the day the work started). Speeds below are the projects' own claims as written in those files, on their hardware; none were reproduced here.

## Other ways people ran Kimi-K3 locally

| Project (file in `k3-research/`) | Approach | Hardware | K3 speed claimed | What we took from it |
|---|---|---|---|---|
| colibri (`JustVugg_colibri-README.md`, `colibri-kimi_k3.md`) | pure C; dense in RAM, routed experts streamed from disk through an LRU; MXFP4 matmul directly on checkpoint bytes, int8 activations per 32; `K3_TOPP` trims the low-weight tail of the top-16; chunked prefill dedups experts (2.7x at C=32) | any CPU, 32/64 GB RAM + 1.56 TB disk | ~9.4 s/token decode (experts at 6.3 GB/s); prefill ~5.3 -> 2.0 s/token | Quantile Balancing makes routing flat ("expect the expert tier to be bandwidth-bound"); AttnRes mix logits have small margins, so judge quantization on real-text logits; top-p trimming = our expert skip idea |
| kimi-k3-in-c (`FareedKhan-dev_kimi-k3-in-c-README.md`, `kimi-k3-in-c-report.md`) | portable C99, CPU only, byte-identical output at every memory budget; AVX2 MXFP4 kernel; trunk pinning | 2x EPYC 7763, 228 GB RAM, NVMe | 19.21-32.69 s/token by memory cap (8-224 GB); README table 5.59 s/token with the trunk resident (other machine) | expert LRU useless below ~36 GB (25.83 GB read per token, flat); trunk int8 error 0.961% vs int4 17.383%: keep dense precision high (matches our finding that ~70% of dense-v1's KLD gap was the dense requant) |
| WARP / WASTE (`sqliteai_warp-README.md`) | C engine, trunk in RAM, experts streamed from SSD, 3-bit residual-VQ experts; lookahead router prefetch | 64 GB MacBook Pro (M5 Pro) | 0.45-0.62 tok/s; experts per token 16 -> 0.59, 12 -> 0.70 (KL 0.007), 8 -> 0.89 (KL 0.037), 4 -> 1.06 (KL 0.118) | top-k vs KL table parallels our skip results; caches larger than real memory get slower; its speculative decoding measured 0.71-0.75x (cited by early research as "don't build it") |
| Moonshine (`outerreaches_moonshine-README.md`) | ROCm engine; Q8 static tier in device memory, MXFP4 experts from NVMe via O_DIRECT + io_uring; per-layer LRU of 30 slots; exact prefix-state reuse | Ryzen AI Max+ 395, 128 GB unified | prefill 4.145-8.126 tok/s; 128k chat prompt/decode 0.438 / 0.498 tok/s | exact state checkpoints (KDA state + MLA KV) for resuming conversations: same idea as our disk prompt cache |
| deltafin (`gavamedia_deltafin-README.md`) | Rust binary, full unpruned K3, Inferact DSpark drafter with K3 verifying every token; int8 resident spine | M1 Max laptop (reference) | 0.2901 tok/s (3.447 s/token) | DSpark drafts exist; we tested RadixArk's (not better than our causal DFlash) |
| AirLLM (`airllm-README.md`) | layer-at-a-time streaming onto one GPU, per-expert streaming for K3 | one GPU (RTX 6000 Ada, 3.72 GB VRAM) | none (disk-bound) | nothing reused |
| kimi-k3-mlx (`PipeNetwork_kimi-k3-mlx-README.md`) | MLX port, REAP expert pruning, AWQ on the shared latent | M3 Ultra 512 GB | pruned REAP73 / REAP80 ~5.5 tok/s; ~73 tok/s prompt | MXFP4 should be passed through bit-exact; AWQ is cheap because all experts share the 3584-wide latent (not used: no requants); 2-bit experts alone gave PPL 17.03 there |
| KTransformers (`kvcache-ai_ktransformers-README.md`) | CPU/GPU hybrid MoE framework (hot experts on GPU, AVX2/AVX-512/AMX kernels) | Xeon/EPYC + GPUs | no K3 support in the README | conceptually closest to our CPU+GPU split; no code reused |
| local-kimi (`RightNow-AI_local-kimi-README.md`) | protocol proxy; serves Kimi-Linear-48B, not K3 | L40S | none for K3 | GEMM-shaped kernels used for single-token decode ran at 4-10% of bandwidth: check decode kernels are real GEMVs |
| gb10-vllm (`ciprianveg_gb10-vllm-README.md`) | vLLM on a 16x DGX Spark (GB10) cluster, DSpark with adaptive draft depth | 16x GB10, 100 GbE | none in the README | - |
| DGX Spark port (`spark_project_report.md`) | PyTorch/Triton runtime, mmap'd MXFP4 experts, int8 spine, slot arena | 1x DGX Spark | 9.62 s/token warm decode after fixes; speculative decoding 0.71x | GPU demand-faulting page-cache memory is ~728x slower than resident reads: never let a GPU fault experts in (we map experts at startup and wait); decode-time arena hit rate 0.0% (no cross-token locality there) |
| Moonshot model card (`moonshot-k3-README.md`) | official | 8x B300 class | - | architecture facts, sampling settings (temp 1.0, top-p 0.95; top-p 1.0 agentic), MXFP4 QAT |
| vLLM blogs (`vllmblog-2026-07-22-...`, `vllmblog-2026-07-27-k3.txt`) | KDA-aware prefix caching, fused AttnRes, fused KDA decode, DSpark | GB300 NVL72 | 111 tok/s per user (TP8), 118 (TP16); with DSpark 331 / 370 tok/s; DSpark ~4.73 accepted per step on code, ~2.61 on creative writing (7 drafts) | prompt-end / interval state checkpoints for hybrid-model prefix caching |
| Unsloth docs (`unsloth_docs.txt`) | GGUFs (our UD-Q2_K_XL and UD-Q4_K_XL) | - | ~20 tok/s on B200s | the quant ladder and KLD table ([quantization](07-quantization-dense-kv.md)) |

Community datapoints collected in `KIMI-K3-LOCAL-REPORT.md` (2026-10-10, their labels): 8x B300 vLLM ~92 tok/s; 16x GB10 + DSpark 21.7-25.4 tok/s; 8x A100 llama.cpp UD-IQ1_S ~9 tok/s; 2x RTX 5090 + 768 GB DDR5 2-bit 3.85 tok/s (second-hand); M3 Ultra 512 GB mixed 389 GiB build 3.36 tok/s decode / 13.26 prompt; streaming engines 0.1-1 tok/s. AtomicChat's CPU-only run on an EPYC 9554P (64 cores, 1.5 TB RAM): IQ1_S tg128 1.82 / pp512 8.91, Q3_K_S tg128 1.70 / pp512 10.91 tok/s (`atomic.md`).

**Architecture disagreements between sources** (from the saved files): every source gives ~2.8 T total / 104 B active parameters; the card says "160K" vocab where the real figure is 163,840; whether `A_log` is per-head (3 sources, byte-level evidence) or per-channel (MLX port); how MLA's 64 rope dims are described (cached but not rotated; K is 576 wide either way); the GGUF probably stores the 2 shared experts as one fused tensor.

## Upstream llama.cpp support (LOCAL-K3 §3)

K3 text support landed as `ad1de39e0` (PR #26185, 2026-08-15, a delta against Kimi-Linear-48B: cross-layer residual attention via `ggml_dsv4_hc_pre`, latent MoE, SiTU, MLA output gate, full-rank KDA gate); `LLAMA_MAX_EXPERTS` was raised to 1024 (#26192); recurrent-state rollback for speculative decoding came with #28466 (2026-09-07). Text only (the MoonViT vision tower is in Unsloth's fork). Everything on this wiki builds on that support.

## Other Kimi-K3 drafts on Hugging Face (from `dflash-distill-research.md`, 2026-11-04)

| Repo | Type and backbone | Drafted | Reported acceptance length | Runs on our fork? |
|---|---|---|---|---|
| modal-labs/Kimi-K3-DFlash (ours) | DFlash, Qwen3 GQA, 6 layers, 2.6 B | 15 (block 16) | block 8 greedy: gsm8k 5.91, mt-bench 4.20 | yes |
| Inferact/Kimi-K3-DSpark | DSpark, MLA, 5 layers | 7 | t=1.0: mean of 14 benchmarks 3.73 | no (MLA) |
| RadixArk/Kimi-K3-DSpark | DSpark, Qwen3 GQA, 5 layers, 2.25 B, confidence head | 7 | GSM8K 5.42, HumanEval 5.51, MT-Bench 3.93 | yes (tested: not better) |
| RedHatAI/Kimi-K3-speculator.dspark | DSpark, Qwen3 GQA, 5 layers | 8 | HumanEval 4.96, writing 3.14 | likely (not tested) |
| lightseekorg/kimi-k3-dspark, kimi-k3-dflash2, eagle3(.1)-mla | MLA backbones | 3-7 | e.g. dflash2 t=1.0 GSM8K 5.90, MT-Bench 3.59 | no |
| Inferact/Kimi-K3-DSpark-Block5, slippedJim/Kimi-K3-Draft, ZhuoBohang-AMD/Kimi-K3-Draft-mxfp4 | MLA DSpark variants | 5-7 | 3.31-3.88 means | no |

Per-position cumulative acceptance of Inferact's Block5 draft at t=1.0 (`benchmark_results.json`), e.g. MT-Bench .697 / .442 / .266, GSM8K .929 / .829 / .717, SWE-bench Pro .738 / .500 / .327 for positions 1-3, puts our draft (~0.86 at position 1 and ~0.59 cumulative at position 2 on the bench) in the same range.

## Early research documents (2026-10-10) and how they held up

| Document | Main claims | What happened |
|---|---|---|
| `REPORT.md` (engines beyond llama.cpp) | K3 ships MXFP4 experts (1.56 TB), so there is no lossless smaller K3; GPU serving stacks need 8x B300-class; disk-streaming CPU engines run full K3 at 0.1-0.6 tok/s | correct framing; we took the lossy GGUF route |
| `REPORT-GGUF.md` (llama.cpp + Unsloth route) | quant ladder verified from shard headers; UD-Q2_K_XL experts are IQ2_XS; three tiny quants use undefined ggml types | correct; used to pick UD-Q2_K_XL |
| `LOCAL-K3.md` (consolidated findings) | "the GPU buys very little", "an expert cache buys nothing at all", "speculative decoding is a measured loss on K3 - don't build it" | **all three wrong for this setup**: GPU offload of dense was the biggest win; an 18-slot cache cut steps 306 -> 243 ms; DFlash roughly doubles speed. They were extrapolated from disk-streaming engines with tiny RAM |
| `KIMI-K3-LOCAL-REPORT.md` (community throughput) | community datapoints, bandwidth economics, 1-bit vs 2-bit | context only |
| `K3-IDEAL-BOUND.md` (first-principles bound) | binding constraint = bytes per token; GPU DMA reads are additive (measured); entropy coding dead (measured); speculation +55% at small blocks; 2x 3090 11.47 / 3x 13.22 tok/s ideal | DMA additivity, entropy and small-block speculation held; its 101-125.8 GB/s ingest assumption did not (IQ2_XS decoded at 73-88 GB/s, HANDOFF §14) |
| `SOFTWARE-PLAN-9TPS.md` (staged plan) | 7.5-8.5 tok/s honest target on one 3090, 8.82 ceiling at 125.8 GB/s ingest | **central assumption measured false** (HANDOFF §14), but its stage order (dense to GPU, expert ingest, cache, speculation) roughly matches what was built |

## Earlier work on this box with other models (August-September 2026)

Before K3 the same machine (then with 125 GB of RAM, later 378 GiB, and one 3090) ran **GLM-5.3-Flash** (321 B total / ~18 B active) with ik_llama.cpp (workspace notes `BENCHMARKS.md`, `HOT-EXPERT-*.md`, `PHASE4-*.md`, `PHASE5-*.md`, `moe-expert-cache-notes.md`):

- Hybrid CPU+GPU decode 9.01 tok/s with all experts on CPU, 10.16 with 5 MoE layers' experts on GPU; MTP +62% on pure CPU; `-t 64` (SMT) 3.66 vs 9.14 at `-t 32`.
- Static hot-expert caching was NO-GO (hot sets are session-specific); a dynamic LRU VRAM cache reached 8.09 -> 10.78 tok/s (+33%) after fixing pageable copies, per-layer sync bubbles and host-side classification. Lesson: hit rate is not speed; tune on wall clock.
- Cross-layer expert prediction measured no better than random on GLM (same-step overlap 0.0266 vs 0.0278 random). **On K3 it worked**: layer L+1's experts are predictable from layer L's stream (6.8-7.0 of 16 among the top-8 predicted; random 0.14), though the resulting prefetch gained only ~1.3% because PCIe bandwidth, not prediction, is the ceiling ([MoE page](05-moe-placement-cache-prefetch.md)).
- Hardware lessons that carried over: pinned PCIe host-to-device ~26.2 GB/s; SMT hurts; DIMM thermal throttling seen once with a bad DIMM layout (fixed by spacing the sticks); memory interleaving on this board needs complete channel pairs; session-to-session drift of +-10-20% means same-day controls.

## Hyperserve / Hyperkernel (read 2026-11-08, commit `7048c58`)

[Hyperstitioncc/Hyperkernel](https://github.com/Hyperstitioncc/Hyperkernel): a Rust host (MIT) plus precompiled "megakernels" for NVIDIA B300 (`sm_103a`), serving DeepSeek-V4.1-Flash on 4 GPUs. The kernel sources are not published: only cubins, SASS listings and layout metadata. The repo states no speed numbers. Read only; not built or run.

| Part | What it does | For K3 on 2x 3090 + CPU experts |
|---|---|---|
| One persistent kernel with command rings, sampling and drafting in-kernel | no launches or host round trips per step | little: our experts run on the CPU, so each step returns to the host anyway; draft + sample + other is ~6 ms of 251 |
| Experts in HBM as MXFP4, FP4 / 8-bit tensor-core MMA, TMA loads | tensor-core kernels (`UTCOMMA`, `UTCQMMA`, `UTMALDG`) | needs Blackwell / Hopper features; Ampere has neither TMA nor FP4 / FP8 MMA |
| 4-GPU tensor parallel over a GPU-to-GPU fabric | experts split along K, peer memory | no P2P between our GPUs; already a dead end (`LLAMA_MOE_GPU_KB2`) |
| Engram tables (~189 GiB on host), `idx.*` retrieval layers | DeepSeek-V4.1 model features | K3 has neither |
| Prefix tree over refcounted 256-row KV pages | several prompt branches share KV | our one real-use miss was a compacted prompt with no shared prefix |
| In-kernel draft: 3 MoE layers, block 6, rank-128 Markov logit-bias head (embed [vocab, 128], head [128, vocab]) | the Markov head makes each draft position depend on the previous draft token | **untested idea:** our DFlash draft decodes `[anchor, mask, mask]` in one pass, so position 2 never sees the sampled position 1. RadixArk's K3 DSpark had such a head but differs in everything else, so its result says nothing about the head alone |
