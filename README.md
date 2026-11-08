# k3-wiki: running Kimi-K3 fast on one machine

Everything we tried to make **Kimi-K3** (Moonshot's ~2.8 T-parameter / 104 B-active MoE: 93 layers, 896 routed experts per layer with top-16, KDA linear attention plus MLA) run fast locally, with every measurement and verdict, from 2026-10-10 to 2026-11-07.

**The machine:** one AMD EPYC 7542 (32 Zen 2 cores, AVX2, no AVX-512), ~1 TB of DDR4-2933 LRDIMM (8 channels, ~115-120 GB/s usable), 2x RTX 3090 (24 GB each, PCIe Gen4 x16, no P2P). **The model:** Unsloth's UD-Q2_K_XL expert weights (IQ2_XS, unchanged) in RAM (the model file is 765 GiB and stays in the page cache) with requantized dense tensors on the GPUs, a llama.cpp fork (branch `k3-decode-speed`), and the modal-labs DFlash speculative draft. **The constraints:** no hardware purchases; no requantization of the experts ("I like the quality as is"); lossy speedups only at equal or better KLD than the served config; speed always measured with sampled decoding at Moonshot's recommended settings (temperature 1.0, top-p 0.95), never greedy.

Result: **0.33 tok/s -> ~10 tok/s decode** (bench prompts; ~9 on technical text, ~6.8 measured in real agentic use at 16-48k context), **~12.8 -> ~47 tok/s prompt processing**, at **KLD 0.113** against Moonshot's native 4-bit model.

## Current best configuration (2026-11-07)

`run-kimi.sh` (details: [current setup](pages/02-current-setup.md)): dense-v1 + the dense-v2 overlay; all routed experts in RAM, computed by the CPU (multi-token AVX2 kernel) except a per-layer GPU share streamed over PCIe via HMM and a 9-slot-per-layer VRAM expert cache; union expert skip at tau 0.045; cross-layer expert prefetch; DFlash IQ4_XS draft with 2 drafted tokens and stochastic (lossless) verification; 8-bit KV cache; ub 1024; 96k context; layer split 50/43 over the GPUs.

| | Value |
|---|---|
| Decode, bench prompts (sampled) | 9.80 tok/s as configured (250 ms/step, 2.45 tok/step); ~9.97 with the causal draft mask; 10.08 with causal + fast top-p (mean of the night10 fast-top-p runs, causal on) |
| Decode, technical text / 64k / 90k context | ~9.2-9.9 / ~8.0 / 7.62 tok/s |
| Decode, real use (one user session, 16-48k) | ~6.8 tok/s (2.05 tok/step at ~304 ms/step) |
| Prompt processing | ~47 tok/s; 42.1 tok/s for a 90k-token prompt |
| Quality vs native 4-bit (as served) | KLD 0.113, median 0.059, same top token 84.8% (0.095 / 86.1% without skip) |
| Not yet adopted, measured lossless | causal draft attention (+1.7%), fast exact top-p (+1.5%, in builds after 2026-11-06); distilled draft trained on the user's own sessions (+5.4% expected tok/step on held-out sessions, online confirmation pending) |

## Biggest wins

| # | What | Gain (as measured) | Date | Commit | Lossy? | Page |
|---|---|---|---|---|---|---|
| 1 | Native AVX2 build (the original build had no SIMD) | 0.32-0.35 -> 0.977 tok/s | 10-10 | build flag | no | [CPU kernels](pages/04-cpu-expert-kernels.md) |
| 2 | Batch-1 threading patches + thread pinning | 1.176 -> 1.498 tok/s at `-ngl 28`; 1.600 at `-ngl 36` | 10-10 | `604bafadb` | no | [CPU kernels](pages/04-cpu-expert-kernels.md) |
| 3 | Dense requant (dense-v1) so all dense weights fit the GPU | 1.69 -> 4.19 tok/s | 10-10 | model file | yes: KLD 0.361 vs UD-Q2_K_XL (later fixed by #12) | [quantization](pages/07-quantization-dense-kv.md) |
| 4 | Pooled MUL_MAT_ID + fused SiTU (+ IQ2_XS kernel 1.33x before) | 4.17 -> 4.68 tok/s | 10-10 | `604bafadb` | no | [CPU kernels](pages/04-cpu-expert-kernels.md) |
| 5 | CUDA IQ2_S/IQ3_XXS kernel + deferred shared expert | 4.65 -> 4.99 tok/s | 10-11 | `604bafadb` | no | [GPU kernels](pages/08-gpu-kernels.md), [MoE](pages/05-moe-placement-cache-prefetch.md) |
| 6 | GPU expert share read from the model mmap via HMM | 4.99 -> 5.58 tok/s | 10-11 | `604bafadb` | no | [MoE](pages/05-moe-placement-cache-prefetch.md) |
| 7 | DFlash speculative decoding made efficient (multi-token kernel, GPU share for verify batches) | lossless DFlash ~4.9 -> 6.18 tok/s (greedy) | 10-12 | `dbe3f5f18`, `4f867184b` | no | [speculative decoding](pages/09-speculative-decoding.md) |
| 8 | Union expert skip | 6.18 -> 8.27 tok/s at tau 0.05 (greedy, 1 GPU); today 6.80 -> 9.23 tok/s at tau 0.045 | 10-12 / 11-02 | `f3f84b015`, `a3ea61f0e` | yes: +0.018 KLD at 0.045 (vs native, 3-token batches) | [expert skipping](pages/06-expert-skipping-and-quality.md) |
| 9 | Second GPU + VRAM expert cache | 306 -> 243 ms/step (7.95 -> 10.32 tok/s greedy) | 10-12 | `367f377c6` | no | [MoE](pages/05-moe-placement-cache-prefetch.md) |
| 10 | Stochastic draft verification | acceptance 0.62 -> 0.69; 7.70 -> 8.10 tok/s sampled | 10-25 | `42d85672b` | no | [speculative decoding](pages/09-speculative-decoding.md) |
| 11 | Platform: PCIe reseat x8 -> x16; BMC fans Full (DRAM thermal throttle) | 289 -> 240 ms/step; sustained ~400 -> 233 ms/step | 10-25 / 10-26 | - | no | [thermals](pages/14-memory-bandwidth-thermals.md) |
| 12 | dense-v2 overlay (most damaged dense tensors at Q6_K/Q5_K/Q4_K) | KLD vs native 0.243 -> 0.095 at the same speed (acceptance 0.69 -> 0.76) | 11-01 | `a8405846e` + overlay | quality up | [quantization](pages/07-quantization-dense-kv.md) |
| 13 | Prompt processing: range split, both GPUs per layer, fused MoE op, ub 1024 with 8-bit KV | 12.8 -> 47 tok/s | 10-25 .. 11-04 | `c747ccd50`, `fa0afd606`, `eb1bf7e81`, `dd993055e` | no | [prompt processing](pages/12-prompt-processing.md) |
| 14 | Small step-time wins: multi-column mat-vec, `iommu=pt`, cross-layer prefetch, CPU setup fix | 274-275 -> 266 -> 259-261 -> 256.5 -> 252/253 ms/step | 11-02 .. 11-04 | `cf1f96157`, `7d3132042`, `297e37d9f` | no | [timeline](pages/18-timeline.md) |
| 15 | Causal in-block attention for the draft; fast exact top-p | +1.7%; +1.5% | 11-05 / 11-06 | `934620b52`, `0c587ea0d` | no | [speculative decoding](pages/09-speculative-decoding.md), [sampling](pages/11-sampling.md) |
| 16 | Disk prompt cache (resume chats after restarts) | after a restart a 2.2k-token chat loaded in 1.8 s (its first 2049-token prompt took 72 s); 90k-token chats ~50 min -> seconds | 11-03 | `78c934053`, `867aeb6b9` | no | [current setup](pages/02-current-setup.md) |
| 17 | Distilled draft on the user's own sessions (pending) | +0.046 / +0.055 / +0.060 acceptance per position on held-out sessions | 11-07 | `bc665ba0b` + `k3-research/distill/` | no | [draft distillation](pages/10-draft-distillation.md) |

## Pages

1. [Hardware, model and constraints](pages/01-hardware-model-constraints.md): the box, K3's architecture and byte accounting, model files, the rules.
2. [Current setup](pages/02-current-setup.md): the daily config setting by setting, `serve_k3.sh`, how a step works, VRAM budget, disk prompt cache, troubleshooting, env knob reference.
3. [Where the time goes](pages/03-where-the-time-goes.md): every profile and step budget, from the 300 ms barrier gap to session-3 phase timing.
4. [CPU expert kernels and threading](pages/04-cpu-expert-kernels.md): AVX2 build, threading patches, IQ2_XS kernel rounds, pooled MUL_MAT_ID, SiTU fusion, multi-token kernel, setup fix, SMT.
5. [MoE placement: GPU share, VRAM expert cache, prefetch](pages/05-moe-placement-cache-prefetch.md): HMM zero-copy, batch take rule, the cache, KB tuning, L3 and cross-layer prefetch, second-GPU share.
6. [Expert skipping and quality](pages/06-expert-skipping-and-quality.md): KLD methodology and noise floor, adaptive and union skip, tau curve, per-layer tau, position/importance skip, oracle bound, shared basis.
7. [Quantization, dense overlays, KV cache](pages/07-quantization-dense-kv.md): Unsloth ladder, dense-v1, the precision study (full table), dense-v2, native-expert layers, 8-bit KV.
8. [GPU kernels and fused ops](pages/08-gpu-kernels.md): persistent IQ2_S/IQ3_XXS mat-vec, multi-column version, top-k, residual mix, MLA attention analysis, CUDA graphs.
9. [Speculative decoding](pages/09-speculative-decoding.md): DFlash setup, greedy vs sampled, stochastic verification, n_max, causal mask, DSpark, confidence cutoff and padding.
10. [Draft distillation](pages/10-draft-distillation.md): research, pipeline, distill1-4, the loss mismatch, held-out evaluation.
11. [Sampling](pages/11-sampling.md): settings, sampler cost, fast exact top-p.
12. [Prompt processing](pages/12-prompt-processing.md): range split, PP2, ubatch size, fused MoE range op, fitting ub 1024.
13. [Long context](pages/13-long-context.md): 96k defaults, attention cost, acceptance vs context.
14. [Memory bandwidth, PCIe, thermals and fans](pages/14-memory-bandwidth-thermals.md): the ~120 GB/s wall and why, DMA additivity, x8 link, the DRAM thermal throttle and how it was fixed.
15. [Tools, scripts and methodology](pages/15-tools-and-methodology.md): benchmark rules, noise levels, scripts, step log, TEMP profilers, tests.
16. [Dead ends and negative results](pages/16-dead-ends.md): everything that did not pay, with numbers.
17. [Lessons and pitfalls](pages/17-lessons-and-pitfalls.md): estimates vs results, bugs found, process mistakes.
18. [Timeline and changelog](pages/18-timeline.md): speed milestones, day-by-day log, commit list.
19. [Prior art and other projects](pages/19-prior-art.md): other local-K3 engines, other drafts, early research documents, earlier GLM work on this box.

## Conventions

- **tok/s** = generated tokens per second, sampled unless marked greedy. **ms/step** = time per speculative verify step (a 3-token target batch with 2 drafted tokens); compare ms/step when acceptance differs. **Acceptance** = accepted drafted tokens / drafted tokens; **tok/step** = tokens emitted per step.
- **KLD** = mean KL divergence of next-token distributions against a reference (native 4-bit, or the same config without skip, as stated); K3 is chaotic, so differences are judged against a floor run ([methodology](pages/15-tools-and-methodology.md)).
- **Lossless** = the same function up to float summation order; **lossy** changes always come with a KLD.
- Sources are the project's working notes in the `k3-research/` folder of the user's workspace (not part of this repo): `HANDOFF.md` (chronological lab log; "HANDOFF §31.z" = its section), `K3-SETUP.md` (setup reference), `SESSION3-NOTES.md` (Nov 4-7 session, "Result N" = its numbered results), `dflash-distill-research.md`, plus raw logs in `/data2/k3eval/` and the fork's commit messages. Commit hashes refer to branch `k3-decode-speed` of the llama.cpp fork.
- Experiments were still running when this was written (night14 session benches for the distilled drafts); results are as of 2026-11-07.
