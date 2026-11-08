# Speculative decoding (DFlash draft, verification, draft length)

[Back to index](../README.md) · Related: [draft distillation](10-draft-distillation.md), [sampling](11-sampling.md), [MoE placement](05-moe-placement-cache-prefetch.md) (GPU share for verify batches), [long context](13-long-context.md)

Speculative decoding with the **modal-labs/Kimi-K3-DFlash** draft is the second-largest structural win after moving dense to the GPUs. With stochastic verification it is **lossless** (output distributed exactly as the target's). The served setup verifies **2 drafted tokens per step** (3-token target batches).

## Results at a glance

| Item | Measured | Date / commit | Verdict |
|---|---|---|---|
| DFlash, first working server (IQ4_XS draft, n_max 2, greedy) | 72-83% of drafts accepted, 2.40-2.63 tok/step, but 4.7-5.2 tok/s vs 5.78 without draft | 10-11 | needed batch-efficient experts |
| + multi-token CPU kernel, union skip, GPU share for batches | lossless 5.50 -> 6.18; union 0.05 7.05 -> 8.27 tok/s (greedy) | 10-12 | kept |
| Two GPUs + cache | 10.32 tok/s greedy | 10-12 | kept |
| Sampled re-baseline (temp 1.0, top-p 0.95) | 7.72 tok/s, accept 0.62 (GPU1 at x8) | 10-25 | the honest number |
| Stochastic verification (draft sampled at tau 0.7) | accept 0.62 -> 0.69, 7.70 -> 8.10 tok/s | 10-25, `42d85672b` | kept |
| n_max 3 (3 drafted tokens) | no gain (several tests); OOM since Nov 5 | 10-12 .. 11-05 | n_max 2 |
| Causal in-block attention for the draft | +1.7% bench, ~+3% technical, +3% tok/step | 11-05, `934620b52` | adopt (env knob) |
| RadixArk DSpark draft | 9.67-9.73 vs 9.97 tok/s (causal DFlash); +1.9 GB VRAM at 96k | 11-05, `015994ebc` | dropped |
| Confidence-based draft length (cut draft 2 when unsure) | simulated +3.4% on real use; measured -1.6% on the bench (padded or not) | 11-04 .. 11-06, `ec5c26a52`, `88a810398` | closed |
| Distilled drafts | see [draft distillation](10-draft-distillation.md) | 11-05 .. 11-07 | distill3/4 promising, pending |

## Early analysis vs reality

- Pre-measurement research (LOCAL-K3 §6, citing WASTE's 0.71-0.75x) said "speculative decoding is a measured loss on K3, don't build it", and HANDOFF §5.4 first called it "not the lever" because each verified position routes to its own 16 experts (the expert union grows while only the accepted prefix is banked).
- Correction (HANDOFF §5.4 session 2, K3-IDEAL-BOUND §7): the dense weights and the fixed per-step overhead amortize across a verify block, so small blocks win. K3-IDEAL-BOUND's model (uniform routing, alpha 0.88): n=1 11.35, n=2 9.19, n=4 7.44, n=8 5.79, n=16 4.10 tok/s serial vs 6.04 without speculation: **the optimum is a small block** and block 8-16 is worse than none.
- Measured routing locality (HANDOFF §21): only ~5 of 16 experts are shared with the next token; expert reads per output token drop only ~9% with speculation (B=2: 36.5 slices / 2.52 tok); the gain is amortizing the ~55 ms of fixed GPU time. Projection then: best B = 2-3 drafted tokens; per-token acceptance 0.7 / 0.8 / 0.9 -> ~6.6 / 7.4 / 8.6 tok/s vs 6.1 modelled baseline. Reality ended above that after the multi-token kernel, GPU share for batches and the expert cache.

## Setup (HANDOFF §23)

- Draft: `modal-labs/Kimi-K3-DFlash` (Qwen3-style, 6 layers, hidden 7168, block 16, 2.60 B trainable params, 5.2 GB bf16; no embedding / LM head: it borrows the target's). Converted with `k3-research/convert_dflash_k3.py <target shard1> <hf dir> <out> <type>` (wraps `convert_hf_to_gguf.py`, copies the tokenizer from the target GGUF). Files: Q8_0 2.76 GB, Q4_K_M 1.52 GB, **IQ4_XS 1.39 GB** (used).
- Taps: target layers [19, 37, 54, 66, 78, 90] (GGUF: [20, 38, 55, 67, 79, 91]). `kimi-k3.cpp` sets `t_layer_inp[il] = res_mix(prefix_sum, attn_res_score)` (the pre-norm mixture the layer's attention reads), matching SGLang's `_dspark_capture_stream`.
- Rollback: K3 supports recurrent rollback (`n_rs_seq` = n_max); the KDA state is 443 MiB per copy, and the RS buffer holds 1 + n_max copies on the GPU.
- First server that fit 24 GB (single GPU, no HMM share, no CUDA graphs): `--spec-type draft-dflash --spec-draft-n-max 2 -ngl 99 -ngld 99 -ncmoe 93 --no-op-offload -fit off -c 2048 -b 64 -ub 64`, VRAM peak ~23.9 GB. Per-request `speculative.n_max` is ignored by the DFlash implementation.
- Acceptance (IQ4_XS, n_max 2, thinking on, greedy): code2 2.63, math 2.59, code 2.53, rust 2.46, chat 2.40 tok/step.
- Model card (full-precision K3, greedy): block 8 accept length gsm8k 5.91, math500 4.95, humaneval 6.01, mbpp 5.37, mt-bench 4.20, longbench 3.46; block 16 7.98 / 6.19 / 8.36 / 6.74 / 4.92 / 3.69.

Speed at first: 4.7-5.2 tok/s with DFlash (step ~525 ms) vs 5.78 non-speculative. Target-only llama-bench: tg 223 ms, pp2 421, pp3 588, pp4 775 ms. The fixes that made it pay are on other pages: the multi-token IQ2_XS kernel ([CPU kernels](04-cpu-expert-kernels.md)), union skip ([expert skipping](06-expert-skipping-and-quality.md)), the GPU share for verify batches and the VRAM cache ([MoE placement](05-moe-placement-cache-prefetch.md)).

DFlash server results, greedy, single GPU (HANDOFF §28):

| Config | code | code2 | math | chat | rust | Mean tok/s |
|---|---|---|---|---|---|---|
| old build (§23) | 4.83 | 5.24 | 5.01 | 4.69 | 4.75 | ~4.9 |
| multi-token kernel (2 reps) | 5.45 | 5.94 | 5.75 | 4.87 | 5.47 | **5.50** |
| + union skip 0.04 (2 reps) | 6.03 | 6.80 | 6.45 | 5.30 | 5.94 | **6.11** |
| + union skip 0.05 (1 rep) | 7.13 | 7.02 | 7.75 | 6.31 | 7.02 | **7.05** |
| union 0.04, n_max 3 | 5.66 | 6.53 | 6.35 | 5.05 | 6.31 | 5.98 |
| lossless, n_max 3 | 5.46 | 5.75 | 5.44 | 4.74 | 5.61 | 5.40 |

## Greedy overstated real speed (HANDOFF §30-31)

All DFlash numbers before 2026-10-25 were greedy (temperature 0). The first single-GPU chat server (`serve_k3.sh`, 2026-10-12) gave 7.4 tok/s at default sampling (acceptance 0.63), prompt 9.2 tok/s. Real web-UI chats (temp 0.8, top-k 40, min-p 0.05) got acceptance 0.51-0.64 and 6.0-6.3 tok/s in fast mode, against 0.71-0.78 and ~8 tok/s greedy. Rule since then: **benchmark with Moonshot's settings** (temperature 1.0, top-p 0.95; top-p 1.0 for agentic use), report acceptance with every number, compare **ms per verify step** when acceptance varies.

Sampled baseline (5 prompts x 3 seeds, GPU1 at PCIe x8): H config 7.72 tok/s, accept 0.62, 2.23 tok/step, 288 ms/step (greedy on the same server ~290 ms/step, accept ~0.75).

Sampler A/B on one server (seed 1, x8): greedy 286 ms/step, recommended 289, + top-k 40 284, + top-k 256 285: the sampler chain cost ~3 ms/step then. Acceptance is what changes: greedy 0.74 (2.44 tok/step), recommended 0.53 (2.05), top-k 40 0.66 (2.29; truncation changes the distribution, so not an option). `"backend_sampling": true` per request had no effect.

## Stochastic verification (HANDOFF §31, commit `42d85672b`)

The server verified drafts by exact match (target sample == greedy draft token), so acceptance = p_target(draft token). With a draft sampled from q (DFlash top-10 at temperature tau) and acceptance min(1, p/q), else a sample from max(0, p - q), acceptance = sum min(p, q) and the output is still exactly target-distributed.

- Potential (TEMP `LLAMA_SPEC_RS_STATS`, `rs_stats.py`, 850 steps): exact match 0.731 / 0.716 (pos 0 / 1) -> 2.25 tok/step; rejection sampling at tau 0.7 0.794 / 0.752 -> 2.39; per-position best tau (ceiling) 2.49. The gain is at uncertain positions (p of the greedy draft < 0.2: 0.09 -> 0.38); at confident ones the draft's q is flatter than p (0.998 -> 0.95).
- `rs_math_check.py`: 40 random p/q cases, output matches p (worst |z| 2.3).
- Server (5 prompts x 2 seeds, recommended sampling, x8): **7.70 -> 8.10 tok/s**, accept 0.62 -> 0.69, 2.22 -> 2.34 tok/step, step time unchanged. Second run 8.14 tok/s, accept 0.69.
- Tuning (805 steps): greedy-when-confident adds nothing (theta 0.8-0.9 at tau 0.7: 2.385 vs 2.384 tok/step); tau 0.7-1.0 equal -> plain `LLAMA_SPEC_RS_TAU=0.7`. The 2.49 ceiling would need the target's p at draft time.

## How many drafted tokens? (n_max)

| Test | Result | Source |
|---|---|---|
| Oct 12, greedy, single GPU | union 0.04: n_max 3 5.98 vs n_max 2 6.11; lossless 5.40 vs 5.50 (3-token step 421 ms vs 4-token 509 ms; acceptance does not grow enough) | HANDOFF §28 |
| Oct 26, sampled, x16 sweep | n_max 3: 296 ms/step at 2.80 tok/step vs 242 at KB 2 / n_max 2: no gain on the same prompts (9.51 vs 9.70 tok/s), hits the throttle sooner. 4 verified tokens cost +54 ms over 3 | HANDOFF §31 |
| Nov 2, `DRAFT_N=3` (acceptance had risen to 0.76-0.79) | bench 8.56 tok/s, accept 0.64, 2.87 tok/step, 336 ms/step (vs 267-269 at 2); technical 8.67 tok/s (~3.2 tok/step, ~369 ms/step) vs 8.51-8.57 at 2. A 4-token verify reads ~25% more experts: break-even on technical text, -8% on the bench | HANDOFF 31.zz |
| Nov 5, n_max 3 + cutoff | CUDA OOM on GPU0 in cudaGraphInstantiate (peak 24119 MiB; the 4th KDA state copy + larger buffers). Would need CACHE=8 | SESSION3 |

## Causal in-block attention: a free win (SESSION3; commit `934620b52`)

Found by the distillation research: z-lab's reference `dflash/model.py:366`, SpecForge (`dflash_mask.py`: "full layers bidirectional, sliding layers causal") and SGLang make `sliding_attention` layers causal inside the block when the config has no `is_causal`. All six layers of the draft are sliding. The converter only writes `dflash.attention.causal` when the config says so, and the fork defaulted to bidirectional; verified in code and by parsing the GGUF header. Override: `LLAMA_DFLASH_CAUSAL=1` (server log prints `in-block attention: causal`).

| Config | Bench tok/s | Accept | tok/step | Tech tok/s |
|---|---|---|---|---|
| bidirectional, block cut to 3 (as before) | 9.80 | 0.74 | 2.45 | 9.25 |
| causal, run 1 | 10.01 | 0.77 | 2.51 | 9.87 |
| causal, run 2 | 9.93 | 0.76 | 2.49 | 9.18 |
| bidirectional, full 16-position block | 5.50 | 0.20 | 1.39 | 5.50 |

- Step logs (~950 steps each): pos 1 acceptance 0.859 -> 0.879, pos 2 given pos 1 0.748 -> 0.796 (2.3 standard errors), tok/step 2.50 -> 2.58.
- Pooled: **+1.7% bench, ~+3% technical, +3% tok/step.** The first run's +6.7% on technical text was one noisy 512-token generation.
- The full-block run settles it: under bidirectional attention each position also attends to the masks after it; two extra masks (block cut to 3) cost a little, fifteen wreck acceptance (0.20). With causal attention the truncated `[anchor, mask x n_max]` block is exact.
- Later no-cut causal runs: 10.01, 9.93, 9.89, 10.05 (mean **9.97** tok/s).
- Not yet in `run-kimi.sh` (user decision): add the env knob or write `dflash.attention.causal = true` into the GGUF (or a converter rule: sliding layers without `is_causal` are causal).

## Full-block decoding (`LLAMA_DFLASH_FULL_BLOCK`, commits `505338506`, fixed `3adf5d263`)

Decode the whole trained block (16 for DFlash, 7 for DSpark) and read the first n_max drafts. First version requested outputs at every position, which backend sampling refuses (at most n_max + 1 per sequence): every draft decode failed ("no draft", 4.77 tok/s, 1.00 tok/step, 209 ms/step; useful as a no-draft step-time data point). Fixed: outputs only at readable positions. Results: modal DFlash bidirectional full block 0.20 acceptance (above); DSpark full block below.

## RadixArk DSpark draft: not better (SESSION3)

`RadixArk/Kimi-K3-DSpark`: SpecForge-trained from scratch (0.1 CE + 0.9 L1 + 1.0 confidence BCE), 5 full-attention Qwen3 GQA layers (64 q / 16 kv heads, head 64), hidden 7168, block 7, taps [7, 23, 51, 67, 83], Markov logit-bias head (rank 256) + confidence head, YaRN x16 from 65,536, 2.25 B params. Model card acceptance lengths at 7 drafts: GSM8K 5.42, HumanEval 5.51, MT-Bench 3.93. Converted with the same converter (bf16 4.5 GB in 31 s) and quantized to IQ4_XS (1.21 GB).

Problems found and fixed: the DSpark read loop drafted greedily without proposal probabilities, so the server fell back to exact-match verification (`015994ebc` adds stochastic verification; the in-graph Markov chain conditions on the argmax token, so verification stays exact); full attention sizes the draft KV for the whole 96k context (~20 KB/token, 1920 MiB on GPU1: "failed to create MTP context") -> `LLAMA_SPEC_DRAFT_CTX` (`3adf5d263`), tests at 16384.

| Config (draft ctx 16k) | Bench tok/s | Accept | tok/step | ms/step | Tech tok/s (accepted/drafted) |
|---|---|---|---|---|---|
| DSpark, block cut to 2 positions | 9.67 | 0.72 | 2.42 | 250 | 9.23 (310/401) |
| DSpark, full 7-position block | 9.73 | 0.75 | 2.47 | 254 | 8.87 (306/409) |
| modal DFlash, causal (reference) | 9.97 avg | 0.76-0.77 | 2.50 | 251 | 9.18-9.87 |

Dropped; for 96k use it would also need q8 draft KV (~0.96 GB) plus VRAM freed elsewhere. Other K3 drafts on Hugging Face (most with MLA backbones the fork cannot run) are listed in [prior art](19-prior-art.md).

## Confidence-based draft length: closed (HANDOFF 31.z11, 31.z14, 31.z15, 31.z17; SESSION3)

Idea: each step verifies [last token, draft 1, draft 2]; draft 2 costs extra CPU expert reads and is wasted when rejected. The fork's `--spec-draft-p-min` also cut draft 1 (whole drafts lost). New modes (`LLAMA_SPEC_P_MIN_MODE`, commit `ec5c26a52`): 1 = cut draft 2 by its own top-1 p; 2 = cut when p1 x p2 < p_min. Draft confidence = the draft's top-1 probability over its top-10 at temperature 1.

**Data.** October bench rs-stats (805 steps, `rs_cut_prelim.py`): pos-2 acceptance 0.33 at q 0.2-0.3, 0.46 at 0.4-0.5, 0.60 at 0.6-0.7, 0.99 above 0.95 (34% of tokens); gain of a mode-1 cutoff by assumed saving: 20 ms none, 30 ms +0.1%, 40 ms +0.5%, 50 ms +1.6%. Real use (17,992 steps of one user session at 16-48k context, `LLAMA_SPEC_STEP_LOG` + 10% random cuts via `LLAMA_SPEC_CUT_RAND`, `draft_len.py`):

```text
overall: pos1 0.647, pos2|pos1 0.623, tokens/step 2.050
pos2 acceptance by its top-1 p: 0.10-0.20 0.071 | 0.30-0.40 0.190 | 0.50-0.60 0.389 | 0.70-0.80 0.607 | 0.90-0.95 0.865 | 0.95-1.00 0.986
step time 4k-16k: 2-draft 279.4 ms (n 1774), 1-draft 254.4 ms (n 209) -> saving 25.0 ms
step time 16k-48k: 2-draft 304.2 ms (n 14347), 1-draft 270.4 ms (n 1641) -> saving 33.8 ms
simulated, baseline 6.805 tok/s: mode 2 p_min 0.30 +3.07%, 0.35 +3.35%, 0.40 +3.36%, 0.50 +3.11%; mode 1 best +2.71% at 0.60
```

19% of real-use steps have p > 0.95 at both positions (acceptance ~0.99), which motivated n_max 3 + cutoff (it OOMed).

**Bench A/B** (5 prompts x 2 seeds + 2k technical text):

| Config | Bench tok/s | Tech tok/s | tok/step | ms/step |
|---|---|---|---|---|
| bidirectional, no cutoff | 9.80 | 9.25 | 2.45 | 250 |
| bidirectional + mode 2 p_min 0.4 | 9.37 | 9.19 | 2.41 | 257 |
| causal, no cutoff (2 runs) | 10.01 / 9.93 | 9.87 / 9.18 | 2.51 / 2.49 | 251 |
| causal + mode 2 p_min 0.4 (2 runs, graph-key fix) | 9.80 / 9.84 | 9.24 / 8.97 | 2.40 | 245 / 244 |
| causal + cutoff, CUDA graphs off | 9.17 | 8.85 | 2.42 | 264 |
| causal + cutoff, padded (night7) | 9.80 | 9.58 | 2.34 | 239 |
| causal + cutoff, unpadded (night7) | 9.88 | 9.82 | 2.39 | 242 |
| causal, no cutoff (night7) | 10.05 | 9.16 | 2.52 | 251 |

**Diagnosis: every change of verify batch size cost time.** Step time by (previous -> current drafts), causal + cutoff: graphs on 1->1 207, 2->1 225, 1->2 267, 2->2 263 ms; graphs off 224 / 239 / 284 / 280. Before the CUDA graph key fix (`c52898b5f`) the bidirectional run showed ~20 ms per change (1->1 221, 2->1 241, 1->2 287, 2->2 268); the fix removed ~14 ms from the 2->3 direction. The rest is not CUDA graphs: a symmetric ~10 ms per batch-size change (llama keeps only the previous graph, so a new shape rebuilds, re-splits and re-allocates the K3 graph) plus ~6 ms more rollback after a 3-token step.

**Padded verify batches** (`LLAMA_SPEC_PAD_TOKEN`, commit `88a810398`): always verify n_max + 1 tokens; a cut draft is padded with a reserved token whose routed experts are all skipped (get_rows lookup table -> pad flag, keep mask zeroed after the union rule). **The first "padded" run (night6) was not padded**: only `llama-dflash-dump` had been rebuilt, which relinked `libllama.so` but not the server library. night7 rebuilt `llama-server` and the log showed `padded verify batches: 200 pad tokens so far`. Phase timing (commit `8092e89fb`): the 2->1 penalty is all in the target decode (+18 ms unpadded); padding removes most of it (+5 ms) and brings graph captures back to the no-cut level (2.3% vs 4.9%), but the pad token costs ~7 ms on every 1-draft step (attention, shared expert, head for one more row), so padding nets out to nothing.

Other findings: steps that would be cut are slightly *cheaper*, not dearer (selection check on no-cut runs: cut->cut 246, cut->keep 245, keep->cut 252, keep->keep 259 ms), so the real saving of a cut is ~40 ms (~250 -> ~210), not the 54 ms assumed. Economics: no-cut 2.50 tok / 251 ms = 9.96 tok/s; cut 2.39 tok / 245 ms = 9.76 tok/s; even a zero-penalty cut is worth ~+1% at best on the bench.

**Verdict:** no-cut causal 10.01, 9.93, 9.89, 10.05 (mean 9.97) vs cutoff p_min 0.4 9.80, 9.84, 9.71, 9.88, 9.80 (mean 9.81): **the cutoff costs ~1.6% at n_max 2 on this box, padded or not. Closed.** `LLAMA_SPEC_PAD_TOKEN` and the p_min modes stay as TEMP knobs; `LLAMA_SPEC_CUT_RAND` can leave `run-kimi.sh`. (The plan's original estimate was +3-8%; the simulator ran ~2 points optimistic.)

## Acceptance depends on content, not context length (SESSION3 Result 6, Result 7)

- Same technical-summary task at 2k-64k (night8): 0.69-0.74 accepted per draft from 8k to 64k (2k: 0.76 / 0.85); sample-to-sample spread ~+-0.04. Details on the [long context](13-long-context.md) page.
- Real-use log by context bucket swings without a trend (8-16k pos1 0.441 / pos2|pos1 0.406; 16-24k 0.721 / 0.711; 24-32k 0.632 / 0.579; 32-40k 0.580 / 0.491; 40-48k 0.782 / 0.772): each bucket is mostly one session, so content dominates.
- On-policy generation (K3 answering 293 prompts of a public dataset with max thinking): 0.704 per draft overall (answers hitting the 2048-token limit 0.655, EOS 0.761); per-answer p10-p90 0.62 / 0.67 / 0.79 / 0.88 / 0.93; 126,154 steps at 2.40 tok/step, 256 ms/step. A creative-writing prompt accepted only 0.53 per draft; math ~0.98.
- Bench prompts: 0.77-0.79 per draft (pos1 ~0.86-0.88). So the bench overstates real use (2.05 tok/step), which is what the distillation work targets.

## Not tried yet

N-gram lookup combined with DFlash (`--spec-type draft-dflash,ngram-mod`, content-dependent) and tree verification (top-2 at draft position 1; est. +5-10%, best after a distilled draft) were on the 2026-11-04 plan and have not been run.
