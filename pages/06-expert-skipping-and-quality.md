# Expert skipping and quality (KLD frontier)

[Back to index](../README.md) · Related: [quantization](07-quantization-dense-kv.md), [MoE placement](05-moe-placement-cache-prefetch.md), [tools and methodology](15-tools-and-methodology.md)

Reading fewer expert bytes is the only lever that moves the CPU-bound expert phase once it sits at the DRAM wall. Every method here is **lossy**; each is reported with its KLD. The served configuration uses **union skip at tau 0.045**.

## How quality is measured

- **KLD** (`llama-perplexity --kl-divergence`) of a configuration's next-token distributions against stored reference logits. Several references were used over time; absolute KLDs are only comparable within one harness:

| Harness | Reference | Text | Used for |
|---|---|---|---|
| wikitext (HANDOFF 18.3) | UD-Q2_K_XL logits | wikitext-2 test, 48 x 512 (PPL 1.7744: K3 has largely memorized wikitext) | dense-v1 vs original |
| `run_kld_skip.sh` (§24) | dense-v1 at K=16 (`base-logits-code16-k16.bin`) | novel code, 16 x 256 (PPL 2.309) | single-token skip |
| `run_kld_batch.sh` (§28) | dense-v1, batch 3 (`base-logits-code6-c252.bin`) | code, 6 x 252 | union skip, GPU paths |
| `kld_dense.sh` / `kld_overlay.sh` (31.z) | **native 4-bit** UD-Q4_K_XL (PPL 5.4421) | `unseen-26k.txt` (workspace reports + novel code, ~7k tokens; 4 chunks of 2048; `KLD_EXTRA='-b 3 -ub 3'` for skip runs) | dense precision, skip tau on dense-v2 |
| `kld_pos.sh` (31.z10) | same config **without skip** | calibration_datav3, 20 x 252 at `-b/-ub 3`; prints KLD by ubatch position; fully deterministic | position skip, importance skip, q8 KV |
| `kld_oracle.sh` (31.z16) | same config without skip, every expert on the CPU | calibration_datav3, 20 x 252 at `-b 3` | oracle bound, fused res_mix check |

- **K3 is chaotic in KLD terms** (HANDOFF 20.3, §24): summation-order changes alone give KLD ~0.016 at batch-1 decode (old vs new CUDA kernel 0.0173; same values summed in reverse 0.0162); in the skip study, tau = 1e-6 (2 of 5.6M slots skipped, i.e. last-bit rounding) gave **KLD 0.0706 / same-top 96% / PPL +-1.3%**. Traced with `llama-eval-callback`: the first difference is `ffn_moe_weighted-1` at 1.7e-6 relative, amplified layer by layer through routing near-ties. So compare configurations against a **floor run**, not against 0, and look at the median and p99 too (a few fragile tokens with KLD 9+ dominate the mean even in correct runs).
- Differences below ~0.02 KLD on small sets mean nothing (HANDOFF 20.3); with the deterministic `kld_pos` harness, a pure numerics change (q8 KV) moved KLD by ~0.002 (31.z10).

## Adaptive skip, single token (HANDOFF §24, commit `f3f84b015`)

`ggml_mul_mat_id_set_skip(a, w)`: slots with weight exactly 0 are not read. `LLAMA_MOE_SKIP=tau`: keep = step(w_norm - tau), weights renormalized (`LLAMA_MOE_SKIP_RENORM=0` keeps raw). Works for the CPU part and the GPU share.

| Config | Slots kept /16 | dKLD vs floor | d same-top |
|---|---|---|---|
| tau 0.01 | 15.91 | +0.001 +- 0.010 | -0.4 +- 0.5% |
| tau 0.02 | 15.26 | +0.000 +- 0.011 | -0.7 +- 0.5% |
| tau 0.03 | 13.79 | +0.019 +- 0.011 | -1.8 +- 0.5% |
| tau 0.04 | 11.77 | +0.065 +- 0.013 | -3.9 +- 0.6% |
| tau 0.05 | 9.23 | +0.101 +- 0.017 | -6.2 +- 0.9% (PPL +5%) |
| fixed K=12 (top-12) | 12 | +0.055 +- 0.016 | -2.9 +- 0.6% |

Adaptive skip is free up to ~5% skipped and cheaper than fixed K at ~14%, but no better than fixed K=12 at ~25%: quality loss tracks the number of dropped experts more than their router weights (the normalized top-16 weights are flat; the smallest is typically 0.02-0.05).

Speed (FRAC 0.4, K_gpu 3): tau 0.03 5.77, 0.04 5.86, 0.05 6.20 vs 5.78 base: the GPU share becomes the critical path as the CPU part shrinks, so rebalance K_gpu. Rebalanced sweep (some background load, baseline 5.61): tau 0.04 K_gpu 2 **6.38**, K_gpu 1 6.21; tau 0.05 K_gpu 2 **7.20**; no share: base 5.17, tau 0.04 5.98. Later (prefetch sweep, graphs on) tau 0.05 K_gpu 2 measured **7.56 tok/s** vs 5.79 lossless.

Fixed expert count, speed only (HANDOFF 18.4): K=16 4.63, K=12 5.62, K=8 7.20, K=1 12.73 tok/s (~9.6 ms per expert slot).

## Union skip for verify batches (HANDOFF §28, commit `a3ea61f0e`)

`LLAMA_MOE_SKIP_UNION=1` keeps a sub-tau expert if another token of the batch selected it (it is read anyway). Offline (`union_skip.py`): at equal dropped slots per token it reads 5-13% fewer experts than plain skip.

| Batch-3 config (6 x 252 code) | Experts read per batch | dKLD vs floor | d same-top |
|---|---|---|---|
| floor (no skip) | 34.40 | - (KLD 0.044) | - |
| plain 0.03 | 29.82 | +0.021 +- 0.018 | -1.3 +- 1.0% |
| plain 0.04 | 25.56 | +0.051 +- 0.016 | -4.5 +- 1.3% |
| union 0.04 | 26.82 | +0.005 +- 0.018 | -3.1 +- 1.2% |
| union 0.05 | 22.43 | +0.035 +- 0.025 | -2.9 +- 0.8% |

DFlash server, greedy (n_max 2): multi-token kernel 5.50 tok/s; + union 0.04 **6.11**; + union 0.05 **7.05**. Union 0.05 (7.05 tok/s at ~+0.035 KLD) beat single-token plain tau 0.05 (7.56 at +0.10) on quality per speed. Acceptance drops a little with union skip (code 0.75 -> 0.71, chat 0.61 -> 0.54).

## Skip tau on dense-v2 vs native (HANDOFF 31.z, 2026-11-01/02)

KLD vs native 4-bit at 3-token batches (`v2_b3_*`) and server speed (`speed_test.sh`, sampled):

| tau (`SKIP`) | KLD | Median | p99 | Same top | PPL | Decode tok/s | ms/step | Accept |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.1243 | 0.0628 | 0.863 | 83.3% | 5.971 | 10.17 (9.52 in a later control) | 245 (258) | 0.76 (0.74) |
| **0.045** | **0.1132** | 0.0594 | 0.768 | **84.8%** | 5.896 | **9.23** | 274 | 0.78 |
| 0.04 | 0.1066 | 0.0543 | 0.711 | 84.5% | 5.870 | 8.20 | 302 | 0.75 |
| 0.03 | 0.0985 | 0.0501 | 0.716 | 85.56% | 5.794 | 7.17 | 337 | 0.72 |
| off | 0.0954 | - | - | 85.6% | 5.794 | 6.80 | 361 | 0.74 |

- Without skip, 3-token batches give the same KLD as batch 2048 (0.0954 vs 0.0953).
- Union skip at 0.05 costs +0.029 KLD / -2.3 points same-top, about as much as the `attn_output` upgrade gained.
- The speed curve matches the single-token slot counts: tau 0.03 / 0.04 / 0.05 keep 13.8 / 11.8 / 9.2 of 16 experts per token, so 0.04 does ~28% more CPU expert work than 0.05 (step time +23%). No skip costs ~1.1 ms/step per layer. Both knees sit at 0.04-0.05 because the normalized top-16 weights are flat.
- **Decision (2026-11-02 03:30): tau 0.045**, the best quality above 9 tok/s (KLD 0.113 / 84.8% vs 0.124 / 83.3% at 0.05 for ~0.3-0.9 tok/s).

### Per-layer tau (`LLAMA_MOE_SKIP_LAYERS`, commit `c7fb4f4cd`)

tau 0.05 everywhere except no skip in one layer group (recovery of the 0.029 that tau 0.05 costs):

| No skip in | KLD | Median | p99 | Same top | PPL | Recovered | Per layer |
|---|---|---|---|---|---|---|---|
| 1-19 | 0.1176 | 0.0619 | 0.780 | 84.0% | 5.897 | 0.0067 (23%) | 0.00035 |
| 20-39 | 0.1204 | 0.0636 | 0.822 | 83.6% | 5.932 | 0.0039 (13%) | 0.00020 |
| 40-59 | 0.1140 | 0.0588 | 0.733 | 84.5% | 5.884 | 0.0103 (36%) | 0.00052 |
| 60-79 | 0.1130 | 0.0576 | 0.834 | 85.2% | 5.915 | 0.0113 (39%) | 0.00057 |
| 80-92 | 0.1175 | 0.0593 | 0.849 | 84.1% | 5.944 | 0.0068 (24%) | 0.00052 |

The five recoveries sum to 0.039 (sub-additive; noise ~0.003 each). Layers 40-92 are ~2x as sensitive as 1-39, but no small group dominates. No skip in 20 layers adds ~1.5 slots/token, the same as global tau 0.045 (~1.3), and gives the same KLD (0.113), so per-layer "off" is not a clear win. A gradient (0.05 in 1-39, 0.04 in 40-92, est. -0.002) was not tested.

### The frontier

Skip tau, per-layer skip and native-expert layers ([quantization](07-quantization-dense-kv.md)) all trade about **0.008 KLD per tok/s** (HANDOFF 31.z). At a fixed speed, more quality needs a faster step, which is why later work targeted step time.

## Draft-position skip: dropped (HANDOFF 31.z10, commit `10b684e22`)

`LLAMA_MOE_SKIP_POS=t0,t1,t2`: stricter tau for later draft positions (the 3rd token is kept only ~59% of the time). Harness `kld_pos.sh` (reference = same config without skip):

| Config (union skip) | CPU experts/layer | KLD pos 0 / 1 / 2 | Mean KLD | Same top |
|---|---|---|---|---|
| no skip (ref) | ~25 | - | - | - |
| u045 (daily) | 15.0 | 0.033 / 0.034 / 0.031 | 0.0327 | 93.2% |
| pos 0.045, 0.045, 0.07 | 12.2 | 0.050 / 0.039 / 0.048 | 0.0455 | 92.6% |
| pos 0.04, 0.05, 0.07 | 12.3 | 0.049 / 0.043 / 0.050 | 0.0475 | 91.8% |
| pos 0.045, 0.06, 0.08 | 10.1 | 0.061 / 0.053 / 0.066 | 0.0601 | 91.0% |

Raising only position 2's tau raises position 0's KLD too: accepted drafts keep the KV computed with their skip, so every later token sees it (the harness models this, teacher-forced). Uniform-tau curve on the same harness: u0.04 17.0 experts / KLD 0.0246; u0.045 15.0 / 0.0327; u0.05 13.1 / 0.0377; u0.055 11.2 / 0.0464. At 12.2 experts the uniform curve gives ~0.042 vs 0.0455: **worse than a uniform tau at equal experts read.**

## Importance-aware skip: inconclusive, opt-in (HANDOFF 31.z10, commit `5c0c99c2a`)

`LLAMA_MOE_SKIP_NORMS=<file>`: keep = (w x mean output norm of the expert, normalized over the token's selected experts) > tau; norms from a calibration run (`GGML_MOE_NORM_DUMP`, every expert on the CPU, 20 x 512 tokens of calibration_datav3; `norm_stats.py`). Mean expert output norm varies across a layer's experts by CV ~0.17 median (0.09-0.4; max/min 2-8x), less than one expert's token-to-token variation (CV 0.2-0.4).

| tau | Experts read | KLD | Uniform curve at the same experts |
|---|---|---|---|
| i0.04 | 16.4 | 0.0268 | 0.0271 |
| i0.045 | 14.25 | 0.0329 | 0.0347 |
| i0.05 | 12.3 | 0.0406 | 0.0414 |

1-5% lower KLD (~0.8 fewer CPU experts per layer, ~5%, at today's quality), but a pure numerics change moved KLD by ~0.002, the same size: not conclusive with 2500 scored tokens. Left opt-in.

## Oracle bound for any output-norm predictor: dropped (HANDOFF 31.z16, SESSION3; commit `d3f8c075c`)

`LLAMA_MOE_SKIP_ORACLE=1` computes every selected expert, then drops slots whose weight x **actual** output norm (normalized per token) <= tau, with the same union rule and renormalization. Every expert on the CPU, reference in the same config, 20 x 252 tokens at `-b 3`, 48 slots per call:

| Rule | tau | Slots kept | KLD vs no-skip |
|---|---|---|---|
| router weight (today) | 0.045 | 76.6% (11.2 of 48 skipped, 23.6 experts read) | 0.0335 (+- 0.0015) |
| router weight | 0.055 | 67.5% (15.6 of 48 skipped, 19.3 read) | 0.0469 |
| oracle: weight x actual norm | 0.045 | 73.4% | 0.0343 |
| oracle | 0.06 | 61.7% | 0.0547 |
| oracle | 0.08 | 53.8% (same top 89.96%) | 0.0838 |

At 73.4% kept the oracle's KLD (0.0343) is ~10% below the weight-only curve (~0.038 interpolated); at equal KLD it keeps ~2.6 points fewer slots (~3.5% fewer experts). That is the ceiling for any norm predictor (a learned one would get maybe half): ~1-2% speed, not the +5-10% estimated. **Router weight alone is nearly as good as knowing each expert's real contribution.** (The CPU-only weight skip at 0.045, 0.0335, matches the GPU-config value 0.0327.)

## Shared input basis for experts (MoBE-style): negative (HANDOFF §25)

Question: can experts be represented in a shared basis to cut bandwidth (cf. MoBE, arXiv 2508.05257)?

- **Weights look like random matrices** (layers 2, 10, 85; `expert_basis.py`, IQ2_XS dequant): mean expert 0.15% of energy, expert-expert cosine ~0; a shared input basis of rank 1024 / 1792 / 2560 keeps 32-37 / 53-60 / 74-80% of energy (random 29 / 50 / 71); per-expert rank-1024 energy 0.71-0.73 (Gaussian 0.710). The latent MoE (7168 -> 3584) already took the cheap part.
- **Inputs are concentrated** (`GGML_MMID_DUMP_X`, 2048 code tokens): rank 256 / 512 / 1024 keeps 65 / 72 / 79% on average (early layers 77 / 85 / 91%); 12-50% of the energy is the mean direction; subspaces of different layers are unrelated.
- **Quality test** (TEMP `LLAMA_K3_XPROJ`, x -> P P^T x on the routed input, bases fit on 8192 calibration tokens): held-out energy r1536 68% / r2048 78% / r2560 87%; KLD on novel code r2560 **0.351** (PPL +27%, same-top 83%); r3584 (identity) 0.079 = floor. So 19% fewer expert reads for +0.28 KLD, vs skip tau 0.04: 26% fewer for +0.065. Dropped.

## Summary of skip-like levers

| Lever | Lossy? | Best result | Verdict |
|---|---|---|---|
| Union skip, tau 0.045 | yes: KLD 0.113 vs 0.095 (native ref) | 9.23 tok/s vs 6.80 without skip (Nov 2) | **in use** |
| Union skip, tau 0.05 | +0.029 KLD vs no skip | 9.5-10.2 tok/s | earlier default; superseded by 0.045 |
| Per-layer tau | same frontier | - | option kept, not used |
| Draft-position skip | worse than uniform | - | dropped |
| Importance-aware (static norms) | 1-5% lower KLD, within noise | - | opt-in |
| Output-norm predictor | oracle ceiling ~3.5% fewer experts | ~1-2% at best | dropped |
| Fixed top-K | K=12 +0.055 KLD | 5.62 vs 4.63 tok/s (Oct 10) | worse than adaptive at low skip |
| Shared input basis | +0.28 KLD for 19% fewer reads | - | dropped |
