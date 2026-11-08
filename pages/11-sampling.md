# Sampling (settings, cost, fast exact top-p)

[Back to index](../README.md) · Related: [speculative decoding](09-speculative-decoding.md), [tools and methodology](15-tools-and-methodology.md)

## Settings

- Moonshot's recommended settings (model card): **temperature 1.0, top-p 0.95** for chat; top-p 1.0 for agentic tasks; no top-k / min-p. Thinking is always on (reasoning effort low / high / max, default max). Both launchers set `--temp 1.0 --top-p 0.95 --top-k 0 --min-p 0.0`.
- **Benchmarks must sample with these settings** (user rule, 2026-10-12): greedy inflated draft acceptance (0.71-0.78 vs 0.51-0.64 in real chats) and overstated tok/s.
- Top-k truncation raises acceptance (top-k 40: 0.66 vs 0.53 with the recommended settings, HANDOFF §31) but changes the output distribution: not an option.
- Speculative verification samples exactly from the target ([stochastic verification](09-speculative-decoding.md)).

## Results at a glance

| Change | Measured | Date / commit | Lossless? |
|---|---|---|---|
| Top-p: grow to a bucket-mass bound before the full-vocab sort | 3.1 -> 2.6 ms per call (real K3 distributions) | 10-25, `42d85672b` | yes (250/250 kept sets equal up to ties) |
| Fast exact top-p (AVX2 exp + gather within 8/16/32 nats + small sort) | sample phase **6.05 -> 2.2 ms/step**; **9.93 -> 10.08 tok/s (+1.5%)** | 11-06, `0c587ea0d` | yes (exact top-p set; more accurate normalization) |

## October: the full-vocab sort fallback (HANDOFF §31)

Top-p without top-k fell back to sorting the whole 163,840-token vocab when the top 256 tokens held < p (~15 ms per call). On real K3 distributions (`samplertime/test_topp_real.cpp` on `base-logits-code6-c252.bin`): median kept 2 tokens, p90 77, p99 4743; 7.6% of calls need > 256; mean 3.1 ms per call -> ~5-9 ms per verify step. Fix: grow to a bucket-mass bound first (`llama_token_data_array_top_p_bound`), full sort only as a fallback: exact (250/250 real cases, kept sets and order equal up to ties), 3.1 -> 2.6 ms per call. Sampler A/B on one server then: greedy 286 ms/step, recommended 289, + top-k 40 284, + top-k 256 285: the chain cost ~3 ms/step.

## November: fast exact top-p (SESSION3 Result 5, Result 11; commit `0c587ea0d`)

The step-log phase timing put verification sampling at 4.3 / 6.1 ms per step (2 / 3 positions): ~2 ms per position. Micro-benchmarks (the library's sampler chain on a 163,840-token vocab, one Zen 2 core; inputs = dumped top-32 log-probs plus a smooth tail `l32 - c*log(r/32)` holding the remaining mass) put all of it in top-p:

- Chain per position: mean 2.6-3.2 ms, median 2.2 ms, p99 26 ms; every other sampler ~0 (penalties, top-k 0, min-p 0, typical 1, xtc 0, temp 1 return early); dist 0.004 ms; filling the candidate array 0.15 ms.
- Inside top-p: max 0.05 ms, scalar `expf` + sum 0.54, normalize 0.08, bucket index + histogram 0.35 (allocates a 655 KB vector per call), then the bucket sort. The buckets are fixed to logits in [-10, 10], so the cost depends on the raw-logit offset (old path 2.0-3.3 ms at offsets -15..+35). ~2-4% of positions keep > 256 tokens and take 20-35% of the time.
- An earlier synthetic test with a capped tail made thousands of ties at the 32nd logit and inflated the sort; the smooth tail is realistic.

New path (`llama_sampler_top_p_fast`, for unsorted arrays > 1024): copy logits + max; one AVX2 pass for exp (`ggml_v_expf`) and the sum (double accumulation; runtime dispatch, scalar fallback); gather tokens within 8 nats of the max (then 16, 32 if their mass cannot reach p); sort them; the usual running sum decides. Every token above the threshold is gathered, so the cut is exactly the top-p cut; if p is not reached, the old path runs.

| Check | Old | Fast |
|---|---|---|
| library chain, mean / median / p99 per position | 2.68 / 2.23 / 25.6 ms | 0.67 / 0.40 / 13.7 ms |
| max relative p error vs a double-precision reference (3 offsets) | 0.98-1.34e-3 | 1.62-1.68e-4 |
| 300 cases incl. masked (-inf) tokens, +17 offset: same top token | | 300/300 |
| identical kept sets | | 289/300 (others differ by 1-2 boundary tokens within the old error; flat cases; ties) |

In-server phase timing (first measurement, night9 generation server): sample phase 1.8 ms per 2-draft step vs 6.1 ms (decode 246.0 vs 245.7 ms). Interleaved server A/B (night10, no throttling):

```text
speed-dab10_topp_old_a  | 9.97 tok/s | accept 0.78 | 2.53 tok/step | 253 ms/step   (step log median 260 ms)
speed-dab10_topp_fast_a | 10.24 tok/s | accept 0.79 | 2.54 tok/step | 248 ms/step  (median 252 ms)
speed-dab10_topp_old_b  | 9.88 tok/s | accept 0.76 | 2.48 tok/step | 251 ms/step   (median 255 ms)
speed-dab10_topp_fast_b | 9.91 tok/s | accept 0.75 | 2.46 tok/step | 249 ms/step   (median 253 ms)
```

2-draft step phases (median ms; step / pre / draft / decode / sample): old_a 260 / 0.2 / 3.2 / 251.3 / 6.1; fast_a 252 / 0.2 / 3.2 / 246.3 / 2.2; old_b 255 / 0.2 / 3.2 / 245.9 / 6.0; fast_b 253 / 0.2 / 3.2 / 247.4 / 2.2. **Sample phase 6.05 -> 2.2 ms (-3.9 ms/step), tok/s 9.93 -> 10.08 (+1.5%)**, consistent in both pairs. Kept, default on (`LLAMA_SAMPLER_TOPP_FAST=0` restores the old path).
