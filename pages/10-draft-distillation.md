# Draft distillation (fine-tuning the DFlash draft against our target)

[Back to index](../README.md) · Related: [speculative decoding](09-speculative-decoding.md), [tools and methodology](15-tools-and-methodology.md)

Goal: raise draft acceptance by fine-tuning the modal-labs DFlash draft on our exact quantized target (dense-v2 overlay, 2-bit experts, skip). It is **lossless** for output quality (stochastic verification keeps the target's distribution); only speed depends on it. Status on 2026-11-07: generic data does not help; the user's own agentic sessions give **+0.046 / +0.055 / +0.060 acceptance per block position on held-out sessions** (distill3), confirmed online but noisily (24-probe paired +0.054 +- 0.028). Recommended: distill3.

## Results at a glance

| Draft | Data | Loss | Held-out offline pos1 / pos2 / pos3 (expected tok/step) | Online session bench (per draft) | Generic bench (per draft) | Verdict |
|---|---|---|---|---|---|---|
| modal (reference) | - | - | 0.7399 / 0.5483 / 0.4417 (2.145) | 0.603; repeat 0.583; 24-probe 0.572 | 0.774 | baseline |
| distill1 | K2.5-written chat text (602k tokens) | 0.9 TV + 0.1 CE | 0.7111 / 0.5327 / 0.4324 (2.090) | - | 0.775 vs same-night ref 0.766 | rejected |
| distill2 | on-policy K3 answers (354k tokens) | 0.9 TV + 0.1 CE | 0.7000 / 0.5190 / 0.4130 (2.063) | 0.542 | 0.747 (-0.027) | rejected |
| distill2b | same as distill2 | acceptance loss | 0.7240 / 0.5295 / 0.4204 (2.107) | 0.556 | 0.774 (0) | rejected |
| **distill3** | user's agentic sessions (526k tokens) | acceptance loss | **0.7856 / 0.6029 / 0.5015 (2.260, +5.4%)** | 0.612; 24-probe 0.594 (paired per probe +0.054 +- 0.028) | 0.787 (+0.013) | **recommended** (positive on every measure) |
| **distill4** | 2.3x the session data (1.18M tokens) | acceptance loss | **0.7887 / 0.6047 / 0.5032** | 0.643 (+4.6% tok/step vs ref mean); 24-probe 0.571 (paired per probe +0.051 +- 0.029) | 0.764 (-0.010) | matches distill3 offline, less consistent online |

Decision rule used throughout: keep a distilled draft only if acceptance rises by **>= 0.02 per draft**, measured from step logs or paired evaluations, not a single run.

## Research (2026-11-04; `k3-research/dflash-distill-research.md`)

- The code that trained `modal-labs/Kimi-K3-DFlash` is **not public** (Modal: 28 B300 nodes producing hidden states, 4 training; "trained only on a generic data mix"). z-lab's repo is inference-only but its `dflash/model.py` (MIT) is the reference forward. Usable open trainers: SpecForge (MIT; offline features, warm start, CPU-offloaded optimizer) and vLLM speculators (Apache-2.0).
- Architecture: 6 taps -> one `fc` [7168, 43008] + norm, fed into every draft layer's K/V ("KV injection"); block = [anchor, mask x (block-1)], mask token id 163592; the draft sees context features only before the anchor.
- Recipes are teacher-forced ("online" means features computed on the fly, not generation). DFlash paper loss: hard-label CE with per-position decay exp(-(k-1)/gamma). Newer K3 drafts use 0.1 CE + 0.9 L1, and L1 = 2 x TV; under speculative sampling acceptance = 1 - TV. "Regenerating responses with the target is the single largest lever on acceptance" (SpecForge v0.3).
- Sizing: 2.60 B trainable params (embedding and LM head, 1.17 B each, frozen and borrowed from the target); full fine-tune with the optimizer in RAM ~16-19 GB per 3090 (~31 GB host RAM); features 86,016 B/token (6 x 7168 x bf16); training ~1.8-2.9 M tokens/hour on 2 GPUs (est.), so **feature dumping (~170k tokens/hour at 47 tok/s) is the bottleneck** and needs server downtime.
- Expected gain revised from +10-25% to **+4-13%** (n_max 3 then pays a little).
- Free fixes it found first: the causal in-block mask (adopted: +1.7%, see [speculative decoding](09-speculative-decoding.md)) and the DSpark draft (dropped).
- User pointer: Soft-MTP (github.com/JaydenTeoh/Soft-MTP, arXiv 2610.38331, MIT) trains MTP heads on soft targets (gradient of KL(q||p)); supports training on the target's full distributions.

## Pipeline (`k3-research/distill/` + one fork tool)

1. `prep_data.py`: range slices of `lightseekorg/kimi-mtp-dataset` (Apache-2.0, K2.5-regenerated conversations with `<think>` reasoning) -> 5,313 conversations (3,022 single-turn, the rest 2-14 turns), rendered with **K3's own jinja chat template** from the GGUF (thinking-effort system message, think/response channels). Every assistant span is a train segment: 48.8 M chars, 87% in assistant spans. `docs_to_seg.py` writes the length-prefixed input format.
2. **`llama-dflash-dump`** (commit `bc665ba0b`): prompt processing over the documents with the daily target (overlay, GPU share, cache, prefetch; **per-token** expert skip 0.045 without the union rule, because union skip at 1024-token batches keeps nearly everything). Per token: the 6 tap inputs (bf16, the same `llama_get_embeddings_layer_inp` data the drafter reads), the target's top-32 log-probs, the token and the train flag. Documents cut at 4096 tokens (the draft's window). Files: `tokens.i32`, `mask.u8`, `taps.bf16` [n, 6, 7168], `topk_id.i32` [n, 32], `topk_lp.f32` [n, 32], `docs.idx`.
3. `extract_head.py`: dequantizes the target's `token_embd` (Q8_0) and `output` (Q4_K) into bf16 (20 s): exactly what the draft borrows at inference.
4. **`train_draft.py`**: the reference architecture with causal blocks, warm-started from the modal weights (all 69 tensors load by name). 192 anchors per document in one pass with a block mask; block position i = 1..3 predicts token a+i against the target's distribution at a+i-1 after top-p 0.95; weights exp(-(i-1)/4). Metric = expected acceptance of the drafter's real proposal (top-10 at tau 0.7). bf16 weights and grads on cuda:0, frozen head on cuda:1, fp32 master + AdamW in RAM. Saves HF safetensors + `is_causal: true`, so the converter writes the causal key. Then `convert_dflash_k3.py` + IQ4_XS (27 s).

**Validation** (night5/6): test dump 6 docs, 8,202 tokens at ~44 tok/s; 89% train tokens; mean target top-1 p 0.79; top-32 mass 0.979; target top-1 == K2.5's next token 80.5% in assistant spans. Eval-only baseline of the untrained modal draft (causal, tau 0.7 top-10): pos1 0.892, pos2 0.752, pos3 0.647, vs the bench step log's measured pos1 0.879 / pos2|pos1 0.796: taps, head and masks are right. Smoke training on 4 docs (8 steps): held-out pos2 0.859 -> 0.864, pos3 0.782 -> 0.787; memory fits; saves 5.2 GB HF weights.

## distill1: K2.5-written text, TV loss (night6, 2026-11-05/06)

- Dump: 275 documents, **602,058 tokens in 12,812 s (47.0 tok/s)**, 49 GB.
- Training: 262 train / 13 val documents; 2 epochs = 131 optimizer steps at 13.2 s/step (~30 min); lr 3e-5 peak, warmup 20, cosine to ~3e-6; 192 anchors, accum 4, block 4.

| Step | pos1 acc | pos1 TV | pos2 acc | pos2 TV | pos3 acc | pos3 TV |
|---|---|---|---|---|---|---|
| 0 (modal) | 0.8556 | 0.1537 | 0.7008 | 0.3270 | 0.5977 | 0.4305 |
| 50 | 0.8540 | 0.1398 | 0.7121 | 0.2978 | 0.6126 | 0.4017 |
| 100 | 0.8586 | 0.1362 | 0.7148 | 0.2946 | 0.6145 | 0.4001 |
| 131 | 0.8590 | 0.1356 | 0.7156 | 0.2937 | 0.6155 | 0.3987 |

Gains pos1 +0.003, pos2 +0.015, pos3 +0.018; TV -9-12%. Predicted +0.018 tok/step (+0.7%) at n_max 2.

A/B (`dab6_distill1` vs `dab6_causal_ref`): 9.85 tok/s, accept 0.76, 2.50 tok/step, 254 ms/step (tech 9.50) vs 9.89, 0.76, 2.48, 251. Step logs (2-draft steps, +- 1 binomial s.e.):

| Run | Steps | pos1 | pos2 given pos1 | Accepted per draft | tok/step |
|---|---|---|---|---|---|
| ab6-distill1 | 946 | 0.864 +- 0.011 | 0.794 +- 0.014 | 0.775 | 2.550 |
| ab6-causal_ref (same night) | 914 | 0.863 +- 0.011 | 0.776 +- 0.015 | 0.766 | 2.533 |
| ab4-causal_gstats | 946 | 0.871 +- 0.011 | 0.779 +- 0.014 | 0.775 | 2.550 |
| ab-causal | 933 | 0.879 +- 0.011 | 0.796 +- 0.014 | 0.789 | 2.579 |

+0.009 per draft vs the same-night reference, -0.002 vs the mean of three causal runs (0.777): **rejected**. Likely reasons: off-policy data (K3 teacher-forced on K2.5 text; at inference the draft only sees K3's own samples), the modal draft already near what 6 layers can do on this distribution, too little training.

## distill2: on-policy K3 answers, TV loss (night9/9b, 2026-11-06)

- Generation (`distill/gen_onpolicy.py`): K3 answers the first user turn of dataset conversations (4,168 usable prompts) via raw `/completion` on the K3-template prompt, thinking effort max, temp 1.0 / top-p 0.95, n_predict 2048, until 300k tokens. **Bug:** `/completion`'s `content` drops special tokens, so the channel markers vanished ("...= 18 years old.thinkresponse**Freddy..."); fixed with `"return_tokens": true` + `/detokenize` (re-tokenized to identical ids, 51/51) and restarted. Result: **293 answers, 301,024 tokens** (incl. throttle pauses and a fan break); 76 hit the 2048-token limit (155,648 tokens), 217 ended with EOS (145,376).
- Dump2: 293 docs, 353,772 tokens in 8,513 s (41.6 tok/s). Training: 279 train / 14 val, lr 5e-5, 3 epochs, 209 steps.
- Validation: step 0 pos1 0.8725 / pos2 0.7231 / pos3 0.6178 (TV 0.1391 / 0.3097 / 0.4138); step 50 0.8584 / 0.7086 / 0.5939; step 100 0.8602 / 0.7117 / 0.5982; step 150 0.8641 / 0.7176 / 0.6018 (TV 0.1362 / 0.2949 / 0.4122); final step 209 0.8657 / 0.7204 / 0.6053 (TV 0.1344 / 0.2920 / 0.4081): **TV improves while acceptance falls**.

A/B (interleaved):

```text
speed-dab9_ref_a        | 9.80 tok/s | accept 0.77 | 2.51 tok/step | 256 ms/step
speed-dab9_distill2_a   | 9.62 tok/s | accept 0.75 | 2.46 tok/step | 255 ms/step
speed-dab9_ref_b        | 9.67 tok/s | accept 0.76 | 2.49 tok/step | 258 ms/step
speed-dab9_distill2_b   | 6.84 tok/s | accept 0.73 | 2.44 tok/step | 356 ms/step   <- THROTTLED (speed invalid, acceptance valid)
```

Pooled step logs: ref 0.774 +- 0.007 per draft (pos1 0.873, pos2|pos1 0.774, 2.549 tok/step) vs distill2 0.747 +- 0.007 (pos1 0.848, pos2|pos1 0.761, 2.493): **-0.027 per draft (~4 s.e.), rejected.** Validation said pos1 -0.007; the bench said -0.025 (val under-predicted by ~3x here).

## The training-loss mismatch (found 2026-11-06 20:47)

The `tv` loss (0.9 TV + 0.1 CE) matches the draft's **temperature-1** distribution to p, but the proposal that verification sees is the **top-10 at tau 0.7**. Once q(T=1) is closer to p, the tau-sharpened proposal overshoots, so acceptance falls while TV improves (the same pattern as distill1). New `train_draft.py --loss acc`: loss = 0.9 x (1 - sum min(p, q_tau,top10)) + 0.1 x CE(p, softmax(logits / tau)), i.e. directly the expected acceptance of the actual proposal (differentiable through the top-10 logits), plus a tau-consistent CE term that pulls target tokens into the top-10.

- **distill2b** (dump2 retrained with `--loss acc`, lr 3e-5, 3 epochs, 209 steps): val pos1 0.8712 / pos2 0.7176 / pos3 0.6060 vs modal 0.8725 / 0.7231 / 0.6178: flat to slightly worse. With the right loss, generic chat/math/code data still gives nothing (the modal draft was likely trained on similar data).

## distill3 / distill4: the user's own sessions (2026-11-06/07)

**Data.** The user shared three folders of their own agentic coding sessions (Kimi Code transcripts, 121 transcripts in total; K3 used through the API with thinking effort high; ~10.76 M K3-generated tokens in the first folder). A secret scan found no real secrets; the builder redacts secret-like patterns anyway, and data and draft weights stay local. `distill/prep_sessions.py` rebuilds each conversation, renders it with K3's template (thinking_effort=high, tools), marks only assistant spans (K3's output, incl. its tool-call markup) as train, clips tool results to 4000 chars (head 65% / tail 35%), drops the 95k-char system prompt + tool list (the draft's 4096 window never sees it in long sessions), and cuts ~24k-char windows at message boundaries. Holdout: whole sessions (final build: one per project among sessions with >= 5 windows, after two builds picked unsuitable holdouts). Final docs3: **80 train windows (2.07 M chars, 68% generated)**, holdout 268 windows from 3 sessions.

- Dump3: 80 documents, **525,555 tokens in 13,846 s** (38.0 tok/s; heavy user work overlapped the first ~2 h), 43 GB (`DUMP_MAX_DOC 8192`, `-c 8192`).
- train3: 76 train / 4 val documents, `--loss acc`, lr 3e-5, 6 epochs, 114 steps at ~14.5 s.

| Step | pos1 acc | pos1 TV | pos2 acc | pos2 TV | pos3 acc | pos3 TV |
|---|---|---|---|---|---|---|
| 0 (modal) | 0.7209 | 0.3029 | 0.5101 | 0.5397 | 0.4073 | 0.6370 |
| 50 | 0.7559 | 0.3744 | 0.5455 | 0.6086 | 0.4389 | 0.6933 |
| 100 | 0.7587 | 0.3669 | 0.5512 | 0.5946 | 0.4488 | 0.6777 |
| 114 | 0.7583 | 0.3661 | 0.5518 | 0.5931 | 0.4500 | 0.6761 |

On the user's sessions the modal draft accepts far less than on the bench (pos1 0.72 vs 0.87): predicted 1 + a1 + a1 x a2 = 2.09 tok/step, matching the measured real-use 2.05. distill3 predicts 2.18 (+4.2%). With the acc loss acceptance rises while TV (of the T=1 distribution) rises, as intended. Caveat: these val docs come from the training sessions.

**Session bench** (`distill/session_bench.py`: replays held-out windows up to an assistant turn, ~3.3-3.8k tokens of context; K3 generates 384 tokens with real-use sampling; acceptance from the server's draft counters; 8 probes):

```text
11-07_04:31 session-bench ref:       accepted per draft 0.603 (1240/2058), gen 7.91 tok/s
11-07_04:58 session-bench distill3:  accepted per draft 0.612 (1453/2373), gen 8.06 tok/s
11-07_05:25 session-bench distill2b: accepted per draft 0.556 (1287/2314), gen 7.66 tok/s
11-07_05:52 session-bench distill2:  accepted per draft 0.542 (1277/2358), gen 7.38 tok/s
11-07_16:47 session-bench distill4:  accepted per draft 0.643 (1546/2404), gen 8.19 tok/s
11-07_17:13 session-bench ref2:      accepted per draft 0.583 (1276/2190), gen 7.83 tok/s
```

Step logs: ref 1028 two-draft steps, pos1 0.732, pos2|pos1 0.644, 0.602 per draft, 2.204 tok/step; distill3 1185 steps, 0.756, 0.619, 0.612, 2.224; distill2b 1157, 0.699, 0.591, 0.556, 2.112; distill4 pos1 0.770, pos2|pos1 0.669, 0.643, 2.286 tok/step vs ref mean 2.185 (**+4.6%**). The bench reproduces real-use acceptance (Nov 4 log: pos1 0.65, pos2|pos1 0.62), but it is noisy: two identical ref runs differ by 0.02, per-probe acceptance spans 0.35-0.81, and K3 samples its own continuation.

Generic bench speed tests (no throttling: 0.6% / 0.2% of steps > 350 ms): `dab12_distill3` 10.10 tok/s, accept 0.79, 2.54 tok/step, 252 ms/step; `dab12_distill2b` 9.92 / 0.76 / 2.49 / 251. Step logs vs ref (ab9 ref_a+b 0.774 +- 0.007): distill3 0.787 +- 0.009 (pos1 0.891, pos2|pos1 0.767); distill2b 0.774 +- 0.010.

**Held-out offline eval** (night13; 24 windows of the 3 held-out sessions, `dump3h` at 46.9 tok/s; every draft scored by `train_draft.py --eval-only --val-frac 1.0` on the same text and anchors: paired, no sampling noise; bf16 weights; n ~6,100 positions per block position, s.e. ~0.006):

| Draft | pos1 | pos2 | pos3 | Expected tok/step (1 + a1 + a1 x a2) |
|---|---|---|---|---|
| modal | 0.7399 | 0.5483 | 0.4417 | 2.145 |
| **distill3** | **0.7856** | **0.6029** | **0.5015** | **2.260 (+5.4%)** |
| distill4 | 0.7887 | 0.6047 | 0.5032 | |
| distill2b | 0.7240 | 0.5295 | 0.4204 | 2.107 |
| distill2 | 0.7000 | 0.5190 | 0.4130 | 2.063 |
| distill1 | 0.7111 | 0.5327 | 0.4324 | 2.090 |

- distill3: **+0.046 / +0.055 / +0.060 on sessions it never saw**, well over the +0.02 bar. The 8-probe online bench (+0.009) was noise-dominated.
- Generic data hurts real-use acceptance (pos1: distill2b -0.016, distill2 -0.040, distill1 -0.029).
- **distill4** (dump4n: 100 new windows, 657,379 tokens in ~4 h at 46.9 tok/s; trained on dump3 + dump4n = 180 documents, 1.18 M tokens, acc loss, lr 3e-5, 4 epochs, 171 steps; own val step 50 0.8012 / 0.6335 / 0.5238, step 171 0.8123 / 0.6466 / 0.5421): only +0.002-0.003 over distill3 on the held-out set: the gain comes from **the domain + the acceptance loss**, not data volume.

Caveats before adopting: the offline eval uses bf16 draft weights (the server runs IQ4_XS), and API-K3 text with local-K3 targets.

## 24-probe online check (night14, 2026-11-07 evening)

Same 24 held-out prompts (seed 7) for all three drafts; speeds are DRAM-throttle-affected (session benches had no
cool-down), acceptance is not:

```text
session-bench ref24:       24 probes, accepted per draft 0.572 (4054/7089)
session-bench distill3_24: 24 probes, accepted per draft 0.594 (4084/6873)
session-bench distill4_24: 24 probes, accepted per draft 0.571 (3863/6766)
```

Pooled: distill3 +0.022, distill4 -0.001. Paired per probe (mean of per-prompt differences): **distill3 +0.054 +-
0.028 (t 1.94, better on 13/24)**, distill4 +0.051 +- 0.029 (t 1.77, 12/24); distill4's flat pooled number comes from
a few long generations where it did worse.

| Draft | Held-out offline pos1 / pos2 | Online 8 probes | Online 24 probes pooled / paired | Generic bench |
|---|---|---|---|---|
| distill3 | +0.046 / +0.055 | +0.019 | +0.022 / +0.054 | +0.013 |
| distill4 | +0.049 / +0.056 | +0.050 | -0.001 / +0.051 | -0.010 (0.764 +- 0.010; 9.96 tok/s) |

**Recommendation: distill3**, positive on every measure (~+2-5% tokens per step in agentic sessions, lossless).
Adopted 2026-11-07 (`-md` pointing at `K3-dflash-distill3-iq4_xs.gguf`; its metadata already sets the
causal mask). Online session benches are noise-limited (sampled continuations, per-probe acceptance 0.35-0.9); the
paired offline evaluation is the reliable instrument. Possible follow-ups: retune `LLAMA_SPEC_RS_TAU` for the
distilled draft, revisit n_max 3.

## First real use (2026-11-07, SESSION3 Result 16)

One user session with distill3 (7,325 two-draft steps, 13-79k context): pos1 0.614, pos2|pos1 0.606, **0.494 accepted per draft, 1.99 tok/step**. This is below the ~0.52 per draft of the Nov 4 session with the modal draft, but the content differs, so it does not show whether distill3 helped there. Acceptance swung 0.33-0.78 per window or request. A paired offline evaluation of distill3 against the modal draft on this session's text would settle it; not done yet.

## Lessons from this work

- Measure acceptance with **paired offline evaluation on held-out sessions** (same text, same anchors): far less noisy than online benches where K3 samples its own continuation.
- Validation on documents from the training sessions overstates gains (in-session repetition of files and code).
- Train the loss that verification actually scores (top-10 at tau 0.7), not TV at temperature 1.
- `/completion`'s `content` drops special tokens; request token ids and detokenize when building training text.
- A waiter `while pgrep -f llama-dflash-dump` never ends because `pgrep -f` matches the waiter's own command line; use `pgrep -x` or a PID.
