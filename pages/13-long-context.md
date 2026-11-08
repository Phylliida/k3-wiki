# Long context

[Back to index](../README.md) · Related: [GPU kernels](08-gpu-kernels.md) (MLA attention), [speculative decoding](09-speculative-decoding.md), [prompt processing](12-prompt-processing.md), [current setup](02-current-setup.md)

The user's real sessions run ~90-100k tokens, so the server defaults to a **96k context** since 2026-10-31. K3 makes this affordable: 69 of 93 layers are KDA linear attention with a fixed-size recurrent state; only the 24 MLA layers keep a KV cache (27.6 KB/token at f16, half with q8).

## Results at a glance

| Measurement | Value | Source |
|---|---|---|
| KV cache per GPU at 96k | 1296 MiB f16 -> 689 MiB q8 | HANDOFF 31.z |
| 96k defaults (Oct 31, dense-v1, cache 16, ub 512) | 9.83 tok/s, 242 ms/step, accept 0.70 (measured during a download); peak 23666 / 23199 MiB | HANDOFF 31.z |
| Decode at 90k (Nov 4, q8 KV, ub 1024) | 7.62 tok/s | HANDOFF 31.z10 |
| 90,119-token prompt | 2138 s = 42.1 tok/s | HANDOFF 31.z10 |
| MLA flash attention per call (3 tokens) | 34 us at 512 KV, 127 at 8k, 434 at 32k, 844 at 64k, 1257 at 96k | HANDOFF 31.zz |
| Attention cost per step (x23 MLA layers) | ~19 ms at 64k, ~29 ms at 96k | HANDOFF 31.zz |
| Target decode per step vs context (same task) | 265 ms (2-8k), 268 (16k), 275 (32k), 296 (64k) | SESSION3 Result 6 |
| Draft time vs context | 3.3-3.4 ms at every length (sliding window 4096) | SESSION3 Result 6 |
| Acceptance vs context (same task) | flat: 0.69-0.74 per draft from 8k to 64k | SESSION3 Result 6 |
| Disk prompt-cache file at 90k | 3.1 GB with q8 KV | HANDOFF 31.z10 |

## Step time: content dominates, context adds attention (HANDOFF 31.zz)

`long_ctx.py` (llama.cpp docs/sources as context, "summarize", 512 tokens; `GGML_SCHED_TIMING`), 2026-11-02:

| Context | tok/s | Accept | ~ms/step |
|---|---|---|---|
| 64k | 6.12 | 0.70 | ~390 (incl. a stall ~token 135) |
| 90k (prefix from the prompt cache) | 7.64 | 0.73 | ~322 |
| same task at 2k | 7.85 | 0.84 | ~340 |
| bench prompts right after | 9.23 | | 266 |

At 64k the CPU split compute was 233-269 ms/step (vs 186) and the CPU wait 77-80 ms (vs 62). **The per-step cost depends mostly on content** (technical summary ~340 vs bench ~266 ms/step), not context length: technical text has flatter router weights and fewer cache hits, so the CPU reads ~20.5-22.6 instead of ~17.3-19.6 experts per layer. Bench prompts are optimistic for real technical work (~7.6-7.9 tok/s then). No major faults in the server (all expert pages resident).

Earlier hints (single-GPU and x8 era): real chats ran 335-370 ms/step at 600-1100 tokens of context vs ~307 at ~150 (HANDOFF §30); a 1875-token prompt gave 299 ms/step generation (+4% at ~2k context) (§31).

## Same task at 2k-64k (night8, SESSION3 Result 6)

Causal draft, no cutoff, old sampler (sample phase 6.0 ms), 2 samples per length:

```text
ctx ~2000:  prompt 2119 tok in 50.6 s (41.91 tok/s, cached 0),     gen 512 tok 9.08 tok/s, draft 308/404
ctx ~2000:  (cached),                                              gen 512 tok 9.70 tok/s, draft 321/380
ctx ~8000:  prompt 7024 tok in 148.6 s (47.27 tok/s, cached 1095), gen 512 tok 8.96 tok/s, draft 305/411
ctx ~8000:  (cached),                                              gen 512 tok 8.60 tok/s, draft 296/429
ctx ~16000: prompt 9024 tok in 194.7 s (46.35 tok/s, cached 7095), gen 512 tok 8.41 tok/s, draft 294/433
ctx ~16000: (cached),                                              gen 512 tok 8.50 tok/s, draft 297/427
ctx ~32000: prompt 17024 tok in 381.0 s (44.69 tok/s, cached 15095), gen 512 tok 8.40 tok/s, draft 302/417
ctx ~32000: (cached),                                              gen 512 tok 8.44 tok/s, draft 299/423
ctx ~64000: prompt 33024 tok in 795.4 s (41.52 tok/s, cached 31095), gen 512 tok 7.98 tok/s, draft 305/410
ctx ~64000: (cached),                                              gen 512 tok 8.01 tok/s, draft 304/413
```

| ctx | Steps | pos1 | pos2 given pos1 | Accepted per draft | tok/step | Step | Draft | Decode | Sample |
|---|---|---|---|---|---|---|---|---|---|
| 2k | 202 | 0.851 | 0.791 | 0.762 | 2.52 | 273 | 3.3 | 264.4 | 6.1 |
| 2k | 190 | 0.900 | 0.877 | 0.845 | 2.69 | 275 | 3.3 | 265.8 | 6.1 |
| 8k | 206 | 0.863 | 0.718 | 0.741 | 2.48 | 274 | 3.4 | 265.1 | 6.0 |
| 8k | 215 | 0.822 | 0.676 | 0.689 | 2.38 | 272 | 3.4 | 263.7 | 6.0 |
| 16k | 217 | 0.782 | 0.734 | 0.678 | 2.36 | 276 | 3.4 | 267.8 | 6.0 |
| 16k | 214 | 0.845 | 0.644 | 0.695 | 2.39 | 277 | 3.4 | 269.1 | 5.9 |
| 32k | 209 | 0.851 | 0.701 | 0.724 | 2.45 | 288 | 3.4 | 278.3 | 6.0 |
| 32k | 211 | 0.829 | 0.703 | 0.706 | 2.41 | 280 | 3.4 | 271.2 | 6.0 |
| 64k | 205 | 0.849 | 0.753 | 0.744 | 2.49 | 307 | 3.4 | 298.5 | 6.0 |
| 64k | 207 | 0.845 | 0.741 | 0.735 | 2.47 | 303 | 3.4 | 293.2 | 6.0 |

- Acceptance does not fall with context length (sample-to-sample spread ~+-0.04). The real-use log's low acceptance (pos1 0.44-0.78 by session) comes from content: a distilled draft needs data like real use ([draft distillation](10-draft-distillation.md)).
- Target decode grows +31 ms at 64k, matching the MLA attention profile: compute-bound (~62 GFLOP per layer at 96k for 96 heads x 3 tokens, ~49 TFLOPS = ~70% of the 3090's FP16-with-FP32-accumulate peak), not bandwidth-bound.

## Attention levers (checked by code reading; SESSION3)

P*V already accumulates in FP16; Q*K^T accumulates in FP32 and takes ~69% of the attention time. int8 Q*K^T would save ~17% of attention (~3 ms/step at 64k, ~5 at 96k) for a new MMA kernel; FP16 accumulation for Q*K^T is too lossy (scores of ~20-50); splitting heads across GPUs needs the KV on both GPUs. Verdict: near the practical limit. Details on the [GPU kernels](08-gpu-kernels.md) page.

## VRAM at long context

- Compute buffers are reserved for the full context: the daily config peaks at 24010 / 23164 MiB at 90k, the same as at 7k.
- q8 KV halves the MLA cache (-0.6 GB per GPU at 96k; flash attention then needs +0.2 GB of compute buffer; net -0.4 GB). At 128k context q8 would still use less VRAM than f16 at 96k (not run).
- Full-attention drafts (DSpark) need KV for the whole context (~20 KB/token, 1.9 GB at 96k): one reason DSpark was dropped. The DFlash draft uses a 4096-token sliding window.

## Conversation state at long context

The disk prompt cache stores target KV + KDA state (443 MiB per state/checkpoint) + draft KV + 4 checkpoints: ~3.1 GB per conversation at 90k with q8 KV (estimate before q8 and before slim checkpoints: ~6.8 GB; ~15 GB with a draft copy per checkpoint). Without it, a restart meant re-processing 90k tokens (~50 min at the older ~29.5 tok/s). See [current setup](02-current-setup.md).
