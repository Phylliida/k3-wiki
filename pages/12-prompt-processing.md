# Prompt processing

[Back to index](../README.md) · Related: [MoE placement](05-moe-placement-cache-prefetch.md), [current setup](02-current-setup.md) (VRAM budget, disk prompt cache), [long context](13-long-context.md)

Prompt batches (512-1024 tokens per ubatch) touch nearly every expert in every layer, so the question is who streams which experts once per batch. Result: **~12.8 tok/s CPU-only -> ~47 tok/s** today. All changes are lossless (KLD of the GPU paths matches the CPU within noise).

## Results at a glance

| Step | Prompt tok/s | Date / commit | Source |
|---|---|---|---|
| CPU only, ub 64 / 128 / 512 (pp512) | 10.52 / 11.26 / **12.77** | 10-25 | HANDOFF §31 |
| single-GPU chat server (fast mode, ub 64) | 9.2 | 10-12 | HANDOFF §30 |
| range split, GPU0 only, PP 0.5 (x16) | 22.52 | 10-25, `c747ccd50` | HANDOFF §31 |
| two GPUs layer split + adaptive share (GPU1 at x8), ub 2048 | pp2048 36.45 | 10-25 | HANDOFF §31 |
| x16, one GPU per layer (PP 1.0, PP_HALF 450), ub 512 | pp512 23.05 / pp2048 22.12 | 10-26 | HANDOFF 31.y |
| **both GPUs on every layer** (PP2 0.5, PP_HALF 225), ub 512 | **31.87 / 31.06 (+38%)** | 10-26, `fa0afd606` | HANDOFF 31.y |
| ub 1024 (llama-bench) | pp1024 48.62 / pp2048 47.59 | 10-30 | HANDOFF 31.y |
| ub 2048 + fused MoE op (llama-bench) | pp2048 69.53 | 10-31, `eb1bf7e81` | HANDOFF 31.y |
| server, ub 512, deployed | 29.45 / 29.40 (3.6k / 7.2k prompts) | 10-30 | HANDOFF 31.y |
| server, ub 1024 + fused op on the main stream | 41.5 / 44.1 | 10-31, `dd993055e` | HANDOFF 31.z |
| `iommu=pt` | 29.5-29.9 vs 28.8-29.2 (ub 512) | 11-03 | HANDOFF 31.z7 |
| MUL_MAT_ID setup off thread 0 | 2k 29.6-30.0 -> 31.3-31.5; 7.2k -> 31.1 (+5.5%) | 11-04, `297e37d9f` | HANDOFF 31.z9 |
| **8-bit KV + ub 1024 (daily)** | **47.0 / 47.7**; 90,119-token prompt in 2138 s = **42.1** | 11-04 | HANDOFF 31.z10 |

Op-offload reference (stock llama.cpp, imatrix run on UD-Q2_K_XL, Oct 10): ub 2048 34 tok/s vs 11.9 at ub 512 and 6.5 with `-nopo` (HANDOFF 18.2).

## Range split (HANDOFF §31, commit `c747ccd50`)

`LLAMA_MOE_GPU_PP=<frac>` gives batches > 4 tokens a GPU share of the experts [0, frac x 896) ("range split": split k = n_expert; no expert cache for these batches, so both state copies stay untouched). CUDA `ggml_cuda_mul_mat_id_range` stages chunks of 64 experts in VRAM with wide loads from host memory, remaps ids per chunk and calls `ggml_cuda_mul_mat_q(..., ids_partial=true)`. Tests: MUL_MAT_ID_SPLIT +36 range cases (104/104), MUL_MAT_ID 932/932.

- PP 0.5, GPU0 only: **22.52 tok/s** (`GGML_SCHED_DEFER=0`: 11.95 = serial). The range MUL_MAT_ID streamed 448 experts x 3.18 MB in 66 ms per call = 21.6 GB/s at x16.
- PP 0.5, two GPUs (layer split, GPU1 at x8): 16.72; + cache 18 + union skip 0.05: pp512 17.08, **pp128 4.91**: the GPU share costs the same per batch whatever its size, so short prompts got slow.
- Adaptive share `LLAMA_MOE_GPU_PP_HALF=h`: fraction n/(n + h), capped by PP. PP_HALF 600, cap 0.5: pp512 17.24, pp128 11.68 (was 4.91).
- ub 2048 (FRAC 1.0, cap 1.0, PP_HALF 600, cache 18, union skip, GPU1 x8): **pp2048 36.45** (CPU only ~13), pp512 17.58.
- The TEMP op profiler serializes CPU and GPU (and broke with 2 GPUs): do not judge overlap with it on.

**Latent bug fixed (commit `c1cfdfd7f`):** `ggml_mul_mat_id_set_split` stored e1 in `op_params[3]`, which MUL_MAT_ID also uses for precision; an e1 of 10 read as `GGML_PREC_F32` made CUDA refuse the op (likely with two GPUs, where e1 is half the share). Split params moved to op_params 4-7; new offset-range test cases 128/128 on both GPUs.

## Both GPUs on every layer (`LLAMA_MOE_GPU_PP2`, HANDOFF 31.y)

With a layer split only one GPU's link was busy at a time. Now the layer's GPU takes experts [0, e1), the other GPU takes the **top** range [e2, 896), the CPU the middle.

| x16, ub 512 | pp512 | pp2048 |
|---|---|---|
| base (PP 1.0, PP_HALF 450, one GPU per layer) | 23.05 | 22.12 |
| PP2 0.5 + PP_HALF 225 | **31.87** | **31.06** |
| PP2 0.5 + PP_HALF 150 | 31.18 | 30.68 |
| PP2 0.5 + PP_HALF 450 (same total GPU share as base) | 23.74 | 23.10 |

The gain comes from moving more experts to the GPUs (the CPU part is the limit). The balance point h = G/C (GPU cost per batch / CPU cost per token) is independent of the batch size in n/(n + h); ~225 at ub 512, 200 in the server.

Correctness (`kld_pp.sh`, b 252, no skip, vs `base-logits-code6-c252`): pp1 (single-GPU share) KLD 0.0272, PPL 2.4965, same top 95.87%; pp2 0.0260, 2.4923, 96.00%: equal within noise.

**First-touch mapping:** the startup prefetch mapped each expert tensor only on its layer's GPU; with PP2 the other GPU faulted the other layers in on the first prompt (first 1091 tokens took 216 s). Fix: the other GPU takes the **top** experts (so it only reads the top PP x PP2 = 50% of the other layers' experts) and the startup prefetch maps that range on it too (startup ~6 -> ~10 min). KLD of this layout (`pp2-top`): mean 0.0434 (PPL 2.548) but median 0.0030 (lowest of all runs), p99 0.381: the mean is 1-2 fragile tokens (max 8.8); the CPU-only b3-floor run had the same mean 0.0437 and max 9.46: not a bug.

Server with ub 512 + PP2 0.5 + PP_HALF 200 + cache 18 (top layout + prefetch): long prompt (3.6k) **29.1 tok/s** (first request 22.4 incl. warm-up), generation 10.15 tok/s, VRAM peak 23456 / 24045 MiB.

## Batch size: bigger ubatches win (HANDOFF 31.y)

The GPUs stream the same bytes per batch, so bigger batches halve their cost per token.

| Config (llama-bench, no draft) | Result |
|---|---|
| ub 1024, PP2 0.5, PP_HALF 200, double-buffered, cache 18 | **pp1024 48.62 / pp2048 47.59** (+53% over ub 512) |
| ub 2048 + cache 18 | CUDA OOM in mmq's pool |
| ub 2048 + cache 15 | 69.90 double-buffered, **70.76** serial |
| ub 2048 + cache 18 + fused op | **pp2048 69.53**; pp4096 died in a cuBLAS BF16 matmul (attention side: the second ubatch has 4096 tokens of context) |

**Double-buffered range copies** (second stream copies the next chunk while mmq runs; `GGML_CUDA_MMID_RANGE_SERIAL=1` = old path): no gain at ub 512 (h225 31.56 / 30.15, h150 31.74 / 29.89 vs serial 31.87 / 31.06, 31.18 / 30.68) or ub 2048: reverted.

## Fused MoE range op (`GGML_OP_MOE_FFN_RANGE`, commit `eb1bf7e81`)

`ggml_moe_ffn_range(up, gate, down, x, ids, weights, skip, e0, e1, beta, lb)` -> [n_out, n_tokens]: the weighted sum over the slots in [e0, e1). CUDA: routing planned on the host (ids/skip copied down, one sync, counting sort by expert); per chunk of `GGML_CUDA_MOE_RANGE_CHUNK` (16) experts: stage up/gate/down, then per <= `GGML_CUDA_MOE_RANGE_ROWS` (2048) rows: gather x rows, mmq up/gate with compact ids, SiTU, mmq down, atomicAdd weighted rows into dst. It replaces separate MUL_MAT_ID nodes whose [3072 x 16 slots x tokens] intermediates took 1-3 GB at large batches; the fused op needs a few hundred MB of pool. Disables CUDA graphs for the graph that holds it. Tests MOE_FFN_RANGE 34/34 (also with chunk 3/1 and rows 5/1); a mutation (weight dropped in the scatter) fails as expected.

- KLD (`kld_pp.sh fused`): mean 0.0348, median 0.0033, p99 0.285, PPL 2.484 (best of all runs; the max 9.4 is the same fragile token).
- Speed: ub 1024 cache 18: pp1024 48.65 / pp2048 47.73 (same as the MUL_MAT_ID nodes, 48.62 / 47.59); ub 2048 cache 18: 69.53 (the MUL_MAT_ID nodes OOMed).
- Compute buffers with the fused op (server, ctx 8192): ub 512 CUDA0 384 / CUDA1 500 MiB; ub 1024 767 / 1006; ub 2048 1534 / 1982: ~0.75 / ~1 MB per token, all attention side (96-head MLA every 4th layer, KDA elsewhere, dense layer FFN 33792 wide), not the routed experts.
- **Main stream instead of the side stream** (commit `dd993055e`): the side stream gave the fused op nothing to overlap with, while its temporaries went to the side stream's own VMM pool. Keeping it on the main stream let ub 1024 fit: GPU1 peak 24058 -> 23896 MiB at the first long prompt. TEMP `GGML_CUDA_POOL_LOG=1` logs VMM pool growth.

## Fitting ub 1024 into the server (HANDOFF 31.y, 31.z, 31.z10)

- Draft compute buffers on CUDA1: 334 + 655 MiB at `LLAMA_SPEC_DRAFT_UBATCH=512`, 84 + 164 at 128 (default 128 since). RS buffer 732 / 597 MiB with the draft (rollback states) vs 244 / 199 without.
- Practical VRAM ceiling **~24050 MiB used per GPU**: OOMs at 24056-24058 (cuBLAS workspace for the side stream on CUDA1; cudaGraphInstantiate). ub 1024 + cache 18 failed in several variants (1350 MiB compute pp buffers with ~570 free; OOM in cudaGraphInstantiate at 24056). A crashed server sat ~10 min in teardown holding VRAM.
- Deployed 2026-10-30 (`serve_k3.sh fast2`): ub 512, cache 18, PP2 0.5, PP_HALF 200, fused op, draft-ub 128, -ts 49,44: idle 22936 / 23021, peak 23332 / 23578 MiB; prompts 3.6k / 7.2k **29.45 / 29.40 tok/s** (first request 25.1 with warm-up); generation 10.22 tok/s, accept 0.69, 230 ms/step.
- 2026-10-31, fused op on the main stream, ub 1024 + cache 18 + draft-ub 128: prompts **41.5 / 44.1 tok/s**, generation 10.12 tok/s, 229 ms/step, peak 23596 / 24067 (no margin). With `LLAMA_MOE_GPU_K=2` (frees a scratch slot): prompts 41.4 / 43.9, generation 10.15, peak 23290 / 23970.
- Then 96k context and dense-v2 took the VRAM back (ub 512, ~29 tok/s), until **8-bit KV (2026-11-04)** freed ~0.4 GB per GPU: ub 1024 at cache 9: 251 ms/step decode, **prompts 47.0 / 47.7 tok/s (+52%)**, peak 24012 / 23162; validated at 90k: 90,119-token prompt in 2138 s (**42.1 tok/s** vs ~29.5 before), decode 7.62 tok/s at 90k, peak 24010 / 23164 MiB.

## Prompt speed vs context (SESSION3 Result 6)

Same server, technical-summary prompts with prefix caching: 2119 tokens in 50.6 s (41.91 tok/s, cold); 7024 new tokens at 47.27 tok/s (1095 cached); 9024 at 46.35 (7095 cached); 17,024 at 44.69 (15,095 cached); 33,024 at 41.52 (31,095 cached). The distillation dumps (prompt processing over documents) ran at 47.0, 41.6, 38.0 (with outside load) and 46.9 tok/s.

## Real use (SESSION3 Result 16, 2026-11-07)

Server log of one session: a 13,212-token prompt at 46.3 tok/s; 3,089 and 1,679-token prompts at 28.6 and 26.2 tok/s (throttled); short prompts (36-457 tokens) at 7-22 tok/s (per-batch overhead). One request re-processed 51,174 tokens in 19.5 min (43.65 tok/s): a different or compacted prompt after a 78,797-token conversation, where the best disk-cache match was 14,151 tokens (f_sim 0.105, ~5.5k reused). That is inherent to the changed prompt; only faster prompt processing would shorten it (ub 2048 in the server, estimated +45%, not yet measured there).

Across 18,460 of the user's requests there were 63.8M new input tokens vs 13.5M output (4.7:1; median 821 new / 306 output), which puts prompt processing at ~43% of model time at 47 / 7.4 tok/s, or ~36% at the measured real-use 5.6 tok/s.

## Notes

- An earlier conclusion that "prompt processing slows decode" (decode ~400 ms/step after some requests) was the DRAM bandwidth throttle, not PP (HANDOFF §31, 31.x).
- Prompt re-processing after a restart is avoided by the disk prompt cache ([current setup](02-current-setup.md)).
- Other projects' prompt tricks (chunk-union expert dedup, reading the route union in storage order) are listed in [prior art](19-prior-art.md); our range split is the same idea for a RAM-resident model.
