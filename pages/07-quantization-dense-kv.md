# Quantization: dense requant, overlays, native experts, KV cache

[Back to index](../README.md) · Related: [hardware and model](01-hardware-model-constraints.md), [expert skipping and quality](06-expert-skipping-and-quality.md), [current setup](02-current-setup.md)

The routed experts were never requantized: every config uses Unsloth's IQ2_XS expert bytes (plus 13 IQ3_XXS tensors) unchanged. What changed is the **dense** (always-read) part, so that it fits in VRAM, and later how much of it to buy back at higher precision. Since 2026-11-04 the user's rule is **no more requants** (quality as is; lossy speedups only at equal or better KLD).

## Unsloth quant ladder (reference, from Unsloth's docs; LOCAL-K3 §4, `unsloth_docs.txt`)

| Quant | Size | Mean KLD | PPL | Top-1 agreement | Expert type |
|---|---|---|---|---|---|
| UD-IQ1_S | 594.0 GB | 0.5645 | 2.5789 | 78.875% | IQ1_S |
| UD-IQ1_M | 648.9 GB | 0.4789 | 2.3639 | 81.219% | IQ2_XXS + IQ1_S |
| UD-IQ2_XXS | 711.1 GB | 0.3784 | 2.1266 | 84.127% | IQ2_XXS + IQ1_M |
| **UD-Q2_K_XL** (our experts) | 861.3 GB | **0.1779** | 1.7359 | **90.390%** | IQ2_XS (+ IQ3_XXS) |
| UD-Q4_K_XL (our quality reference) | 1,510 GB | - | 1.4579 | - | MXFP4 (native) + Q8_0 rest |
| UD-Q8_K_XL (lossless) | 1,560 GB | - | 1.4581 | - | MXFP4 + BF16 rest |

(LOCAL-K3 printed 0.3943 / 2.1160 for UD-IQ2_XXS; the Unsloth docs saved in `unsloth_docs.txt` give 0.3784 / 2.1266.) Three smaller Unsloth quants (UD-Q1_0, UD-TQ1_0, UD-TQ2_0) use ggml type ids 64-66 that mainline does not define and do not load.

## Why dense had to shrink (HANDOFF §4)

Dense is ~61 GB per token (81% of traffic) at Q8_0 and does not fit 24 GB:

| Dense precision | Size | Layers that fit (~21 GB usable) | Dense left on CPU | CPU cost |
|---|---:|---:|---:|---:|
| Q8_0 (as shipped) | 59.1 GB | 33 | 38.1 GB | 323 ms/token |
| Q6_K | 45.6 GB | 43 | 24.6 | 209 ms |
| Q5_K | 38.2 GB | 51 | 17.2 | 146 ms |
| Q4_K | 31.3 GB | 62 | 10.3 | 87 ms |
| Q3_K_M | 27.1 GB | 72 | 6.1 | 52 ms |
| Q3_K_S | 23.6 GB | 83 | 2.6 | 22 ms |
| Q2_K | 18.2 GB | 107 (all) | 0 | 0 |

Every GB moved from CPU to GPU removed ~8.5 ms/token. A first mixed Q2_K/Q4_K plan (`k3mix.txt`) dry-ran to 27.27 GiB dense / 732.99 GiB experts.

### `llama-quantize` blockers (HANDOFF §5.3)

1. `--tensor-type-file` does not support comments (`malformed tensor type '#'`): keep a stripped `.types` file.
2. A type name in the wrong argument slot prints usage and **exits 0**.
3. Over-broad regexes hit the wrong tensors (`.*exps\.weight` also matched dense `blk.0.ffn_gate.weight`); rules are first-match-wins on lowercased names. Always dry-run and grep.
4. **An IQ target type demands an imatrix even for a same-type copy** (`requires_imatrix` is derived from the target type before the same-type byte-copy check). Workarounds: a 1-3 line patch, or a fresh imatrix.
5. The 13 IQ3_XXS expert tensors must be pinned by name before the IQ2_XS catch-all, or they are requantized downward.

`-ot` / `-ncmoe` control placement only; they cannot keep a tensor byte-identical in a rewritten file.

## dense-v1 (2026-10-10; HANDOFF 18.2-18.3)

User approved a lossy dense-only requant. Rules (`k3dense-v1.txt`): experts byte-identical (13 IQ3_XXS pinned first, then `_exps=iq2_xs`); `attn_output`, KDA `attn_v`, `ffn_down_shexp` -> IQ3_XXS; `output`, MLA `q_a/q_b/kv_a`, layer-0 FFN -> Q4_K; KDA `attn_q/k`, `ssm_g`, shared-expert gate/up, `attn_gate`, latent projections `ffn_routed_up/down` -> IQ2_S; router F32 -> F16 (needs a `llama-quant.cpp` patch); `k_b/v_b/f_a/f_b` stay Q8_0 (128-wide rows); `token_embd` stays Q8_0 (CPU gather). Budget: 21.23 GB GPU-resident dense. Output 765 GiB in 19 shards.

- **imatrix:** `/data2/k3eval/imatrix-k3-calv3.gguf` from bartowski's `calibration_datav3.txt`, 126 x 512 chunks, `-ngl 8 -ncmoe 93 -b 2048 -ub 2048`, 53 min. Op-offload at ub 2048 gave 34 tok/s prompt processing vs 11.9 at ub 512 and 6.5 with `-nopo`.
- **Speed:** `-ngl 99 -ncmoe 93 -nopo 1` pinned: **4.19 tok/s** (reps 4.237 / 4.183 / 4.200 / 4.154 / 4.185); real generation 4.23 tok/s. (UD-Q2_K_XL best: 1.69 tok/s.)
- **Quality vs UD-Q2_K_XL** (wikitext-2 test, 48 x 512): PPL 2.543 vs 1.774 (**+43%**), mean KLD **0.361**, median 0.087, p90 1.04, p99 3.33, max 9.4, same top token **82.1%**, RMS dp 26%. Heavy tail: most tokens close, some far. A fast-doubling Fibonacci prompt still gave correct code and the exact fib(100).
- **Later, vs native 4-bit** (31.z): KLD **0.243**, median 0.142, p99 1.50, same top 78.4%, PPL 6.66 (native 5.44).

## Dense precision study vs native (2026-10-31 to 11-01; HANDOFF 31.z)

Tools: loader override `LLAMA_TENSOR_SOURCE=<other GGUF>` + `LLAMA_TENSOR_SOURCE_RE=<regex>` (commit `a8405846e`; matching tensors read from the native UD-Q4_K_XL, shapes equal, types may differ); overlays written by `llama-quantize` with `LLAMA_QUANT_ONLY=<regex>` (commit `8a1e0e5a5`, requantized from the native Q8_0 with the imatrix); `kld_dense.sh` / `kld_overlay.sh` / `kld_table.py`. Reference: native, PPL 5.4421 on `unseen-26k.txt`; experts on the CPU in every run.

- The native model is 1.4 TB and streams from the SSDs at only 0.3-0.8 GB/s through page faults: 2637 s for the first reference pass, 2 h 50 min in all. Loader fix `aa0525637`: tensor-source files were mapped with MAP_POPULATE, so each run read the whole 1.4 TB and evicted dense-v1 (17 min load + 34 min first pass).
- **dense_all** (every non-expert tensor from native at Q8_0, experts still 2-bit): KLD **0.071** -> ~70% of dense-v1's gap to native was the dense requant, not the 2-bit experts. But all dense at Q8_0 = +37 GiB of VRAM.

Full results (`kld_table.py` output; dKLD vs dense-v1 0.2433; +GiB = VRAM the swapped tensors add; "nan" = overlay requantized, GiB in the notes):

| Run | KLD | dKLD | Median | p99 | Top % | PPL | +GiB | dKLD/GiB |
|---|---|---|---|---|---|---|---|---|
| v1 | 0.2433 | +0.0000 | 0.1421 | 1.503 | 78.37 | 6.661 | 0.00 | |
| dense_all (Q8_0) | 0.0708 | -0.1725 | 0.0316 | 0.522 | 88.39 | 5.654 | 37.00 | 0.0047 |
| attn_output (Q8_0) | 0.1879 | -0.0554 | 0.1062 | 1.083 | 80.16 | 6.337 | 5.19 | 0.0107 |
| attn_v | 0.2410 | -0.0024 | 0.1384 | 1.380 | 78.91 | 6.645 | 3.85 | 0.0006 |
| ssm_g | 0.2402 | -0.0031 | 0.1382 | 1.452 | 78.67 | 6.653 | 4.20 | 0.0007 |
| shexp_gate_up | 0.2181 | -0.0253 | 0.1282 | 1.293 | 80.18 | 6.522 | 5.60 | 0.0045 |
| shexp_down | 0.2379 | -0.0055 | 0.1460 | 1.423 | 78.94 | 6.632 | 2.56 | 0.0021 |
| latent_proj (`ffn_routed_up/down`) | 0.1596 | -0.0838 | 0.0847 | 0.969 | 82.31 | 6.168 | 3.27 | 0.0256 |
| attn_gate | 0.2433 | -0.0001 | 0.1413 | 1.451 | 78.50 | 6.670 | 1.46 | 0.0001 |
| small_q4k (MLA q_a/q_b/kv_a, output, layer-0 FFN) | 0.2394 | -0.0039 | 0.1413 | 1.427 | 78.62 | 6.673 | 1.37 | 0.0029 |
| router (F32) | 0.2482 | +0.0048 | 0.1386 | 1.523 | 78.42 | 6.702 | 1.10 | |
| kv_q8 (8-bit KV) | 0.2440 | +0.0006 | 0.1409 | 1.411 | 77.79 | 6.681 | 0.00 | |
| exps_00_19 (native MXFP4 experts, layers 0-19) | 0.2486 | +0.0052 | 0.1409 | 1.395 | 78.57 | 6.725 | 125.97 (RAM) | |
| lat_q6k (overlay) | 0.1616 | -0.0818 | 0.0849 | 0.938 | 82.41 | 6.171 | 2.20 | |
| lat_q5k | 0.1700 | -0.0734 | 0.0863 | 0.978 | 82.31 | 6.227 | 1.62 | |
| lat_ao_q8 (latent + attn_output Q8_0) | 0.1061 | -0.1372 | 0.0565 | 0.681 | 85.00 | 5.869 | 8.45 | 0.0162 |
| kda_qk (KDA attn_q/k Q8_0) | 0.2398 | -0.0035 | 0.1369 | 1.388 | 78.62 | 6.680 | 8.40 | 0.0004 |
| ao_q5k | 0.1905 | -0.0528 | 0.1073 | 1.139 | 79.91 | 6.359 | 2.32 | |
| sgu_q5k (shexp gate/up) | 0.2146 | -0.0288 | 0.1265 | 1.320 | 80.30 | 6.492 | 2.77 | |
| sgu_q4k | 0.2156 | -0.0277 | 0.1280 | 1.236 | 79.98 | 6.471 | 1.83 | |
| router_q8 | 0.2471 | +0.0037 | 0.1439 | 1.447 | 77.88 | 6.686 | -0.52 | |
| **c3r = dense-v2** | **0.0953** | -0.1481 | 0.0482 | 0.662 | 86.09 | 5.774 | 5.84 | |
| c2r | 0.1099 | -0.1335 | 0.0576 | 0.755 | 84.58 | 5.875 | ~4.0 | |

Readings:

- **latent_proj, attn_output and shexp gate/up hold ~all of the dense damage** (-0.084, -0.055, -0.025 of dense_all's -0.172); everything else is noise-level.
- Gains are **additive**: lat_ao_q8 (-0.137) = the sum of the two singles.
- Cheaper types get nearly all of the gain: latent Q6_K 98% of Q8_0's gain; attn_output Q5_K 95%; shexp gate/up Q4_K = Q8_0 within noise.
- Router: F16 is enough (F32 no better), and Q8_0 changes nothing measurable while freeing ~0.52 GiB of VRAM.
- 8-bit KV cache: free quality-wise at 2k context (+0.0006).

Candidate configs (additive prediction): C1 latent Q6_K (KLD ~0.162, +2.2 GiB); C2 + attn_output Q5_K (~0.108, +4.5); C3 + shexp gate/up Q4_K (~0.080, +6.35); the "r" variants add router Q8_0.

| Speed test (96k ctx, f16 KV) | tok/s | ms/step | Accept | tok/step | 7.2k prompt | Peak VRAM MiB |
|---|---|---|---|---|---|---|
| c2r (latent Q6_K + attn_output Q5_K + router Q8_0), cache 11, -ts 49,44 | 10.21 | 245 | 0.77 (was 0.69-0.70) | 2.50 | 29.3 tok/s | 23186 / 23707 |
| c3r (c2r + shexp gate/up Q4_K), cache 9, -ts 49,44 | 9.93 | 247 | 0.74 | 2.46 | 29.25 tok/s | 23274 / 23791 |
| dense-v2 final (c3r, cache 9, -ts 50,43, ub 512) | 10.17 | 245 | 0.76 | 2.49 | 29.15 tok/s | 23688 / 23486 |

**The quality upgrade paid for itself:** a target closer to native agrees more with the DFlash draft (trained on native K3): acceptance 0.69-0.70 -> 0.74-0.77, which offset the 7 cache slots given up (16 -> 11 slots cost only ~3 ms/step). c3r measured KLD 0.0953 (less gain than the additive 0.080 estimate) but **-61% KLD at the same speed**. Installed as **dense-v2**: `/data2/kimi-k3/K3-dense-v2-overlay.gguf`, 13.7 GiB (latent Q6_K, attn_output Q5_K, shexp gate/up Q4_K, router Q8_0), default since 2026-11-01. Loader commit `b4f56553a` applies the source only to models of its architecture (the DFlash draft has same-named tensors).

## Native-precision experts per layer range (HANDOFF 31.z)

Native MXFP4 experts read from UD-Q4_K_XL (`LLAMA_TENSOR_SOURCE2`, commit `6bf6407e6`) on top of dense-v2 (0.0954 at batch 3 without skip):

| Layers at native | KLD | Median | Top % | PPL | dKLD vs dense-v2 | Per layer |
|---|---|---|---|---|---|---|
| 1-19 | 0.0956 | 0.0478 | 85.85 | 5.776 | none | |
| 20-39 | 0.0955 | 0.0487 | 85.85 | 5.781 | none | |
| 40-59 | **0.0778** | 0.0412 | 87.05 | 5.718 | -0.018 | |
| 60-79 | 0.0819 | 0.0404 | 86.46 | 5.697 | -0.013 | |
| 80-92 | 0.0829 | 0.0392 | 86.83 | 5.711 | -0.012 | |
| 40-49 | 0.0922 | 0.0471 | 85.80 | 5.792 | -0.0031 | -0.00031 |
| 50-59 | 0.0808 | 0.0420 | 86.22 | 5.723 | -0.0145 | -0.00145 |
| 60-69 | 0.0881 | 0.0441 | 86.44 | 5.738 | -0.0072 | -0.00072 |
| 70-79 | 0.0894 | 0.0455 | 86.46 | 5.738 | -0.0059 | -0.00059 |
| 80-86 | 0.0897 | 0.0434 | 85.97 | 5.727 | -0.0056 | -0.0008 |
| 87-92 | 0.0885 | 0.0432 | 86.29 | 5.763 | -0.0068 | -0.00113 |

Native experts in layers 0-19 on top of dense-v1 also changed nothing (+0.0052). Best 16 layers (50-59 + 87-92) predicted KLD ~0.074.

Speed (native layer experts are 14.6 GiB vs ~7 at 2 bits):

| Config | tok/s | ms/step | Accept | Prompt |
|---|---|---|---|---|
| dense-v2 | 10.17 | 245 | 0.76 | 29.15 |
| x4_5054 (native layers 50-54, cache 8, -ts 51,42, page cache warmed) | 8.50 | 292 | 0.76 | 26.4 |
| x4_5059 (native layers 50-59, cache 7, -ts 51,42) | 7.19 | 336 | 0.72 | 24.7 |

Cost is linear, **~9 ms/step per native layer** (~0.3 tok/s each), from CPU bandwidth, not RAM overflow (65 GiB free when warmed; the first x4_5059 run also had page-cache misses because process memory had been ignored when sizing). 50-59 buy -0.0146 KLD for -3 tok/s: poor value next to the skip curve; rejected (~0.008 KLD per tok/s or worse, like every lever). A middle ground (3-bit IQ3_XXS/Q3_K experts for 50-59) was proposed but falls under "no requants".

## 8-bit KV cache (HANDOFF 31.z, 31.z10)

- Quality: kv_q8 +0.0006 vs dense-v1 at 2k context (above); with the deterministic `kld_pos` harness, u045 f16 0.0327 vs u045 + q8 KV 0.0307: quality-neutral.
- **Nov 1: it crashed the server** (CUDA illegal memory access on GPU0 in decode after 2 requests; q8 KV forces flash attention for the 576/512 MLA heads). Dropped then.
- **Nov 4: works.** New K3-shaped `FLASH_ATTN_EXT` test cases (576/512, {96,1} GQA, kv 512-98304, 1-4 tokens, V view of K) all pass for q8_0 (commit `10b684e22`; the f16 cases at kv >= 32k exceed the 5e-4 NMSE tolerance on random data = flat attention, not a bug). Quantized KV turns on the Hadamard attention rotation (`LLAMA_ATTN_ROT_DISABLE=1` disables). `speed_test` kvq8_rot / kvq8_norot: no crash, 251 / 252 ms/step, prompts 31.3 / 31.2 tok/s, **peak VRAM 23414 / 22555 vs f16 23816 / 22975 MiB (-0.4 GB per GPU)**. HANDOFF 31.z10 guesses the old crash "(~2026-10-26)" was "the deferred-split input-copy bug or similar since fixed", but that fix (`4f33592bc`) dates from 2026-10-11, before the crash (logged 2026-11-01 in 31.z and in `serve_k3.sh`); the cause was never identified.
- The freed VRAM was spent on ub 1024 (prompts 31 -> 47 tok/s); cache 10 instead gave 249 vs 251 ms/step. KV at 96k: 1296 -> 689 MiB per GPU; at 128k q8 would still use less than f16 at 96k.

## Other quantization findings

| Item | Result | Source |
|---|---|---|
| Lossless entropy coding of IQ2_XS expert bytes | order-0 entropy 568.41 of 592 bits per block: at most ~4.0% compressible vs 12.3% needed: dead. (The `iq2xs_entropy.py` printed total double-counts; the per-field numbers are right.) | K3-IDEAL-BOUND §5, HANDOFF §10 |
| Draft quantization: Q8_0 vs IQ4_XS DFlash draft | Q8_0 8.90 tok/s, accept 0.76, 286 ms/step vs IQ4_XS 9.01 / 0.77 / 283 (run D vs C, greedy): no gain from the bigger draft. (The distillation research doc of 2026-11-04 says this was never measured; run D did measure it, though greedy and on dense-v1) | HANDOFF §30, dflash-distill-research §7 |
| GPU mat-vec cost of low-bit types | per weight, IQ2_S ~ Q4_K in time (decode-bound on CUDA too); see [GPU kernels](08-gpu-kernels.md) | HANDOFF 18.3 |
| Better 2-bit expert format / per-layer bit allocation | planned 2026-11-03 (est. +5-10%), dropped 2026-11-04 under "no requants" | HANDOFF "Agreed plan", memory note |
