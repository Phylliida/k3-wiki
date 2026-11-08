# GPU kernels and fused ops

[Back to index](../README.md) · Related: [where the time goes](03-where-the-time-goes.md) (serial GPU path), [MoE placement](05-moe-placement-cache-prefetch.md) (expert-major kernel, cache kernels), [prompt processing](12-prompt-processing.md) (range split, fused MoE op), [long context](13-long-context.md)

The GPU runs attention, router, norms, latent projections and shared experts: the serial path the CPU waits on each layer (~60 ms of a ~250-273 ms step). All changes here are lossless up to float summation order.

## Results at a glance

| Change | Measured | Date / commit | Verdict |
|---|---|---|---|
| Persistent IQ2_S / IQ3_XXS mat-vec (sign trick, grid in smem) | e.g. iq2_s 12288x7168 63.9 -> 44.6 us; model 4.65 -> 4.78 tok/s | 10-11, `604bafadb` | kept |
| 5-op residual mix (concat + score + mix in one pass) | 5.70 -> 5.79 tok/s | 10-11, `604bafadb` | kept |
| Warp top-k for the router (k <= 32, 896 columns) | 22 -> 12.3 us; 5.76 tok/s (within noise) | 10-11, `604bafadb` | kept |
| Expert-major MoE mat-vec (`mul_mat_vec_q_moe_xm`) | each streamed expert crosses PCIe once per verify batch | 10-12, `4f867184b` | kept |
| MXFP4 in the expert-major kernel | enables native-expert layers on GPU paths | 10-31, `299c3ecbc` | kept |
| Multi-column persistent mat-vec (1-4 columns) | n=3 iq2_s 68.8 -> 53.5 us; 274-275 -> 266 ms/step | 11-02, `cf1f96157` | kept |
| CUDA graph key fix | removed ~14 ms from 2->3 token batch switches | 11-05, `c52898b5f` | kept |
| Gated delta net 4-columns-per-warp rewrite | 33 -> 33.7 us | 11-02 | reverted |
| Rows per CUDA block 4 / 8 (generic table) | +-1 us at 4, worse at 8 | 11-02 | kept 2 |
| Fused residual mix op `GGML_OP_RES_MIX` | no step-time change (decode 245.3-245.5 vs 245.4-245.8 ms) | 11-06, `c9666450c` | off |

## Persistent IQ2_S / IQ3_XXS mat-vec (HANDOFF §20)

**Diagnosis** (standalone harness `mmvq_bench.cu`):

- K3 dense GPU bytes per token in dense-v1: IQ2_S 10.1 GiB (k=7168: m=12288 6.2 GiB, m=6144 2.4, m=3584 0.7; k=3584, m=7168 0.7), IQ3_XXS 6.7 GiB.
- The old kernel took ~64 us at m=12288, k=7168 for IQ2_S, IQ3_XXS and Q4_K alike: bound by work per element, not bytes.
- Probes: same loads + trivial math = 37 us (streaming read, ~800 GB/s: the load pattern is fine); full math on L1-resident weights = 50 us. So decode is the limit.
- SASS: 165 instructions per 32-weight unit, ~88 INT ALU; `__vcmpne4` / `__vsub4` are emulated on sm_86 (~10 ops per 4 weights). The kernel draws 350 W (power cap) and runs at ~1.5 GHz.
- Did not help: warp-per-row, cp.async staging of weights into smem, a nibble-coded IQ2_S grid in smem (+ALU > saved LSU), more warps per block.

**Fix** (`mul_mat_vec_q_pers*` in `mmvq.cu`):

- Signs: `dot(g*s, u) = dot(g ^ m, u) - dot(m, u)` with m = 0xFF where the sign bit is set; m built with `(rep & 0x08040201) + 0x7F7F7F7F` then `prmt.b32 ... 0xBA98`. One XOR + one extra dp4a per 4 weights instead of vcmpne4 + xor + vsub4. Integer result exact.
- Persistent blocks (occupancy x SMs), 4 warps, 2 rows per iteration; grid table copied to shared memory once per block (8 KB IQ2_S / 1 KB IQ3_XXS); each thread keeps its q8_1 chunks of y in registers.
- Used only for IQ2_S/IQ3_XXS, one column, no ids, no fusion, NVIDIA Ampere+. `GGML_CUDA_MMVQ_PERS=0` disables. MUL_MAT 1297/1297 OK.

| Shape (test-backend-ops, us) | Old | New |
|---|---:|---:|
| iq2_s 12288x7168 | 63.9 | 44.6 |
| iq2_s 6144x7168 | 34.3 | 25.4 |
| iq2_s 3584x7168 | 22.5 | 17.2 |
| iq2_s 7168x3584 | 21.4 | 18.2 |
| iq2_s 7168x12288 | 62.6 | 43.6 |
| iq2_s 7168x6144 | 34.9 | 25.3 |
| iq3_xxs (same six shapes) | 63.4 / 34.0 / 22.2 / 24.7 / 61.6 / 35.0 | 49.0 / 27.3 / 18.6 / 19.3 / 48.8 / 27.2 |

End to end (dense-v1, llama-bench tg32): **4.65 -> 4.78 tok/s**; real generation 4.83 tok/s. The greedy sample diverged from the previous one after ~60 tokens (float sum order changed; output still correct), which led to measuring the KLD noise floor ([expert skipping and quality](06-expert-skipping-and-quality.md)).

### Multi-column version for verify batches (HANDOFF 31.zz, commit `cf1f96157`)

The persistent kernel was single-token only, so 3-token verify steps used the generic mmvq. Each 32-weight unit is now decoded once and dotted with every column (y in registers, units_per_thread x ncols <= 12). `GGML_CUDA_MMVQ_PERS_MAXCOLS=1` restores single-token only.

| n=3 (us) | Old | New |
|---|---:|---:|
| iq2_s 12288x7168 | 68.8 | 53.5 |
| iq3_xxs 12288x7168 | 67.2 | 55.7 |
| iq3_xxs 7168x6144 | 36.9 | 32.5 |

n=1 unchanged; 40/40 new eval cases OK. Server bench: 274-275 -> **266 ms/step** (9.23 tok/s, accept 0.74).

Isolated serial-path kernels at n=3 after this: iq2_s 53 us x2.6 per layer, q5_K `attn_output` 77 (~790 GB/s), q6_K latent 31 + 32 (~670 GB/s), iq3_xxs MLA q 55 x0.75, gated delta net 33 x0.75, router q8_0 10. **Rows per CUDA block** for 2-8 columns (generic table): 2 (default) vs 4 vs 8 at n=3: within +-1 us at 4, worse at 8 (q4_K 35.6 -> 40.3, q5_K 76.9 -> 81.8 us). Not L2-bound; the small mat-vecs are at their practical limit (~5-8 us fixed launch/tail each).

## Router top-k (HANDOFF 20.8)

CUDA ARGSORT(896) was 22 us (bitonic), TOP_K the same (argsort + copy). New warp top-k fast path in `top-k.cu` (k <= 32, ncols <= 1024, one warp per row, registers + taken bitmask, descending, ties -> lower index): 12.3 us in test-backend-ops including launch; TOP_K 525/525. K3 uses `ggml_top_k` in `build_moe_ffn`. End to end 5.76 tok/s (within noise of 5.79).

## Residual mix (HANDOFF 20.8; SESSION3 night11)

K3's cross-layer residual attention mixes the banked checkpoints and the current stream twice per layer (186 `res_mix` per token).

- **5-op version (2026-10-11):** the current stream is concatenated onto the checkpoint stack and scored/mixed in one pass (concat, rms_norm x w, sum_rows, soft_max, dsv4_hc_pre = 5 ops instead of 11). Block size 12, so the stack holds <= 8 + 1. **5.70 -> 5.79 tok/s.**
- **Fused op `GGML_OP_RES_MIX` (2026-11-06, commit `c9666450c`):** `ggml_res_mix(stack, x, score_w, eps)`, no concat; CPU reference; CUDA = 2 kernels. Motivation: the Nov 2 graphs-off profile put the chain at ~37 us per call, ~199 calls/step (~7 ms/step).
  - CPU check: max abs diff 2.6e-4 on outputs of magnitude ~1-30 (24 cases). test-backend-ops CUDA vs CPU: 20/20 RES_MIX cases OK.
  - Perf (decode shape): 9.6 / 10.2 / 12.0 us per call at n_ckpt 5 / 12 / 23 (3 tokens); 231-462 us at 512 tokens. (The RES_MIX_UNFUSED perf lines, 2.1-2.8 us, are invalid: test-backend-ops perf re-runs only the graph's last node.)
  - **KLD vs the `kld_oracle` reference: off -0.000000 (bit-identical); on 0.006535 +- 0.000293** (97.52% same top), by position 0.0063 / 0.0068 / 0.0065, repeated exactly (deterministic). Control `GGML_CUDA_DISABLE_FUSION=1` was bit-identical (so not a rounding control). Unresolved whether this is K3 amplifying a ~1e-5 relative difference (different summation order over 7168 terms, then softmax) or a real bug; would need a double-precision reference.
  - **Server A/B (interleaved, no throttling):** resmix_off_a 10.10 tok/s 247 ms/step, on_a 10.22 / 247, off_b 10.01 / 248, on_b 10.20 / 248; 2-draft step median 251 ms in all four; decode 245.3 / 245.5 (off) vs 245.4 / 245.8 (on). The on runs' higher tok/s is acceptance variance (2.53 vs 2.49 tok/step). **No gain: dropped** (stays opt-in via `LLAMA_K3_FUSED_RESMIX=1`, off). With CUDA graphs on, the ~800 tiny kernels per step it removes cost ~nothing. The similar KDA short-conv fusion idea (est. ~2-3 ms/step, ~690 fewer kernels) was dropped for the same reason.

## Gated delta net (KDA) (HANDOFF 31.zz)

Isolated 33 us at K3's decode shape (96 x 128, 3 tokens), ~770 GB/s; the 84 us in the op profile was host-side launch prep counted with graphs off. A 4-columns-per-warp rewrite gave 33.7 us and was reverted; its K3-shape eval/perf cases stay (45/45 OK).

## MLA flash attention at long context (HANDOFF 31.zz; SESSION3 "Long-context MLA attention")

K 576 / V 512, 96 heads on one KV head, 3 tokens: **34 us at 512 KV, 127 at 8k, 434 at 32k, 844 at 64k, 1257 at 96k** (x23 MLA layers: ~19 ms/step at 64k, ~29 at 96k). Compute-bound: ~62 GFLOP per layer at 96k, ~49 TFLOPS = ~70% of the 3090's FP16-with-FP32-accumulate peak. `GGML_PREC_DEFAULT` is the same speed.

Levers checked by code reading (no runs): the path is `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>`; on Ampere `T_C_VKQ = tile<16,8,half2>`, so **P*V already accumulates in FP16** (142 TFLOPS); only Q*K^T accumulates in FP32 (71 TFLOPS), ~69% of the attention time.

| Lever | Estimate | Verdict |
|---|---|---|
| int8 Q*K^T (q8 Q x q8_0 K) | I2F conversions at ~1/8 FMA rate: ~-25% of Q*K^T, ~-17% of attention (~-3 ms/step at 64k, ~-5 at 96k) for a new MMA kernel | low value |
| FP16 accumulation for Q*K^T | scores ~20-50 in FP16 shift probabilities by percents | too lossy |
| Split heads over both GPUs | needs the KV cache on both GPUs (VRAM full) | no |

Verdict: the long-context attention cost is near the practical limit here.

## CUDA graphs

- Worth **~23 ms/step** (~9%) at decode: causal draft without graphs 9.08 tok/s, 274 ms/step vs 251 with graphs (SESSION3); in the step-transition table graphs saved ~16 ms in every case; at K=1 (Oct 10) ~8 ms.
- **Graph key collision (commit `c52898b5f`):** the key was the first node's address, which collides between graphs of different batch sizes rebuilt into the same memory, so alternating 2- and 3-token verify batches re-captured graphs every switch (step log: 1->1 drafts 221 ms, 2->2 268, but 2->1 241, 1->2 287: ~20 ms per size change). The key now includes the node count and first/last node shapes; this removed ~14 ms from the 2->3 direction. A residual ~10 ms per switch comes from llama-side graph rebuild + scheduler allocation ([speculative decoding](09-speculative-decoding.md)).
- `GGML_CUDA_GRAPH_OPT=1`: hurts (4.96 vs ~5.58 tok/s, HANDOFF 20.7) / no gain (20.5).
- `GGML_CUDA_GRAPH_STATS` counters (commit `0099fa178`), e.g. at 140k calls: padded cutoff run 125,500 replays / 3,243 captures / 11,257 direct (2.3% captures); unpadded cutoff 118,627 / 6,867 / 14,506 (4.9%); no cutoff 123,258 / 3,524 / 13,218 (2.5%). They printed via INFO, which the server log drops; switched to WARN in `8092e89fb`.
- Early single-GPU DFlash configs had to run with graphs off for VRAM; graphs came back with the second GPU (run B: 8.69 -> 9.04 tok/s, HANDOFF §30).

## Remaining GPU ideas (HANDOFF "Plan for the next sessions")

Faster small-batch kernels for IQ2_S / IQ3_XXS / Q6_K attention-side weights (~40-50% of bandwidth now; est. +4-6%; partly done by `cf1f96157`); splitting a layer's dense work across both GPUs (the other GPU idles during each layer; no P2P makes it hard).
