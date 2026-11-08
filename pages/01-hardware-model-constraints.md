# Hardware, model and constraints

[Back to index](../README.md) · Related: [current setup](02-current-setup.md), [memory bandwidth and thermals](14-memory-bandwidth-thermals.md), [quantization](07-quantization-dense-kv.md)

## The machine

| Part | Fact | Source |
|---|---|---|
| CPU | AMD EPYC 7542, 32 cores / 64 threads, Zen 2, AVX2 + FMA, **no AVX-512**. 4 CCDs = 8 CCXs x 4 cores, 16 MB L3 per CCX (128 MB total), 1 NUMA node. Boost max 3408 MHz, `scaling_max_freq` 2900 MHz | HANDOFF §13, K3-IDEAL-BOUND §1, HANDOFF 31.z6 |
| Board | Supermicro H12SSL-i, BIOS 3.7 (05/2026); supports EPYC 7002 (Rome) and 7003 (Milan), up to 280 W | HANDOFF 31.z6 |
| RAM | 995 GiB: 8 x 128 GB DDR4-2933 **LRDIMM**, one DIMM per channel (DIMMA1-DIMMH1). Theoretical peak 187.7 GB/s; measured streaming read **119.6 GB/s** (32 threads, 4K pages); the 4-CCD GMI links cap it (39.1 GB/s per CCD, 123.9 GB/s with all 4 CCDs) | HANDOFF §13, 31.z6, 31.z7 |
| GPUs | 2x RTX 3090 (24576 MiB each, sm_86). GPU0 at `81:00.0`, GPU1 at `c1:00.0` (own root complex `c0:03.1`). PCIe Gen4 x16 each, **no P2P** between them (`nvidia-smi topo -p2p r`: CNS), so cross-GPU tensors go through host RAM. GPU1 also drives the display (~0.5-1 GB VRAM). One card until 2026-10-12, the second borrowed for a day, both permanent from 2026-10-25 | HANDOFF §13, §30, §31, 31.zz |
| PCIe host reads | DMA 25.0 GB/s per card, **additive** to CPU reads (CPU 101.0 GB/s with DMA vs 106.3 solo); HMM in-kernel reads 25.2 GB/s (uint4 loads); both GPUs together 52.5 GB/s | HANDOFF §13, 20.6, 20.7, §30 |
| HMM | NVIDIA open kernel module 580.82 on Linux 6.12, `PageableMemoryAccess = 1`: GPU kernels can read the model's read-only file mmap directly. Mapping runs at ~2 GiB/s (driver limit); GPU page tables cost ~1.9-2 MiB of VRAM per GiB mapped | HANDOFF 20.7, §31, 31.z |
| Disks | Working model on `/data2` (XFS, NVMe; ~1.0 TB free at the start, ~20 GB free by 2026-11-04). The 1.4 TB native reference model is spread over `/data2`, `/mnt/models` and `/mnt/items` (SSDs); streaming it through page faults ran at only 0.3-0.8 GB/s (a 990 Pro idled at 324 MB/s). `/mnt/items` holds the distillation dumps | HANDOFF §5.4, §13, 31.z, 31.z10 |
| Cooling | Supermicro BMC fan control. Fan mode must be **Full** (`ipmitool raw 0x30 0x45 0x01 0x01`): on "Optimal" the DRAM throttles after ~3 min of load. A third fan was added on 2026-10-26 | HANDOFF 31.x, [thermals page](14-memory-bandwidth-thermals.md) |
| OS details | Transparent huge pages `madvise`; `ulimit -l` 8 MB (not enforced by the NVIDIA driver for `cudaHostRegister`); IOMMU switched from DMA-FQ to passthrough (`iommu=pt`) on 2026-11-03 (~2-2.5% faster, see [thermals page](14-memory-bandwidth-thermals.md)) | HANDOFF 18.4, 20.6, 31.z5, 31.z7 |

This is a **streaming-read box, not a compute box** (HANDOFF §9): IQ-quant decoding needs AVX-512 to be cheap, so on this CPU the most compressed weights are the slowest to decode, and DRAM bandwidth (~115-120 GB/s usable) bounds the expert work.

## The model: Kimi-K3

Architecture facts as recorded in the notes (LOCAL-K3 §2 read them from Moonshot's `config.json`; K3-SETUP "The model"; HANDOFF §4, §23):

| Property | Value |
|---|---|
| Total / active parameters | ~2.8 T total (llama.cpp labels it `LLM_TYPE_2_8T_A50B`; the kimi-k3-mlx accounting gives 2,780.0 B); **104 B active** per Moonshot's README (LOCAL-K3 §2 explains why llama.cpp's "A50B" label counts only routed experts) |
| Layers | 93: 1 dense lead layer + 92 MoE layers (`first_k_dense_replace` 1) |
| Attention | 69 KDA (Kimi Delta Attention, linear, recurrent state) + 24 MLA layers (every 4th layer). MLA: 96 heads on one latent KV head, K 576 / V 512, with an output gate |
| Width | hidden 7168; dense lead layer FFN 33792 wide |
| Routed experts | 896 per layer, top-16, in a **3584-wide latent space** (3584 -> 3072 -> 3584, `ffn_routed_down/up` project into and out of it); ~9.55 MB per expert per layer at 2 bits (3 matrices x 3,182,592 B) |
| Shared experts | 2 (`num_shared_experts`), always active; the GGUF stores them fused as one 6144-wide (2 x 3072) `ffn_{gate,up,down}_shexp` set per layer (`k3q2_tensors.json`: `blk.5.ffn_gate_shexp.weight` [7168, 6144]), which is why one speed sheet in `k3-research/sheet/` says "1 shared expert" |
| Activation | SiTU instead of SwiGLU (beta 4, linear beta 25) |
| Residual stream | cross-layer residual attention: checkpoints banked every 12 layers (`attn_res_block_size` 12), mixed by softmax scores (`res_mix`, `ggml_dsv4_hc_pre`) |
| Router | sigmoid scores + bias, trained with Quantile Balancing (deliberately flat expert usage); stored F32 in the GGUF (2.36 GB) |
| Vocab / context | 163840 tokens; 1,048,576 max positions |
| Native release | MXFP4 routed experts (quantization-aware training from SFT on, MXFP8 activations), ~1.56 TB |
| Per-sequence state | KDA recurrent state 443 MiB per copy (R 29 + S 414 MiB); MLA KV 27.6 KB/token at f16 (HANDOFF 31.z8) |

Note: HANDOFF §23 describes the KDA state as "70 layers x 96 heads x 128x128 f32"; 69 KDA layers x 96 x 128 x 128 x 4 B = 414 MiB matches the "S 414 MB" of 31.z8, so "70" looks like a typo.

### Per-token byte accounting (UD-Q2_K_XL, from `k3q2_tensors.json`, HANDOFF §4 / K3-IDEAL-BOUND §2)

| Class | GB on disk | Read per token |
|---|---:|---:|
| Routed experts (IQ2_XS 749.971 + IQ3_XXS 49.094) | 799.065 | **14.269 GB** (16 of 896) |
| Non-expert ("dense"): attention, shared experts, router, latent projections, head | 62.2 (56.7 G params) | ~61 GB (81% of per-token weight traffic) |
| Whole file | 861.27 (802.13 GiB) | ~75.2 GB |

Biggest dense classes: `attn_output` 8.70 GB; KDA `attn_q/k/v` 6.46 each; `ssm_g` 6.46; shared-expert down/gate/up 4.30 each; router `ffn_gate_inp` (F32) 2.36; latent projections `ffn_routed_down/up` 2.51 each; `attn_gate` 2.25; `output` 1.25; `token_embd` 1.25 (gather only).

The 13 IQ3_XXS expert tensors in UD-Q2_K_XL: `blk.{12,24,25,28,36,37,48,60,72,84,91}.ffn_down_exps.weight` plus `blk.91.ffn_{gate,up}_exps.weight` (HANDOFF §5.3).

## Model files used

| File | What | Notes |
|---|---|---|
| `Kimi-K3-UD-Q2_K_XL` (Unsloth) | 19 shards, 802.13 GiB. Experts IQ2_XS (+13 IQ3_XXS), dense Q8_0, router F32 | Starting point. Unsloth measured it at KLD 0.178 / top-1 90.4% vs lossless. Deleted 2026-10-31 after its experts were verified byte-identical to dense-v1's (276/276 tensors) |
| `K3-dense-v1` | 19 shards, 765 GiB. Same expert bytes, dense requantized to fit ~20 GB of VRAM (IQ2_S / IQ3_XXS / Q4_K mix, router F16) | Built 2026-10-10 with a new imatrix. KLD 0.361 vs UD-Q2_K_XL on wikitext; KLD 0.243 vs native. See [quantization](07-quantization-dense-kv.md) |
| `K3-dense-v2-overlay.gguf` | 13.7 GiB overlay read on top of dense-v1 (`LLAMA_TENSOR_SOURCE`): latent projections Q6_K, `attn_output` Q5_K, shared-expert gate/up Q4_K, router Q8_0 | Default since 2026-11-01. KLD vs native 0.0953 without skip |
| `Kimi-K3-UD-Q4_K_XL` (Unsloth) | 32 shards, 1405.06 GiB. Moonshot's MXFP4 experts untouched, Q8_0 for the rest | The **quality reference** ("native"): PPL 1.4579 vs 1.4581 for lossless UD-Q8_K_XL (Unsloth). Downloaded 2026-10-31 |
| `K3-dflash-iq4_xs.gguf` | modal-labs/Kimi-K3-DFlash draft, IQ4_XS, 1.39 GB (also Q8_0 2.76 GB, Q4_K_M 1.52 GB) | 6 Qwen3-style layers, 2.60 B trainable params, block 16, taps [19,37,54,66,78,90] (+1 in the GGUF: [20,38,55,67,79,91]). See [speculative decoding](09-speculative-decoding.md) |
| `K3-dspark-radixark-iq4_xs.gguf` | RadixArk/Kimi-K3-DSpark, IQ4_XS, 1.21 GB | Tested 2026-11-05, dropped |
| `K3-dflash-distill{1,2,2b,3,4}-iq4_xs.gguf` | our fine-tuned drafts | See [draft distillation](10-draft-distillation.md) |

## Constraints (the rules every experiment is judged by)

| Constraint | Since | Detail |
|---|---|---|
| **No hardware purchases** | 2026-11-03 | An 8-CCD CPU (e.g. EPYC 7532 / 7543 / 7713) would likely lift the memory wall, but there is no budget; software levers only. Memory note `no-hardware-budget`; HANDOFF 31.z6 |
| **No requants; keep quality** | 2026-11-04 | "I like the quality as is": no expert requantization work. Lossy speedups only at **equal or better KLD** than the served config (measured with the KLD-by-position harness). Memory note `no-requant-keep-quality`; HANDOFF 31.z10 |
| **Benchmark with sampling** | 2026-10-12 | Speed is measured with Moonshot's recommended sampling (temperature 1.0, top-p 0.95, no top-k/min-p), never greedy: greedy inflates draft acceptance (0.71-0.78 vs 0.51-0.64 in real chats). Memory note `benchmark-with-sampling`; HANDOFF §30 |
| **Fans on Full forever** | 2026-10-26 | Required for sustained bandwidth. HANDOFF 31.x |
| Quality goal | 2026-10-31 | "As close to full quality as possible while staying at 9-10 tok/s"; same-top 95% wanted if possible (ceiling with 2-bit experts is ~88-89%). HANDOFF 31.z |
| Early speed goals | 2026-10-10 | "8.82 tok/s is fine, or even 5-6" (HANDOFF §1); later a 12-13 tok/s target was analysed in K3-IDEAL-BOUND (needed a third GPU in that model) |
