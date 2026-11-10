# Current setup, serving config and knobs

[Back to index](../README.md) · Related: [where the time goes](03-where-the-time-goes.md), [MoE placement](05-moe-placement-cache-prefetch.md), [speculative decoding](09-speculative-decoding.md), [tools](15-tools-and-methodology.md)

State as of 2026-11-07. Sources: `k3-research/K3-SETUP.md` (written 2026-10-31, updated through 2026-11-04), the daily launcher `run-kimi.sh`, `k3-research/serve_k3.sh`, HANDOFF 31.z7-31.z10, SESSION3-NOTES.

## At a glance

| | Value | Conditions / source |
|---|---|---|
| Decode, bench prompts | **9.80 tok/s** (250 ms/step, accept 0.74, 2.45 tok/step) as configured in `run-kimi.sh`; **~9.97** with the causal draft mask; **10.08** with causal + fast top-p (night10 fast-top-p runs, causal on: 10.24 / 9.91) | 5 prompts x 2 seeds, temp 1.0 / top-p 0.95 (`speed_test.sh`). `dab_base` 2026-11-05; causal mean of 4 runs; Nov 7 baselines 10.10 / 10.01 tok/s at 247-248 ms/step (SESSION3 Results 4, 11, 12) |
| Decode, technical text (2k context) | 9.25 tok/s (bidirectional draft); 9.18-9.87 causal | same runs, 512 generated tokens |
| Decode, real use | **~5.6 tok/s** overall with the adopted config: 1.99 tok/step, median 357 ms/step, throttled about half the time (~6.9 tok/s unthrottled); before the adopted config ~6.8 tok/s (2.05 tok/step at ~304 ms/step) | 7,325 steps at 13-79k context (SESSION3 Result 16); 17,992 steps at 16-48k (SESSION3 "Confidence-based draft length"). Different sessions and content, so not a paired comparison |
| Decode at long context | 7.98-8.01 tok/s at 64k (night8); 7.62 tok/s at 90k (Nov 4) | technical summary task; HANDOFF 31.z10, SESSION3 Result 6 |
| Prompt processing | **~47 tok/s** (2052 tokens in 43.7 s); 42.1 tok/s for a 90,119-token prompt | 8-bit KV + ub 1024 (HANDOFF 31.z10) |
| Quality as served | **KLD 0.113**, median 0.059, same top token 84.8%, PPL 5.90 (native 5.44) vs native 4-bit | dense-v2 overlay + union skip 0.045, 3-token batches (HANDOFF 31.z, "v2_b3_skip045") |
| Quality without skip | KLD 0.095, same top 86.1% | dense-v2 alone |
| Context | 96k tokens (`-c 98304`) | q8 KV would fit 128k in less VRAM than f16 at 96k |
| Startup | ~10 min | maps ~1.1 TiB of expert weights into both GPUs' page tables (HMM, ~2 GiB/s) |
| Peak VRAM | 24010 / 23164 MiB at 90k (allocations fail at ~24050 MiB used) | HANDOFF 31.z10 |

Where it started: 0.32-0.35 tok/s CPU-only on 2026-10-10 (a build without AVX2), 1.2 tok/s after the first fixes. See the [timeline](18-timeline.md).

## The daily configuration (`run-kimi.sh`)

Every setting, with the reason (comments condensed from the script):

| Setting | Value | Why / measured effect |
|---|---|---|
| Model | `K3-dense-v1` + overlay `K3-dense-v2-overlay.gguf` (`LLAMA_TENSOR_SOURCE`, `_RE='.*'`) | 2-bit experts (IQ2_XS) + better dense tensors; KLD vs native 0.243 -> 0.095 at the same speed ([quantization](07-quantization-dense-kv.md)) |
| Draft | `K3-dflash-distill3-iq4_xs.gguf` (since 2026-11-07; before: `K3-dflash-iq4_xs.gguf`), `--spec-type draft-dflash --spec-draft-n-max 2` | 2 drafted tokens per step; 3 is no faster ([speculative decoding](09-speculative-decoding.md)) |
| Vision (since 2026-11-09, pending first end-to-end test) | `--mmproj mmproj-BF16.gguf --no-mmproj-offload --image-max-tokens 1024` (`MMPROJ=` disables, `IMG_MAX_TOKENS` sets the cap) | MoonViT tower on the CPU (no VRAM to spare): 448x448 = 256 tokens, ~5.5 s to encode on 4 threads plus prompt time; see [Vision input](#vision-input-2026-11-09) |
| Placement | `-ngl 99 -ngld 99 -ncmoe 93 --no-op-offload -fit off -sm layer -ts 50,43 -devd CUDA1` | all dense on GPUs (layer split), all routed experts in RAM; GPU1 holds the draft and output head |
| `LLAMA_MOE_GPU_FRAC=1.0` | any expert may go to the layer's GPU | the GPU reads host memory in place via HMM |
| `LLAMA_MOE_GPU_K=2` | experts streamed for single tokens | the cache reserves max(K, KB) scratch slots; K 2 frees a slot's VRAM for nothing |
| `LLAMA_MOE_GPU_KB=1` | experts streamed per 2-4 token verify batch | 1 with the prefetch below (KB 2 without it) |
| `LLAMA_MOE_GPU_CACHE=9` | VRAM expert cache, slots per layer (LRU) | ~0.45 GB per slot per GPU |
| `LLAMA_MOE_GPU_PREFETCH=4`, `_ADMIT=1`, `_ASYNC=1`, `GGML_CUDA_PF_BLOCKS=1 GGML_CUDA_PF_THREADS=64` | cross-layer prefetch of 4 predicted experts of layer L+2 | -1.3% ms/step, +126/+166 MB VRAM; kept on at the user's request ([MoE page](05-moe-placement-cache-prefetch.md)) |
| `LLAMA_MOE_GPU_PREFETCH_WAIT=1` | finish the HMM mapping before serving | otherwise the first requests fault experts in |
| `LLAMA_MOE_SKIP=0.045`, `LLAMA_MOE_SKIP_UNION=1` | lossy expert skip | best quality above 9 tok/s ([expert skipping](06-expert-skipping-and-quality.md)) |
| `LLAMA_MOE_GPU_PP=1.0`, `_PP_HALF=200`, `_PP2=0.5`, `GGML_CUDA_MMID_RANGE_CHUNK=32` | prompt batches: GPUs take n/(n+200) of the experts, split over both GPUs | [prompt processing](12-prompt-processing.md) |
| `LLAMA_SPEC_RS_TAU=0.7` | stochastic draft verification | acceptance 0.62 -> 0.69 under sampling, lossless |
| `LLAMA_SPEC_DRAFT_UBATCH=128` | draft context ubatch | draft compute buffers 989 MiB at 512 vs 248 at 128 on GPU1 |
| `LLAMA_SPEC_STEP_LOG=...`, `LLAMA_SPEC_CUT_RAND=0.10` | TEMP data collection (2026-11-04) | ~1% slower while on; the cutoff study is closed, so `CUT_RAND` can go (SESSION3 Result 4) |
| `OMP_PLACES=cores OMP_PROC_BIND=close`, `-t 32` | one thread per physical core | +12.6% on the expert kernel; SMT (64 threads) is unusable ([CPU kernels](04-cpu-expert-kernels.md)) |
| `-c 98304 -b 1024 -ub 1024 -np 1 --cache-ram 32768` | 96k context, 1024-token prompt batches, 32 GB RAM prompt cache | ub 1024 made possible by the 8-bit KV |
| `-ctk q8_0 -ctv q8_0` | 8-bit KV cache | -0.4 GB peak VRAM per GPU, same speed, KLD unchanged (HANDOFF 31.z10) |
| `--cache-disk /data2/k3eval/prompt-cache --cache-disk-size 24576` | disk prompt cache | resumes conversations after a restart (below) |
| `--temp 1.0 --top-p 0.95 --top-k 0 --min-p 0.0` | Moonshot's recommended chat sampling | |

### Adopted 2026-11-07 (lossless; `run-kimi.sh` edited with the user's approval)

| Change | Gain | How | Source |
|---|---|---|---|
| Distilled draft `distill3` (trained on the user's own sessions, acceptance loss) | held-out offline +0.046 / +0.055 per position (+5.4% expected tok/step); online 24-probe paired +0.054 +- 0.028 per draft; generic bench +0.013 | `DRAFT=/mnt/items/distill/K3-dflash-distill3-iq4_xs.gguf` (previous: `/data2/kimi-k3/K3-dflash-iq4_xs.gguf`) | SESSION3 Results 13-15 |
| Causal in-block attention for the DFlash draft | +1.7% bench, ~+3% technical, +3% tok/step | comes with distill3 (`dflash.attention.causal = true` in its metadata); `LLAMA_DFLASH_CAUSAL=1` for other drafts | SESSION3 "Draft attention direction" |
| Fast exact top-p | +1.5% (sample phase 6.05 -> 2.2 ms/step) | `LLAMA_SAMPLER_TOPP_FAST=1` (default on since `0c587ea0d`) | SESSION3 Result 11 |
| Removed the TEMP `LLAMA_SPEC_CUT_RAND=0.10` | ~+1% | random draft cuts were only for collecting cutoff data | SESSION3 Result 4 |

With all of these: bench ~10.1 tok/s (distill3 measured 10.10). The estimate for real agentic use was ~7.4 tok/s (~300 ms/step, ~2.2 tok/step); the first real session measured **~5.6 tok/s** (~6.9 without the DRAM throttle), see [speculative decoding](09-speculative-decoding.md) and [thermals](14-memory-bandwidth-thermals.md). `LLAMA_SPEC_STEP_LOG` stays on.

## Vision input (2026-11-09)

K3 is natively multimodal (MoonViT-V2 tower, ~0.45 B parameters). The fork gained projector type `kimik3` (commit `986568475`), ported from Unsloth's llama.cpp PR #48 and used with Unsloth's `mmproj-BF16.gguf` (unsloth/Kimi-K3-GGUF, 904 MB, sha256 `c858aec6...`). Two fixes over the PR:
- the image prefix follows Moonshot's processor: `<|media_begin|>image {W}x{H}<|media_content|>` (original image size), embeddings, `<|media_end|>`
- `--image-max-tokens` now overrides the GGUF's pixel limit (12.8 MP, ~16k tokens per image, ~6 min of prompt at 47 tok/s). The daily cap is 1024: 2000x1500 -> 972 tokens, 1920x1080 -> 1008.

Measured without the text model (encoder-only test on the CPU, GPUs hidden): 448x448 -> 256 tokens (matches the MLX port's numbers), 5.4-5.5 s on 4 niced threads with the server busy; 640x480 -> 391 tokens, 9.1 s; deterministic, all finite. Known deviation: mtmd resizes to multiples of 28 px where Moonshot's processor pads (640x480: 391 tokens vs 414).

Compatibility checked in code: the server turns OpenAI `image_url` parts into a media marker that the K3 template passes through as text; the DFlash draft injects target features for embedding batches (K3 uses 1-D positions); the disk prompt cache stores image chunks and its fingerprint ignores the projector, so text-only states saved before stay valid. Loading a projector disables context shift and cache reuse (both unused here). Not yet tested end to end (needs a server restart).


## KV splice: compaction without re-processing (2026-11-09)

`POST /kv/splice` (commits `df6b9d6c3`, `e0dc04902`) replaces a span of a cached conversation with a span of another cached conversation, directly in the KV cache. Use: compact ~90k to ~65k by swapping old messages for a memory note K3 wrote in a separate session, then continue with a cache hit instead of re-processing everything after the swap (10+ minutes at 47 tok/s).

- Request: `src` and `dst` contexts (chat request bodies, rendered like `/v1/chat/completions`, or a raw `prompt`) pick the cached states by longest common prefix (slot, RAM cache, disk cache); `src_span` / `dst_span` are found in the states' own tokens (or `*_range` token indices). Whole message contents make the safest spans (special tokens on both sides).
- K3 specifics: the MLA cache uses no RoPE, so moved entries are exact copies and need no re-rotation (the generic K-shift would wrongly rotate them, so the splice moves positions without one). The KDA recurrent state stays the conversation's own; context checkpoints after the span are moved (their stored position patched), those inside it dropped. The DFlash draft cache of the sequence is cleared (it refills from the next tokens).
- Response flags `canonical: false` when the source's sampled tokens are not the tokenizer's split of their text (the compacted prompt then matches only up to the first difference).
- Tests (LFM2-350M hybrid, CPU): identity splice bit-identical; moved RoPE keys match a shifted prefill (first attention layer rel 2.9e-4); 205 -> 100 tokens in 62 ms, then the compacted conversation reused 73 of 77 prompt tokens. On K3 (2026-11-09): the compacted conversation is a cache hit (~215 tokens reused, 4 replayed) and K3 answers correctly from the memory note (it dropped a detail the note had dropped, though the KDA state still holds it). Two K3-only bugs fixed on the way: an assertion with the vision projector loaded, and a segfault after moving the DFlash draft cache, which is now cleared instead (acceptance 0.78 on the next reply). Conversations with images can be spliced too (`f4c8480ac`). Quality: splicing identical text from another session leaves answers unchanged (first-token KL ~0.013, the noise floor), but a memory note is not equivalent to a fresh prefill of the compacted conversation (KL 0.09-0.8 in 5 questions, 1/5 identical answers): K3 answers from the note yet keeps details only the original had, because its KDA state and the tail's cache entries still carry the original. Details: `k3-research/kv-splice/README.md`.

## KV splice2: building a request's cache from pieces (2026-11-09)

`POST /kv/splice2` (commit `086b4a78b`) implements a design from the user's connectome-web project: the client sends the next request plus pieces (a memory by its reply text, or a stretch of a cached conversation by its request), each with an anchor. The server keeps the request's cached prefix, copies each piece's MLA entries from its source while the tokens agree, evaluates only the tokens between pieces, and takes each piece's KDA state from its source (its own end state, a checkpoint, a checkpoint advanced over up to `max_advance` tokens, or as is). On K3: a fold with 3 pieces and an L1 memory request with 2 placed every piece and the requests then reused the built cache; built vs fresh prefill gave the same fold answer (first-token KL 0.25) and nearly the same memory text (KL 0.44). Calls took 24-30 s, not yet profiled. Usage: `kimi-harness-api.md` in llm-tests.

## `serve_k3.sh` (the experiment launcher)

`serve_k3.sh [fast2|lossless2|fast|lossless]` starts a detached server on port 8091 (log `/data2/k3eval/serve.log`, pid file `/data2/k3eval/serve.pid`). `speed_test.sh` drives it. Modes:

| Mode | What | Measured |
|---|---|---|
| `fast2` (default) | two GPUs, layer split, VRAM cache, KB 2, union skip (`SKIP`, default 0.045), stochastic verification, prompt split | see above |
| `lossless2` | same without skip, KB 4 | untuned since the PCIe x16 reseat (KB was tuned at x8) |
| `fast` / `lossless` | the single-GPU setups of HANDOFF §29 (router on CPU, no CUDA graphs, FRAC 0.4) | ~8 / ~6.2 tok/s greedy (less sampled) |

Environment overrides: `OVERLAY` (empty = plain dense-v1), `PORT`, `HOST`, `CTX` (98304), `UB` (512), `CACHE` (9), `KB` (2), `K1` (2), `PP` (1), `PP_HALF` (200), `PP2` (0.5), `TS` (50,43), `CRAM` (32768), `DRAFT_UB` (128), `KV` (f16 or q8), `SKIP` (0.045), `DRAFT_N` (2), `X4` (layer regex: those layers' experts from the native model), `DRAFT_MODEL`, `SPEC_TYPE` (`draft-dflash` / `draft-dspark`), `SPEC_ARGS`, `SERVE_ENV="VAR=x ..."`. Note: `serve_k3.sh` still carries a comment that KV=q8 crashes; that crash was gone by 2026-11-04 ([quantization page](07-quantization-dense-kv.md)).

## How a generation step works

One step verifies 2 drafted tokens plus the next one: a 3-token batch through the target (K3-SETUP "How a generation step works").

1. **On the layer's GPU** (layers split 50/43): attention, router, norms, latent projections. The shared expert is *deferred* (`GGML_TENSOR_FLAG_DEFER`) so it runs while the CPU does the routed experts.
2. **Routed experts, split three ways:** the VRAM expert cache (experts already resident cost nothing), the GPU share over PCIe (the KB experts most shared within the batch, streamed from host memory via HMM and admitted to the cache), and the CPU (everything else, with a multi-token IQ2_XS kernel that decodes each weight row once for all tokens). CPU and GPU keep identical copies of the cache state, updated by the same deterministic rule.
3. **Expert skip** (lossy): experts with normalized router weight <= tau are dropped unless another token of the batch selected them.
4. **Verification** is lossless speculative sampling: draft proposals sampled at temperature 0.7 from its top-10, accepted with probability min(1, p/q), else a residual sample.

Prompt batches use a different path (range split across CPU and both GPUs, fused MoE op): see [prompt processing](12-prompt-processing.md).

## VRAM budget

Per GPU, measured in the server (K3-SETUP "VRAM budget", MiB from nvidia-smi):

| | 96k defaults, dense-v2 (ub 512, 9 slots, -ts 50,43, f16 KV) | short sessions, dense-v1 (8k, ub 1024, 18 slots, -ts 49,44) |
|---|---|---|
| Dense weights | ~14.0 / 12.0 GB (incl. overlay ~3.1 / 2.7 GB) | 10.7 / 9.5 GB |
| Expert cache (slots + 2 scratch per layer) | ~5.3 / 4.5 GB | ~9.4 / 8.4 GB |
| KV cache (MLA, f16) | 1.27 / 1.27 GB (q8: ~0.69) | 0.11 / 0.11 GB |
| Recurrent state (KDA, incl. draft rollback copies) | 0.73 / 0.60 GB | same |
| Compute buffers | 0.49 / 0.62 GB | 0.77 / 1.01 GB |
| HMM page tables | ~1.1 / 1.1 GB | same |
| Draft model + buffers (GPU1) | 1.3 + 0.35 GB | same |
| Display (GPU1) | ~1 GB | same |
| **Peak used** | **23.7 / 23.5 GB** | **23.3 / 24.0 GB** |

Daily config with q8 KV + ub 1024 + prefetch: peak 24010 / 23164 MiB at 90k (same as at 7k: compute buffers are reserved for the full context).

Levers per GPU: a cache slot ~0.45 GB (worth only ~0.6 ms/step at 9-11 slots: 16 -> 11 slots cost ~3 ms/step); ub 1024 -> 512 saves 0.3-0.6 GB; one layer moved with `TS` ~0.42 GB; `LLAMA_MOE_GPU_K=2` frees a scratch slot; draft ubatch 512 -> 128 ~0.5 GB on GPU1; 8-bit KV -0.4 GB net. Allocations fail at **~24.05 GB used per GPU** (nvidia-smi), not at 24.58; GPU1 grows 0.5-0.7 GB during the first long prompt, so check peaks after a long prompt.

## Disk prompt cache (`--cache-disk`, commits `78c934053`, `867aeb6b9`, `8a44b3ad9`)

Before this, a server restart lost every conversation, and K3 re-processes prompts at ~29.5 tok/s (then; ~47 now): 12k tokens ~6.5 min, 90k ~50 min (HANDOFF 31.z8).

- 2 s after a response (or on a chat switch / stop) the slot state is written in the background: target KV + KDA state + DFlash draft KV + the newest 4 context checkpoints. A state the new one continues replaces the old file; LRU eviction above the size limit; files carry a fingerprint (models, overlay, KV types).
- Measured on K3: a 2049-token first turn took 72 s; after a restart the stored 2248-token state loaded in 1812 ms, 44 tokens were processed, decode 9.28 tok/s at acceptance 0.78 (draft state restored). Slim checkpoints (no draft copy per checkpoint) verified with two more restarts (acceptance 0.87 / 0.81).
- File sizes: ~2.3 GB at 2k tokens; 3.1 GB at 90k with q8 KV (estimate before q8: ~6.8 GB at 90k, ~15 GB without the slim checkpoints). Only conversations of 2048+ tokens are stored (short chats cost ~1.4 GB each and filled the 24 GB budget).
- Upstream `--slot-save-path` was not usable: manual, target only (no draft state, no checkpoints).

## Startup checklist (K3-SETUP "Quick start")

1. Fans on Full: `ipmitool raw 0x30 0x45 0x00` prints `01` (both launchers re-set it).
2. Both GPUs at x16: `nvidia-smi --query-gpu=pcie.link.width.current --format=csv` (one bad reseat trained at x8 and cost ~17%).
3. Page cache warm (~800 GB `buff/cache`); after a reboot `ls *.gguf | xargs -P 4 -I{} dd if={} of=/dev/null bs=16M` (~2.5 min).
4. No leftover server holding VRAM (`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`): teardown of a server with ~800 GB HMM-mapped takes 1-3 min, a crashed one up to ~16 min.

## Troubleshooting (K3-SETUP)

| Symptom | Cause / fix |
|---|---|
| Fast at first, then ~40% slower after minutes | DRAM bandwidth throttle: check the fan mode, then back-to-back `cpu_load kern 64 25` (~115 GB/s expected); pause the server for ~4-5 min to recover ([thermals](14-memory-bandwidth-thermals.md)) |
| First long prompt very slow, later ones fine | expert mapping not finished (no `LLAMA_MOE_GPU_PREFETCH_WAIT`, or a config touching unmapped experts) |
| `cudaMalloc failed: out of memory` / `failed to allocate compute pp buffers` | VRAM ceiling: lower `UB` or `CACHE`, rebalance `TS`, check for a leftover process |
| Server won't start after a crash | the old process is still tearing down (state R, holding VRAM): wait |
| `pgrep -f` / `pkill -f` in scripts | can match the script's own command line: use PIDs |

## Open items (as of 2026-11-07)

- Real-use DRAM throttle: about half of the first real session; airflow onto DIMM bank A-D is the open lever ([thermals](14-memory-bandwidth-thermals.md)).
- distill3 adopted 2026-11-07; real-use acceptance 0.494 per draft (one session). Open: a paired offline eval against the modal draft on that content, retuning `LLAMA_SPEC_RS_TAU` for the distilled draft, revisiting n_max 3.
- ub 2048 prompt-processing A/B in the server (~+45% estimated; needs about 2 expert-cache slots of VRAM).
- `lossless2` mode is untuned since the x16 reseat.
- Before any upstreaming: remove TEMP code (op profilers, sched timing, stats hooks, dumps, `LLAMA_SPEC_RS_STATS`, `GGML_CUDA_POOL_LOG`, experimental knobs) and move K3-specific knobs out of environment variables (K3-SETUP "Open items").
- Unexplained: the fused res_mix op's KLD 0.0065; the in-situ cost of the async prefetch copy beyond the reads-in-flight fix.

## Environment knob reference

Production knobs (defaults are the code defaults; the daily config overrides several). TEMP diagnostics are listed in [tools and methodology](15-tools-and-methodology.md).

| Knob | Default | What it does | Added |
|---|---|---|---|
| `LLAMA_MOE_GPU_FRAC` | off | experts eligible for the GPU share at 1-4 tokens: ids `[0, frac*896)` | 2026-10-11 (`604bafadb`) |
| `LLAMA_MOE_GPU_K` | 3 | experts the GPU takes per single token | 10-11 |
| `LLAMA_MOE_GPU_KB` | = K | experts the GPU takes per 2-4 token batch (batch take rule) | 10-12 (`4f867184b`) |
| `LLAMA_MOE_GPU_CACHE` | 0 | VRAM expert cache slots per layer | 10-12 (`367f377c6`) |
| `LLAMA_MOE_GPU_CACHE_LFU` | 0 | usage-score eviction instead of LRU | 10-26 (`fa0afd606`) |
| `LLAMA_MOE_GPU_PREFETCH_WAIT` | 0 | block until the HMM mapping is done | 10-11 |
| `LLAMA_MOE_SKIP` | off | skip experts with normalized weight <= tau | 10-11 (`f3f84b015`) |
| `LLAMA_MOE_SKIP_RENORM` | 1 | renormalize the kept weights | 10-11 |
| `LLAMA_MOE_SKIP_UNION` | 0 | keep a sub-tau expert another token of the batch selected | 10-12 (`a3ea61f0e`) |
| `LLAMA_MOE_SKIP_LAYERS` | unset | `a-b:tau[,...]` per layer range (0 = no skip there) | 11-02 (`c7fb4f4cd`) |
| `LLAMA_MOE_SKIP_POS` | unset | per-position tau in 2-4 token batches (measured worse) | 11-04 (`10b684e22`) |
| `LLAMA_MOE_SKIP_NORMS` | unset | importance-aware skip by weight x mean expert output norm (opt-in, inconclusive) | 11-04 (`5c0c99c2a`) |
| `LLAMA_MOE_GPU_PP` / `_PP_HALF` | off / - | prompt-batch GPU share cap / half-point n/(n+h) | 10-25 (`c747ccd50`) |
| `LLAMA_MOE_GPU_PP2` | 0 | the other GPU's part of the prompt share (top expert range) | 10-26 (`fa0afd606`, `eb1bf7e81`) |
| `LLAMA_MOE_GPU_PP_FUSED` | 1 | fused MoE range op for prompt batches | 10-31 (`eb1bf7e81`) |
| `LLAMA_MOE_GPU_KB2` | 0 | second-GPU share at 1-4 tokens (exact, no gain without P2P) | 11-02 (`8a6c6d4cf`) |
| `LLAMA_MOE_GPU_PREFETCH` (+ `_ADMIT`, `_ASYNC`, `_AHEAD` (2), `_SPLIT`) | 0 | cross-layer expert prefetch into VRAM | 11-03 (`12f3d199e`, `56801b5b3`, `7d3132042`) |
| `GGML_CUDA_PF_ABORT` | 1 | abortable in-order prefetch copy (0 = run to completion, 2 = never stopped, for tests) | 11-03 |
| `GGML_CUDA_PF_BLOCKS` / `_THREADS` | 2 per slice / 256 | prefetch copy grid (1 x 64 keeps ~48 KB in flight) | 11-03 |
| `LLAMA_SPEC_RS_TAU` | 0 | draft sampling temperature for stochastic verification (0 = exact match) | 10-25 (`42d85672b`) |
| `LLAMA_SPEC_RS_GREEDY_ABOVE` | - | draft greedy where its top-1 p is above this (no gain) | 10-25 |
| `LLAMA_SPEC_DRAFT_UBATCH` | - | cap on the draft context's ubatch | 10-26 |
| `LLAMA_SPEC_DRAFT_CTX` | - | cap on the draft context (full-attention drafts' KV) | 11-05 (`3adf5d263`) |
| `LLAMA_SPEC_P_MIN_MODE` | 0 | which drafts `--spec-draft-p-min` may cut: 0 all, 1 from the 2nd by own p, 2 from the 2nd by product | 11-04 (`ec5c26a52`) |
| `LLAMA_SPEC_PAD_TOKEN` | - | pad cut drafts to n_max (constant batch shape) | 11-05 (`88a810398`) |
| `LLAMA_DFLASH_CAUSAL` | GGUF key, else bidirectional | in-block attention direction of the DFlash draft | 11-04 (`934620b52`) |
| `LLAMA_DFLASH_FULL_BLOCK` | 0 | decode the whole trained block, read n_max drafts | 11-04 (`505338506`, fixed `3adf5d263`) |
| `LLAMA_SAMPLER_TOPP_FAST` | 1 | fast exact top-p (0 = old path) | 11-06 (`0c587ea0d`) |
| `LLAMA_K3_FUSED_RESMIX` | 0 | fused residual-mix op (no gain, off) | 11-06 (`c9666450c`) |
| `LLAMA_TENSOR_SOURCE` / `_RE`, `LLAMA_TENSOR_SOURCE2` / `_RE` | unset | read matching tensors from another GGUF (overlay, native expert layers) | 10-31 / 11-01 (`a8405846e`, `aa0525637`, `b4f56553a`, `6bf6407e6`) |
| `LLAMA_QUANT_ONLY` | unset | `llama-quantize` writes only matching tensors (overlays) | 10-31 (`8a1e0e5a5`) |
| `LLAMA_ATTN_ROT_DISABLE` | 0 | disable the Hadamard KV rotation that quantized KV turns on | upstream |
| `GGML_MMID_DYN` / `GGML_MMID_CHUNK` | 1 / 128 | pooled decode MUL_MAT_ID and its rows per work item | 10-10 |
| `GGML_MMID_MULTI` | 1 | multi-token IQ2_XS kernel | 10-12 (`dbe3f5f18`) |
| `GGML_CPU_FUSE_SITU` | 1 | fused CPU gate+up+SiTU | 10-11 |
| `GGML_SCHED_DEFER` | 1 | deferred splits (GPU share / shexp overlap the CPU experts) | 10-11 |
| `GGML_CUDA_HOST_SRC_STREAM` | 1 | host-weight GPU nodes on a second stream | 10-11 |
| `GGML_CUDA_MMVQ_PERS` / `_PERS_MAXCOLS` | 1 / up to 4 | persistent IQ2_S/IQ3_XXS mat-vec; max columns | 10-11 / 11-02 (`cf1f96157`) |
| `GGML_CUDA_MMID_GATHER` | 0 | gather-into-VRAM path for the single-token share (slower) | 10-11 |
| `GGML_CUDA_MMID_RANGE_CHUNK` / `_RANGE_SERIAL` | 64 / - | prompt range-split staging chunk; serial copy | 10-25 / 10-30 |
| `GGML_CUDA_MOE_RANGE_CHUNK` / `_ROWS` | 16 / 2048 | fused op experts per chunk / rows per pass | 10-31 |
| `GGML_CUDA_DISABLE_GRAPHS` | 0 | CUDA graphs off (graphs save ~16-23 ms/step) | upstream |
