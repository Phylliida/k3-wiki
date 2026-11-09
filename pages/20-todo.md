# To do

[Back to index](../README.md) · Related: [speculative decoding](09-speculative-decoding.md), [draft distillation](10-draft-distillation.md), [prompt processing](12-prompt-processing.md), [dead ends](16-dead-ends.md)

Ideas worth testing, roughly in the order to try them (cheap and promising first). Sources for items 1 and 5-7: `k3-research/LIT-REVIEW-2026-11-08.md`. Smaller open items are in [current setup](02-current-setup.md#open-items-as-of-2026-11-07). Before adding one here, check [dead ends](16-dead-ends.md).

## 1. Block verification of the 2-token draft

- **Why:** verification currently judges drafted tokens one at a time. Block verification ([Sun et al., arXiv 2403.10444](https://arxiv.org/abs/2403.10444)) judges the block jointly from the same target and draft distributions, so one position's ratio can rescue the other. It is provably lossless and optimal among valid verifiers for a fixed draft.
- **Source measured:** +3-8% accepted tokens per step at 4-8 drafts (PALM-2, Vicuna), more at higher temperature. Nothing was measured at 2 drafts.
- **Gain (estimate, extrapolated):** +1-3% tok/step, i.e. +1-3% decode tok/s; no VRAM and no extra step time.
- **Cost:** ~40 lines in `common/sampling.cpp` (`common_sampler_sample_and_accept_n`). Both positions' distributions are already computed by the 3-token verify batch.
- **Test (no A/B noise):** extend the TEMP `LLAMA_SPEC_RS_STATS` log with r1, r2 (p/q at the drafted tokens) and S = sum over x of max(p1 * P2(x) - Q2(x), 0). Then compare, over the same real steps, the token-wise expectation `min(r1,1) * (1 + min(r2,1))` with the block expectation `2*p2 + (1-p2)*h1` (p1 = min(r1,1), p2 = min(p1*r2,1), h1 = S/(S+1-p1)). Implement only if the mean gain is at least 1%. Needs a server restart to add the log.

## 2. ub 2048 prompt processing in the server

- **Why:** `llama-bench` reached pp2048 69.53-70.76 tok/s at ub 2048 vs ~47 at ub 1024 in the server (estimated +45%). Prompt processing is ~36% of model time at the measured real-use decode speed.
- **Cost:** +770 MB (GPU0) and +980 MB (GPU1) of compute buffer. One expert-cache slot is ~440 MB per GPU, so going from 9 to 7 slots frees ~0.9 GB. Peaks are 24012 / 23162 MiB against a ceiling of ~24050.
- **Test:** A/B decode and prompt speed at realistic contexts, since fewer cache slots may slow decode. Needs server downtime.

## 3. Markov logit-bias head on the DFlash draft

- **Why:** our draft decodes `[anchor, mask, mask]` in one pass, so position 2 never sees the token sampled for position 1. Hyperserve's DeepSeek-V4.1 drafter ([prior art](19-prior-art.md)) has a rank-128 head that adds a bias to each position's logits based on the previous draft token.
- **Gain (arithmetic on an assumed improvement):** at real-use pos1 0.614 and 1.99 tok/step, pos2|pos1 +0.02 / +0.05 gives +0.6% / +1.5%.
- **Cost:** ~170 MB of VRAM uncompressed for rank 128 and K3's 163,840-token vocab (two tables of ~84 MB). Quantizing them would reduce that.
- **Lossless only if** position 2's proposal is conditioned on the token actually sampled for position 1. The existing `build_dspark_markov_head` chains on the argmax, so this needs a two-stage draft: sample position 1, apply the bias, then sample position 2.
- **Evidence so far:** RadixArk's K3 DSpark has such a head and lost to DFlash (9.67-9.73 vs 9.97 tok/s), but it differs in everything else, so that says little about the head alone.
- **Test:** train the head on distill3's hidden states with the acceptance loss and the real previous token, then score it with the paired held-out offline eval (distill3: 0.7856 / 0.6029 / 0.5015). Implement it in the server only if position 2 gains at least 0.02.

## 4. Chunked KDA kernel for prompt processing

- **Why:** `ggml/src/ggml-cuda/gated_delta_net.cu` has no chunked form (`//TODO: Add chunked kernel for even faster pre-fill`), so prompts run KDA one token at a time. Chunkwise forms update the recurrent state once per chunk; [LinQuant](19-prior-art.md) and flash-linear-attention have reference implementations.
- **Gain (estimate, not measured at prompt shape):** KDA is ~0.76 ms/token (11 us per token per layer x 69 layers, from the decode-shape timing): ~3.6% of prompt time at 47 tok/s, ~5.3% at ub 2048's 69.5. Removing half to three-quarters of that gives ~2-4% faster prompts, ~1% of total model time.
- **Cost:** a new CUDA kernel, including a per-chunk Gram matrix and triangular solve.
- **Not bit-exact:** rounding changes slightly in FP32, so it needs a KLD check. INT8 operands are not part of this item; they would be lossy.
- **Test:** first time the op with `test-backend-ops perf` at K3's shape (96 heads, 128 x 128 state, n_tokens 1024). Build the kernel only if KDA is at least ~3% of prompt time.

## 5. Expert deferral: KLD test first

- **Why:** [KTransformers](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf) (SOSP '25) adds a token's lowest-ranked experts one layer late, so their CPU work overlaps the next layer's GPU phase. They measured +33% decode on DeepSeek-V3 with 3 of 8 experts deferred, and task scores within ~0.2% for up to 5 deferred (task accuracy, not KLD).
- **Gain (estimate, upper end):** at deferred fraction f = 0.2, ~251 -> ~218 ms/step (+15%) if the CPU is the only critical path. The wiki has seen waits move to the GPU splits' copies when work left the CPU, and cross-layer prefetch already uses DRAM during the GPU window, so the real gain is likely lower.
- **Lossy:** under the equal-or-better-KLD rule, extra KLD must be bought back with speed at ~0.008 KLD per tok/s. Net gain ~= gross tok/s - dKLD / 0.008, so dKLD 0.012 is break-even. A useful win at f = 0.2 needs dKLD of about 0.006 or less (~+7% net).
- **Cost:** large for the real overlap (the CPU split must run concurrently with a later GPU split; the allocator hazard hit by `LLAMA_MOE_DEFER_LATE` applies). The KLD test is small.
- **Test:** emulate in the graph only. Compute the deferred slots as a second MUL_MAT_ID via the skip-mask mechanism, add their sum one layer late, and run the deterministic `kld_pos.sh` at batch 3 for f = 0.1 / 0.2 / 0.3. Build the overlap only if dKLD at f = 0.2 is about 0.006 or less.

## 6. N-gram proposal at draft position 2

- **Why:** a long n-gram match against the context (re-edited files, echoed tool output) could beat DFlash at position 2 where DFlash is unsure, and a point-mass proposal keeps verification exact.
- **Evidence so far:** on the held-out dump (156,666 tokens, windows capped at 8192), an 8-gram match exists at 23% of positions (mean target probability of the proposal 0.857), but at only 5.5% of the uncertain positions (top-1 p < 0.8), where it averages 0.265. So it mostly proposes where DFlash is probably already right. Real-use contexts (13-79k) are longer than the dump's.
- **Gain (estimate, low confidence):** 0 to +2% tok/step. Lossless; no VRAM.
- **Cost:** moderate. The fork uses the first draft type that produces anything (`common_speculative_draft`), so a per-position policy (n-gram only when DFlash's top-1 q is low and the match is >= 8 tokens) has to be written.
- **Test:** score the hybrid policy against DFlash alone at position 2 in the paired offline eval, ideally on a long-context dump.

## 7. EPYC fabric power state (`APBDIS=1`) and the `performance` governor

- **Why:** pinning the data fabric to its top P-state is common EPYC tuning advice for sustained bandwidth, and the DRAM throttle (bandwidth roughly halves after minutes; cause not found) might plausibly be a power-state effect. Unverified: the advice comes from a search snippet, and no measured source was found. The CPU governor is currently `schedutil` on `acpi-cpufreq`.
- **Gain (estimate):** 0-3% direct; mostly useful as a test of the throttle. Lossless.
- **Cost:** a BIOS change (the user's call; needs a reboot and server downtime).
- **Test:** `cpu_load kern` and `readbw` before and after, plus a throttle soak.

