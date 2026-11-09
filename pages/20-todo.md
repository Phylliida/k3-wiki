# To do

[Back to index](../README.md) · Related: [speculative decoding](09-speculative-decoding.md), [draft distillation](10-draft-distillation.md), [prompt processing](12-prompt-processing.md), [dead ends](16-dead-ends.md)

Ideas worth testing, most promising first. Smaller open items are in [current setup](02-current-setup.md#open-items-as-of-2026-11-07). Before adding one here, check [dead ends](16-dead-ends.md).

## 1. ub 2048 prompt processing in the server

- **Why:** `llama-bench` reached pp2048 69.53-70.76 tok/s at ub 2048 vs ~47 at ub 1024 in the server (estimated +45%). Prompt processing is ~36% of model time at the measured real-use decode speed.
- **Cost:** +770 MB (GPU0) and +980 MB (GPU1) of compute buffer. One expert-cache slot is ~440 MB per GPU, so going from 9 to 7 slots frees ~0.9 GB. Peaks are 24012 / 23162 MiB against a ceiling of ~24050.
- **Test:** A/B decode and prompt speed at realistic contexts, since fewer cache slots may slow decode. Needs server downtime.

## 2. Markov logit-bias head on the DFlash draft

- **Why:** our draft decodes `[anchor, mask, mask]` in one pass, so position 2 never sees the token sampled for position 1. Hyperserve's DeepSeek-V4.1 drafter ([prior art](19-prior-art.md)) has a rank-128 head that adds a bias to each position's logits based on the previous draft token.
- **Gain (arithmetic on an assumed improvement):** at real-use pos1 0.614 and 1.99 tok/step, pos2|pos1 +0.02 / +0.05 gives +0.6% / +1.5%.
- **Cost:** ~170 MB of VRAM uncompressed for rank 128 and K3's 163,840-token vocab (two tables of ~84 MB). Quantizing them would reduce that.
- **Lossless only if** position 2's proposal is conditioned on the token actually sampled for position 1. The existing `build_dspark_markov_head` chains on the argmax, so this needs a two-stage draft: sample position 1, apply the bias, then sample position 2.
- **Evidence so far:** RadixArk's K3 DSpark has such a head and lost to DFlash (9.67-9.73 vs 9.97 tok/s), but it differs in everything else, so that says little about the head alone.
- **Test:** train the head on distill3's hidden states with the acceptance loss and the real previous token, then score it with the paired held-out offline eval (distill3: 0.7856 / 0.6029 / 0.5015). Implement it in the server only if position 2 gains at least 0.02.
