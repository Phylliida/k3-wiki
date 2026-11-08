# Memory bandwidth, PCIe, thermals and fans

[Back to index](../README.md) · Related: [hardware](01-hardware-model-constraints.md), [CPU kernels](04-cpu-expert-kernels.md), [tools and methodology](15-tools-and-methodology.md)

The CPU expert phase runs at the DRAM wall, so anything that lowers usable bandwidth (load from other processes, a PCIe link at x8, a hidden thermal throttle) shows up directly in tok/s. Several "results" in this project were really machine-state effects; this page is the reference for checking them.

## Results at a glance

| Finding | Numbers | Source |
|---|---|---|
| Streaming read wall | **119.6 GB/s** best (32 threads, 4K pages, 16 GiB buffer, best of 3); 48-64 threads no better; `MADV_HUGEPAGE` slower (111.9) | HANDOFF §13, K3-IDEAL-BOUND §11, §13 |
| Theoretical peak | 8 ch x DDR4-2933 x 8 B = 187.7 GB/s (measured is 64% of it) | HANDOFF §9 |
| Why ~120 | 4-CCD Rome: each CCD's GMI link caps at **~39 GB/s** (1 CCD 39.1, 2 CCDs 77.2, 3 106.9, 4 123.9 GB/s); a second limit (IO die / LRDIMM) flattens the total | HANDOFF 31.z6, 31.z7 |
| GPU PCIe reads are additive | CPU 92.6 -> 93.9 GB/s with DMA, DMA 25.0 -> 22.5; total 116.4 vs serial sum 117.6. CPU 101.0 under DMA vs 106.3 solo (-5%) | K3-IDEAL-BOUND §6, §11 |
| Two GPUs + CPU | total DRAM ~142-146 GB/s (CPU kernel 95.7 + GPUs ~46-50) | HANDOFF §30 |
| `iommu=pt` | 266-267 -> 259-261 ms/step; prompts 28.8-29.2 -> 29.5-29.9 tok/s (~2-2.5%, free) | HANDOFF 31.z7 |
| GPU1 at PCIe x8 after a reboot | 288 vs 238 ms/step (+~50 ms); reseat to x16: 289 -> 240 ms/step (~17%) | HANDOFF §31 |
| DRAM thermal throttle (fans "Optimal") | bandwidth 115 -> **55 GB/s** after ~2.5-3 min of load; decode ~243 -> ~400 ms/step | HANDOFF 31.x |
| Fix: BMC fans Full + third fan | 1 h soak: 55 requests, 0 throttled, 233/232/233/232 ms/step per quarter | HANDOFF 31.x |
| Throttle again with fans Full (Nov 6) | 251 -> ~450-465 ms/step after a rebuild on top of decode; managed by `thermostat.sh`, then a better fan layout (45 min continuous decode, no throttle) | SESSION3 "DRAM throttle" |

## Bandwidth measurements

- `readbw 16 0` (C streaming read): 119.6 GB/s at load average 3.0; the same binary gave 92.6 GB/s at load average 7.4. Always record `uptime` beside a bandwidth number (K3-IDEAL-BOUND §13). A pre-existing 4 GiB STREAM triad gave 95.6 GB/s at 64 threads.
- Software prefetch does not raise it: raw read stays ~120 GB/s for prefetch distances 0-8 KB at 32 or 64 threads (THP + 4K pf 124, noise) (HANDOFF 18.3).
- The IQ2_XS kernel: 32 threads 115.7 GB/s, 64 SMT 120.3 (`cpu_load kern`, 31.z7); the multi-token kernel is compute-bound below the wall (x2 91.8, x3 73.6, x4 59.6 GB/s; [CPU kernels](04-cpu-expert-kernels.md)).
- RAM is 8 x 128 GB DDR4-2933 **LRDIMM**, one DIMM per channel (user-verified twice): ideal population; LRDIMMs have higher latency, consistent with a memory-level-parallelism limit (HANDOFF §13, §26).
- `numpy.uint8.sum()` is ALU-bound (~2.5 GB/s per core) and must not be used as a bandwidth probe; pinning worker i to CPU i puts workers >= 32 on SMT siblings (K3-IDEAL-BOUND §13).

### Per-CCD test (HANDOFF 31.z7)

`cpu_load read`, pinned with taskset: 1 CCD 39.1 GB/s (4 or 8 threads alike), 2 CCDs 77.2, 3 CCDs 106.9, 4 CCDs 123.9; 8 threads spread 2 per CCD 112.2; 4 threads 1 per CCD 73.1. So each CCD's GMI link caps at ~39 GB/s, and an 8-CCD CPU on this board (H12SSL-i supports Rome and Milan, e.g. EPYC 7532 / 7543 / 7713) would gain maybe **+10-15% bandwidth** (not the 25% first estimated). Recorded as a future option only: **no hardware budget** (2026-11-03). Consequence for software: uneven expert work across the 4 CCDs wastes link bandwidth.

### CPU vs GPU DMA contention (K3-IDEAL-BOUND §6, HANDOFF 20.6)

| Condition | CPU read | GPU DMA |
|---|---:|---:|
| alone | 92.6 GB/s | 25.0 GB/s |
| both at once | 93.9 GB/s (+1.4%) | 22.5 GB/s (-9.8%) |
| concurrent total | 116.4 GB/s | (serial sum 117.6) |

An earlier version of this test reported "separate paths" from a broken verdict rule (it fired on a run where the GPU got 23.4 GB/s, less than alone). With the IQ2_XS kernel instead of a plain read: 104.5 GB/s solo -> 101.3 GB/s under 24.9 GB/s of DMA (-3%) when the host thread does not spin; a spinning host thread cost the kernel 25-30% (it shares a core with a worker).

## PCIe

- Each RTX 3090 is on its own Gen4 x16 root complex; no P2P between them. Host reads: DMA 25.0 GB/s, HMM in-kernel 25.2-26.3 GB/s (16 B loads), both GPUs 52.5 GB/s together. Idle links show 2.5 GT/s (power saving; normal).
- **2026-10-25: GPU1 trained at Gen4 x8** after a reboot (sysfs `current_link_width` = 8, root port `c0:03.1` also 8/16). GPU1 streams the share for the second half of the layers, so this alone added ~50 ms per verify step (288 vs 238 ms, greedy too). After a reseat both GPUs ran x16: the pre-reboot config went 289 -> 240 ms/step (9.46 tok/s). Check the width before trusting any two-GPU number.
- **IOMMU passthrough** (2026-11-03): groups were "DMA-FQ" (translated). After `iommu=pt` (groups `identity`): isolated HMM/pinned/DMA interference unchanged (HMM +7.0 / +6.9%, pinned +6.9 / +6.8%, DMA +2.1%), HMM mapping a bit faster (3.3 vs ~4.0 s for 8.55 GB), the in-situ async prefetch cost unchanged (wait 1438 vs 1460 us), but the **baseline improved ~2-2.5%**: 259-261 ms/step and 29.5-29.9 tok/s prompts vs 266-267 and 28.8-29.2. Kept.

## Machine-state traps seen

- **Outside workload** (2026-10-11 ~15:10): streaming read 120.8 -> 81.8 GB/s, IQ2_XS kernel 104 -> 64.7 GB/s, load average ~6 with nothing of ours running; tok/s fell to ~4.1. Re-check `readbw_pf 16 0` (~120 GB/s expected) before trusting new numbers.
- An outside process swapping ~9k pages/s in, ~5k out (with 160 GB free), ~2.5-3 CPUs of outside load: not the throttle trigger. Later the box swapped ~50 MB/s from something else; our server had VmSwap 0.
- Load average drifts numbers 5-15% (HANDOFF §9).

## The DRAM thermal throttle (HANDOFF 31.x, 2026-10-26)

**Symptom:** every long server run: first 3-7 requests ~243 ms/step, then ~400 ms/step for the rest. `GGML_SCHED_TIMING`: the CPU split compute per step 130-142 ms -> 280-320 ms; the copy (waiting for the GPU) flat at ~60 ms. Standalone `cpu_load kern 64 25` back to back: 114-115 GB/s for ~2.5-3 min, then **55 GB/s**, staying there; back to 115 after ~3-4 min idle (1 min is not enough). 20 s runs with ~12 s gaps did not trip it in 4.5 min.

**Ruled out:** GPU throttling (61-65 C, 1920 MHz, no throttle reasons), CPU temperature (Tctl 68-76 C), page cache / huge pages, NUMA balancing (off), VRAM (asymmetric split), UVM parameters. BMC sensors: DIMM A-D 66-67 C, E-H ~61 C (thresholds 85/90), CPU 70-76, VRMs 60-65; only FAN1/FAN5 present (700-980 RPM), the SEL full of "Lower Critical going low" fan events. A clean 2x drop with a minutes-long time constant looked like a power budget; it had never happened before the 2026-10-25 reboot.

**Cause and fix: thermal.** The BMC fan mode was "Optimal" (`ipmitool raw 0x30 0x45 0x00` -> 02): fans follow the CPU temperature, which stays low on this memory-bound load. A third fan was added and the mode set to **Full** (`ipmitool raw 0x30 0x45 0x01 0x01`; the BMC keeps it across reboots; both launchers re-set it if it drifts). After: 12 x 25 s back-to-back `cpu_load` held 112.5-115.2 GB/s; under load CPU 56-57 C (was 70-76), DIMM A-D 54 (66), E-H 57 (61), SOC VRM 53 (60-62). The throttling part (LRDIMM buffers or the IO die) is not reported by any sensor.

- Sustained check (25 requests back to back, no cooldown): all at 223-240 ms/step, **10.21 tok/s sustained**, accept 0.69; DIMM A-D climbs ~0.5 C/min to 71 C.
- **One-hour soak** (`long_run.sh`, n_predict 512): 55 requests, 28,160 tokens, **0 throttled**, ms/step per quarter 233 / 232 / 233 / 232, 9.81 tok/s (accept 0.65 on longer replies). DIMM A-D 52 -> 71 C at 10 min -> plateau 73-74 C from ~20 min; E-H 54; CPU 59; VRMs 57-64; GPUs 67-70. Fans at 100% duty (FAN1 1400, FAN3/FAN5 1120 RPM).
- Before the fix, sweeps used `COOLDOWN=240` and `MAX_MS=300` (drop throttled requests). The earlier "prompt processing slows decode" conclusion was this throttle.

## The throttle comes back (SESSION3, 2026-11-06)

During on-policy generation the step log showed 251 ms (05:23-05:33), 264 (during a nice'd 20-job rebuild), then ~450-465 ms from 05:43 on, with fans on Full, DIMM A-D 74 C, CPU clocks ~3.3 GHz. The trigger was the rebuild's CPU load on top of decode (night8's 35 min of continuous decode had not throttled). **Pausing the server with SIGSTOP for 5 min, then SIGCONT, restored a 254 ms median at once** (clients just wait). Lesson: no heavy builds while a long decode run or a measurement runs.

It throttled again ~10 min after resuming with no build (DIMM A-D 76 C; the box had been under near-continuous load since 22:00). Software guard **`k3-research/thermostat.sh`**: every 20 s reads DIMM A-D (`ipmitool sensor`) and the step log; SIGSTOPs the heavy process (llama-server / llama-dflash-dump / llama-perplexity) pre-emptively at >= 75 C (resume <= 70 C, ~40 s) or when the step-log median exceeds 350 ms (pause >= 240 s, resume <= 65 C or after 10 min); never during a speed test. Trace data: under decode the bank heats ~1 C per 20 s (57 -> 74 C in ~4 min) and steps stay unthrottled up to at least 74 C; a 42 s pause after a throttle did not clear it (430 ms), a 258 s pause did (248 ms, DIMM 57 C). The pre-emptive mode ran ~2 min on / ~43 s off (~74% duty) and held 4 h without a throttle.

- The memory is LRDIMM, so ~75 C is normal: a probe with HOT=78 ran 20 min at 77 C at 243-257 ms/step, so the DIMM sensor is not the trigger. Detection-only mode (HOT=82) throttled after ~27 min of near-continuous load at 77-78 C; another cycle stayed unthrottled 33+ min (plateau 75 C). Hypothesis: slow heat build-up in an unsensed part (LRDIMM data buffers / SoC) or a power budget, worse when a run starts warm.
- **Fan layouts:** layout 2 (FAN5 1260 RPM, reoriented): DIMM A-D 55 -> 74 C in ~4 min and a throttle after only ~5 min (and again 3.5 min after a resume): worse. Layout 3: from 55 C, 73 C at ~9.5 min and flat, E-H 62 C, GPUs 66 / 71 C; **45 min of continuous decode with no throttle** (237-288 ms/step), better than the original layout.
- Back-to-back speed tests: the 4th run of a night throttled (356 ms/step) although each run started with DIMM A-D <= 70 C: the sensor cools in seconds, the throttling part needs minutes. `speed_test.sh` now waits for DIMM A-D <= 62 C **plus 3 idle minutes** before each run and appends a step-log throttle line (`step log: N steps, median X ms, Y% > 350 ms`, THROTTLED if > 5%).

**How to check:** `ipmitool raw 0x30 0x45 0x00` prints `01` (Full); back-to-back `k3-research/cpu_load kern 64 25` should hold ~115 GB/s for 5+ minutes; in the server, watch the step log for medians > 350 ms.
