# Experiments

What we tried, how, what happened, and what we learned. Newest at the bottom. Failed ideas belong here too.

Recall and precision come from `topo_agent.eval` (a found hold counts when one center lies inside the other's
outline). "Drawn walls" are the hard sample walls in `topo_agent/baseline/bench.py` (9 scenes × 7 marked holds;
seeds 11 and 12 give 18 walls, 126 holds). They were also used while tuning, so real photos are the real test.

## Template

```markdown
## E<n>: <short title> (<date>, <author>, <commit or PR>)
- **Question:** what we wanted to know
- **Setup:** data, methods, command, settings
- **Results:** numbers (recall / precision / seconds), per scene if useful
- **Takeaways:** what we keep, what we drop, what to try next
```

## E1: Color-only baseline on drawn walls (2026-10-27, Yuzuki, f52900a)
- **Question:** how good is the pre-hackathon color method?
- **Setup:** `--bench --methods color --seeds 11,12`
- **Results:** recall 85%, precision 96%, 0.8 s. Weak scenes: `twins` (same color on the next wall) recall 57%,
  `tape` (holds of mixed colors marked by tape) recall 14% / precision 33%.
- **Takeaways:** color can't tell problems apart when the color repeats elsewhere or doesn't mark the problem at all.

## E2: SAM 3.1 candidates (2026-10-27, Yuzuki, dc23afb)
- **Question:** does SAM 3.1 find the holds, and what does its output look like?
- **Setup:** Responses API, concept prompt `climbing hold`, on `twins` and `tape` (seed 11). Point prompt at the tap.
- **Results:** 34 and 29 candidates; all 7 marked holds were among them in both. Confidence only 0.77–0.90, so it
  can't separate problems. 2–5 s per request. A point prompt must start with `<0f>1` (otherwise HTTP 400).
  Without `store: false` the response is stored at Meta.
- **Takeaways:** SAM finds holds well; choosing *which* holds belong together is the agent's job.

## E3: First Muse Spark agent (2026-10-27, Yuzuki, e3d2ee0)
- **Question:** can an agent with tools beat color on the hard scenes?
- **Setup:** tools `inspect_candidates`, `find_similar`, `look_closer`, `submit_holds`; default reasoning effort;
  `--bench --methods color,sam_color,meta --seeds 11` (9 walls).
- **Results:** color 89% / 97% / 0.9 s; sam_color 90% / 85% / 2.1 s; **meta 94% / 100% / 36 s**.
  `tape` 100% (color 14%). `twins` 4/7 (7/7 in a separate single run: high variance). `light` 6/7: SAM missed a
  washed-out hold.
- **Takeaways:** the agent wins where color fails and adds nothing from other problems, but 36 s is too slow.

## E4: Faster agent (2026-10-28, Yuzuki, aba31c1)
- **Question:** can we cut the number of model turns?
- **Setup:** send a table of the candidates closest in color (color, position, size, linked) in the first message;
  `reasoning_effort: low` (reasoning tokens 732 → 297 on a small test prompt); 18 walls.
- **Results:** meta 94% / 100% / **6.1 s**. With the table alone, `tape` broke (2/7): the agent trusted the colors.
  Adding a close-up of the tapped hold and "check for tape first" fixed it (7/7). New failure: `twins` seed 12
  found only the tapped hold, because the tap was the far end of its problem and nothing was "linked".
- **Takeaways:** most clear cases need one turn. Hints steer the agent strongly, so they must not be wrong.

## E5: Linked chain and second look (2026-10-28, Yuzuki, b266a14)
- **Question:** fix the `twins` misses without losing precision.
- **Setup:** linked = same-color chain with gaps up to 30% of the photo; if the chain has fewer than 3 holds, jump to
  the nearest same color; a `gap` column; answers with fewer than 3 holds are sent back once; a text-only reply is
  reminded once. 18 walls.
- **Results:** color 85% / 96% / 0.8 s; sam_color 90% / 86% / 1.8 s; **meta 99% / 100% / 7.6 s**. The only miss is
  the washed-out hold in `light` seed 11 that SAM never segmented.
- **Takeaways:** next are real photos (these walls were used for tuning) and SAM misses.

## E6: Real gym photos (2026-10-29, Siho, PR #3 and 6bc820c)
- **Question:** how do SAM 3.1 and the agent do on real gym photos?
- **Setup:** `--photos` with a multi-route `annotations.json`; 8 labelled images / 68 routes derived from a Kaggle
  indoor climbing hold dataset (CC BY-SA 4.0). Photos stay out of git for now and are shared from Siho's
  experiment server.
- **Results:** on all 68 routes, `meta` reached 60.1% recall / 68.3% precision / 63.9% F1
  at 30.9 s per route; the single-pass winner reached 56.9% / 56.9% / 56.9% at 16.1 s.
  The initial 8-route subset is detailed in E9 below; complete runs are in the
  [leaderboard snapshot](leaderboard.md).

The following detailed studies were recorded before this shared log existed. Their
dates show when the runs happened; their E numbers follow the order added to this log.
Results on eight selected routes are separate from the full 68-route leaderboard.

## E7: Turn-budget optimization (2026-10-27, PR #5)

Goal: cut `meta` wall time on a real gym photo (956x1054, 86 SAM candidates). A trace showed
27.4 s in 3 Muse Spark turns vs 3.8 s in SAM: turns are ~90% of the time.

Tried prompt pressure, then reverted it:

| Variant | Holds | Seconds | Turns | Outcome |
|---|---|---|---:|---|
| Base prompt | 11-12 | 43-45 | 3 | Main wall only, correct |
| v1: submit in the first turn | 6 | 14.5 | 1-2 | Dropped the top group |
| v2: include every linked=yes | 22 | 17.2 | 1-2 | Included the next wall |

v1 under-picks (answers before looking far), v2 over-picks (the `linked` chain already reaches
the next wall on this photo, so "trust linked" equals a color filter). The base prompt was
restored and quality re-verified (12 holds, 43 s, 4 turns, left wall excluded).

Kept (structural caps, no policy change):

- `MAX_TURNS` 6 → 4 (typical runs use ≤4; the 90 s tail is gone).
- `MAX_LOOKS` = 2 `look_closer` calls per run (each call appends an image to every later turn).
- Tests: default turn cap, per-run look cap (`tests/test_agent.py`).

## E8: Image diet (2026-10-27, PR #5)

Knobs (all optional, defaults unchanged): `META_AGENT_MODEL`, `META_AGENT_EFFORT`,
`META_AGENT_IMAGE_SIDE` (1280), `META_AGENT_JPEG_QUALITY` (85).

Smaller first-message images, bench `meta` on tape,twins,similar × seed 11:

| Variant | Payload | Recall | Precision | Seconds/photo |
|---|---|---|---|---|
| base 1280/85 | 100% | 100% | 100% | 8.2 |
| mid 960/70 | ~60% | 100% | 100% | 9.6 |
| low 640/55 | ~30% | 100% | 96% | 10.8 |

Rejected: less detail means more turns (tape: base 2 turns + 1 `look_closer` vs low 4 turns +
3 `look_closer`, the third refused by `MAX_LOOKS`), so the run gets slower, not faster, and
low answers wobble between runs (6 vs 8 holds on tape). Payload bytes were measured without
API calls; bench numbers above used the real APIs.

## E9: Real-photo subset bench (meta) (2026-10-28, PR #5)

First real-photo scores: 8 Kaggle route cases (largest route per image), `--photos` mode,
team code (6-turn cap):

| Case | Truth | Found | Hits | Seconds |
|---|---|---|---|---|
| bh0003_r1 | 3 | 3 | 1 | 23.6 |
| bh0074_r13 | 20 | 9 | 3 | 39.0 |
| bh0075_r1 | 12 | 10 | 10 | 32.9 |
| bh0457_r1 | 18 | 7 | 7 | 33.2 |
| bh0502_r4 | 15 | 8 | 7 | 36.7 |
| bh0518_r3 | 8 | 5 | 5 | 34.6 |
| bh0530_r3 | 16 | 7 | 7 | 29.3 |
| bh0917_r7 | 12 | 11 | 11 | 36.8 |
| TOTAL | 104 | 60 | 51 | mean 33.3, recall 49%, precision 85% |

Re-runs wobble (0457: 7 → 14 hits; 0917: 11 → 9). All 8 traces saved with the overlays;
analysis in [learnings.md](learnings.md).

## E10: SAM phrase grounding (2026-10-28, PR #5)

Color- and tape-qualified concept prompts on a real gym photo (assets/119720094.2.jpg):

| Prompt | Returned | Actually matching |
|---|---|---|
| `red climbing hold` | 16 | 15 red |
| `yellow climbing hold` | 33 | 29 yellow (+4 orange) |
| `tape` | 0 | — |
| `colored tape strip` | 13 | mixed tape + bolt-hole lookalikes |

Color adjectives ground well enough for hints; tape needs the longer phrase plus
color+adjacency filtering. Detail in [learnings.md](learnings.md).

## E11: Three-stage pipeline (meta3) vs ReAct agent (meta) (2026-10-28, PR #5)

Same 8 Kaggle route cases, one run each. `meta` numbers from the earlier run (team
6-turn code); `meta3` = SAM + CV color grouping + single-pass Spark (`feat/three-stage`).
Per-case traces, prompts, raw Spark replies and found holds saved under `out/three-stage/`.

| Case | meta truth/found/hits/s | meta3 truth/found/hits/s |
|---|---|---|
| bh0003_r1 | 3/3/1/23.6 | 3/2/1/12.5 |
| bh0074_r13 | 20/9/3/39.0 | 20/3/3/13.6 |
| bh0075_r1 | 12/10/10/32.9 | 12/10/10/21.3 |
| bh0457_r1 | 18/7/7/33.2 | 18/5/4/29.4 |
| bh0502_r4 | 15/8/7/36.7 | 15/5/2/21.1 |
| bh0518_r3 | 8/5/5/34.6 | 8/16/4/13.3 |
| bh0530_r3 | 16/7/7/29.3 | 16/4/3/18.1 |
| bh0917_r7 | 12/11/11/36.8 | 12/18/12/18.2 |
| TOTAL | recall 49%, precision 85%, mean 33.3 s | recall 38%, precision 62%, mean 18.4 s |

`meta3` stage means: SAM 2.8 s (3 calls), CV colors 0.4 s, Spark 15.2 s (1 call),
post ~0 s. Spark prompts 3.3k-11.5k tokens (110 candidates on 0074), completions
1.7k-5.3k tokens, mostly reasoning (1.6k-4.8k) despite `reasoning_effort: low`.
Zero fallbacks: all 8 replies were valid JSON holding the tapped route.
Analysis in [learnings.md](learnings.md).

## E12: Synth bench with meta3 (9 scenes x seeds 11, 12) (2026-10-28, PR #5)

Same run, team 6-turn code, one run per method. Traces and raw replies saved under
`out/three-stage-synth/`.

| Method | Recall | Precision | Mean s/photo |
|---|---|---|---|
| sam_color | 90% | 86% | 1.5 |
| meta | 99% | 100% | 6.5 |
| meta3 | 85% | 96% | 12.2 |

`meta3` misses: tape-11 (1/7), tape-12 (1/7), volume-11 (2/7), busy-11 (5/7); all
other scenes 7/7. Stage means: SAM 1.5 s, CV colors 0.4 s, Spark 10.3 s.
Analysis in [learnings.md](learnings.md).

## E13: Tap-only meta3 re-run (equal criteria, same session) (2026-10-28, PR #5)

v1 asked for ALL routes (unfair task vs the agent's tap-only question, temperature 0,
no tap close-up). v2 fixes all three and re-runs both methods in one session.
Outputs under `out/three-stage-v2/` (real) and `out/three-stage-synth-v2/`.

Synth (18 problems):

| Method | Recall | Precision | Mean s/photo |
|---|---|---|---|
| sam_color | 90% | 86% | 1.6 |
| meta | 99% | 100% | 6.8 |
| meta3 | 99% | 96% | 5.1 |

`meta3` stages: SAM 1.5 s, CV 0.4 s, Spark 3.2 s (was 10.3 s). Tape 7/7 both.
Only miss vs meta: twins-12 over-pick (12 found, 7 hits).

Real (8 cases, same session):

| Case | meta truth/found/hits/s | meta3 truth/found/hits/s |
|---|---|---|
| bh0003_r1 | 3/4/2/25.4 | 3/3/1/8.7 |
| bh0074_r13 | 20/9/3/17.8 | 20/3/3/5.6 |
| bh0075_r1 | 12/10/10/25.2 | 12/5/5/7.3 |
| bh0457_r1 | 18/14/14/23.3 | 18/16/5/14.9 |
| bh0502_r4 | 15/8/7/31.9 | 15/20/5/14.9 |
| bh0518_r3 | 8/9/8/40.6 | 8/33/5/11.3 |
| bh0530_r3 | 16/7/7/28.6 | 16/9/9/12.6 |
| bh0917_r7 | 12/12/11/36.4 | 12/12/9/13.6 |
| TOTAL | recall 60%, precision 85%, mean 28.6 s | recall 40%, precision 42%, mean 11.1 s |

`meta3` stages: SAM 2.9 s, CV 0.4 s, Spark 7.8 s (was 15.2 s), zero fallbacks.
Note the baseline wobble: same code scored 49%/85%/33.3 s last session.

## E14: DINOv2 hold embeddings vs HSV (local retrieval, no API) (2026-10-28, PR #5)

Leave-one-hold-out nearest neighbor: is the closest other hold on the same image
from the same route? GT holds of all 68 routes (587 holds), DINOv2 ViT-S/14
pretrained weights, cosine on crops vs HSV color_distance. Script `/tmp/emb_eval.py`,
results `out/embed-proto/retrieval.json`.

| Image | Holds/routes | HSV | DINO masked | DINO bbox |
|---|---|---|---|---|
| bh0003 | 6/2 | 100% | 83% | 67% |
| bh0074 | 217/17 | 44% | 72% | 69% |
| bh0075 | 58/8 | 83% | 90% | 90% |
| bh0457 | 72/7 | 60% | 76% | 68% |
| bh0502 | 68/11 | 69% | 69% | 68% |
| bh0518 | 43/8 | 60% | 60% | 56% |
| bh0530 | 52/5 | 81% | 81% | 83% |
| bh0917 | 71/10 | 65% | 70% | 77% |
| TOTAL | 587/68 | 60% | 74% | 72% |

Masked crops (background blacked out) beat bbox slightly overall. torch 2.14+cu126
installed in project `.venv` (local GPU RTX 3080 Ti); not yet a repo dependency.

## E15: Subagent optimization sprint on 8 real photos (v3 matrix) (2026-10-29, PR #5)

Three parallel subagents (own worktrees, merged to `feat/three-stage`): prompt
variants v2/v3 (`META3_PROMPT`), DINOv2 embed hints (`META3_EMBED=1`), latency
flags (`TOPO_SKIP_COVERED_POINT`, `TOPO_PRUNE_PROMPT`, payload knobs). 67 tests
+ pyflakes green. Same-session matrix, one run each, outputs `out/three-stage-v3/`:

| Variant | Recall | Precision | Mean s | Spark s |
|---|---|---|---|---|
| v1 (baseline rerun) | 36% | 57% | 12.6 | 9.1 |
| v2 | 47% | 71% | 14.3 | 10.9 |
| v3 | 51% | 70% | 13.7 | 9.8 |
| v3-embed | 49% | 76% | 16.3 | 10.4 |
| v3-embed-fast (winner) | 57% | 75% | 16.7 | 11.4 |

Winner per case: 0003 1/3, 0074 3/20, 0075 9/12, 0457 14/18, 0502 8/15, 0518 4/8,
0530 11/16, 0917 9/12. Stages: SAM 3.0 s (2 calls on 7/8, 3 on 0074), CV 0.4 s,
embed 2.0 s, Spark 11.4 s. Zero fallbacks. Reference: meta 60%/85%/28.6 s.

Rejected: `TOPO_AI_SIDE=960` on the winner (`out/three-stage-v4/`): recall fell
57% to 43% with no time saved (16.8 s). Synth regression on the winner: 100% /
96% / 5.9 s (was 99%/96%/5.1 s), no regression.

<!-- topo-agent-mcp:begin -->
## E16: Full 68-route bench + leaderboard submission (2026-10-30, Siho)

All 68 Kaggle routes, one run each, same session. Server rescored identically
(local holds submitted as x/y/r circles). Outputs `out/bench68/`, leaderboard
runs as siho9o9o (meta d3d6a169, meta3 7f1d69ed, meta3-winner 6505b7da).

| Method | Recall | Precision | F1 | Mean s | Hits/found/truth |
|---|---|---|---|---|---|
| meta (ReAct, 6 turns) | 60% | 68% | 0.639 | 30.9 | 353/517/587 |
| meta3 (single-pass v1) | 54% | 46% | 0.501 | 12.7 | 319/687/587 |
| meta3-winner (v3+embed+fast) | 57% | 57% | 0.569 | 16.1 | 334/587/587 |

8-case subset overstated the gaps: full-set meta precision is 68% (was 85%),
winner precision 57% (was 75%). Winner still leads meta3 everywhere and halves
meta's time at -3pp recall / -11pp precision.

## E17: Three improvement experiments on 68 routes (2026-10-30, Siho)

Built by 3 parallel subagents on `feat/three-stage` (110 tests green), one run
each, outputs `out/bench68x/`, leaderboard runs as siho9o9o.

| Method | Recall | Precision | F1 | Mean s | Hits/found |
|---|---|---|---|---|---|
| exp-verify (2nd drop-only call) | 50% | 65% | 0.568 | 23.3 | 295/451 |
| exp-shortlist (embed shortlist 24) | 54% | 55% | 0.545 | 15.9 | 319/583 |
| exp-rank (no-LLM hybrid) | 61% | 24% | 0.349 | 5.6 | 360/1478 |

Reference: meta3-winner 57%/57%/16.1 (F1 0.569), meta 60%/68%/30.9 (F1 0.639).
Verify ties the winner on F1 but slower; shortlist loses recall (pruning cuts
true holds); ranker has the best recall of any method at 5.6 s with zero Spark
calls, but precision 24% (nearly 1500 found) makes it a retriever, not an answer.

## E18: YOLO11n-seg hold detector via D2 teacher pseudo-labels (2026-10-31, Siho)

Only 19 of 2273 Kaggle images are labeled; YOLO11n-seg on GT alone underfit
(Box/Mask mAP50 0.096/0.100, zero detections on gym photos). Converted the
Kaggle Detectron2 Mask R-CNN weights (segm AP50 74.5) to torchvision (307
tensors, key/channel/normalization mapping, no compiler needed) and
pseudo-labeled 2254 unlabeled images at score>=0.6: 63,641 hold instances.
Retrained nano on GT 14 + pseudo 2254, gold val 5: Box 0.753 / Mask 0.700.

LEAKAGE NOTE: all 8 scoreboard photos (68 route cases) were inside the
training data (5 in GT train, 3 in GT val). Clean retrain `y11n-seg-c`
excludes all 8 (GT train 8 + pseudo 2254 / gold val 3, disjoint). Do not
report board numbers from `y11n-seg-b`.

Exported best.pt to ExecuTorch 0.6 `model.pte` (11 MB, ultralytics exporter
with torch gate bypassed + bundled flatc); `yolo val` on the .pte matches
torch (0.741/0.699). On-device (Galaxy S23, XNNPACK): 160-204 ms per photo,
61-100 holds, Kotlin NMS + mask assembly, tap-to-inspect + camera capture.

(Clean retrain cancelled per user 2026-10-31: `y11n-seg-c` stopped after a few
epochs and removed. `y11n-seg-b` weights + 2254 pseudo-labels kept; the 8
scoreboard photos remain seen-data for that model.)

## E19: Clean n/s/m training excluding all 8 scoreboard photos (2026-10-31, Siho)

Retrained YOLO11-seg n/s/m on GT-clean 8 + pseudo 2247 (8 eval photos fully
excluded; gold val 3 incl. dense bh_0006 with 312 holds). All three
early-stopped (~45 epochs, 6h total). Clean val looked bad (Mask 0.33-0.39)
but the control settled it: leaky run-b weights score only 0.39/0.33 on the
same clean val vs 0.75/0.70 on their own val — the clean val is harder, not
the training worse (all three clean runs beat run-b there).

Fair held-out eval on the 8 scoreboard photos (GT hold masks):

| Weights | Seen 8? | Box mAP50 | Mask mAP50 |
|---|---|---|---|
| y11n-seg-b (leaky) | yes | 0.807 | 0.751 |
| yolo11n-seg-clean | no | 0.708 | 0.658 |
| yolo11s-seg-clean | no | 0.822 | 0.735 |
| yolo11m-seg-clean | no | 0.834 | 0.735 |

Leakage is worth ~10pp Box / ~9pp Mask at nano. Clean-s already beats
leaky-n on Box and ties on Mask; clean-m adds +1pp Box only. Exported
clean-s to ET 0.6 `model.pte` (40 MB); `yolo val` on the .pte matches torch
(0.816/0.736).

On-device clean-s (Galaxy S23, XNNPACK, sample photo, 3 runs): 55-80
holds, infer 390-710 ms, render 220-270 ms. Nano (leaky 11 MB) was
160-370 ms / 58-74 holds. Device inference is nondeterministic across runs
(two stable candidate modes, 343 vs 423) while the .pte is bit-identical on
desktop: XNNPACK threading on the phone, not an export bug.

## E20: Hold-detection head-to-head: SAM 3.1 concept vs YOLO-s clean (2026-10-31, Siho)

Same 8 photos, same greedy recall/precision @ IoU 0.5 vs GT hold masks
(outputs `out/sam-detect-8/`). SAM: concept prompt "climbing hold", 1 call
per photo (~2-4 s). YOLO-s clean (never saw the 8): conf 0.5.

| Photo | GT | SAM R/P | YOLO-s R/P |
|---|---|---|---|
| bh_0003 | 7 | 0.57/0.67 | 0.71/0.83 |
| bh_0074 | 236 | 0.47/1.00 | 0.50/0.85 |
| bh_0075 | 63 | 0.94/1.00 | 0.94/0.98 |
| bh_0457 | 76 | 0.88/0.92 | 0.87/0.96 |
| bh_0502 | 72 | 0.86/0.95 | 0.71/0.91 |
| bh_0518 | 48 | 0.92/0.90 | 0.77/0.82 |
| bh_0530 | 55 | 0.87/0.89 | 0.75/0.82 |
| bh_0917 | 79 | 0.82/1.00 | 0.65/0.93 |
| MEAN | | 0.79/0.92 | 0.74/0.89 |

SAM 3.1 leads pure hold detection by ~5pp recall / ~3pp precision, but needs
the API (~3 s/photo); YOLO-s runs on-device in 465 ms with no API. Both miss
hardest on dense bh_0074 (236 tiny holds).

## E21: YOLO-s + Spark route finding on 68 cases (2026-10-31, Siho)

New `yolo3s` finder (`feat/three-stage` 7541da7): YOLO11s-seg clean masks
through the existing CV + single-pass Spark path, no segmentation API.
Same 68 real-route cases, both submitted as siho9o9o:

| Method | Recall | Precision | F1 | Mean s |
|---|---|---|---|---|
| yolo3s (v1 prompt) | 42% | 52% | 0.461 | 7.1 |
| yolo3s-winner (v3 + embed) | 51% | 66% | 0.573 | 9.5 |

yolo3s-winner takes leaderboard #2 (meta 0.639 stays #1), beating
meta3-winner 0.569 at nearly half the time (9.5 s vs 16.1 s). Precision
66% nearly matches meta's 68%; the gap is recall (51% vs 60%), mostly
tapped holds YOLO misses (3 zero-find cases on bh0917).

## E22: Kaggle triplet net as color-grouping feature, rejected (2026-10-31, Siho)

Tested the Kaggle triplet checkpoint (`triplet_network_final.pt`, ResNet50 +
256-d head) as an illumination-robust replacement for HSV in same-route
retrieval: 587 truth holds across 68 real-route cases, nearest neighbor
must share the route. Masked 128 px crops, ImageNet norm:

| Feature | Same-route NN accuracy |
|---|---|
| HSV baseline | 354/587 = 60.3% |
| Triplet final layer (masked) | 250/587 = 42.6% |
| Triplet final layer (bbox) | 178/587 = 30.3% |
| Triplet mid layer fc.0+ReLU (masked) | 321/587 = 54.7% |
| Triplet mid layer (bbox) | 248/587 = 42.2% |
| DINOv2-S (reference, earlier run) | 436/587 = 74.3% |

Even the mid-layer embedding loses to plain HSV on 6/8 walls, and the
ResNet50 backbone is too heavy for on-device anyway. Rejected; HSV stays
the practical feature. Raw per-image numbers in
`out/trip-proto/retrieval_early.json`.

## E23: CV color follow-ups: histogram beats DINOv2 (2026-10-31, Siho)

Same 587-hold same-route NN protocol as the triplet test. Gray-world
white balance per wall, HS histograms (16x16, Bhattacharyya/correlation,
8x8 EMD), gray-world Lab means, DINOv2-S re-run, and z-score HSV+DINO
ensembles:

| Feature | Accuracy |
|---|---|
| HSV baseline | 352/587 = 60.0% |
| Gray-world + HSV | 375/587 = 63.9% |
| Gray-world + Lab mean | 416/587 = 70.9% |
| Gray-world + ab mean | 367/587 = 62.5% |
| HS hist correlation | 445/587 = 75.8% |
| HS hist Bhattacharyya | 457/587 = 77.9% |
| HS hist EMD | 429/587 = 73.1% |
| HS hist Bhattacharyya (gray-world) | 456/587 = 77.7% |
| HSV 3D hist (8x8x4) | 364/587 = 62.0% |
| DINOv2-S masked | 433/587 = 73.8% |
| Ensemble DINO 0.3 + HSV 0.7 | 460/587 = 78.4% |
| Ensemble DINO 0.5 + HSV 0.5 | 459/587 = 78.2% |
| Ensemble DINO 0.7 + HSV 0.3 | 465/587 = 79.2% |

HS histogram + Bhattacharyya alone beats DINOv2 by 4pp with zero model
and ~12 s total pairwise CPU time (EMD included). Winners: hist_bhatt
for on-device (no weights, OpenCV only), DINO 0.7 + HSV 0.3 ensemble
for the server path. Raw numbers in `out/cv-proto/retrieval.json`.

## E24: HS-histogram hints in the pipeline, yolo3s-hist (2026-11-01, Siho)

Ported the retrieval winner into `topo_agent/pipeline.py` (`feat/hist-color`):
`classify_colors` stores a normalized 16x16 HS histogram per hold, ranking
similarities prefer it over the HSV mean, and `META3_HIST=1` adds a `hist`
top-3 column to the Spark table (mirrors the `embed` column, no model needed).
125 unittests pass, pyflakes clean.

Two findings on the way: union-find grouping on hist over-merges (fallback
F1 0.27-0.35 vs 0.356 for HSV-only, measured offline without API), so groups
stay HSV-based; and a hints-off rerun scored 46%/65% vs the earlier 51%/66%,
pinning Spark single-pass run-to-run noise near ±5pp recall.

68-case leaderboard (`yolo3s-hist`, run 1172a18a, siho9o9o):

| Method | Recall | Precision | F1 | Mean s |
|---|---|---|---|---|
| yolo3s-winner (before) | 51% | 66% | 0.573 | 9.5 |
| yolo3s-hist (hist hints) | 50% | 69% | 0.581 | 12.4 |

Precision 69% is the best on the board (meta 68%); the +0.008 F1 gain sits
inside Spark noise but takes #2 behind meta 0.639. Extra time is histogram
building on dense walls. Raw traces in `out/bench68hist2/`.

## E25: Offline tap-to-group route finder in the mobile apps (2026-11-01, Siho)

Same histogram idea, fully on-device (`feat/hist-route-app`): after YOLO
detection the app builds a 16x16 HS histogram per hold over mask-interior
pixels and tapping a hold keeps matches (Bhattacharyya >= 0.6) bright while
the rest dim; wall tap clears. Threshold calibrated on 3,075 same-route /
31,577 diff-route truth pairs: 0.6 keeps 84% recall at 14% false positives.
Android verified with a real `assembleDebug` build (25 MB APK); iOS mirrors
the logic in `YoloSeg.swift`/`ContentView.swift` (needs an Xcode build).

## E26: Muse Spark direct from the Android app (2026-11-01, Siho)

First tried serving `yolo3s` over a Cloudflare tunnel for the phone browser
(`feat/yolo3s-web`: web GUI + `meta3`/`yolo3s` methods, smoke-tested 9 holds
in 20.5 s), then dropped it: the phone can call the Meta API directly, which
saves the tunnel relay and works without the PC. New `SparkRoute.kt`
(`feat/hist-route-app`) ports the server's Spark stage — colors, groups,
neighbors, hist hints, prompt table, v3 system prompt verbatim, annotated
photo + tap close-up, one `muse-spark-1.3` call, strict answer parsing.
Taps group offline instantly, then the Spark answer replaces the grouping;
the key is pasted in-app (SharedPreferences, never baked into the APK) and
any failure keeps the offline grouping. Verified with `assembleDebug` plus
6 Robolectric unit tests (table shape, request contract, parser); on-device
run pending (no device attached at build time).

## E27: Phone UI: tap frame, stopwatch, grayscale route (2026-11-01, Siho)

`feat/hist-route-app`: tapped hold gets a black-and-white target frame,
Spark waits show a ticking `asking spark… N.N s` stopwatch, and a confirmed
Spark route renders in color on a grayscaled wall. Verified end-to-end via
adb-driven taps + screenshots: tap #35 → stopwatch → `spark route 7/100 ·
conf 0.90 · 20596 ms` with the route in color on gray. Fixed one crash on
the way (`createBitmap` is immutable; overlay now copies mutable) with a
Robolectric regression test.

## E28: Close-up ablation: no gain either way (2026-11-01, Siho)

`TOPO_NO_CLOSEUP=1` drops the 256px tap close-up (4th content part) to test
whether 2 images suffice (`exp/no-closeup`). Same 10 cases, yolo3s-hist
config, both directions:

| Config | Recall | Precision | F1 | Mean s |
|---|---|---|---|---|
| With close-up | 42% | 70% | 0.525 | 7.9 |
| Without close-up | 44% | 63% | 0.518 | 8.0 |

No meaningful difference: scores sit inside Spark noise and time is
identical (the tile is tiny). Kept the close-up; phone latency must be
attacked elsewhere (selective Spark calls, not smaller payloads).
Raw rows in `out/bench10noclose/`.

## E29: Tap-miss recovery with a synthetic tap box (2026-11-01, Siho)

YOLO ignores point prompts, so when it missed the tapped hold (17 of 68 cases)
the pipeline answered [] without calling Spark. `find_with` now appends a
small tap-centered box (`tap_made` trace flag) that rides through colors,
hints, table, close-up and answer like any other candidate.

| Method | Recall | Precision | F1 | Mean s |
|---|---|---|---|---|
| yolo3s-hist (before) | 50% | 69% | 0.581 | 12.4 |
| yolo3s-tapfix (after) | 60% | 60% | 0.603 | 17.4 |

All 17 rescued cases return holds; recall matches meta (60%). Extra time is
the rescued cases now reaching Spark. Submitted as `yolo3s-tapfix` (new
frontier). Raw traces in `out/bench68tapfix/`.

## Shared MCP experiment records

These entries were exported from the team MCP server. The server database
retains the full record; this section is the repository snapshot.

### Benchmark record: exp-rank

- Author: siho9o9o
- Recorded: 2026-10-29T17:18:48+00:00
- Commit: `88c0463`
- Tags: benchmark, frontier

> Run bef941cd5f7b4a6db993299a306854d1: F1 34.9%, recall 61.3%, precision 24.4%, mean time 5.6s. No-LLM hybrid ranker: color*embed score + LINK chain (rank method). 0 Spark calls.

### Benchmark record: exp-shortlist

- Author: siho9o9o
- Recorded: 2026-10-29T17:18:48+00:00
- Commit: `88c0463`
- Tags: benchmark, frontier

> Run 80eba4f6f12d42f38a5177548ff836f9: F1 54.5%, recall 54.3%, precision 54.7%, mean time 15.9s. Winner + embedding-first shortlist to 24 cands (META3_SHORTLIST=1). 1 call.

### Benchmark record: meta3-winner

- Author: siho9o9o
- Recorded: 2026-10-29T15:33:07+00:00
- Commit: `6bc820c`
- Tags: benchmark, frontier

> Run 6505b7da6eb34efeba50afdb99654361: F1 56.9%, recall 56.9%, precision 56.9%, mean time 16.1s. Single-pass v3 prompt + DINOv2 embed hints + skip-covered-point + prune-40. 1 Spark call.

### Benchmark record: meta3

- Author: siho9o9o
- Recorded: 2026-10-29T15:33:07+00:00
- Commit: `6bc820c`
- Tags: benchmark, frontier

> Run 7f1d69eda29145bda845f250cfc9f95f: F1 50.1%, recall 54.3%, precision 46.4%, mean time 12.7s. Single-pass pipeline v1 prompt, no embed, no prune. 1 Spark call.

### Benchmark record: meta

- Author: siho9o9o
- Recorded: 2026-10-29T15:33:07+00:00
- Commit: `6bc820c`
- Tags: benchmark, frontier

> Run d3d6a16969cb488dbcf6403ec1d27a0d: F1 63.9%, recall 60.1%, precision 68.3%, mean time 30.9s. ReAct agent (SAM 3.1 + Muse Spark, 6 turns, tools). Team code; env-gated opts off.
<!-- topo-agent-mcp:end -->

## 2026-10-30: Align MCP scoring with upstream main

The experiment server now scores against upstream commit `9bd7b02` rather than an older
local checkout. The five scoring functions used by the server (`holds_of`, `match`,
`summary`, `_inside`, `_covers`) have identical executable bodies in both checkouts.
After backing up the server database, six existing runs were moved from evaluator
version `a16e466e` to `23985f59`. Their scores and reported times were not changed;
the models were not rerun. Future submissions should use the version from
`get_benchmark`.
