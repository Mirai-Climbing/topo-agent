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

## E6: Real gym photos (in progress, Siho, PR #3)
- **Question:** how do SAM 3.1 and the agent do on real gym photos?
- **Setup:** `--photos` with a multi-route `annotations.json`; 8 labelled images / 68 routes derived from a Kaggle
  indoor climbing hold dataset (CC BY-SA 4.0). Photos stay out of git for now and are shared from Siho's
  experiment server.
- **Results:** pending.
