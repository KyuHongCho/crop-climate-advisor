---
name: crop-suitability
description: Assess whether a crop (basil, lettuce, strawberry, tomato, cucumber, sweet-pepper or kale) suits a location's climate, and what temperature it needs. Use when someone asks if a crop can be grown somewhere, whether a place's climate is right for it, or what its optimal temperature is. Runs this project's crop-climate-advisor CLI on real climate data and reports every published temperature claim with its source.
---

# Crop suitability

## Run the CLI

From the project root, with the location's coordinates in decimal degrees:

```bash
python -m crop_advisor.cli --crop <slug> --lat <lat> --lon <lon> --place "<name>"
```

`<slug>` is one of `basil`, `lettuce`, `strawberry`, `tomato`, `cucumber`, `sweet-pepper`, `kale`.
Run it once per location. Answer from what it prints; do not substitute remembered figures.

If the CLI exits with an error, say the assessment could not be run and why. Do not estimate the
climate yourself.

## Report the answer

1. **Every temperature claim, attributed.** The CLI lists each source's optimal window. Name each
   source, give its window, and give the condition it states (for basil, two figures hold only at a
   stated daily light integral, DLI). Never present one window as the crop's optimum.
2. **If you lean on one source, say which and why.** An unattributed "the optimal range is…" is wrong.
3. **Opposite directions.** When the CLI prints `Sources disagree:`, tell the user that the sources
   point in opposite directions at this location: one calls it too warm, another too cool.
4. **No unqualified "optimal".** Do not call the site optimal, ideal, or a good match unless you name
   the source(s) that support it, or say that at least one source places it outside its optimal window.
   The CLI's `Conservative read:` line is the safe summary.
5. **Rainfall and the verdict.** Report the rainfall status and the CLI's verdict line, which is on
   FAO ECOCROP's bands only.

Why basil's sources disagree, and what the DLI conditions mean: `reference/temperature-claims.md`.
Read it when the user asks about the disagreement itself.
