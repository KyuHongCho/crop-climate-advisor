# Crop-Climate Advisor

Given a **crop** and a **location**, retrieve the crop's agronomic requirements and the
location's real climate, then reason about **suitability** and what a controlled-environment
growing chamber would need to correct for any gap.

A small project exploring how to combine **retrieval over structured agronomic data**, a
**live climate-data tool call (MCP)**, and **Agent Skills** into an honest decision-support
tool — built as real, working code with transparently reported status.

> **Status: Stage 0 (early). Building in progress — not finished.**
> A working end-to-end slice runs today for **basil**. See [What works / What's planned](#what-works--whats-planned).
> This README describes only what actually runs; planned work is labelled as such.

## What works / what's planned

| | Component | Status |
|---|-----------|--------|
| ✅ | Live climate lookup — NASA POWER climatology API (free, no auth) | **working** |
| ✅ | Structured crop-requirements lookup — FAO ECOCROP, scraped once → bundled JSON | **working** (basil) |
| ✅ | Suitability + chamber-correction reasoning (temperature, rainfall) | **working** |
| ✅ | CLI + offline unit tests | **working** |
| ⏳ | Second crop (*Catharanthus roseus*, ECOCROP id 652) | planned (Stage 1) |
| ⏳ | Eval harness — flag the basil optimal-temp source discrepancy (see below) | planned (Stage 1) |
| ⏳ | Narrative RAG over ECOCROP free-text + peer-reviewed papers | planned (Stage 1) |
| ⏳ | NASA POWER wrapped as an **MCP server** | planned (Stage 2) |
| ⏳ | Packaged as a **Claude Code Agent Skill** (`SKILL.md`) | planned (Stage 2) |

## Quickstart

```bash
pip install -r requirements.txt          # just `certifi` (for TLS verification)

# Assess basil at a location (live NASA POWER call):
python3 -m crop_advisor.cli --crop basil --lat 51.5 --lon -0.13 --place London

# Run the offline unit tests:
python3 -m unittest discover -s tests

# Regenerate the bundled ECOCROP data (one-time scrape):
python3 scripts/scrape_ecocrop.py --id 1547 --slug basil --common-name basil
```

Example output (basil @ London):

```
Location climate (NASA POWER): annual mean 10.39 °C, warmest month 17.63 °C, annual precip 708.6 mm
Crop needs (FAO ECOCROP id 1547): temp opt 18–27 °C, rain opt 1000–1600 mm/yr

  temperature  10.39 °C     → SURVIVABLE  (needs +7.6 °C to reach optimal)
  rainfall     708.6 mm/yr  → SURVIVABLE  (needs +291.4 mm/yr to reach optimal)
  note: even the warmest month stays below the crop's optimal minimum.

Verdict: Basil is marginal outdoors at London; a controlled-environment chamber
would need to close the gaps below.
```

## Architecture & design decisions

- **Structured lookup vs RAG.** ECOCROP's environmental fields (temperature, rainfall, pH) are
  *structured numbers*, so they get a **structured lookup**, not text retrieval. RAG is reserved
  for genuinely narrative text (ECOCROP free-text descriptions + peer-reviewed papers — planned).
- **Scrape once, not live per request.** ECOCROP's numeric fields are scraped **once** into
  `data/ecocrop/*.json` rather than fetched on every request — cleaner, and it bounds third-party
  calls (a deliberate context-engineering / cost choice). FAO terms permit non-commercial research
  reuse with attribution.
- **Runtime: hybrid.** The core is a standalone Python app with a clean interface; it will also be
  packaged as a **Claude Code Agent Skill** that sequences the procedure via progressive disclosure.
  (The Claude API code-execution container was ruled out — its sandbox has no network access, which
  conflicts with the live NASA POWER call.)
- **TLS.** Network calls verify against `certifi`'s CA bundle; verification is never disabled.

Each of these was a deliberate, recorded design decision rather than an incidental choice.

## The eval case (planned)

ECOCROP lists basil's optimal temperature as **18–27 °C**; a peer-reviewed source gives **25–30 °C**.
The planned eval harness checks that the system **notices and flags** this cross-source disagreement
rather than silently picking one — a real, pre-identified benchmark, not a synthetic one.

## Data sources & attribution

- **Climate:** [NASA POWER](https://power.larc.nasa.gov/) climatology API (T2M, PRECTOTCORR).
- **Crop requirements:** © **FAO ECOCROP** — used for non-commercial research with attribution,
  per [FAO Terms and Conditions](https://www.fao.org/contact-us/terms/en/).

## Project layout

```
crop_advisor/           # the app: climate.py, ecocrop.py, suitability.py, cli.py
data/ecocrop/           # bundled, scraped-once crop requirements (JSON)
scripts/scrape_ecocrop.py   # one-time ECOCROP scraper
tests/                  # offline unit tests for the suitability logic
```
