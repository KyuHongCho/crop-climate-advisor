# Crop-Climate Advisor

[![CI](https://github.com/KyuHongCho/crop-climate-advisor/actions/workflows/ci.yml/badge.svg)](https://github.com/KyuHongCho/crop-climate-advisor/actions/workflows/ci.yml)

Given a **crop** and a **location**, retrieve the crop's agronomic requirements and the
location's real climate, then reason about **suitability** and what a controlled-environment
growing chamber would need to correct for any gap.

A small project exploring how to combine **retrieval over structured agronomic data**, a
**live climate-data tool call (MCP)**, and **Agent Skills** into an honest decision-support
tool — built as real, working code with transparently reported status.

> **Status: early. Building in progress — not finished.**
> A working end-to-end slice runs today for **basil** — including its optimal temperature reported
> as three attributed, disagreeing published claims rather than one — and the NASA POWER climate
> lookup is exposed as an **MCP server**. The second crop, the eval harness that would *score* how
> that disagreement is handled, the RAG layer, and the Agent Skill are **not built**.
> See [What works / What's planned](#what-works--whats-planned).
> This README describes only what actually runs; planned work is labelled as such.

## What works / what's planned

| | Component | Status |
|---|-----------|--------|
| ✅ | Live climate lookup — NASA POWER climatology API (free, no auth) | **working** |
| ✅ | Structured crop-requirements lookup — FAO ECOCROP, scraped once → bundled JSON | **working** (basil) |
| ✅ | Suitability + chamber-correction reasoning (temperature, rainfall) | **working** |
| ✅ | CLI + offline unit tests | **working** |
| ✅ | NASA POWER wrapped as an **MCP server** (official SDK, stdio) | **working** — the CLI still calls `fetch_climate` directly, not via MCP |
| ✅ | Basil's optimal temperature carried as **three attributed claims**, not one — each with its stated condition and the paper it was verified through (see [The eval case](#the-eval-case)) | **working** — the CLI prints all three and derives from their intersection that no single window satisfies them all; for the queried location it also reports where that location falls relative to each claim, flags it when the sources point in opposite directions there, and prints a conservative read that never calls a site optimal unless every source does |
| ⏳ | Second crop (*Catharanthus roseus*, ECOCROP id 652) | planned |
| ⏳ | Eval harness — turning that disagreement into a scored, repeatable pass/fail run | planned |
| ⏳ | Narrative RAG over ECOCROP free-text + peer-reviewed papers | planned |
| ⏳ | Packaged as a **Claude Code Agent Skill** (`SKILL.md`) | planned |

## Quickstart

```bash
# Requires Python 3.10+ (the `mcp` SDK does; macOS's stock /usr/bin/python3 is 3.9 and won't work)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # certifi (TLS), mcp (the MCP SDK), anyio

# Assess basil at a location (live NASA POWER call):
python3 -m crop_advisor.cli --crop basil --lat 51.5 --lon -0.13 --place London

# Run the MCP server (stdio; speaks the Model Context Protocol on stdin/stdout):
python3 -m crop_advisor.mcp_server

# Run the offline unit tests (no network):
python3 -m unittest discover -s tests

# Also run the live NASA POWER test (opt-in; makes a real API call):
CROP_ADVISOR_LIVE_TESTS=1 python3 -m unittest discover -s tests

# Regenerate the bundled ECOCROP data (one-time scrape):
python3 scripts/scrape_ecocrop.py --id 1547 --slug basil --common-name basil
```

Example output (basil @ London):

```
Crop-Climate Advisor — basil @ London
========================================================
Location climate (NASA POWER): annual mean 10.39 °C, warmest month 17.63 °C, annual precip 708.6 mm
Crop needs — temperature: no single window satisfies all 3 published sources; every claim is listed below.
Crop needs — rainfall (FAO ECOCROP id 1547): opt 1000–1600 mm/yr

  temperature  10.39 °C     → each published optimal window, and the change it would need:
      18–27 °C  → needs +7.6 °C   FAO ECOCROP (id 1547)  [no condition stated]
      25–30 °C  → needs +14.6 °C  Chang, Alderson & Wright (2005)  [at DLI 20–22 mol·m⁻²·d⁻¹]
      29–35 °C  → needs +18.6 °C  Walters & Currey (2019)  [at DLI 19.5 mol·m⁻²·d⁻¹]
  rainfall     708.6 mm/yr  → SURVIVABLE  (needs +291.4 mm/yr to reach optimal)
  note: even the warmest month stays below FAO ECOCROP's optimal minimum.

Verdict (on FAO ECOCROP's bands): Basil is marginal outdoors at London; a controlled-environment chamber would need to close the gaps below.
Conservative read: 10.39 °C is below optimal.

Data: NASA POWER (climate) · © FAO ECOCROP (crop requirements).
Temperature claims:
  · FAO ECOCROP data sheet, id 1547 (read directly): https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=1547
  · Chang, Alderson & Wright (2005), J. Hortic. Sci. Biotechnol. 80:593–598 (via Barickman et al. (2021), Plants 10(6):1072): https://pmc.ncbi.nlm.nih.gov/articles/PMC8226578/
  · Walters & Currey (2019), HortScience 54(11):1915 (via Walters, Tarr & Lopez (2023), PLoS One 18(11):e0294905): https://pmc.ncbi.nlm.nih.gov/articles/PMC10688745/
```

This block is not hand-written: `tests/test_cli_format.py` extracts it from this README and
asserts it equals what the CLI actually prints, so it cannot drift.

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

## The eval case

Three sources give three different optimal temperature ranges for basil:

| Source | Optimal | Stated condition |
|---|---|---|
| FAO ECOCROP (id 1547) | 18–27 °C | none stated |
| Chang et al. (2005), via [Barickman et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8226578/) | 25–30 °C | DLI 20–22 mol·m⁻²·d⁻¹ |
| Walters & Currey (2019), via [Walters et al. 2023](https://doi.org/10.1371/journal.pone.0294905) | 29–35 °C | DLI 19.5 mol·m⁻²·d⁻¹ |

No single window satisfies all three — 18–27 and 25–30 overlap at 25–27, but 29–35 does not intersect
ECOCROP's range at all. The ranges are also not strictly comparable: the two journal figures are
conditioned on a stated daily light integral (DLI), while ECOCROP states none.

**What runs today:** the CLI reports all three of these claims side by side — each with its stated
condition, and, for the two it could not read directly, the open-access paper it *was* read through.
The "no single window" sentence above is computed from the three bands rather than asserted
(`crop_advisor/claims.py`); if the sources ever did overlap, the CLI would name the overlap instead.
Nothing picks a winner. The CLI also reports, for the queried location, whether the sources point in
*opposite* directions there — one calling the site too cold while another calls it too hot — which is a
narrower question than whether they share a window at all, and can be false even where the answer to that
one is no. Between 27 and 29 °C they do point in opposite directions: that is above ECOCROP's optimal
maximum but below Walters & Currey's optimal minimum, so the CLI flags the disagreement and refuses to call
the site optimal (the interval is derived from the published bands and pinned by
`tests/test_divergence.py`).

**What is still planned:** the eval harness itself — a scored, repeatable run that checks the system
**notices and flags** this disagreement rather than silently picking one. A real, pre-identified
benchmark, not a synthetic one. Reporting the three claims (above) is not yet the same as being
scored on handling them.

## Data sources & attribution

- **Climate:** [NASA POWER](https://power.larc.nasa.gov/) climatology API (T2M, PRECTOTCORR).
- **Crop requirements:** © **FAO ECOCROP** — used for non-commercial research with attribution,
  per [FAO Terms and Conditions](https://www.fao.org/contact-us/terms/en/).
- **Basil optimal-temperature claims** — cited in the CLI's own output (see the example above):
  Barickman et al. (2021), *Plants* 10(6):1072,
  doi:10.3390/plants10061072 ([open access via PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8226578/)) — CC BY 4.0. The 25–30 °C
  figure originates with Chang, Alderson & Wright (2005), *J. Hortic. Sci. Biotechnol.* 80:593–598;
  the 29–35 °C figure with Walters & Currey (2019), *HortScience* 54(11):1915. Both are cited here
  **via** the open-access papers linked above, which is how they were verified.

## Project layout

```
crop_advisor/           # the app: climate.py, ecocrop.py, suitability.py, cli.py,
                        #          claims.py (published temperature claims + provenance),
                        #          mcp_server.py (MCP wrapper around the climate call)
data/ecocrop/           # bundled, scraped-once crop requirements (JSON)
scripts/scrape_ecocrop.py   # one-time ECOCROP scraper
tests/                  # offline unit tests (+ one opt-in live NASA POWER test)
```

## The MCP server

`crop_advisor/mcp_server.py` exposes one tool, `get_climate(lat, lon)`, over the
Model Context Protocol using the official Python SDK (`FastMCP`, stdio transport).
It is a thin adapter: it wraps the existing `fetch_climate()` and adds no climate
logic of its own. The tool returns the full climate summary — annual and monthly
mean temperature, warmest-month temperature, annual precipitation, and the source
string — as structured content. The return type is a `TypedDict`, so all seven
fields and their types appear in the tool's `outputSchema`, and a client can
discover the shape without calling it.

The tool is `async` and offloads the blocking HTTP call to a worker thread. This
matters: `FastMCP` 1.x runs a *sync* tool inline on the event loop, so a slow
NASA POWER response would otherwise freeze the whole server — every other
request, including cancellation — for up to the 30s timeout.

The SDK is pinned to `mcp>=1.29,<2`. Version 2.0.0 (released 2026-07-28) replaced
`FastMCP` with `MCPServer`; upstream recommends a `<2` bound until migrating, and
the migration for a stdio server this small is a two-line change.

Tests drive the server through a real MCP client session (`initialize` →
`list_tools` → `call_tool`) rather than calling the Python function directly — a
direct call would exercise `fetch_climate`, not the MCP layer.
