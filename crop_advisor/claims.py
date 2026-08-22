"""Published optimal-temperature claims for basil, each with its provenance.

Three sources give three different optimal temperature ranges (see the README's
"The eval case"). This module keeps all three as *claims* — a number plus who
said it, under what stated condition, and whether we read the source ourselves
— so nothing downstream can present one of them as unattributed fact.

Why a module and not `data/`: `data/` holds requirements **scraped** from a
provider (see the README's "Scrape once, not live per request"). These two
journal figures were not scraped; they were read by hand out of open-access
papers, and the citation chain that makes them checkable is the interesting
part. Hard-coding them beside that chain, frozen, keeps the two inseparable.

Everything here is pure and offline.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

#: Where a location sits relative to one claim's optimal band. "below" and
#: "above" describe the *location*, not the band.
Direction = Literal["below", "optimal", "above"]

#: Every reading conservative() can return. get_args() on this gives callers
#: (and tests) an authoritative allowed-set that cannot drift from the code.
ConservativeRead = Literal[
    "optimal",
    "below optimal",
    "above optimal",
    "not optimal (sources disagree in direction)",
]


@dataclass(frozen=True)
class Claim:
    """One source's claimed optimal band for one metric, with its provenance."""

    source: str            # short label for inline use, e.g. "Walters & Currey (2019)"
    reference: str         # full bibliographic reference, for the citation footer
    opt_min: float
    opt_max: float
    condition: str | None  # the condition the source states, or None if it states none
    read_directly: bool    # False when we read the figure in a *different* paper
    via: str | None        # the open-access paper it was read through, when not direct
    url: str               # where the figure can actually be checked

    @property
    def band(self) -> tuple[float, float]:
        return (self.opt_min, self.opt_max)

    def __post_init__(self) -> None:
        # cite() renders a read_directly claim as first-hand and never looks at
        # `via`, so this combination would credit the wrong URL and drop the
        # citation chain in silence. Make the state unrepresentable instead.
        if self.read_directly and (self.via or "").strip():
            raise ValueError(
                f"{self.source!r} is marked read_directly=True but also names a 'via' "
                f"source; rendering it would credit the via paper's URL as read directly "
                f"and silently drop the chain"
            )


def cite(claim: Claim) -> str:
    """Render a claim's attribution.

    A claim we did not read directly renders a `(via ...)` form naming the
    paper we did read. If such a claim names no `via`, that is a broken claim
    and this raises: a placeholder would let an unprovenanced number travel
    quietly, which is the exact failure this module exists to prevent.
    """
    if claim.read_directly:
        return f"{claim.reference} (read directly): {claim.url}"
    via = (claim.via or "").strip()
    if not via:
        raise ValueError(
            f"{claim.source!r} is marked read_directly=False but names no 'via' source; "
            f"an indirect claim must say which paper it was read through"
        )
    return f"{claim.reference} (via {via}): {claim.url}"


# --- the two journal claims, frozen ------------------------------------------
# Both were read via an open-access paper, not from the (paywalled) original.

CHANG_2005 = Claim(
    source="Chang, Alderson & Wright (2005)",
    reference="Chang, Alderson & Wright (2005), J. Hortic. Sci. Biotechnol. 80:593–598",
    opt_min=25,
    opt_max=30,
    condition="DLI 20–22 mol·m⁻²·d⁻¹",
    read_directly=False,
    via="Barickman et al. (2021), Plants 10(6):1072",
    url="https://pmc.ncbi.nlm.nih.gov/articles/PMC8226578/",
)

WALTERS_CURREY_2019 = Claim(
    source="Walters & Currey (2019)",
    reference="Walters & Currey (2019), HortScience 54(11):1915",
    opt_min=29,
    opt_max=35,
    condition="DLI 19.5 mol·m⁻²·d⁻¹",
    read_directly=False,
    via="Walters, Tarr & Lopez (2023), PLoS One 18(11):e0294905",
    url="https://pmc.ncbi.nlm.nih.gov/articles/PMC10688745/",
)

#: The journal claims, in ascending order of optimal band. A tuple, of frozen
#: dataclasses: nothing downstream can quietly edit a published figure.
JOURNAL_TEMPERATURE_CLAIMS: tuple[Claim, ...] = (CHANG_2005, WALTERS_CURREY_2019)

#: The two journal claims are studies of *Ocimum basilicum*; nothing in them
#: applies to another species. This is the crop they are about.
JOURNAL_CLAIMS_ECOCROP_ID = 1547


def temperature_claims(crop: dict) -> tuple[Claim, ...]:
    """The published optimal-temperature claims that apply to `crop`.

    The FAO ECOCROP claim is built from the crop dict actually in hand, so it
    tracks the bundled data file. The two journal claims are studies of one
    species and are attached only to that species (see
    JOURNAL_CLAIMS_ECOCROP_ID); any other crop gets its ECOCROP claim alone.
    """
    band = crop["temperature_c"]
    ecocrop_id = crop.get("ecocrop_id")
    source = crop.get("_source") or {}
    ecocrop = Claim(
        source=f"FAO ECOCROP (id {ecocrop_id})",
        reference=f"FAO ECOCROP data sheet, id {ecocrop_id}",
        opt_min=band["opt_min"],
        opt_max=band["opt_max"],
        condition=None,          # ECOCROP states none
        read_directly=True,      # bundled from the data sheet itself
        via=None,
        url=source.get("url", ""),
    )
    # Read through the module-level name at call time, so the journal claims
    # stay substitutable, and gate on the crop they are actually about.
    journal = JOURNAL_TEMPERATURE_CLAIMS if ecocrop_id == JOURNAL_CLAIMS_ECOCROP_ID else ()
    return (ecocrop,) + journal


def consensus(claims: Iterable[Claim]) -> tuple[float, float] | None:
    """The band satisfying every claim at once, or None if there is no such band.

    For basil's real three this is None: max(18, 25, 29) = 29 is above
    min(27, 30, 35) = 27, so the intersection is empty.
    """
    claims = tuple(claims)
    if not claims:
        return None
    low = max(float(c.opt_min) for c in claims)
    high = min(float(c.opt_max) for c in claims)
    if low > high:
        return None
    return (low, high)


def classify(value: float, claim: Claim) -> Direction:
    """Where `value` sits relative to one claim's optimal band.

    "below"/"above" describe the *location*, not the band: 10.39 °C is below
    ECOCROP's 18–27, 28.0 °C is above it. Band edges are inclusive, matching
    _assess_band's optimal test.
    """
    if value < claim.opt_min:
        return "below"
    if value > claim.opt_max:
        return "above"
    return "optimal"


def divergence(value: float, claims: Iterable[Claim]) -> frozenset[Direction]:
    """The distinct directions the claims put `value` in.

    Feeds conservative() directly. Sources disagree *in direction* at `value`
    exactly when this contains both "below" and "above" — one source says the
    site is too cold while another says it is too hot.
    """
    return frozenset(classify(value, c) for c in claims)


def conservative(verdicts: Iterable[Direction]) -> ConservativeRead:
    """The reading that never calls a site optimal unless every claim does.

    Total: every input has a defined result, including the empty one, which
    falls through to the disagreement string — a default that errs away from
    "optimal" rather than raising. The CLI cannot reach that case, because
    temperature_claims() always returns at least the ECOCROP claim.
    """
    s = set(verdicts)
    if s == {"optimal"}:
        return "optimal"
    non_opt = s - {"optimal"}
    if non_opt == {"below"}:
        return "below optimal"
    if non_opt == {"above"}:
        return "above optimal"
    return "not optimal (sources disagree in direction)"
