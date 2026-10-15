"""Record NASA POWER responses for replay via CROP_ADVISOR_CLIMATE_REPLAY.

Each response is trimmed to the two parameters `fetch_climate` reads; SOURCE.md records its provenance.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from crop_advisor import _http
from crop_advisor.climate import power_url, replay_path

SITES = {
    "london": (51.5, -0.13),
    "lagos": (6.52, 3.38),
    "bangkok": (13.75, 100.5),
    "djibouti": (11.59, 43.15),
}

_KEPT_PARAMETERS = ("T2M", "PRECTOTCORR")

_POWER_REFERENCE = (
    "The data was obtained from National Aeronautics and Space Administration (NASA) "
    "Langley Research Center's Prediction Of Worldwide Energy Resources (POWER) project "
    "funded through the NASA Earth Science Division."
)


def trim(response: dict) -> dict:
    param = response["properties"]["parameter"]
    return {"properties": {"parameter": {k: param[k] for k in _KEPT_PARAMETERS}}}


def _provenance(response: dict) -> dict:
    header = response.get("header", {})
    api = header.get("api", {})
    return {"service": api.get("name", "unknown service"),
            "version": api.get("version", "unknown version"),
            "period": header.get("range", "unknown period")}


def _data_reference(service: str, version: str, accessed: str) -> str:
    # The guide's template reads "the POWER Project's Hourly 2.x.x version on YYYY/MM/DD".
    product = service.replace("POWER ", "").replace(" API", "")
    return f"The data was obtained from the POWER Project's {product} {version} version on {accessed.replace('-', '/')}."


def source_md(rows: list, accessed: str) -> str:
    services = sorted({(r["service"], r["version"]) for r in rows})
    service = "; ".join(f"{name} {version}" for name, version in services)
    periods = "; ".join(sorted({r["period"] for r in rows}))
    data_refs = " ".join(_data_reference(name, version, accessed) for name, version in services)
    lines = [
        "# NASA POWER fixtures",
        "",
        "Trimmed NASA POWER climatology responses, read by `crop_advisor.climate.fetch_climate`",
        "when `CROP_ADVISOR_CLIMATE_REPLAY` names this directory. Each file keeps only",
        "`properties.parameter.T2M` and `properties.parameter.PRECTOTCORR`; everything else in the",
        "response was dropped. Written by `python -m evals.record_fixtures`, which rewrites this file.",
        "",
        f"- **Service:** {service}",
        f"- **Accessed:** {accessed} (UTC, recording machine's clock)",
        f"- **Climatology period:** {periods}",
        "- **Request:** `" + power_url("<lat>", "<lon>") + "`",
        "",
        "| site | latitude | longitude | file |",
        "|---|---|---|---|",
    ]
    lines += [f"| {r['site']} | {r['lat']} | {r['lon']} | `{r['file']}` |" for r in rows]
    lines += [
        "",
        "## References",
        "",
        "Per the [POWER Referencing Guide](https://power.larc.nasa.gov/docs/referencing/):",
        "",
        f"- **POWER's Reference:** {_POWER_REFERENCE}",
        f"- **POWER's Data Reference:** {data_refs}",
        "",
        "## Terms",
        "",
        "From the [AWS Open Data registry entry](https://registry.opendata.aws/nasa-power/), "
        "licence listed as [Creative Commons BY 4.0](https://creativecommons.org/licenses/by/4.0/):",
        "",
        "> There are no restrictions on the use, access, and/or download of data from the NASA POWER Project.",
        "> We request that you cite the NASA POWER Project when using the data provided from NASA POWER Project.",
        "",
        "The Referencing Guide also says \"the Project requests notification if POWER data is transmitted",
        "to other researchers\"; the contact address it gives, for publication notices, is",
        "larc-power-project@mail.nasa.gov. No notification has been sent from here.",
        "",
    ]
    return "\n".join(lines)


def record(sites, out_dir, accessed=None) -> list:
    out = Path(out_dir)
    keep = {replay_path(out, *SITES[site]).name for site in sites}
    # SOURCE.md is rewritten from this run alone, so a file left out would lose its provenance.
    stale = sorted(p.name for p in out.glob("*.json") if p.name not in keep)
    if stale:
        raise FileExistsError(f"{out} already holds fixtures not being re-recorded: {', '.join(stale)}. "
                              "Record every site again, or write to an empty directory.")
    out.mkdir(parents=True, exist_ok=True)
    accessed = accessed or datetime.now(timezone.utc).date().isoformat()
    # Fetch everything before writing anything, so a failed fetch cannot leave files out of step with SOURCE.md.
    responses = {site: _http.get_json(power_url(*SITES[site])) for site in sites}
    payloads = {site: json.dumps(trim(r), indent=2, sort_keys=True) + "\n" for site, r in responses.items()}
    rows, written = [], []
    for site, response in responses.items():
        lat, lon = SITES[site]
        path = replay_path(out, lat, lon)
        path.write_text(payloads[site], encoding="utf-8")
        rows.append({"site": site, "lat": lat, "lon": lon, "file": path.name, **_provenance(response)})
        written.append(path)
    source = out / "SOURCE.md"
    source.write_text(source_md(rows, accessed), encoding="utf-8")
    return written + [source]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Record NASA POWER responses for replay.")
    ap.add_argument("--sites", nargs="+", required=True,
                    help=f"site ids, comma- or space-separated: {', '.join(sorted(SITES))}")
    ap.add_argument("--out", required=True, help="directory to write the fixtures and SOURCE.md")
    args = ap.parse_args(argv)
    sites = [s for arg in args.sites for s in arg.split(",") if s]
    unknown = sorted(set(sites) - set(SITES))
    if unknown or not sites:
        ap.error(f"unknown site id(s): {', '.join(unknown) or '(none given)'}; choose from {', '.join(sorted(SITES))}")
    try:
        written = record(list(dict.fromkeys(sites)), args.out)
    except FileExistsError as e:
        ap.error(str(e))
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
