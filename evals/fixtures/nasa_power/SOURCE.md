# NASA POWER fixtures

Trimmed NASA POWER climatology responses, read by `crop_advisor.climate.fetch_climate`
when `CROP_ADVISOR_CLIMATE_REPLAY` names this directory. Each file keeps only
`properties.parameter.T2M` and `properties.parameter.PRECTOTCORR`; everything else in the
response was dropped. Written by `python -m evals.record_fixtures`, which rewrites this file.

- **Service:** POWER Climatology API v2.10.1
- **Accessed:** 2026-10-15 (UTC, recording machine's clock)
- **Climatology period:** 25-year Meteorological and Solar Monthly & Annual Climatologies (January 2001 - December 2025)
- **Request:** `https://power.larc.nasa.gov/api/temporal/climatology/point?parameters=T2M,PRECTOTCORR&community=AG&longitude=<lon>&latitude=<lat>&format=JSON`

| site | latitude | longitude | file |
|---|---|---|---|
| london | 51.5 | -0.13 | `51.5_-0.13.json` |
| lagos | 6.52 | 3.38 | `6.52_3.38.json` |
| bangkok | 13.75 | 100.5 | `13.75_100.5.json` |
| djibouti | 11.59 | 43.15 | `11.59_43.15.json` |

## References

Per the [POWER Referencing Guide](https://power.larc.nasa.gov/docs/referencing/):

- **POWER's Reference:** The data was obtained from National Aeronautics and Space Administration (NASA) Langley Research Center's Prediction Of Worldwide Energy Resources (POWER) project funded through the NASA Earth Science Division.
- **POWER's Data Reference:** The data was obtained from the POWER Project's Climatology v2.10.1 version on 2026/10/15.

## Terms

From the [AWS Open Data registry entry](https://registry.opendata.aws/nasa-power/), licence listed as [Creative Commons BY 4.0](https://creativecommons.org/licenses/by/4.0/):

> There are no restrictions on the use, access, and/or download of data from the NASA POWER Project.
> We request that you cite the NASA POWER Project when using the data provided from NASA POWER Project.

The Referencing Guide also says "the Project requests notification if POWER data is transmitted
to other researchers"; the contact address it gives, for publication notices, is
larc-power-project@mail.nasa.gov. No notification has been sent from here.
