# WHO mpox surveillance data

`WHO_global_mpox_cases_2025-12-31.csv` is the WHO global country-day surveillance snapshot used by the analysis.

Source: World Health Organization, *Global Mpox Trends* and the WHO mpox topic page:

- https://worldhealthorg.shinyapps.io/mpx_global/
- https://www.who.int/health-topics/mpox
- Public API: https://xmart-api-public.who.int/MPX/V_MPX_VALIDATED_DAILY

## Snapshot scope

This repository preserves the 2025 analysis snapshot; it is not a live mirror of the API. The dated records run from 2022-01-05 through 2025-12-14. The source export also contains one aggregate row with `DATE=NA`; it is retained unchanged for provenance and should be excluded when constructing a dated time series.

WHO may revise historical records. Re-downloading the API today may therefore produce different values from this frozen snapshot.

## Contents

Variables are country, date, ISO3 code, WHO region, confirmed and probable case counts, confirmed and probable death counts, and their new-count fields. The file contains no individual-level records or direct identifiers.

The file is included to support reproduction of the country-level case-series preparation.
