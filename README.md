# Bayesian reconstruction of mpox introduction pressure into Shenzhen

This repository provides the model, preprocessing, posterior propagation, country screening and plotting code for the Shenzhen 2023–2025 analysis.

## Installation

Use Python 3.12 in a new environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

The complete dependency set is pinned in `requirements-lock.txt`. The shorter `requirements.txt` records the direct dependencies. NumPy, Numba and llvmlite are pinned together for binary compatibility.

The sampler is nutpie 0.16.11, called through PyMC 5.27.1. It uses NUTS with low-rank adaptation, four chains, 2,000 tuning iterations and 2,000 retained draws per chain. The target acceptance probability is 0.95 and the maximum tree depth is 10. A country's seed is 2026 plus the sum of the Unicode code points in its harmonised name.

## Included public data

- `data/who/WHO_global_mpox_cases_2025-12-31.csv`: the archived country-level WHO case snapshot.
- `data/population/World_Bank_population_2024.csv`: the archived World Bank population data. The analysis uses the `2024` column throughout the study period.

The OAG passenger and capacity data require a commercial licence. Shenzhen imported-case surveillance is institutionally restricted. These files and individual-level records are not distributed here. Access to the original inputs is required for a complete replication.

## Inputs

Place restricted inputs outside the repository or in `private_data/`.

The transport-capacity CSV must contain source country, flight date and seat count. Accepted OAG field names include `Dep DOT Country Name`, `Time series` and `Seats (Total)`. The passenger CSV must contain source country, month and passenger count. Accepted OAG names include `Dep Country Name`, `Time Series` and `Total Est. Pax`. Inputs must already be restricted to Shenzhen-bound direct international routes.

Monthly passengers are allocated to daily capacity records in proportion to monthly seats, then summed into Monday–Sunday weeks. Each complete country-month's passenger total is preserved before the study-period cut-off is applied. Only country-months with positive direct seat capacity enter the allocation. Unmatched passenger country-months are exported to an audit CSV.

Case files are generated from the included WHO snapshot by default. Daily new confirmed counts are grouped into Monday–Sunday weeks. Negative surveillance corrections are excluded, as in the analysis script, and recorded in a separate audit file. A custom case directory may be supplied with `--cases-dir`; each CSV filename is the country name, and each file must contain `date,cases` or `dates,reported_cases`.

## Primary and sensitivity analyses

```bash
python run_all.py --transport-capacity-file private_data/Transport_capacity.csv --monthly-passenger-file private_data/Passenger.csv --population-file data/population/World_Bank_population_2024.csv --include-sensitivity --output-dir outputs
```

The primary analysis uses maximum R-hat strictly below 1.01, minimum bulk ESS 400, minimum tail ESS 400, zero divergences, no samples reaching the maximum tree depth and minimum BFMI 0.30. Missing diagnostic quantities are not treated as passing. Only countries with positive direct passenger flow, reported cases and passing diagnostics are propagated.

The sensitivity analysis changes only minimum tail ESS to 100. It reuses the same saved country posterior draws. Bulk ESS and all other criteria remain unchanged. Its outputs are written to `outputs/sensitivity/`. The primary outputs are in `outputs/model/`.

To reuse the original analysis posteriors, supply `--posterior-input-dir /path/to/posterior_2023`. Saved files may use either `Country_posterior.nc` or the safe country filename produced by this repository. Reuse requires the original delay parameters. The program verifies the nutpie origin, chain/draw counts, observed case series, target acceptance probability and requested tree depth. No country is silently substituted if a posterior is missing.

## Separate stages

```bash
python prepare_passenger_data.py --transport-capacity-file private_data/Transport_capacity.csv --monthly-passenger-file private_data/Passenger.csv --output-file outputs/weekly_passengers.csv
python prepare_cases.py --passenger-file outputs/weekly_passengers.csv --output-dir outputs/cases
python bayesian_importation_pipeline.py --cases-dir outputs/cases --passenger-file outputs/weekly_passengers.csv --population-file data/population/World_Bank_population_2024.csv --output-dir outputs/model
python bayesian_importation_pipeline.py --cases-dir outputs/cases --passenger-file outputs/weekly_passengers.csv --population-file data/population/World_Bank_population_2024.csv --posterior-input-dir outputs/model/posteriors --tail-ess-threshold 100 --output-dir outputs/sensitivity
python make_figures.py --model-dir outputs/model --figure-dir outputs/figures
```

## Model and outputs

The incubation and reporting delays use log-normal distributions. Their respective `(meanlog, sdlog)` pairs are `(2.001, 0.234)` and `(1.6057319, 0.5964992)`. The code retains the analysis's daily CDF discretisation, FFT convolution and consecutive seven-day aggregation. It does not introduce a different weekly-delay kernel or a new observation model.

The observation model is Poisson. Latent log incidence follows a non-centred Gaussian random walk with an Exponential(1) innovation scale. The initial-state centre depends on the first four observed weeks, using the same rule as the original fitting script. Delay distributions, passenger data and population are fixed inputs.

All retained posterior draws are propagated. A country's contribution is calculated separately in each draw using the corresponding total across the retained countries. The reported mean share is the mean of these draw-wise ratios. Credible intervals use the empirical 2.5th and 97.5th percentiles. They do not include fixed-input or country-selection uncertainty. Results under tail ESS 100 require cautious interpretation of tail quantiles.

`screening_audit.csv` records inclusion and failed diagnostic criteria. `weekly_summary.csv`, `country_summary.csv`, `key_results.csv` and `peak_week_distribution.csv` report the propagated results. `posterior_draws.npz` retains the underlying incidence and introduction draws. `manifest.json` records versions, thresholds and sampler settings. Retained-country posterior predictive summaries are exported to `diagnostics/*_ppc.csv`; the plotting stage also generates their predictive-check figures. NetCDF files retain the original nutpie inference metadata.

## Exploratory scaling and plots

An optional `--observed-file` must contain dates and counts for the non-Hong Kong imported cases represented in the manuscript comparison. Filter the observed data to this scope before running. The scaling factor is the observed mature-period count divided by the posterior mean mature-period cumulative expected introductions. It matches the cumulative endpoint by construction and is not independent validation.

Optional world maps require `--world-shapefile` with an ISO3 field. The included WHO country codes provide default name-to-ISO3 mappings. A two-column custom mapping may instead be supplied with `--country-map-file`. Plots are exported in PNG, PDF and SVG formats. Manuscript typography and multi-panel page layout can be applied separately to these generated plots.

Exact floating-point results and MCMC diagnostics can vary across environments. Reusing the archived posterior files preserves the study's inference draws. Fresh fitting uses the stated model and settings, then applies the diagnostic screen to the newly obtained samples.
