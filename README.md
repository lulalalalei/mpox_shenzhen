# Bayesian mpox importation model

This directory contains the reproducible code for the Shenzhen mpox importation analysis. It includes frozen public snapshots of WHO surveillance data and World Bank population data; passenger data, posterior files, and generated figures are not included.

## Files

- `run_all.py` runs preprocessing, country-specific Bayesian fitting, posterior propagation, diagnostics, calibration, and figure generation.
- `prepare_passenger_data.py` converts flight-level capacity and monthly passenger totals into weekly source-country passenger flows.
- `bayesian_importation_pipeline.py` fits the PyMC model and propagates complete posterior draws through the importation-risk calculation.
- `make_figures.py` generates temporal risk, source-country ranking, calibration, and optional world-map figures.
- `combine_figure.py` combines generated panel images into a submission figure.
- `requirements.txt` lists the Python dependencies.

## Input layout

The scripts read input files through command-line arguments. The tracked public snapshots under `data/who/` and `data/population/` are included for provenance; other input files should remain outside the public repository or under the ignored parts of `data/`.

The case directory contains one CSV per source country. Each file must contain `date,cases`; the aliases `dates,reported_cases` are also accepted.

The transport-capacity file must contain source country, flight date, and seats. The original column names `Dep DOT Country Name`, `Time series`, and `Seats (Total)` are accepted.

The monthly-passenger file must contain source country, month, and passenger count. The original column names `Dep Country Name`, `Time Series`, and `Total Est. Pax` are accepted.

The population file must contain country and population. The original columns `Country Name` and `2024` are accepted. The included snapshot is `data/population/World_Bank_population_2024.csv`.

The optional observed-case file must contain weekly or daily dates and imported-case counts. It is used only for the calibration panel.

## Run

```bash
python run_all.py --cases-dir data/cases --transport-capacity-file data/Transport_capacity.csv --monthly-passenger-file data/Passenger.csv --population-file data/population/World_Bank_population_2024.csv --observed-file data/observed_imported_cases.csv --world-shapefile data/world.shp --country-map-file data/country_iso3.csv --output-dir outputs
```

The default sampler uses four chains, 2,000 tuning iterations per chain, and 2,000 retained posterior draws per chain. These can be changed from the command line.

## Outputs

The model directory contains one NetCDF posterior per country, full MCMC diagnostics, trace and rank plots, weekly posterior summaries, country summaries, peak-week probabilities, and `posterior_draws.npz`.

`posterior_draws.npz` stores the complete country-level latent-incidence draws and the propagated draws for expected introductions, introduction probability, mature-period cumulative introductions, and country contribution fractions.

All final intervals are computed directly from these propagated posterior draws.
