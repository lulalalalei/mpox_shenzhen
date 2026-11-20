from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bayesian_importation_pipeline import run_pipeline
from combine_figure import combine_panels
from make_figures import generate_figures
from prepare_passenger_data import build_weekly_passengers
from prepare_cases import export_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-dir", type=Path, default=None)
    parser.add_argument("--who-file", type=Path, default=Path(__file__).resolve().parent / "data/who/WHO_global_mpox_cases_2025-12-31.csv")
    parser.add_argument("--include-sensitivity", action="store_true")
    parser.add_argument("--transport-capacity-file", type=Path, required=True)
    parser.add_argument("--monthly-passenger-file", type=Path, required=True)
    parser.add_argument("--population-file", type=Path, required=True)
    parser.add_argument("--population-column", default=None)
    parser.add_argument("--observed-file", type=Path, default=None)
    parser.add_argument("--world-shapefile", type=Path, default=None)
    parser.add_argument("--country-map-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--start", default="2023-01-09")
    parser.add_argument("--end", default="2025-10-26")
    parser.add_argument("--incubation-meanlog", type=float, default=2.001)
    parser.add_argument("--incubation-sdlog", type=float, default=0.234)
    parser.add_argument("--reporting-meanlog", type=float, default=1.6057319)
    parser.add_argument("--reporting-sdlog", type=float, default=0.5964992)
    parser.add_argument("--reporting-threshold", type=float, default=0.95)
    parser.add_argument("--incubation-lag-weeks", type=int, default=6)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--max-treedepth", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tail-ess-threshold", type=float, default=400.0)
    parser.add_argument("--posterior-input-dir", type=Path, default=None)
    parser.add_argument("--skip-diagnostic-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.include_sensitivity and args.tail_ess_threshold != 400:
        raise ValueError("Use tail ESS 400 for the primary run when requesting sensitivity analysis.")
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    passenger_file = args.output_dir / "weekly_passengers.csv"
    model_dir = args.output_dir / "model"
    figure_dir = args.output_dir / "figures"
    build_weekly_passengers(
        transport_capacity_file=args.transport_capacity_file,
        monthly_passenger_file=args.monthly_passenger_file,
        output_file=passenger_file,
        start=start,
        end=end,
    )
    cases_dir = args.cases_dir or args.output_dir / "cases"
    if args.cases_dir is None:
        export_cases(args.who_file, cases_dir, start, end, passenger_file)
    run_pipeline(
        cases_dir=cases_dir,
        passenger_file=passenger_file,
        population_file=args.population_file,
        output_dir=model_dir,
        start=start,
        end=end,
        population_column=args.population_column,
        incubation_meanlog=args.incubation_meanlog,
        incubation_sdlog=args.incubation_sdlog,
        reporting_meanlog=args.reporting_meanlog,
        reporting_sdlog=args.reporting_sdlog,
        reporting_threshold=args.reporting_threshold,
        incubation_lag_weeks=args.incubation_lag_weeks,
        chains=args.chains,
        tune=args.tune,
        draws=args.draws,
        target_accept=args.target_accept,
        max_treedepth=args.max_treedepth,
        seed=args.seed,
        tail_ess_threshold=args.tail_ess_threshold,
        posterior_input_dir=args.posterior_input_dir,
        diagnostic_plots=not args.skip_diagnostic_plots,
    )
    if args.include_sensitivity:
        run_pipeline(
            cases_dir=cases_dir, passenger_file=passenger_file, population_file=args.population_file,
            output_dir=args.output_dir / "sensitivity", start=start, end=end,
            population_column=args.population_column, incubation_meanlog=args.incubation_meanlog,
            incubation_sdlog=args.incubation_sdlog, reporting_meanlog=args.reporting_meanlog,
            reporting_sdlog=args.reporting_sdlog, reporting_threshold=args.reporting_threshold,
            incubation_lag_weeks=args.incubation_lag_weeks, chains=args.chains, tune=args.tune,
            draws=args.draws, target_accept=args.target_accept, max_treedepth=args.max_treedepth,
            seed=args.seed, tail_ess_threshold=100,
            posterior_input_dir=args.posterior_input_dir or model_dir / "posteriors",
            diagnostic_plots=not args.skip_diagnostic_plots,
        )
    generate_figures(
        model_dir=model_dir,
        figure_dir=figure_dir,
        observed_file=args.observed_file,
        shapefile=args.world_shapefile,
        country_map_file=args.country_map_file,
        seed=args.seed,
    )
    panels = [
        figure_dir / "temporal_importation_risk.png",
        figure_dir / "world_map_cumulative_importation.png",
        figure_dir / "calibrated_importation_model.png",
    ]
    panels = [path for path in panels if path.exists()]
    if len(panels) >= 2:
        combine_panels(panels, figure_dir, "figure_1_combined")


if __name__ == "__main__":
    main()
