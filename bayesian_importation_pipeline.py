from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.signal import fftconvolve
from scipy.stats import lognorm


def discretize_lognormal_pmf(meanlog: float, sdlog: float, max_days: int) -> np.ndarray:
    edges = np.arange(max_days + 2, dtype=float)
    pmf = np.diff(lognorm.cdf(edges, s=sdlog, scale=np.exp(meanlog)))
    pmf = np.maximum(pmf, 0.0)
    total = pmf.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("The delay distribution has no probability mass.")
    return pmf / total


def build_weekly_delay_pmf(
    incubation_meanlog: float,
    incubation_sdlog: float,
    reporting_meanlog: float,
    reporting_sdlog: float,
    max_incubation_days: int = 40,
    max_reporting_days: int = 40,
) -> np.ndarray:
    incubation = discretize_lognormal_pmf(
        incubation_meanlog,
        incubation_sdlog,
        max_incubation_days,
    )
    reporting = discretize_lognormal_pmf(
        reporting_meanlog,
        reporting_sdlog,
        max_reporting_days,
    )
    daily = fftconvolve(incubation, reporting)
    daily = np.maximum(daily, 0.0)
    daily /= daily.sum()
    n_weeks = int(np.ceil(len(daily) / 7))
    weekly = np.array(
        [daily[7 * i : min(7 * (i + 1), len(daily))].sum() for i in range(n_weeks)],
        dtype=float,
    )
    return weekly / weekly.sum()


def build_convolution_matrix(delay_pmf: np.ndarray, n_weeks: int) -> np.ndarray:
    matrix = np.zeros((n_weeks, n_weeks), dtype=float)
    for report_week in range(n_weeks):
        for lag in range(min(report_week + 1, len(delay_pmf))):
            matrix[report_week, report_week - lag] = delay_pmf[lag]
    return matrix


def reporting_completeness(delay_pmf: np.ndarray, n_weeks: int) -> np.ndarray:
    cumulative = np.cumsum(delay_pmf)
    output = np.zeros(n_weeks, dtype=float)
    for week in range(n_weeks):
        max_lag = n_weeks - 1 - week
        if max_lag >= len(delay_pmf) - 1:
            output[week] = 1.0
        elif max_lag >= 0:
            output[week] = cumulative[max_lag]
    return output


def build_weekly_incubation_weights(
    meanlog: float,
    sdlog: float,
    max_lag_weeks: int = 6,
) -> np.ndarray:
    weights = []
    for lag_week in range(max_lag_weeks + 1):
        total = 0.0
        for infection_day in range(7):
            for target_day in range(7):
                age = 7 * lag_week + target_day - infection_day
                if age >= 0:
                    total += lognorm.sf(age, s=sdlog, scale=np.exp(meanlog))
        weights.append(total / 49.0)
    return np.asarray(weights, dtype=float)


def build_lag_matrix(weights: np.ndarray, n_weeks: int) -> np.ndarray:
    matrix = np.zeros((n_weeks, n_weeks), dtype=float)
    for week in range(n_weeks):
        for lag, weight in enumerate(weights):
            if week >= lag:
                matrix[week, week - lag] = weight
    return matrix


def load_weekly_cases(
    path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    frame = pd.read_csv(path)
    frame.columns = frame.columns.astype(str).str.strip()
    date_column = next(
        (name for name in ("date", "dates", "Date", "Dates") if name in frame.columns),
        None,
    )
    case_column = next(
        (
            name
            for name in ("cases", "reported_cases", "Cases", "reported cases")
            if name in frame.columns
        ),
        None,
    )
    if date_column is None or case_column is None:
        raise ValueError(f"{path} must contain a date column and a case-count column.")
    frame = frame[[date_column, case_column]].rename(
        columns={date_column: "date", case_column: "cases"}
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["cases"] = pd.to_numeric(frame["cases"], errors="coerce")
    frame = frame.dropna().loc[lambda x: x["cases"] >= 0]
    frame["cases"] = frame["cases"].round().astype(int)
    daily = frame.groupby("date")["cases"].sum()
    calendar = pd.date_range(start, end, freq="D")
    daily = daily.reindex(calendar, fill_value=0)
    weekly = daily.resample("W-MON", label="left", closed="left").sum()
    weekly.index.name = "date"
    return weekly.astype(int)


def load_weekly_passengers(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = frame.columns.astype(str).str.strip()
    country_column = next(
        (
            name
            for name in ("country", "Country", "source_country", "Dep Country Name")
            if name in frame.columns
        ),
        None,
    )
    date_column = next(
        (name for name in ("date", "Date", "week", "Time Series") if name in frame.columns),
        None,
    )
    passenger_column = next(
        (
            name
            for name in (
                "passengers",
                "Estimated_Weekly_Pax",
                "Estimated Pax",
                "Total Est. Pax",
            )
            if name in frame.columns
        ),
        None,
    )
    if country_column is None or date_column is None or passenger_column is None:
        raise ValueError(f"{path} must contain country, week, and passenger columns.")
    frame = frame[[country_column, date_column, passenger_column]].rename(
        columns={
            country_column: "country",
            date_column: "date",
            passenger_column: "passengers",
        }
    )
    frame["country"] = frame["country"].astype(str).str.strip()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["passengers"] = pd.to_numeric(frame["passengers"], errors="coerce")
    frame = frame.dropna()
    frame = frame.loc[frame["passengers"] >= 0].copy()
    frame["date"] = frame["date"] - pd.to_timedelta(frame["date"].dt.weekday, unit="D")
    return (
        frame.groupby(["country", "date"], as_index=False)["passengers"]
        .sum()
        .sort_values(["country", "date"])
    )


def load_population(path: Path, population_column: str | None = None) -> dict[str, float]:
    frame = pd.read_csv(path)
    frame.columns = frame.columns.astype(str).str.strip()
    country_column = next(
        (name for name in ("country", "Country", "Country Name") if name in frame.columns),
        None,
    )
    if country_column is None:
        raise ValueError(f"{path} must contain a country column.")
    if population_column is None:
        population_column = next(
            (
                name
                for name in ("population", "Population", "2024", "2023", "2022")
                if name in frame.columns
            ),
            None,
        )
    if population_column is None or population_column not in frame.columns:
        raise ValueError(f"{path} must contain a population column.")
    frame[population_column] = pd.to_numeric(frame[population_column], errors="coerce")
    frame = frame.dropna(subset=[country_column, population_column])
    frame = frame.loc[frame[population_column] > 0]
    return dict(zip(frame[country_column].astype(str).str.strip(), frame[population_column].astype(float)))


def adaptive_initial_state(observed_cases: np.ndarray) -> tuple[float, float]:
    early_mean = float(np.mean(observed_cases[: min(4, len(observed_cases))] + 1.0))
    return np.log(max(early_mean, 1.0)), 2.0 if len(observed_cases) < 40 else 1.0


def fit_country(
    observed_cases: np.ndarray,
    delay_pmf: np.ndarray,
    chains: int = 4,
    tune: int = 2000,
    draws: int = 2000,
    target_accept: float = 0.95,
    max_treedepth: int = 10,
    seed: int = 2026,
) -> az.InferenceData:
    observed_cases = np.asarray(observed_cases, dtype=np.int64)
    n_weeks = len(observed_cases)
    if n_weeks < 10 or observed_cases.sum() <= 0:
        raise ValueError("The weekly case series must contain at least 10 weeks and one case.")
    convolution = pt.as_tensor_variable(build_convolution_matrix(delay_pmf, n_weeks))
    init_mu, init_sigma = adaptive_initial_state(observed_cases)
    with pm.Model() as model:
        sigma_rw = pm.Exponential("sigma_rw", 1.0)
        log_i0 = pm.Normal("log_I0", mu=init_mu, sigma=init_sigma)
        innovations = pm.Normal("eps", mu=0.0, sigma=1.0, shape=n_weeks - 1)
        log_i = pt.concatenate([log_i0[None], log_i0 + pt.cumsum(sigma_rw * innovations)])
        incidence = pm.Deterministic("I", pt.exp(log_i))
        reporting_rate = pm.Deterministic(
            "lambda_report",
            pt.clip(pt.dot(convolution, incidence), 1e-8, np.inf),
        )
        pm.Poisson("reported_cases", mu=reporting_rate, observed=observed_cases)
        step = pm.NUTS(target_accept=target_accept, max_treedepth=max_treedepth)
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=min(chains, os.cpu_count() or 1),
            step=step,
            random_seed=seed,
            progressbar=True,
            compute_convergence_checks=True,
            return_inferencedata=True,
        )
    idata.posterior.attrs.update(
        {
            "n_chains": int(idata.posterior.sizes["chain"]),
            "draws_per_chain": int(idata.posterior.sizes["draw"]),
            "tune_requested": int(tune),
            "draws_requested_per_chain": int(draws),
            "target_accept": float(target_accept),
            "max_treedepth": int(max_treedepth),
            "complete_posterior_draws": 1,
        }
    )
    return idata


def save_diagnostics(
    idata: az.InferenceData,
    output_dir: Path,
    country: str,
    ess_threshold: float = 400.0,
    rhat_threshold: float = 1.01,
    bfmi_threshold: float = 0.30,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = az.summary(
        idata,
        var_names=["sigma_rw", "log_I0", "I", "lambda_report"],
        kind="diagnostics",
    )
    summary.to_csv(output_dir / f"{safe_name(country)}_diagnostics.csv")
    max_rhat = float(np.nanmax(summary["r_hat"].to_numpy()))
    min_bulk_ess = float(np.nanmin(summary["ess_bulk"].to_numpy()))
    min_tail_ess = float(np.nanmin(summary["ess_tail"].to_numpy()))
    divergences = int(idata.sample_stats["diverging"].sum().values)
    if "tree_depth" in idata.sample_stats:
        tree_depth = idata.sample_stats["tree_depth"].to_numpy()
        max_tree_depth = int(np.nanmax(tree_depth))
        max_tree_depth_hits = int(np.sum(tree_depth >= int(idata.posterior.attrs["max_treedepth"])))
    else:
        max_tree_depth = np.nan
        max_tree_depth_hits = np.nan
    try:
        min_bfmi = float(np.nanmin(np.asarray(az.bfmi(idata))))
    except Exception:
        min_bfmi = np.nan
    diagnostic_pass = bool(
        max_rhat <= rhat_threshold
        and min_bulk_ess >= ess_threshold
        and min_tail_ess >= ess_threshold
        and divergences == 0
        and (np.isnan(min_bfmi) or min_bfmi >= bfmi_threshold)
        and (np.isnan(max_tree_depth_hits) or max_tree_depth_hits == 0)
    )
    selected_coords = None
    time_dims = [dim for dim in idata.posterior["I"].dims if dim not in {"chain", "draw"}]
    if len(time_dims) == 1:
        time_dim = time_dims[0]
        n_weeks = idata.posterior["I"].sizes[time_dim]
        selected_coords = {time_dim: sorted({0, n_weeks // 2, n_weeks - 1})}
    import matplotlib.pyplot as plt

    try:
        az.plot_trace(
            idata,
            var_names=["sigma_rw", "log_I0", "I"],
            coords=selected_coords,
            compact=True,
            combined=False,
        )
        plt.gcf().savefig(output_dir / f"{safe_name(country)}_trace.png", dpi=180, bbox_inches="tight")
        plt.close("all")
    except Exception:
        plt.close("all")
    try:
        az.plot_rank(
            idata,
            var_names=["sigma_rw", "log_I0", "I"],
            coords=selected_coords,
            kind="vlines",
        )
        plt.gcf().savefig(output_dir / f"{safe_name(country)}_rank.png", dpi=180, bbox_inches="tight")
        plt.close("all")
    except Exception:
        plt.close("all")
    result = {
        "country": country,
        "n_chains": int(idata.posterior.sizes["chain"]),
        "draws_per_chain": int(idata.posterior.sizes["draw"]),
        "total_posterior_draws": int(idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]),
        "max_rhat": max_rhat,
        "min_bulk_ess": min_bulk_ess,
        "min_tail_ess": min_tail_ess,
        "divergences": divergences,
        "max_tree_depth": max_tree_depth,
        "max_tree_depth_hits": max_tree_depth_hits,
        "min_bfmi": min_bfmi,
        "diagnostic_pass": diagnostic_pass,
    }
    return result


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value).strip("_") or "country"


def posterior_incidence_draws(idata: az.InferenceData) -> np.ndarray:
    variable = idata.posterior["I"]
    time_dims = [dim for dim in variable.dims if dim not in {"chain", "draw"}]
    if len(time_dims) != 1:
        raise ValueError("Posterior I must have exactly one time dimension.")
    time_dim = time_dims[0]
    return (
        variable.stack(sample=("chain", "draw"))
        .transpose("sample", time_dim)
        .values.astype(float)
    )


def propagate_country_draws(
    incidence_draws: np.ndarray,
    dates: pd.DatetimeIndex,
    passenger_panel: pd.DataFrame,
    population: float,
    incubation_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isfinite(population) or population <= 0:
        raise ValueError("Population must be positive and finite.")
    passenger = (
        passenger_panel.set_index("date")["passengers"]
        .reindex(dates)
        .fillna(0.0)
        .to_numpy(float)
    )
    lag_matrix = build_lag_matrix(incubation_weights, len(dates))
    incubation_pool = incidence_draws @ lag_matrix.T
    prevalence = incubation_pool / float(population)
    expected_imports = np.clip(prevalence * passenger[None, :], 0.0, np.inf)
    probability = -np.expm1(-expected_imports)
    return incubation_pool, expected_imports, probability


def quantiles(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        draws.mean(axis=0),
        np.quantile(draws, 0.025, axis=0),
        np.quantile(draws, 0.5, axis=0),
        np.quantile(draws, 0.975, axis=0),
    )


def run_pipeline(
    cases_dir: Path,
    passenger_file: Path,
    population_file: Path,
    output_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    population_column: str | None,
    incubation_meanlog: float,
    incubation_sdlog: float,
    reporting_meanlog: float,
    reporting_sdlog: float,
    reporting_threshold: float,
    incubation_lag_weeks: int,
    chains: int,
    tune: int,
    draws: int,
    target_accept: float,
    max_treedepth: int,
    seed: int,
) -> None:
    files = sorted(cases_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No case CSV files found in {cases_dir}.")
    passengers = load_weekly_passengers(passenger_file)
    populations = load_population(population_file, population_column)
    delay_pmf = build_weekly_delay_pmf(
        incubation_meanlog,
        incubation_sdlog,
        reporting_meanlog,
        reporting_sdlog,
    )
    incubation_weights = build_weekly_incubation_weights(
        incubation_meanlog,
        incubation_sdlog,
        incubation_lag_weeks,
    )
    posterior_dir = output_dir / "posteriors"
    diagnostics_dir = output_dir / "diagnostics"
    posterior_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    country_names = []
    country_dates = None
    country_incidence = []
    country_expected = []
    country_probability = []
    diagnostics = []
    for index, case_file in enumerate(files):
        country = case_file.stem
        weekly_cases = load_weekly_cases(case_file, start, end)
        if country not in populations:
            raise ValueError(f"Population data are missing for {country}.")
        idata = fit_country(
            weekly_cases.to_numpy(),
            delay_pmf,
            chains=chains,
            tune=tune,
            draws=draws,
            target_accept=target_accept,
            max_treedepth=max_treedepth,
            seed=seed + index,
        )
        posterior_file = posterior_dir / f"{safe_name(country)}.nc"
        idata.to_netcdf(posterior_file)
        diagnostics.append(
            save_diagnostics(idata, diagnostics_dir, country)
        )
        incidence_draws = posterior_incidence_draws(idata)
        panel = passengers.loc[passengers["country"].eq(country), ["date", "passengers"]].copy()
        _, expected_imports, probability = propagate_country_draws(
            incidence_draws,
            weekly_cases.index,
            panel,
            populations[country],
            incubation_weights,
        )
        if country_dates is None:
            country_dates = weekly_cases.index
        elif not country_dates.equals(weekly_cases.index):
            raise ValueError("Country case series do not share the same weekly calendar.")
        country_names.append(country)
        country_incidence.append(incidence_draws)
        country_expected.append(expected_imports)
        country_probability.append(probability)
    draw_counts = {array.shape[0] for array in country_expected}
    if len(draw_counts) != 1:
        raise ValueError("All countries must have the same number of posterior draws.")
    expected_array = np.stack(country_expected, axis=0)
    probability_array = np.stack(country_probability, axis=0)
    incidence_array = np.stack(country_incidence, axis=0)
    dates = pd.DatetimeIndex(country_dates)
    completeness = reporting_completeness(delay_pmf, len(dates))
    mature_mask = completeness >= reporting_threshold
    right_truncated = ~mature_mask
    total_expected = expected_array.sum(axis=0)
    total_probability = -np.expm1(-total_expected)
    country_mature_expected = expected_array[:, :, mature_mask].sum(axis=2)
    total_mature_expected = country_mature_expected.sum(axis=0)
    country_contribution = np.divide(
        country_mature_expected,
        total_mature_expected[None, :],
        out=np.zeros_like(country_mature_expected),
        where=total_mature_expected[None, :] > 0,
    )
    weekly_expected_mean, weekly_expected_q025, weekly_expected_median, weekly_expected_q975 = quantiles(total_expected)
    weekly_probability_mean, weekly_probability_q025, weekly_probability_median, weekly_probability_q975 = quantiles(total_probability)
    weekly_summary = pd.DataFrame(
        {
            "date": dates,
            "expected_imports_mean": weekly_expected_mean,
            "expected_imports_q025": weekly_expected_q025,
            "expected_imports_median": weekly_expected_median,
            "expected_imports_q975": weekly_expected_q975,
            "probability_mean": weekly_probability_mean,
            "probability_q025": weekly_probability_q025,
            "probability_median": weekly_probability_median,
            "probability_q975": weekly_probability_q975,
            "right_truncated": right_truncated,
        }
    )
    country_rows = []
    for index, country in enumerate(country_names):
        expected = country_mature_expected[index]
        probability = -np.expm1(-expected)
        contribution = country_contribution[index]
        country_rows.append(
            {
                "country": country,
                "cumulative_expected_imports_mean": float(expected.mean()),
                "cumulative_expected_imports_q025": float(np.quantile(expected, 0.025)),
                "cumulative_expected_imports_median": float(np.quantile(expected, 0.5)),
                "cumulative_expected_imports_q975": float(np.quantile(expected, 0.975)),
                "cumulative_probability_mean": float(probability.mean()),
                "cumulative_probability_q025": float(np.quantile(probability, 0.025)),
                "cumulative_probability_median": float(np.quantile(probability, 0.5)),
                "cumulative_probability_q975": float(np.quantile(probability, 0.975)),
                "contribution_fraction_mean": float(contribution.mean()),
                "contribution_fraction_q025": float(np.quantile(contribution, 0.025)),
                "contribution_fraction_median": float(np.quantile(contribution, 0.5)),
                "contribution_fraction_q975": float(np.quantile(contribution, 0.975)),
            }
        )
    country_summary = pd.DataFrame(country_rows).sort_values(
        "cumulative_expected_imports_mean",
        ascending=False,
    )
    mature_expected = total_expected[:, mature_mask]
    mature_probability = total_probability[:, mature_mask]
    peak_indices = mature_probability.argmax(axis=1)
    peak_probability = mature_probability[np.arange(len(mature_probability)), peak_indices]
    peak_expected = mature_expected[np.arange(len(mature_expected)), peak_indices]
    peak_dates = dates[mature_mask].to_numpy()[peak_indices]
    peak_distribution = (
        pd.Series(pd.to_datetime(peak_dates))
        .value_counts()
        .rename_axis("peak_week")
        .reset_index(name="draw_count")
    )
    peak_distribution["posterior_probability"] = peak_distribution["draw_count"] / len(peak_dates)
    peak_distribution = peak_distribution.sort_values("posterior_probability", ascending=False)
    key_results = pd.DataFrame(
        [
            {
                "metric": "peak_weekly_probability",
                "mean": float(peak_probability.mean()),
                "q025": float(np.quantile(peak_probability, 0.025)),
                "median": float(np.quantile(peak_probability, 0.5)),
                "q975": float(np.quantile(peak_probability, 0.975)),
                "mean_curve_peak_week": str(dates[mature_mask][mature_probability.mean(axis=0).argmax()].date()),
            },
            {
                "metric": "peak_weekly_expected_imports",
                "mean": float(peak_expected.mean()),
                "q025": float(np.quantile(peak_expected, 0.025)),
                "median": float(np.quantile(peak_expected, 0.5)),
                "q975": float(np.quantile(peak_expected, 0.975)),
                "mean_curve_peak_week": str(dates[mature_mask][mature_probability.mean(axis=0).argmax()].date()),
            },
            {
                "metric": "cumulative_expected_imports_mature",
                "mean": float(total_mature_expected.mean()),
                "q025": float(np.quantile(total_mature_expected, 0.025)),
                "median": float(np.quantile(total_mature_expected, 0.5)),
                "q975": float(np.quantile(total_mature_expected, 0.975)),
                "mean_curve_peak_week": "",
            },
            {
                "metric": "cumulative_probability_mature",
                "mean": float((-np.expm1(-total_mature_expected)).mean()),
                "q025": float(np.quantile(-np.expm1(-total_mature_expected), 0.025)),
                "median": float(np.quantile(-np.expm1(-total_mature_expected), 0.5)),
                "q975": float(np.quantile(-np.expm1(-total_mature_expected), 0.975)),
                "mean_curve_peak_week": "",
            },
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    weekly_summary.to_csv(output_dir / "weekly_summary.csv", index=False)
    country_summary.to_csv(output_dir / "country_summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output_dir / "diagnostics_summary.csv", index=False)
    peak_distribution.to_csv(output_dir / "peak_week_distribution.csv", index=False)
    key_results.to_csv(output_dir / "key_results.csv", index=False)
    np.savez_compressed(
        output_dir / "posterior_draws.npz",
        dates=dates.to_numpy(dtype="datetime64[D]"),
        countries=np.asarray(country_names),
        n_chains=np.int64(chains),
        draws_per_chain=np.int64(draws),
        country_latent_incidence=incidence_array,
        country_expected_imports=expected_array,
        country_probability=probability_array,
        country_cumulative_expected_imports_mature=country_mature_expected,
        country_contribution_fraction_mature=country_contribution,
        overall_expected_imports=total_expected,
        overall_probability=total_probability,
        overall_cumulative_expected_imports_mature=total_mature_expected,
        overall_cumulative_probability_mature=-np.expm1(-total_mature_expected),
        reporting_completeness=completeness,
        right_truncated=right_truncated,
    )
    manifest = {
        "n_countries": len(country_names),
        "n_weeks": len(dates),
        "n_chains": chains,
        "tune_per_chain": tune,
        "draws_per_chain": draws,
        "total_draws_per_country": int(expected_array.shape[1]),
        "interval_method": "complete_posterior_draws",
        "poisson_probability_transform": "-expm1(-lambda)",
        "marginal_interval_resampling": False,
        "analysis_start": str(start.date()),
        "analysis_end": str(end.date()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-dir", type=Path, required=True)
    parser.add_argument("--passenger-file", type=Path, required=True)
    parser.add_argument("--population-file", type=Path, required=True)
    parser.add_argument("--population-column", default=None)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        cases_dir=args.cases_dir,
        passenger_file=args.passenger_file,
        population_file=args.population_file,
        output_dir=args.output_dir,
        start=pd.Timestamp(args.start),
        end=pd.Timestamp(args.end),
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
    )


if __name__ == "__main__":
    main()
