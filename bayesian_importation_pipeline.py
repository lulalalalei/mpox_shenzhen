from __future__ import annotations

import argparse
import json
import os
import hashlib
import warnings
from importlib.metadata import version
from countries import canonical_country
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
        (name for name in ("date", "dates", "Date", "Dates", "DATE") if name in frame.columns),
        None,
    )
    case_column = next(
        (
            name
            for name in ("cases", "reported_cases", "Cases", "reported cases", "NEW_CONF_CASES")
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
    if frame.isna().any().any() or (frame["cases"] < 0).any() or not np.equal(frame["cases"], np.floor(frame["cases"])).all():
        raise ValueError(f"{path}: invalid dates or non-negative integer case counts.")
    frame["cases"] = frame["cases"].astype(int)
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
    frame["country"] = frame["country"].map(canonical_country)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["passengers"] = pd.to_numeric(frame["passengers"], errors="coerce")
    frame = frame.dropna()
    if (frame["passengers"] < 0).any():
        raise ValueError("Passenger counts must be non-negative.")
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
    return dict(zip(frame[country_column].map(canonical_country), frame[population_column].astype(float)))


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
    observed_cases = np.asarray(observed_cases)
    if not np.isfinite(observed_cases).all() or (observed_cases < 0).any() or not np.equal(observed_cases, np.floor(observed_cases)).all():
        raise ValueError("Case counts must be finite non-negative integers.")
    observed_cases = observed_cases.astype(np.int64)
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
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=min(chains, os.cpu_count() or 1),
            nuts_sampler="nutpie",
            nuts={"target_accept": target_accept},
            nuts_sampler_kwargs={"maxdepth": max_treedepth, "adaptation": "low_rank"},
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
            "sampling_backend": "nutpie",
            "nutpie_version": version("nutpie"),
            "pymc_version": pm.__version__,
            "nutpie_adaptation": "low_rank",
        }
    )
    return idata


def save_posterior(idata: az.InferenceData, path: Path) -> None:
    for group_name in idata.groups():
        dataset = getattr(idata, group_name)
        for name in list(dataset.data_vars):
            if np.issubdtype(dataset[name].dtype, np.bool_):
                dataset[name] = dataset[name].astype(np.int8)
        for name, value in list(dataset.attrs.items()):
            if isinstance(value, (bool, np.bool_)):
                dataset.attrs[name] = int(value)
            elif isinstance(value, (dict, list, tuple, set)):
                dataset.attrs[name] = json.dumps(value, default=str)
    for name, value in list(idata.attrs.items()):
        if isinstance(value, (dict, list, tuple, set)):
            idata.attrs[name] = json.dumps(value, default=str)
        elif isinstance(value, (bool, np.bool_)):
            idata.attrs[name] = int(value)
    idata.to_netcdf(path)


def save_diagnostics(
    idata: az.InferenceData,
    output_dir: Path,
    country: str,
    ess_threshold: float = 400.0,
    rhat_threshold: float = 1.01,
    bfmi_threshold: float = 0.30,
    tail_ess_threshold: float = 400.0,
    plots: bool = True,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = az.summary(
        idata,
        var_names=["sigma_rw", "log_I0", "I"],
        kind="diagnostics",
        round_to=4,
    )
    summary.to_csv(output_dir / f"{safe_name(country)}_diagnostics.csv")
    rhat_values = summary["r_hat"].to_numpy()
    bulk_values = summary["ess_bulk"].to_numpy()
    tail_values = summary["ess_tail"].to_numpy()
    max_rhat = float(rhat_values.max()) if np.isfinite(rhat_values).all() else np.nan
    min_bulk_ess = float(bulk_values.min()) if np.isfinite(bulk_values).all() else np.nan
    min_tail_ess = float(tail_values.min()) if np.isfinite(tail_values).all() else np.nan
    divergences = int(idata.sample_stats["diverging"].sum().values)
    depth_name = "tree_depth" if "tree_depth" in idata.sample_stats else "depth" if "depth" in idata.sample_stats else None
    if depth_name is not None:
        tree_depth = idata.sample_stats[depth_name].to_numpy()
        max_tree_depth = int(np.nanmax(tree_depth))
        max_tree_depth_hits = int(np.sum(tree_depth >= int(idata.posterior.attrs["max_treedepth"])))
    else:
        max_tree_depth = np.nan
        max_tree_depth_hits = np.nan
    try:
        bfmi_values = np.asarray(az.bfmi(idata))
        min_bfmi = float(bfmi_values.min()) if np.isfinite(bfmi_values).all() else np.nan
    except Exception:
        min_bfmi = np.nan
    failures = []
    for passed, reason in [
        (np.isfinite(max_rhat) and max_rhat < rhat_threshold, "rhat"),
        (np.isfinite(min_bulk_ess) and min_bulk_ess >= ess_threshold, "bulk_ess"),
        (np.isfinite(min_tail_ess) and min_tail_ess >= tail_ess_threshold, "tail_ess"),
        (divergences == 0, "divergences"),
        (np.isfinite(min_bfmi) and min_bfmi >= bfmi_threshold, "bfmi"),
        (np.isfinite(max_tree_depth_hits) and max_tree_depth_hits == 0, "tree_depth"),
    ]:
        if not passed:
            failures.append(reason)
    diagnostic_pass = not failures
    if plots:
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
        except Exception as error:
            plt.close("all")
            warnings.warn(f"{country}: diagnostic plot could not be generated: {error}", RuntimeWarning)
        try:
            az.plot_rank(
                idata,
                var_names=["sigma_rw", "log_I0", "I"],
                coords=selected_coords,
                kind="vlines",
            )
            plt.gcf().savefig(output_dir / f"{safe_name(country)}_rank.png", dpi=180, bbox_inches="tight")
            plt.close("all")
        except Exception as error:
            plt.close("all")
            warnings.warn(f"{country}: diagnostic plot could not be generated: {error}", RuntimeWarning)
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
        "failed_criteria": ";".join(failures),
        "tail_ess_threshold": tail_ess_threshold,
    }
    return result


def save_predictive_summary(idata: az.InferenceData, cases: pd.Series, output_dir: Path, country: str, seed: int) -> None:
    rate = idata.posterior["lambda_report"].values.reshape(-1, len(cases))
    if not np.isfinite(rate).all() or (rate < 0).any():
        raise ValueError(f"{country}: invalid posterior predictive rates.")
    predictive = np.random.default_rng(seed).poisson(np.clip(rate, 0, 1e7))
    frame = pd.DataFrame({"date": cases.index, "reported_cases": cases.to_numpy(), "ppc_reported_median": np.median(predictive, axis=0), "ppc_reported_lower_95": np.quantile(predictive, .025, axis=0), "ppc_reported_upper_95": np.quantile(predictive, .975, axis=0)})
    frame.to_csv(output_dir / f"{safe_name(country)}_ppc.csv", index=False)


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
    tail_ess_threshold: float = 400.0,
    posterior_input_dir: Path | None = None,
    diagnostic_plots: bool = True,
) -> None:
    if start > end or start.dayofweek != 0 or end.dayofweek != 6:
        raise ValueError("Use a Monday start and Sunday end, with start <= end.")
    if not 0 < reporting_threshold <= 1 or tail_ess_threshold <= 0:
        raise ValueError("Reporting completeness and tail ESS thresholds are invalid.")
    if posterior_input_dir is not None and (incubation_meanlog, incubation_sdlog, reporting_meanlog, reporting_sdlog) != (2.001, .234, 1.6057319, .5964992):
        raise ValueError("Archived posterior reuse requires the original delay parameters.")
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
        country = canonical_country(case_file.stem)
        panel = passengers.loc[passengers["country"].eq(country), ["date", "passengers"]].copy()
        weekly_cases = load_weekly_cases(case_file, start, end)
        if panel["passengers"].sum() <= 0:
            continue
        if weekly_cases.sum() <= 0:
            diagnostics.append({"country": country, "diagnostic_pass": False, "failed_criteria": "no_reported_cases"})
            continue
        if country not in populations:
            raise ValueError(f"Population data are missing for {country}.")
        posterior_file = posterior_dir / f"{safe_name(country)}.nc"
        if posterior_input_dir is None:
            idata = fit_country(
                weekly_cases.to_numpy(), delay_pmf, chains=chains, tune=tune,
                draws=draws, target_accept=target_accept, max_treedepth=max_treedepth,
                seed=seed + sum(ord(character) for character in country),
            )
            idata.posterior.attrs["case_counts_sha256"] = hashlib.sha256(weekly_cases.to_numpy(dtype=np.int64).tobytes()).hexdigest()
            idata.posterior.attrs["analysis_start"] = str(start.date())
            idata.posterior.attrs["analysis_end"] = str(end.date())
            save_posterior(idata, posterior_file)
        else:
            candidates = [posterior_input_dir / f"{country}_posterior.nc", posterior_input_dir / f"{safe_name(country)}.nc"]
            found = next((path for path in candidates if path.exists()), None)
            if found is None:
                raise FileNotFoundError(f"No saved posterior for {country} in {posterior_input_dir}")
            idata = az.from_netcdf(found)
            for group in idata.groups():
                dataset = getattr(idata, group)
                dataset.load()
                dataset.close()
            if str(idata.posterior.attrs.get("inference_library", "")) != "nutpie":
                raise ValueError(f"{country}: saved posterior does not identify nutpie.")
            if idata.posterior.sizes.get("chain") != chains or idata.posterior.sizes.get("draw") != draws:
                raise ValueError(f"{country}: saved chain/draw dimensions differ from requested settings.")
            observed_names = list(idata.observed_data.data_vars) if hasattr(idata, "observed_data") else []
            if len(observed_names) != 1 or not np.array_equal(idata.observed_data[observed_names[0]].values, weekly_cases.to_numpy()):
                raise ValueError(f"{country}: saved posterior observations do not match the current case series.")
            if idata.posterior["I"].shape[-1] != len(weekly_cases):
                raise ValueError(f"{country}: saved posterior has a different weekly calendar length.")
            if float(idata.posterior.attrs.get("target_accept", np.nan)) != target_accept:
                raise ValueError(f"{country}: saved target_accept differs from requested value.")
            saved_depth = idata.posterior.attrs.get("max_treedepth", idata.posterior.attrs.get("max_treedepth_requested"))
            if saved_depth is None or int(saved_depth) != max_treedepth:
                raise ValueError(f"{country}: saved maximum tree depth differs from requested value.")
            for key, expected in [("analysis_start", str(start.date())), ("analysis_end", str(end.date()))]:
                if key in idata.posterior.attrs and idata.posterior.attrs[key] != expected:
                    raise ValueError(f"{country}: saved {key} differs from requested calendar.")
            idata.posterior.attrs["max_treedepth"] = max_treedepth
        diagnostic = save_diagnostics(idata, diagnostics_dir, country, tail_ess_threshold=tail_ess_threshold, plots=diagnostic_plots)
        diagnostics.append(diagnostic)
        if not diagnostic["diagnostic_pass"]:
            continue
        save_predictive_summary(idata, weekly_cases, diagnostics_dir, country, seed + sum(ord(character) for character in country) + 1)
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
        if country in country_names:
            raise ValueError(f"Duplicate case series after country harmonisation: {country}")
        if not np.isfinite(incidence_draws).all() or not np.isfinite(expected_imports).all():
            raise ValueError(f"{country}: non-finite posterior propagation.")
        country_names.append(country)
        country_incidence.append(incidence_draws)
        country_expected.append(expected_imports)
        country_probability.append(probability)
    pd.DataFrame(diagnostics).to_csv(output_dir / "screening_audit.csv", index=False)
    if not country_expected:
        raise ValueError("No source countries passed the diagnostic screen. See screening_audit.csv.")
    draw_counts = {array.shape[0] for array in country_expected}
    if len(draw_counts) != 1:
        raise ValueError("All countries must have the same number of posterior draws.")
    expected_array = np.stack(country_expected, axis=0)
    probability_array = np.stack(country_probability, axis=0)
    incidence_array = np.stack(country_incidence, axis=0)
    dates = pd.DatetimeIndex(country_dates)
    completeness = reporting_completeness(delay_pmf, len(dates))
    mature_mask = completeness >= reporting_threshold
    if not mature_mask.any():
        raise ValueError("No weeks satisfy the reporting-completeness threshold.")
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
        "sampling_backend": "nutpie",
        "nutpie_version": version("nutpie"),
        "pymc_version": pm.__version__,
        "tail_ess_threshold": tail_ess_threshold,
        "bulk_ess_threshold": 400,
        "rhat_threshold_exclusive": 1.01,
        "base_seed": seed,
        "incubation_meanlog": incubation_meanlog,
        "incubation_sdlog": incubation_sdlog,
        "reporting_meanlog": reporting_meanlog,
        "reporting_sdlog": reporting_sdlog,
        "incubation_lag_weeks": incubation_lag_weeks,
        "reporting_completeness_threshold": reporting_threshold,
        "country_names": country_names,
        "country_seed_rule": "base_seed + sum(ord(character) for character in country)",
        "delay_discretisation": "daily CDF differences, discrete convolution, consecutive seven-day sums",
        "posterior_input_dir": str(posterior_input_dir) if posterior_input_dir else None,
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
    parser.add_argument("--tail-ess-threshold", type=float, default=400.0)
    parser.add_argument("--posterior-input-dir", type=Path, default=None)
    parser.add_argument("--skip-diagnostic-plots", action="store_true")
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
        tail_ess_threshold=args.tail_ess_threshold,
        posterior_input_dir=args.posterior_input_dir,
        diagnostic_plots=not args.skip_diagnostic_plots,
    )


if __name__ == "__main__":
    main()
