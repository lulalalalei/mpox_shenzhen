from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.6)
    ax.set_axisbelow(True)


def make_temporal_figure(model_dir: Path, figure_dir: Path) -> None:
    weekly = pd.read_csv(model_dir / "weekly_summary.csv", parse_dates=["date"])
    countries = pd.read_csv(model_dir / "country_summary.csv")
    top = countries.head(6)["country"].tolist()
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.4), sharex=True, gridspec_kw={"hspace": 0.16})
    ax = axes[0]
    x = weekly["date"]
    ax.fill_between(
        x,
        weekly["probability_q025"].to_numpy(float) * 100,
        weekly["probability_q975"].to_numpy(float) * 100,
        color="#ef4444",
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(x, weekly["probability_mean"] * 100, color="#b91c1c", linewidth=2.1)
    peak = weekly.loc[weekly["probability_mean"].idxmax()]
    ax.scatter([peak["date"]], [peak["probability_mean"] * 100], color="#7f1d1d", s=18, zorder=4)
    ax.annotate(
        f"Peak {peak['probability_mean'] * 100:.2f}%\n"
        f"95% CrI {peak['probability_q025'] * 100:.2f}-{peak['probability_q975'] * 100:.2f}%",
        xy=(peak["date"], peak["probability_mean"] * 100),
        xytext=(pd.Timestamp("2024-12-15"), max(weekly["probability_q975"].max() * 100 * 0.85, 0.2)),
        arrowprops={"arrowstyle": "-", "color": "#7f1d1d", "linewidth": 0.8},
        fontsize=8,
    )
    ax.set_ylabel("Probability (%)")
    ax.set_title("A. Overall weekly probability of at least one imported case", loc="left", fontweight="bold")
    style_axes(ax)
    colors = ["#006d77", "#c2410c", "#6d28d9", "#db2777", "#4d7c0f", "#0284c7"]
    ax = axes[1]
    country_weekly = pd.read_csv(model_dir / "country_weekly_summary.csv", parse_dates=["date"])
    for index, country in enumerate(top):
        subset = country_weekly.loc[country_weekly["country"].eq(country)].sort_values("date")
        color = colors[index % len(colors)]
        ax.plot(
            subset["date"],
            subset["probability_mean"] * 100,
            color=color,
            linewidth=1.8 if index < 3 else 1.25,
            label=country,
        )
        if index < 3:
            ax.fill_between(
                subset["date"],
                subset["probability_q025"].to_numpy(float) * 100,
                subset["probability_q975"].to_numpy(float) * 100,
                color=color,
                alpha=0.10,
                linewidth=0,
            )
    ax.set_ylabel("Probability (%)")
    ax.set_xlabel("Week")
    ax.set_title("B. Country-specific weekly probabilities for the six largest contributors", loc="left", fontweight="bold")
    ax.legend(loc="upper left", ncol=3, frameon=False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    style_axes(ax)
    save_figure(fig, figure_dir, "temporal_importation_risk")


def make_country_weekly_summary(model_dir: Path) -> pd.DataFrame:
    draws = np.load(model_dir / "posterior_draws.npz", allow_pickle=False)
    countries = draws["countries"].astype(str)
    dates = pd.to_datetime(draws["dates"])
    probability = draws["country_probability"]
    rows = []
    for index, country in enumerate(countries):
        mean = probability[index].mean(axis=0)
        q025 = np.quantile(probability[index], 0.025, axis=0)
        q975 = np.quantile(probability[index], 0.975, axis=0)
        rows.append(
            pd.DataFrame(
                {
                    "country": country,
                    "date": dates,
                    "probability_mean": mean,
                    "probability_q025": q025,
                    "probability_q975": q975,
                }
            )
        )
    output = pd.concat(rows, ignore_index=True)
    output.to_csv(model_dir / "country_weekly_summary.csv", index=False)
    return output


def make_country_figure(model_dir: Path, figure_dir: Path) -> None:
    summary = pd.read_csv(model_dir / "country_summary.csv")
    top = summary.head(12).sort_values("cumulative_expected_imports_mean")
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    y = np.arange(len(top))
    ax.errorbar(
        top["cumulative_expected_imports_mean"],
        y,
        xerr=[
            top["cumulative_expected_imports_mean"] - top["cumulative_expected_imports_q025"],
            top["cumulative_expected_imports_q975"] - top["cumulative_expected_imports_mean"],
        ],
        fmt="o",
        color="#b91c1c",
        ecolor="#ef4444",
        capsize=2,
    )
    ax.set_yticks(y, top["country"])
    ax.set_xlabel("Cumulative expected introductions")
    ax.set_title("Source-country ranking", loc="left", fontweight="bold")
    style_axes(ax)
    save_figure(fig, figure_dir, "source_country_ranking")


def load_observed_weekly(path: Path, dates: pd.DatetimeIndex) -> np.ndarray:
    frame = pd.read_csv(path)
    frame.columns = frame.columns.astype(str).str.strip()
    date_column = next(
        (name for name in ("date", "week", "dates", "Date") if name in frame.columns),
        None,
    )
    count_column = next(
        (
            name
            for name in (
                "cases",
                "reported_cases",
                "observed_imported_cases",
                "observed_non_hk_imported_cases",
            )
            if name in frame.columns
        ),
        None,
    )
    if date_column is None or count_column is None:
        raise ValueError("The observed-case file must contain date and count columns.")
    frame["date"] = pd.to_datetime(frame[date_column], errors="coerce")
    frame["cases"] = pd.to_numeric(frame[count_column], errors="coerce")
    frame = frame.dropna(subset=["date", "cases"])
    frame["date"] = frame["date"] - pd.to_timedelta(frame["date"].dt.weekday, unit="D")
    weekly = frame.groupby("date")["cases"].sum()
    return weekly.reindex(dates, fill_value=0).to_numpy(float)


def make_calibration_figure(model_dir: Path, figure_dir: Path, observed_file: Path, seed: int) -> None:
    draws = np.load(model_dir / "posterior_draws.npz", allow_pickle=False)
    dates = pd.to_datetime(draws["dates"])
    model_draws = draws["overall_expected_imports"].astype(float)
    observed = load_observed_weekly(observed_file, dates)
    shape = 0.5 + observed.sum()
    rate = np.maximum(model_draws.sum(axis=1), np.finfo(float).tiny)
    alpha = np.random.default_rng(seed).gamma(shape=shape, scale=1.0 / rate)
    calibrated = alpha[:, None] * model_draws
    predictive = np.random.default_rng(seed + 1).poisson(np.clip(calibrated, 0, 1e7))
    expected_mean = calibrated.mean(axis=0)
    expected_q025 = np.quantile(calibrated, 0.025, axis=0)
    expected_q975 = np.quantile(calibrated, 0.975, axis=0)
    predictive_q025 = np.quantile(predictive, 0.025, axis=0)
    predictive_q975 = np.quantile(predictive, 0.975, axis=0)
    result = pd.DataFrame(
        {
            "date": dates,
            "observed_cases": observed,
            "calibrated_expected_mean": expected_mean,
            "calibrated_expected_q025": expected_q025,
            "calibrated_expected_q975": expected_q975,
            "predictive_q025": predictive_q025,
            "predictive_q975": predictive_q975,
        }
    )
    result.to_csv(model_dir / "calibration_weekly.csv", index=False)
    alpha_summary = pd.DataFrame(
        {
            "metric": ["alpha", "observed_total"],
            "mean": [alpha.mean(), observed.sum()],
            "q025": [np.quantile(alpha, 0.025), observed.sum()],
            "q975": [np.quantile(alpha, 0.975), observed.sum()],
        }
    )
    alpha_summary.to_csv(model_dir / "calibration_summary.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True, gridspec_kw={"hspace": 0.16})
    ax = axes[0]
    ax.bar(dates, observed, width=5.2, color="#1f2937", label="Observed")
    ax.plot(dates, expected_mean, color="#b91c1c", label="Calibrated expected introductions")
    ax.fill_between(dates, predictive_q025, predictive_q975, color="#ef4444", alpha=0.18, linewidth=0, label="95% posterior predictive interval")
    ax.set_ylabel("Cases per week")
    ax.legend(frameon=False)
    ax.set_title("A. Calibration against observed imported cases", loc="left", fontweight="bold")
    style_axes(ax)
    ax = axes[1]
    cumulative_expected = result["calibrated_expected_mean"].cumsum()
    cumulative_observed = result["observed_cases"].cumsum()
    ax.plot(dates, cumulative_expected, color="#b91c1c", label="Calibrated cumulative estimate")
    ax.step(dates, cumulative_observed, where="post", color="#1f2937", label="Observed cumulative cases")
    ax.set_ylabel("Cumulative cases")
    ax.set_xlabel("Week")
    ax.legend(frameon=False)
    ax.set_title("B. Cumulative comparison", loc="left", fontweight="bold")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    style_axes(ax)
    save_figure(fig, figure_dir, "calibrated_importation_model")


def make_world_map(model_dir: Path, figure_dir: Path, shapefile: Path, country_map_file: Path | None) -> None:
    import geopandas as gpd

    shapes = gpd.read_file(shapefile)
    summary = pd.read_csv(model_dir / "country_summary.csv")
    if country_map_file is not None:
        mapping = pd.read_csv(country_map_file)
        mapping.columns = mapping.columns.astype(str).str.strip()
        mapping = mapping.rename(columns={mapping.columns[0]: "country", mapping.columns[1]: "iso3"})
        summary = summary.merge(mapping[["country", "iso3"]], on="country", how="left")
    else:
        summary["iso3"] = summary["country"]
    shape_iso = next((name for name in ("ISO_A3", "ADM0_A3", "iso_a3") if name in shapes.columns), None)
    if shape_iso is None:
        raise ValueError("The shapefile must contain an ISO3 column or a country mapping file must be supplied.")
    merged = shapes.merge(summary[["iso3", "cumulative_expected_imports_mean"]], left_on=shape_iso, right_on="iso3", how="left")
    fig, ax = plt.subplots(figsize=(12.0, 6.4))
    merged.plot(
        ax=ax,
        column="cumulative_expected_imports_mean",
        cmap="Reds",
        linewidth=0.3,
        edgecolor="white",
        legend=True,
        missing_kwds={"color": "#eeeeee"},
    )
    ax.set_axis_off()
    ax.set_title("Cumulative expected mpox introductions by source country", loc="left", fontweight="bold")
    save_figure(fig, figure_dir, "world_map_cumulative_importation")


def generate_figures(
    model_dir: Path,
    figure_dir: Path,
    observed_file: Path | None = None,
    shapefile: Path | None = None,
    country_map_file: Path | None = None,
    seed: int = 2026,
) -> None:
    make_country_weekly_summary(model_dir)
    make_temporal_figure(model_dir, figure_dir)
    make_country_figure(model_dir, figure_dir)
    if observed_file is not None:
        make_calibration_figure(model_dir, figure_dir, observed_file, seed)
    if shapefile is not None:
        make_world_map(model_dir, figure_dir, shapefile, country_map_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--observed-file", type=Path, default=None)
    parser.add_argument("--world-shapefile", type=Path, default=None)
    parser.add_argument("--country-map-file", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_figures(
        model_dir=args.model_dir,
        figure_dir=args.figure_dir,
        observed_file=args.observed_file,
        shapefile=args.world_shapefile,
        country_map_file=args.country_map_file,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
