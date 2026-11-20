from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from countries import canonical_country


def export_cases(who_file: Path, output_dir: Path, start: pd.Timestamp, end: pd.Timestamp, passenger_file: Path | None = None) -> list[Path]:
    frame = pd.read_csv(who_file)
    required = {"COUNTRY", "DATE", "NEW_CONF_CASES"}
    if not required.issubset(frame):
        raise ValueError(f"WHO snapshot requires columns {sorted(required)}")
    frame = frame[["COUNTRY", "DATE", "NEW_CONF_CASES"]].rename(columns={"COUNTRY": "country", "DATE": "date", "NEW_CONF_CASES": "cases"})
    frame["country"] = frame["country"].map(canonical_country)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["cases"] = pd.to_numeric(frame["cases"], errors="raise")
    frame = frame.loc[frame["date"].between(start, end)].copy()
    negative = frame.loc[frame["cases"] < 0].copy()
    frame = frame.loc[frame["cases"] >= 0]
    if not np.isfinite(frame["cases"]).all() or not np.equal(frame["cases"], np.floor(frame["cases"])).all():
        raise ValueError("WHO counts must be finite integers.")
    if passenger_file is not None:
        passengers = pd.read_csv(passenger_file)
        country_column = next((name for name in ["country", "Country"] if name in passengers), None)
        count_column = next((name for name in ["passengers", "Estimated_Weekly_Pax"] if name in passengers), None)
        if country_column is None or count_column is None:
            raise ValueError("Passenger file requires country and passenger columns.")
        positive = pd.to_numeric(passengers[count_column], errors="raise") > 0
        countries = sorted(set(passengers.loc[positive, country_column].map(canonical_country)))
    else:
        countries = sorted(frame["country"].unique())
    output_dir.mkdir(parents=True, exist_ok=True)
    negative.to_csv(output_dir.parent / "who_negative_corrections_excluded.csv", index=False)
    calendar = pd.date_range(start, end, freq="D")
    files = []
    for country in countries:
        series = frame.loc[frame["country"].eq(country)].groupby("date")["cases"].sum().reindex(calendar, fill_value=0)
        if "/" in country or "\\" in country or country in {".", ".."}:
            raise ValueError(f"Invalid country filename: {country}")
        path = output_dir / f"{country}.csv"
        pd.DataFrame({"date": calendar, "cases": series.to_numpy(dtype=np.int64)}).to_csv(path, index=False)
        files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--who-file", type=Path, default=Path(__file__).resolve().parent / "data/who/WHO_global_mpox_cases_2025-12-31.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--passenger-file", type=Path)
    parser.add_argument("--start", default="2023-01-09")
    parser.add_argument("--end", default="2025-10-26")
    args = parser.parse_args()
    export_cases(args.who_file, args.output_dir, pd.Timestamp(args.start), pd.Timestamp(args.end), args.passenger_file)


if __name__ == "__main__":
    main()
