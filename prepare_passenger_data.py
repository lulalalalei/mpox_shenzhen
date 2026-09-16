from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def pick_column(frame: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    for name in candidates:
        if name in frame.columns:
            return name
    raise ValueError(f"Missing {label} column. Available columns: {list(frame.columns)}")


def parse_mixed_month(value: pd.Series) -> pd.Series:
    text = value.astype(str).str.strip()
    parsed = pd.to_datetime(text, format="mixed", errors="coerce")
    missing = parsed.isna() & text.str.fullmatch(r"\d{6}")
    parsed.loc[missing] = pd.to_datetime(
        text.loc[missing] + "01",
        format="%Y%m%d",
        errors="coerce",
    )
    missing = parsed.isna() & text.str.fullmatch(r"\d{4}[-/]\d{1,2}")
    parsed.loc[missing] = pd.to_datetime(
        text.loc[missing].str.replace("/", "-", regex=False) + "-01",
        format="%Y-%m-%d",
        errors="coerce",
    )
    return parsed


def build_weekly_passengers(
    transport_capacity_file: Path,
    monthly_passenger_file: Path,
    output_file: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    capacity = pd.read_csv(transport_capacity_file, low_memory=False)
    capacity.columns = capacity.columns.astype(str).str.strip()
    capacity_country = pick_column(
        capacity,
        ("Dep DOT Country Name", "country", "Country"),
        "transport country",
    )
    capacity_date = pick_column(
        capacity,
        ("Time series", "date", "Date"),
        "transport date",
    )
    capacity_seats = pick_column(
        capacity,
        ("Seats (Total)", "seats", "Seats"),
        "seat count",
    )
    capacity = capacity[[capacity_country, capacity_date, capacity_seats]].rename(
        columns={
            capacity_country: "country",
            capacity_date: "date",
            capacity_seats: "seats",
        }
    )
    capacity["date"] = pd.to_datetime(capacity["date"], format="mixed", errors="coerce")
    capacity["seats"] = pd.to_numeric(capacity["seats"], errors="coerce")
    capacity = capacity.dropna()
    capacity = capacity.loc[capacity["seats"] >= 0].copy()
    capacity["country"] = capacity["country"].astype(str).str.strip()
    capacity = capacity.loc[capacity["country"].ne("China")]
    capacity["month"] = capacity["date"].dt.to_period("M")
    capacity["monthly_seats"] = capacity.groupby(["country", "month"])["seats"].transform("sum")
    capacity = capacity.loc[capacity["monthly_seats"] > 0].copy()
    capacity["share"] = capacity["seats"] / capacity["monthly_seats"]

    passengers = pd.read_csv(monthly_passenger_file, low_memory=False)
    passengers.columns = passengers.columns.astype(str).str.strip()
    passenger_country = pick_column(
        passengers,
        ("Dep Country Name", "country", "Country"),
        "passenger country",
    )
    passenger_date = pick_column(
        passengers,
        ("Time Series", "date", "Date"),
        "passenger month",
    )
    passenger_count = pick_column(
        passengers,
        ("Total Est. Pax", "passengers", "Total_Est_Pax"),
        "passenger count",
    )
    passengers = passengers[[passenger_country, passenger_date, passenger_count]].rename(
        columns={
            passenger_country: "country",
            passenger_date: "month_date",
            passenger_count: "monthly_passengers",
        }
    )
    passengers["country"] = passengers["country"].astype(str).str.strip()
    passengers = passengers.loc[passengers["country"].ne("China")]
    passengers["month_date"] = parse_mixed_month(passengers["month_date"])
    passengers["monthly_passengers"] = (
        passengers["monthly_passengers"].astype(str).str.replace(",", "", regex=False)
    )
    passengers["monthly_passengers"] = pd.to_numeric(
        passengers["monthly_passengers"],
        errors="coerce",
    )
    passengers = passengers.dropna()
    passengers = passengers.loc[passengers["monthly_passengers"] >= 0].copy()
    passengers["month"] = passengers["month_date"].dt.to_period("M")
    passengers = (
        passengers.groupby(["country", "month"], as_index=False)["monthly_passengers"]
        .sum()
    )

    mapping = {
        "Iran Islamic Republic of": "Iran (Islamic Republic of)",
        "Korea Republic of": "Republic of Korea",
        "USA": "United States of America",
        "United Kingdom": "The United Kingdom",
    }
    capacity["country"] = capacity["country"].replace(mapping)
    passengers["country"] = passengers["country"].replace(mapping)
    merged = capacity.merge(passengers, on=["country", "month"], how="inner")
    if merged.empty:
        raise ValueError("The transport and passenger tables have no matching country-month rows.")
    merged["estimated_passengers"] = merged["share"] * merged["monthly_passengers"]
    merged = merged.loc[
        (merged["date"] >= start) & (merged["date"] <= end)
    ].copy()
    merged["week"] = merged["date"] - pd.to_timedelta(merged["date"].dt.weekday, unit="D")
    weekly = (
        merged.groupby(["country", "week"], as_index=False)["estimated_passengers"]
        .sum()
        .rename(columns={"week": "date", "estimated_passengers": "passengers"})
        .sort_values(["country", "date"])
        .reset_index(drop=True)
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(output_file, index=False)
    return weekly


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport-capacity-file", type=Path, required=True)
    parser.add_argument("--monthly-passenger-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--start", default="2023-01-09")
    parser.add_argument("--end", default="2025-10-26")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_weekly_passengers(
        transport_capacity_file=args.transport_capacity_file,
        monthly_passenger_file=args.monthly_passenger_file,
        output_file=args.output_file,
        start=pd.Timestamp(args.start),
        end=pd.Timestamp(args.end),
    )


if __name__ == "__main__":
    main()
