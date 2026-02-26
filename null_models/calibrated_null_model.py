#!/usr/bin/env python3
"""
calendar span anchored to its empirical first/last trade dates.

Trade counts per company and insider–company pair, tenure lengths, and the
global buy probability are still matched exactly to the empirical data. Trade
dates are sampled from a global open/blackout calendar (30 business days open
per 91-day quarter) intersected with each insider–firm tenure.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

SEED = 12345
START_DATE = datetime(2014, 1, 1)
END_DATE = datetime(2024, 12, 31)
QUARTER_LENGTH_DAYS = 91
OPEN_BUSINESS_DAYS_PER_QUARTER = 30

COLUMN_ALIASES = {
    "company": ["ISSUERTRADINGSYMBOL", "company", "symbol"],
    "insider": ["RPTOWNERNAME_lower", "insider"],
    "action": ["TRANS_ACQUIRED_DISP_CD", "action"],
    "date": ["TRANS_DATE", "date"],
}


def harmonize_columns(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = canonical
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    missing = {"company", "insider", "action", "date"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {source}: {sorted(missing)}")
    return df


@dataclass
class CalibrationStats:
    num_companies: int
    num_insiders: int
    total_trades: int
    trade_buy_prob: float
    company_trades: np.ndarray
    company_start_dates: np.ndarray  # ordinals
    company_end_dates: np.ndarray    # ordinals
    pair_company_idx: np.ndarray
    pair_insider_idx: np.ndarray
    pair_trade_counts: np.ndarray
    pair_duration_days: np.ndarray


def load_calibration(calibration_csv: Path) -> CalibrationStats:
    df = pd.read_csv(calibration_csv)
    df = harmonize_columns(df, calibration_csv)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.normalize()

    actions = df["action"].astype(str).str.upper()
    trade_buy_prob = float((actions == "A").mean()) if len(df) else 0.5

    companies = np.sort(df["company"].unique())
    insiders = np.sort(df["insider"].unique())
    company_to_idx = {name: idx for idx, name in enumerate(companies)}
    insider_to_idx = {name: idx for idx, name in enumerate(insiders)}

    company_trades = (
        df.groupby("company").size().reindex(companies).fillna(0).to_numpy(dtype=int)
    )
    company_minmax = (
        df.groupby("company")["date"].agg(["min", "max"]).reindex(companies)
    )
    company_start_dates = company_minmax["min"].to_numpy(dtype="datetime64[ns]")
    company_end_dates = company_minmax["max"].to_numpy(dtype="datetime64[ns]")

    pair_stats = (
        df.groupby(["company", "insider"])["date"]
        .agg(["min", "max", "count"])
        .reset_index()
    )
    pair_company_idx = pair_stats["company"].map(company_to_idx).to_numpy(dtype=int)
    pair_insider_idx = pair_stats["insider"].map(insider_to_idx).to_numpy(dtype=int)
    pair_trade_counts = pair_stats["count"].to_numpy(dtype=int)
    pair_duration_days = (
        (pair_stats["max"] - pair_stats["min"]).dt.days.fillna(0).astype(int) + 1
    )
    pair_duration_days = np.maximum(pair_duration_days, 1)

    return CalibrationStats(
        num_companies=len(companies),
        num_insiders=len(insiders),
        total_trades=len(df),
        trade_buy_prob=trade_buy_prob,
        company_trades=company_trades,
        company_start_dates=company_start_dates.astype("datetime64[ns]"),
        company_end_dates=company_end_dates.astype("datetime64[ns]"),
        pair_company_idx=pair_company_idx,
        pair_insider_idx=pair_insider_idx,
        pair_trade_counts=pair_trade_counts,
        pair_duration_days=pair_duration_days,
    )


def build_open_calendar() -> np.ndarray:
    open_days: List[int] = []
    quarter_start = START_DATE
    while quarter_start <= END_DATE:
        quarter_end = min(
            quarter_start + timedelta(days=QUARTER_LENGTH_DAYS - 1), END_DATE
        )
        business_days = pd.bdate_range(quarter_start, quarter_end)
        open_slice = business_days[:OPEN_BUSINESS_DAYS_PER_QUARTER]
        open_days.extend(d.to_pydatetime().toordinal() for d in open_slice)
        quarter_start += timedelta(days=QUARTER_LENGTH_DAYS)
    return np.array(sorted(set(open_days)), dtype=int)


OPEN_DAY_ORDS = build_open_calendar()


def sample_tenure_dates(
    rng: np.random.Generator,
    company_start: datetime,
    company_end: datetime,
    duration_days: int,
) -> Tuple[datetime, datetime]:
    span = (company_end - company_start).days + 1
    duration = int(max(1, min(duration_days, span)))
    max_offset = max(span - duration, 0)
    offset = int(rng.integers(0, max_offset + 1))
    start = company_start + timedelta(days=offset)
    end = min(start + timedelta(days=duration - 1), company_end)
    return start, end


def sample_open_dates(
    rng: np.random.Generator, start: datetime, end: datetime, n_trades: int
) -> Sequence[datetime]:
    if n_trades <= 0:
        return []
    start_ord = start.toordinal()
    end_ord = end.toordinal()
    mask = (OPEN_DAY_ORDS >= start_ord) & (OPEN_DAY_ORDS <= end_ord)
    eligible = OPEN_DAY_ORDS[mask]
    if eligible.size == 0:
        business_days = pd.bdate_range(start, end)
        if business_days.empty:
            return []
        choices = rng.choice(
            business_days.to_pydatetime().tolist(),
            size=n_trades,
            replace=len(business_days) < n_trades,
        )
        return sorted(choices)
    choices = rng.choice(
        eligible,
        size=n_trades,
        replace=eligible.size < n_trades,
    )
    return sorted(datetime.fromordinal(int(val)) for val in choices)


def generate_null_data(
    seed: int,
    calibration_csv: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    calib = load_calibration(calibration_csv)

    num_companies = calib.num_companies
    num_insiders = calib.num_insiders

    company_perm = rng.permutation(num_companies)
    company_perm_inv = np.empty(num_companies, dtype=int)
    company_perm_inv[company_perm] = np.arange(num_companies)
    insider_perm = rng.permutation(num_insiders)

    company_names = [f"COMP_{idx:05d}" for idx in range(num_companies)]
    insider_names = [f"INSIDER_{idx:05d}" for idx in range(num_insiders)]

    company_start_dates = calib.company_start_dates[company_perm_inv]
    company_end_dates = calib.company_end_dates[company_perm_inv]

    company_start_dates = [
        datetime.fromtimestamp(start.astype("datetime64[s]").astype(int))
        for start in company_start_dates
    ]
    company_end_dates = [
        datetime.fromtimestamp(end.astype("datetime64[s]").astype(int))
        for end in company_end_dates
    ]

    slot_company_idx = calib.pair_company_idx.copy()
    slot_company_idx = company_perm[slot_company_idx]
    slot_insider_idx = insider_perm[calib.pair_insider_idx]

    for trade_count in np.unique(calib.pair_trade_counts):
        indices = np.flatnonzero(calib.pair_trade_counts == trade_count)
        rng.shuffle(slot_company_idx[indices])

    slots_by_company: List[List[int]] = [[] for _ in range(num_companies)]
    for slot_id, company_idx in enumerate(slot_company_idx):
        slots_by_company[company_idx].append(slot_id)

    records: List[Tuple[str, str, str, str]] = []
    metadata: List[Dict[str, object]] = []
    trade_id = 0

    for company_idx, slot_indices in enumerate(slots_by_company):
        if not slot_indices:
            continue
        company_name = company_names[company_idx]
        company_start = company_start_dates[company_idx]
        company_end = company_end_dates[company_idx]
        for slot_id in slot_indices:
            insider_idx = slot_insider_idx[slot_id]
            insider_name = insider_names[insider_idx]
            n_trades = int(calib.pair_trade_counts[slot_id])
            duration_days = int(calib.pair_duration_days[slot_id])

            tenure_start, tenure_end = sample_tenure_dates(
                rng, company_start, company_end, duration_days
            )
            trade_dates = sample_open_dates(rng, tenure_start, tenure_end, n_trades)
            for dt in trade_dates:
                action = "A" if rng.random() < calib.trade_buy_prob else "D"
                date_str = dt.strftime("%Y-%m-%d")
                records.append((company_name, insider_name, action, date_str))
                metadata.append(
                    {
                        "trade_id": trade_id,
                        "company": company_name,
                        "insider": insider_name,
                        "action": action,
                        "date": date_str,
                        "is_illegal": 0,
                        "event_id": -1,
                        "trade_type": "BACKGROUND",
                    }
                )
                trade_id += 1

    trades_df = pd.DataFrame(records, columns=["company", "insider", "action", "date"])
    metadata_df = pd.DataFrame(metadata)
    trades_df.sort_values(["date", "company", "insider"], inplace=True)
    metadata_df.sort_values("trade_id", inplace=True)
    return trades_df, metadata_df


def print_summary(trades_df: pd.DataFrame):
    total = len(trades_df)
    purchases = (trades_df["action"] == "A").sum()
    print("Null dataset summary (calibrated v2)")
    print("-------------------------------------")
    print(f"Total trades:     {total:,}")
    if total > 0:
        print(f"Purchases:        {purchases:,} ({purchases / total:.2%})")
        print(f"Sales:            {total - purchases:,} ({1 - purchases / total:.2%})")
        print(f"Date range:       {trades_df['date'].min()} .. {trades_df['date'].max()}")
    print(f"Unique insiders:  {trades_df['insider'].nunique():,}")
    print(f"Unique companies: {trades_df['company'].nunique():,}")


def write_outputs(trades_df: pd.DataFrame, metadata_df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(output_dir / "trades_by_day.csv", index=False)
    metadata_df.to_csv(output_dir / "trades_metadata.csv", index=False)
    pd.DataFrame(
        columns=["event_id", "company", "quarter", "event_date", "is_illegal"]
    ).to_csv(output_dir / "ma_events.csv", index=False)
    pd.DataFrame(
        columns=["event_id", "company", "insider", "trade_id", "quarter"]
    ).to_csv(output_dir / "illegal_participants.csv", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Generate calibrated null dataset with anchored firm spans."
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Output directory for generated CSVs",
    )
    parser.add_argument(
        "--calibration-csv",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "network_cpp"
        / "trades_by_day.csv",
        help="Empirical trades CSV used for calibration",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Suppress console summary output",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Generate data without writing CSV files",
    )
    args = parser.parse_args()

    trades_df, metadata_df = generate_null_data(
        seed=args.seed, calibration_csv=args.calibration_csv
    )
    if not args.no_write:
        write_outputs(trades_df, metadata_df, args.out_dir)
    if not args.no_summary:
        print_summary(trades_df)


if __name__ == "__main__":
    main()

