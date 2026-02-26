#!/usr/bin/env python3
"""
Option A: Insider shuffle within firm/time-bin.

This null model tests whether the same insiders trade together more often than
expected given the actual firm-level timing. For each firm and calendar bin
(month by default) we randomly permute insider IDs across trades, keeping every
timestamp, firm, and trade direction untouched.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

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


def summary(tag: str, trades_df: pd.DataFrame) -> None:
    total = len(trades_df)
    buys = (trades_df["action"].astype(str).str.upper() == "A").sum()
    print(f"{tag} summary")
    print("-------------------------")
    print(f"Total trades:     {total:,}")
    if total > 0:
        print(f"Purchases:        {buys:,} ({buys / total:.2%})")
        print(f"Sales:            {total - buys:,} ({1 - buys / total:.2%})")
        print(f"Date range:       {trades_df['date'].min()} .. {trades_df['date'].max()}")
    print(f"Unique insiders:  {trades_df['insider'].nunique():,}")
    print(f"Unique companies: {trades_df['company'].nunique():,}")


def write_outputs(trades_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(output_dir / "trades_by_day.csv", index=False)
    pd.DataFrame(columns=["event_id", "company", "quarter", "event_date", "is_illegal"]).to_csv(
        output_dir / "ma_events.csv", index=False
    )
    pd.DataFrame(columns=["event_id", "company", "insider", "trade_id", "quarter"]).to_csv(
        output_dir / "illegal_participants.csv", index=False
    )
    pd.DataFrame(
        {
            "trade_id": np.arange(len(trades_df), dtype=int),
            "company": trades_df["company"],
            "insider": trades_df["insider"],
            "action": trades_df["action"],
            "date": trades_df["date"],
            "is_illegal": 0,
            "event_id": -1,
            "trade_type": "BACKGROUND",
        }
    ).to_csv(output_dir / "trades_metadata.csv", index=False)


def shuffle_insiders(
    rng: np.random.Generator, df: pd.DataFrame, freq: str
) -> pd.DataFrame:
    grouped = df.groupby(["company", df["date"].dt.to_period(freq)], group_keys=False)

    shuffled_groups: List[pd.DataFrame] = []
    for _, group in grouped:
        if len(group) <= 1:
            shuffled_groups.append(group)
            continue
        shuffled = group.copy()
        insiders = group["insider"].to_numpy()
        rng.shuffle(insiders)
        shuffled["insider"] = insiders
        shuffled_groups.append(shuffled)

    result = pd.concat(shuffled_groups, ignore_index=True)
    result.sort_values(["date", "company", "insider"], inplace=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Option A: shuffle insiders within firm/time-bin."
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "network_cpp" / "trades_by_day.csv",
        help="Empirical trades CSV to shuffle",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "insider_shuffle_output",
        help="Directory to write shuffled outputs",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="M",
        help="Pandas period frequency for time bins (e.g., 'M' for month, 'Q' for quarter)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Suppress console summaries",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df = harmonize_columns(df, args.input_csv)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])

    rng = np.random.default_rng(args.seed)
    shuffled = shuffle_insiders(rng, df, args.freq)

    if not args.no_summary:
        summary("Original", df)
        print()
        summary("Shuffled", shuffled)

    write_outputs(shuffled, args.out_dir)


if __name__ == "__main__":
    main()

