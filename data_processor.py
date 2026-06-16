# data_processor.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import os

# Canonical name mappings
NUMERIC_MAP = {
    "voltage":     ["voltage", "Voltage", "V"],
    "current":     ["current", "Current", "I"],
    "temperature": ["temperature", "Temperature", "temp", "Temp", "T"],
    "cycle":       ["cycle", "Cycle", "cycle_count", "cycles"],
    "capacity":    ["capacity", "Capacity", "Ah", "SOC", "SoC"],
    "soh":         ["soh", "SOH", "state_of_health"],
    "rul":         ["rul", "RUL", "remaining_life_years"],
}

CAT_MAP = {"battery_type": ["battery_type", "chemistry", "type"]}


def _first(df: pd.DataFrame, names: list[str]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def _canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Rename numeric columns
    for canon, opts in NUMERIC_MAP.items():
        c = _first(df, opts)
        if c and c != canon:
            df.rename(columns={c: canon}, inplace=True)

    # Rename categorical columns
    for canon, opts in CAT_MAP.items():
        c = _first(df, opts)
        if c and c != canon:
            df.rename(columns={c: canon}, inplace=True)

    # Ensure cycle
    if "cycle" not in df.columns:
        df["cycle"] = np.arange(len(df))

    # Derive SOH from capacity if missing
    if "soh" not in df.columns and "capacity" in df.columns:
        mx = df["capacity"].max() or 1.0
        df["soh"] = np.clip(df["capacity"] / (mx + 1e-9), 0, 1)

    # Derive RUL (years) if missing
    if "rul" not in df.columns:
        max_cycle = df["cycle"].max()
        df["rul"] = np.clip((max_cycle - df["cycle"]) / 365.0, 0, None)

    # Always Li-ion — force consistent category
    df["battery_type"] = "Li-ion"

    # Clean & drop impossible rows
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["voltage", "current", "temperature", "cycle"])

    # Standard target names
    df.rename(columns={"soh": "SOH", "rul": "RUL"}, inplace=True)

    # RUL in cycles (always required for 3rd output head)
    max_cycle = df["cycle"].max()
    df["RUL_Cycles"] = (max_cycle - df["cycle"]).clip(lower=0.0)

    # Fill missing capacity
    if "capacity" not in df.columns:
        df["capacity"] = 1.0

    return df


def load_all_csvs(data_dir: str | os.PathLike) -> pd.DataFrame:
    p = Path(data_dir)
    frames = []

    print("\n[Data] Loading datasets:")
    for f in sorted(p.glob("*.csv")):
        print("  -", f.name)
        try:
            df = pd.read_csv(f)
            df["__source__"] = f.name
            frames.append(_canonicalize(df))
        except Exception as e:
            print(f"    [Skipped] {f.name}: {e}")

    if not frames:
        raise FileNotFoundError(f"No CSV files found in {p.resolve()}")

    df = pd.concat(frames, ignore_index=True)
    return df


def split_Xy(df: pd.DataFrame):
    base_feats = ["voltage", "current", "temperature", "cycle", "capacity"]

    X = df[base_feats + ["battery_type"]].copy()
    X = pd.get_dummies(X, columns=["battery_type"], drop_first=False)

    y = df[["SOH", "RUL", "RUL_Cycles"]].copy()
    return X, y
