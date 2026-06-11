from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd


def ensure_parent_dir(path):
    """Create the parent directory for a file if needed and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_attack_day_range(attack_day_start, attack_day_end, *, inclusive = True):
    """Return a dense list of attack-day indices."""
    start = int(attack_day_start)
    end = int(attack_day_end)
    stop = end + 1 if inclusive else end
    if stop < start:
        return []
    return list(range(start, stop))


def coerce_attack_days(attack_days):
    """Normalize an iterable of attack days into a sorted list of integers."""
    if attack_days is None:
        return []
    return sorted({int(day) for day in attack_days})


def _normalize_key_value(value):
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def event_row_key(row, key_columns):
    """Build a stable string key for a row based on the requested columns."""
    return tuple(_normalize_key_value(row.get(column)) for column in key_columns)


def load_existing_event_keys(csv_path, key_columns):
    """Load the keys already present in a CSV event log."""
    path = Path(csv_path)
    if not path.exists() or path.stat().st_size == 0:
        return set()

    existing = pd.read_csv(path, usecols=lambda col: col in set(key_columns), dtype=str)
    if existing.empty:
        return set()

    return {
        tuple(_normalize_key_value(row[column]) for column in key_columns)
        for _, row in existing.iterrows()
    }


def load_existing_event_log(csv_path):
    """Load an existing event log if present."""
    path = Path(csv_path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path)


def append_dataframe_to_csv(df, csv_path):
    """Append a dataframe to a CSV file, creating it if necessary."""
    path = ensure_parent_dir(csv_path)
    if df.empty:
        return path

    write_header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode="a", index=False, header=write_header)
    return path


def flush_rows_buffer(rows, csv_path, *, columns = None):
    """Persist buffered rows to disk and clear the buffer in-place."""
    if not rows:
        return None

    df = pd.DataFrame(rows)
    if columns is not None:
        df = df.reindex(columns=list(columns))
    path = append_dataframe_to_csv(df, csv_path)
    rows.clear()
    return path


def normalize_dataframe_columns(df, columns):
    """Return a copy of ``df`` aligned to a stable column order."""
    return df.reindex(columns=list(columns))
