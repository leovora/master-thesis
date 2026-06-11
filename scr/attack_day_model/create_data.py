from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from notebooks.targeted_experiment_utils import (
    TARGET_SIGNALS,
    evaluate_targeted_attack,
)

from scr.attack_day_model.utils import (
    build_attack_day_range,
    coerce_attack_days,
    flush_rows_buffer,
    load_existing_event_keys,
    normalize_dataframe_columns,
)


DEFAULT_EVENT_LOG_COLUMNS = [
    "setup",
    "model",
    "strategy",
    "assets",
    "attacked_ticker",
    "attack_day",
    "target_signal",
    "target_label",
    "baseline_signal",
    "baseline_label",
    "attacked_signal",
    "attacked_label",
    "success",
    "already_target",
    "valid",
    "min_delta_norm",
    "min_abs_delta_norm",
    "prediction_shift",
    "delta_final_cr",
    "delta_cumulative_return_in_period",
    "objective",
    "timing_policy",
]

EVENT_LOG_KEY_COLUMNS = [
    "setup",
    "model",
    "strategy",
    "attacked_ticker",
    "attack_day",
    "target_signal",
    "objective",
    "timing_policy",
]


def _resolve_attack_days(attack_days, attack_day_start, attack_day_end, *, inclusive = True):
    """Resolve the attack-day schedule from either an iterable or a range."""
    if attack_days is not None:
        return coerce_attack_days(attack_days)

    if attack_day_start is None or attack_day_end is None:
        raise ValueError(
            "Provide either attack_days or both attack_day_start and attack_day_end."
        )

    return build_attack_day_range(attack_day_start, attack_day_end, inclusive=inclusive)


def _setup_name(setup):
    if isinstance(setup, Mapping):
        return str(setup["name"])
    return str(getattr(setup, "name"))


def _iter_attack_jobs(setups, attack_days, target_signals):
    """Yield the evaluation jobs in the same order used by the notebook."""
    for setup in setups:
        for attack_day in attack_days:
            for objective_name, target_signal in target_signals.items():
                yield setup, int(attack_day), objective_name, int(target_signal)


def _job_key(setup, attack_day, objective_name, target_signal, timing_policy):
    """Build the stable key used to decide whether a job can be resumed."""
    return (
        _setup_name(setup),
        str(getattr(setup, "model_name", "")),
        str(getattr(setup, "strategy_name", "")),
        str(getattr(setup, "attacked_ticker", "")),
        str(int(attack_day)),
        str(int(target_signal)),
        str(objective_name),
        str(timing_policy),
    )


def _run_attack_event_log(
    jobs,
    *,
    total_jobs,
    model_folders,
    project_root,
    output_csv,
    target_signals,
    timing_policy,
    data_folder,
    sequence_length,
    max_abs_delta,
    steps,
    simulation_start,
    simulation_end,
    data_start,
    months,
    eval_cr_period,
    train_mode,
    train_window_months,
    ignore_already_target,
    resume,
    checkpoint_every,
    verbose
    ):
    """Shared implementation for the event-log builders."""
    output_path = Path(output_csv)
    existing_keys = set()
    if resume and output_path.exists() and output_path.stat().st_size > 0:
        existing_df = pd.read_csv(output_path)
        if ignore_already_target and "already_target" in existing_df.columns:
            keep_mask = ~existing_df["already_target"].astype(str).str.lower().isin(
                {"true", "1", "yes"}
            )
            existing_df = existing_df.loc[keep_mask].copy()
        existing_df = normalize_dataframe_columns(existing_df, DEFAULT_EVENT_LOG_COLUMNS)
        existing_df.to_csv(output_path, index=False)
        existing_keys = load_existing_event_keys(output_path, EVENT_LOG_KEY_COLUMNS)

    buffer = []
    all_rows = []
    completed_jobs = 0
    written_rows = 0
    skipped_rows = 0

    if verbose:
        print(
            f"Starting attack event-log generation: {total_jobs} trials "
            f"across the requested setups and days."
        )
        if resume and existing_keys:
            print(f"Resuming from {len(existing_keys)} already logged rows.")

    for setup, attack_day, objective_name, target_signal in jobs:
        row_key = _job_key(
            setup=setup,
            attack_day=attack_day,
            objective_name=objective_name,
            target_signal=target_signal,
            timing_policy=timing_policy,
        )

        if resume and row_key in existing_keys:
            completed_jobs += 1
            skipped_rows += 1
            if verbose and completed_jobs % max(1, checkpoint_every) == 0:
                print(
                    f"  {completed_jobs}/{total_jobs} processed, "
                    f"{written_rows} written, {skipped_rows} skipped (already saved)"
                )
            continue

        row = evaluate_targeted_attack(
            setup=setup,
            attack_day=attack_day,
            target_signal=target_signal,
            model_folders=model_folders,
            project_root=project_root,
            sequence_length=sequence_length,
            max_abs_delta=max_abs_delta,
            steps=steps,
            data_folder=data_folder,
            simulation_start=simulation_start,
            simulation_end=simulation_end,
            data_start=data_start,
            months=months,
            eval_cr_period=eval_cr_period,
            train_mode=train_mode,
            train_window_months=train_window_months,
        )
        row["objective"] = objective_name
        row["timing_policy"] = timing_policy

        completed_jobs += 1

        if ignore_already_target and bool(row.get("already_target", False)):
            skipped_rows += 1
            if verbose and completed_jobs % max(1, checkpoint_every) == 0:
                print(
                    f"  {completed_jobs}/{total_jobs} processed, "
                    f"{written_rows} written, {skipped_rows} skipped (already_target)"
                )
            continue

        buffer.append(row)
        all_rows.append(row)
        existing_keys.add(row_key)

        if len(buffer) >= max(1, checkpoint_every):
            flushed = len(buffer)
            flush_rows_buffer(buffer, output_path, columns=DEFAULT_EVENT_LOG_COLUMNS)
            written_rows += flushed
            if verbose:
                print(
                    f"  {completed_jobs}/{total_jobs} processed, "
                    f"{written_rows} written, {skipped_rows} skipped"
                )

    if buffer:
        flushed = len(buffer)
        flush_rows_buffer(buffer, output_path, columns=DEFAULT_EVENT_LOG_COLUMNS)
        written_rows += flushed

    if verbose:
        print(
            f"Completed. Written rows: {written_rows}. "
            f"Skipped rows: {skipped_rows}. Output: {output_path}"
        )

    if output_path.exists() and output_path.stat().st_size > 0:
        df = pd.read_csv(output_path)
        return normalize_dataframe_columns(df, DEFAULT_EVENT_LOG_COLUMNS)

    return pd.DataFrame(all_rows, columns=DEFAULT_EVENT_LOG_COLUMNS)


def create_attack_event_log(
    setups,
    model_folders,
    project_root,
    output_csv,
    attack_days = None,
    attack_day_start = None,
    attack_day_end = None,
    *,
    target_signals = None,
    timing_policy = "manual",
    data_folder = "data/LSTM_3_years",
    sequence_length = 50,
    max_abs_delta = 0.35,
    steps = 35,
    simulation_start = None,
    simulation_end = None,
    data_start = None,
    months = 3,
    eval_cr_period = 10,
    train_mode = "cumulative",
    train_window_months = 3,
    ignore_already_target = True,
    resume = True,
    checkpoint_every = 5,
    verbose = True,
    ):
    """
    Build an attack-day event log and save it incrementally to ``output_csv``.

    The log mirrors the row schema produced by ``evaluate_targeted_attack`` and
    drops rows where ``already_target`` is true.
    """
    target_signals = TARGET_SIGNALS if target_signals is None else target_signals
    attack_days = _resolve_attack_days(
        attack_days,
        attack_day_start,
        attack_day_end,
        inclusive=True,
    )
    total_jobs = len(setups) * len(attack_days) * len(target_signals)
    jobs = _iter_attack_jobs(setups, attack_days, target_signals)
    return _run_attack_event_log(
        jobs,
        total_jobs=total_jobs,
        model_folders=model_folders,
        project_root=project_root,
        output_csv=output_csv,
        target_signals=target_signals,
        timing_policy=timing_policy,
        data_folder=data_folder,
        sequence_length=sequence_length,
        max_abs_delta=max_abs_delta,
        steps=steps,
        simulation_start=simulation_start,
        simulation_end=simulation_end,
        data_start=data_start,
        months=months,
        eval_cr_period=eval_cr_period,
        train_mode=train_mode,
        train_window_months=train_window_months,
        ignore_already_target=ignore_already_target,
        resume=resume,
        checkpoint_every=checkpoint_every,
        verbose=verbose,
    )


def create_attack_event_log_for_plan(
    setups,
    model_folders,
    project_root,
    output_csv,
    attack_day_plan,
    *,
    target_signals = None,
    timing_policy = "manual",
    data_folder = "data/LSTM_3_years",
    sequence_length = 50,
    max_abs_delta = 0.35,
    steps = 35,
    simulation_start = None,
    simulation_end = None,
    data_start = None,
    months = 3,
    eval_cr_period = 10,
    train_mode = "cumulative",
    train_window_months = 3,
    ignore_already_target = True,
    resume = True,
    checkpoint_every = 1,
    verbose = True,
    ):
    """Variant that accepts a per-setup attack-day plan."""
    target_signals = TARGET_SIGNALS if target_signals is None else target_signals
    setup_by_name = {_setup_name(setup): setup for setup in setups}

    expanded_jobs = []
    for setup_name, days in attack_day_plan.items():
        setup = setup_by_name.get(setup_name)
        if setup is None:
            continue
        for attack_day in coerce_attack_days(days):
            for objective_name, target_signal in target_signals.items():
                expanded_jobs.append(
                    (
                        setup,
                        int(attack_day),
                        objective_name,
                        int(target_signal),
                    )
                )

    return _run_attack_event_log(
        expanded_jobs,
        total_jobs=len(expanded_jobs),
        model_folders=model_folders,
        project_root=project_root,
        output_csv=output_csv,
        target_signals=target_signals,
        timing_policy=timing_policy,
        data_folder=data_folder,
        sequence_length=sequence_length,
        max_abs_delta=max_abs_delta,
        steps=steps,
        simulation_start=simulation_start,
        simulation_end=simulation_end,
        data_start=data_start,
        months=months,
        eval_cr_period=eval_cr_period,
        train_mode=train_mode,
        train_window_months=train_window_months,
        ignore_already_target=ignore_already_target,
        resume=resume,
        checkpoint_every=checkpoint_every,
        verbose=verbose,
    )
