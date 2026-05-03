"""
Frontend-ready helpers for the indoor positioning demo.

This module wraps the notebook-style evaluation pipeline into clean,
presentation-friendly data structures that Streamlit can consume directly.
Only Scenarios 1 and 2 are exposed for the final demo because they support the
project claim that WiFi+BLE fusion reduces localization error.
"""

from __future__ import annotations

import contextlib
import io
from functools import lru_cache
from math import sqrt
from typing import Any

import numpy as np

from functions.fusion import prepare_both_datasets
from functions.models import compare_results, run_full_evaluation


DEMO_SCENARIOS = (1, 2)
MODEL_NAMES = ("KNN", "SVR", "Random Forest")


def _positioning_score(mean_error: float, room_diagonal: float) -> float:
    """
    Converts distance error into a normalized presentation score.

    This is not classification accuracy. It is a bounded score where 100 means
    zero localization error and lower values mean larger position error relative
    to the room size.
    """
    if room_diagonal <= 0:
        return 0.0
    score = 100.0 * (1.0 - (mean_error / room_diagonal))
    return float(np.clip(score, 0.0, 100.0))


def _room_metadata(*frames) -> dict[str, float]:
    xs = np.concatenate([frame["x"].to_numpy(dtype=float) for frame in frames])
    ys = np.concatenate([frame["y"].to_numpy(dtype=float) for frame in frames])

    min_x = float(np.min(xs))
    max_x = float(np.max(xs))
    min_y = float(np.min(ys))
    max_y = float(np.max(ys))
    width = max_x - min_x
    height = max_y - min_y

    return {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "width": float(width),
        "height": float(height),
        "diagonal": float(sqrt(width**2 + height**2)),
    }


def _metric_rows(
    wifi_eval: dict[str, Any],
    fused_eval: dict[str, Any],
    room_diagonal: float,
) -> list[dict[str, Any]]:
    rows = []

    for model_name in MODEL_NAMES:
        wifi_error = float(wifi_eval["results"][model_name]["mean_error"])
        fused_error = float(fused_eval["results"][model_name]["mean_error"])
        improvement = wifi_error - fused_error
        improvement_pct = (improvement / wifi_error * 100.0) if wifi_error else 0.0

        rows.append(
            {
                "model": model_name,
                "wifi_error_m": wifi_error,
                "fused_error_m": fused_error,
                "improvement_m": float(improvement),
                "improvement_pct": float(improvement_pct),
                "wifi_positioning_score": _positioning_score(wifi_error, room_diagonal),
                "fused_positioning_score": _positioning_score(fused_error, room_diagonal),
                "better_approach": "WiFi+BLE Fused" if improvement > 0 else "WiFi-Only",
            }
        )

    return rows


def _prediction_rows(data: dict[str, Any], eval_result: dict[str, Any], approach: str):
    actual = data["test"][["x", "y"]].to_numpy(dtype=float)
    locations = data["test"]["location"].tolist()
    rows = []

    for model_name in MODEL_NAMES:
        model_result = eval_result["results"][model_name]
        predictions = model_result["y_pred"]
        errors = model_result["errors"]

        for index, ((actual_x, actual_y), (pred_x, pred_y), error, location) in enumerate(
            zip(actual, predictions, errors, locations)
        ):
            rows.append(
                {
                    "approach": approach,
                    "model": model_name,
                    "test_point": index + 1,
                    "location": location,
                    "actual_x": float(actual_x),
                    "actual_y": float(actual_y),
                    "predicted_x": float(pred_x),
                    "predicted_y": float(pred_y),
                    "error_m": float(error),
                }
            )

    return rows


@lru_cache(maxsize=len(DEMO_SCENARIOS))
def run_scenario(scenario: int) -> dict[str, Any]:
    """
    Runs the complete WiFi-only vs WiFi+BLE evaluation for one demo scenario.

    Returns JSON-like dictionaries/lists so Streamlit can render metric cards,
    charts, and actual-vs-predicted position simulations without touching the
    notebook.
    """
    if scenario not in DEMO_SCENARIOS:
        allowed = ", ".join(str(item) for item in DEMO_SCENARIOS)
        raise ValueError(f"Scenario {scenario} is not part of the final demo. Use: {allowed}.")

    # The lower-level research helpers are intentionally chatty for notebooks.
    # Suppress that output here so a frontend can control its own presentation.
    with contextlib.redirect_stdout(io.StringIO()):
        wifi_data, fused_data = prepare_both_datasets(scenario=scenario)
        wifi_eval = run_full_evaluation(wifi_data, dataset_name=f"Scenario {scenario} WiFi-Only")
        fused_eval = run_full_evaluation(fused_data, dataset_name=f"Scenario {scenario} WiFi+BLE Fused")
        comparison = compare_results(wifi_eval, fused_eval)

    room = _room_metadata(
        wifi_data["db"],
        wifi_data["test"],
        fused_data["db"],
        fused_data["test"],
    )

    return {
        "scenario": scenario,
        "task": wifi_eval["task"],
        "metric_name": "Mean Distance Error",
        "metric_unit": "m",
        "score_name": "Positioning Score",
        "score_note": (
            "Positioning Score is normalized from mean distance error and room size; "
            "it is not classification accuracy."
        ),
        "room": room,
        "metrics": _metric_rows(wifi_eval, fused_eval, room["diagonal"]),
        "predictions": [
            *_prediction_rows(wifi_data, wifi_eval, "WiFi-Only"),
            *_prediction_rows(fused_data, fused_eval, "WiFi+BLE Fused"),
        ],
        "comparison_table": comparison.to_dict(orient="records"),
    }


def run_demo_scenarios() -> list[dict[str, Any]]:
    """Runs all scenarios included in the final demo."""
    return [run_scenario(scenario) for scenario in DEMO_SCENARIOS]
