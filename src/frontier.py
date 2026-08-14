"""Trace and validate the long-only analytical efficient frontier."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.optimize import (
    OptimizationResult,
    load_optimization_inputs,
    solve_minimum_variance,
    solve_target_return_minimum_variance,
)


@dataclass(frozen=True)
class FrontierDiagnostics:
    """Store numerical checks for the solved efficient frontier."""

    requested_points: int
    solved_points: int
    returns_monotonic: bool
    volatility_monotonic: bool
    convex: bool
    minimum_return: float
    maximum_return: float
    minimum_volatility: float
    maximum_volatility: float


@dataclass(frozen=True)
class FrontierResult:
    """Store frontier statistics, weights, failures, and numerical checks."""

    portfolios: pd.DataFrame
    weights: pd.DataFrame
    failed_targets: tuple[float, ...]
    diagnostics: FrontierDiagnostics


def diagnose_frontier(
    portfolios: pd.DataFrame,
    requested_points: int,
    monotonicity_tolerance: float,
    convexity_tolerance: float,
) -> FrontierDiagnostics:
    """Check that the upper frontier is ordered, risk-increasing, and convex."""

    required_columns = {"expected_return", "volatility"}
    if not required_columns.issubset(portfolios.columns):
        missing = sorted(required_columns - set(portfolios.columns))
        raise ValueError(f"Frontier statistics are missing columns: {missing}")
    if len(portfolios) < 3:
        raise ValueError("At least three solved portfolios are needed for checks.")

    expected_returns = portfolios["expected_return"].to_numpy(dtype=float)
    volatilities = portfolios["volatility"].to_numpy(dtype=float)
    if not np.isfinite(expected_returns).all() or not np.isfinite(volatilities).all():
        raise ValueError("Frontier statistics contain non-finite values.")

    return_changes = np.diff(expected_returns)
    volatility_changes = np.diff(volatilities)
    returns_monotonic = bool(
        np.all(return_changes >= -monotonicity_tolerance)
    )
    volatility_monotonic = bool(
        np.all(volatility_changes >= -monotonicity_tolerance)
    )

    if np.any(return_changes <= 0.0):
        convex = False
    else:
        frontier_slopes = volatility_changes / return_changes
        convex = bool(
            np.all(np.diff(frontier_slopes) >= -convexity_tolerance)
        )

    return FrontierDiagnostics(
        requested_points=requested_points,
        solved_points=len(portfolios),
        returns_monotonic=returns_monotonic,
        volatility_monotonic=volatility_monotonic,
        convex=convex,
        minimum_return=float(expected_returns.min()),
        maximum_return=float(expected_returns.max()),
        minimum_volatility=float(volatilities.min()),
        maximum_volatility=float(volatilities.max()),
    )


def _frontier_targets(
    minimum_variance: OptimizationResult,
    expected_returns: pd.Series,
    frontier_config: dict[str, Any],
) -> np.ndarray:
    """Create the configured target-return grid for the efficient upper branch."""

    if frontier_config["target_return_range"] != (
        "minimum_variance_to_maximum_asset"
    ):
        raise ValueError(
            "Unsupported frontier target range: "
            f"{frontier_config['target_return_range']!r}."
        )
    number_of_points = frontier_config["number_of_points"]
    if number_of_points < 3:
        raise ValueError("The frontier requires at least three target points.")

    lower_target = minimum_variance.expected_return
    upper_target = float(expected_returns.max())
    if upper_target <= lower_target:
        raise ValueError("The frontier target range is empty.")
    return np.linspace(lower_target, upper_target, number_of_points)


def trace_efficient_frontier(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    config: dict[str, Any],
) -> FrontierResult:
    """Solve target-return portfolios and return the validated efficient frontier."""

    frontier_config = config["frontier"]
    minimum_variance = solve_minimum_variance(
        expected_returns,
        covariance,
        config,
    )
    targets = _frontier_targets(
        minimum_variance,
        expected_returns,
        frontier_config,
    )

    statistics: list[dict[str, float]] = []
    weight_rows: list[pd.Series] = []
    failed_targets: list[float] = []
    previous_weights = minimum_variance.weights

    for point_number, target in enumerate(targets):
        try:
            result = (
                minimum_variance
                if point_number == 0
                else solve_target_return_minimum_variance(
                    expected_returns,
                    covariance,
                    float(target),
                    config,
                    initial_weights=previous_weights,
                )
            )
        except (RuntimeError, ValueError):
            failed_targets.append(float(target))
            if not frontier_config["skip_failed_targets"]:
                raise
            continue

        statistics.append(
            {
                "target_return": float(target),
                "expected_return": result.expected_return,
                "volatility": result.volatility,
                "variance": result.volatility**2,
                "sharpe_ratio": result.sharpe_ratio,
            }
        )
        weight_rows.append(result.weights)
        previous_weights = result.weights

    portfolios = pd.DataFrame(statistics)
    portfolios.index.name = "frontier_point"
    weights = pd.DataFrame(weight_rows)
    weights.index = portfolios.index
    weights.columns.name = "Ticker"

    diagnostics = diagnose_frontier(
        portfolios,
        frontier_config["number_of_points"],
        frontier_config["monotonicity_tolerance"],
        frontier_config["convexity_tolerance"],
    )
    return FrontierResult(
        portfolios=portfolios,
        weights=weights,
        failed_targets=tuple(failed_targets),
        diagnostics=diagnostics,
    )


def load_frontier_data(
    config_path: str | Path = "config.yaml",
) -> FrontierResult:
    """Load optimization inputs and trace the configured efficient frontier."""

    config, expected_returns, covariance = load_optimization_inputs(config_path)
    return trace_efficient_frontier(expected_returns, covariance, config)


def main() -> None:
    """Run the frontier sweep and print all Phase 4 gate checks."""

    parser = argparse.ArgumentParser(
        description="Trace and validate the analytical efficient frontier."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    arguments = parser.parse_args()

    result = load_frontier_data(arguments.config)
    diagnostics = result.diagnostics
    weight_sums = result.weights.sum(axis=1)

    print("Efficient frontier completed successfully.")
    print(f"Requested points: {diagnostics.requested_points}")
    print(f"Solved points: {diagnostics.solved_points}")
    print(f"Failed targets: {len(result.failed_targets)}")
    print(f"Returns monotonic: {diagnostics.returns_monotonic}")
    print(f"Volatility monotonic: {diagnostics.volatility_monotonic}")
    print(f"Frontier convex: {diagnostics.convex}")
    print(
        "Return range: "
        f"{diagnostics.minimum_return:.6f} to "
        f"{diagnostics.maximum_return:.6f}"
    )
    print(
        "Volatility range: "
        f"{diagnostics.minimum_volatility:.6f} to "
        f"{diagnostics.maximum_volatility:.6f}"
    )
    print(f"Minimum frontier weight: {result.weights.min().min():.12f}")
    print(f"Maximum frontier weight: {result.weights.max().max():.12f}")
    print(f"Minimum weight sum: {weight_sums.min():.12f}")
    print(f"Maximum weight sum: {weight_sums.max():.12f}")


if __name__ == "__main__":
    main()
