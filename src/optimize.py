"""Solve constrained portfolio-optimization problems with SLSQP."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.covariance import CovarianceEstimates, load_covariance_data
from src.data import load_config


@dataclass(frozen=True)
class OptimizationResult:
    """Store one validated portfolio-optimization solution."""

    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    objective_value: float
    iterations: int


def portfolio_return(
    weights: np.ndarray,
    expected_returns: np.ndarray,
) -> float:
    """Return the portfolio's annual expected return."""

    return float(np.dot(weights, expected_returns))


def portfolio_variance(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Return the portfolio's annual variance."""

    return float(weights @ covariance @ weights)


def portfolio_volatility(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Return annual portfolio volatility as the square root of variance."""

    variance = portfolio_variance(weights, covariance)
    if variance < 0.0:
        raise ValueError(f"Portfolio variance cannot be negative: {variance}")
    return float(np.sqrt(variance))


def portfolio_sharpe_ratio(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    risk_free_rate: float,
) -> float:
    """Return annual excess return per unit of annual volatility."""

    volatility = portfolio_volatility(weights, covariance)
    if volatility == 0.0:
        raise ValueError("Sharpe ratio is undefined at zero volatility.")
    excess_return = portfolio_return(weights, expected_returns) - risk_free_rate
    return excess_return / volatility


def _validate_inputs(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    """Validate labels, dimensions, and numeric values used by the optimizer."""

    if expected_returns.empty:
        raise ValueError("Expected returns cannot be empty.")
    if covariance.shape != (len(expected_returns), len(expected_returns)):
        raise ValueError(
            "Covariance dimensions do not match expected returns: "
            f"{covariance.shape} versus {len(expected_returns)} assets."
        )
    if list(covariance.index) != list(expected_returns.index):
        raise ValueError("Covariance row labels must match expected-return labels.")
    if list(covariance.columns) != list(expected_returns.index):
        raise ValueError("Covariance column labels must match expected-return labels.")
    if not np.isfinite(expected_returns.to_numpy(dtype=float)).all():
        raise ValueError("Expected returns contain non-finite values.")
    if not np.isfinite(covariance.to_numpy(dtype=float)).all():
        raise ValueError("Covariance contains non-finite values.")


def _make_initial_weights(
    asset_count: int,
    initial_weights: pd.Series | np.ndarray | None,
    configured_method: str,
) -> np.ndarray:
    """Return initial weights from a supplied portfolio or configured method."""

    if initial_weights is not None:
        values = np.asarray(initial_weights, dtype=float)
        if values.shape != (asset_count,):
            raise ValueError(
                f"Initial weights must have shape {(asset_count,)}, "
                f"found {values.shape}."
            )
        return values.copy()

    if configured_method != "equal":
        raise ValueError(
            f"Unsupported initial-weight method: {configured_method!r}."
        )
    return np.full(asset_count, 1.0 / asset_count)


def _make_bounds(
    optimization_config: dict[str, Any],
    asset_count: int,
) -> list[tuple[float, float]]:
    """Create one configured lower and upper bound pair per asset."""

    lower = float(optimization_config["weight_bounds"]["minimum"])
    upper = float(optimization_config["weight_bounds"]["maximum"])
    if lower >= upper:
        raise ValueError("The minimum weight bound must be below the maximum.")
    if not optimization_config["allow_shorting"] and lower < 0.0:
        raise ValueError("Long-only optimization cannot use a negative lower bound.")
    if lower * asset_count > 1.0 or upper * asset_count < 1.0:
        raise ValueError("Configured weight bounds cannot satisfy sum(weights) = 1.")
    return [(lower, upper) for _ in range(asset_count)]


def _sector_indices(
    asset_names: pd.Index,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Map each configured sector to positions in the optimizer's asset order."""

    position_by_ticker = {
        str(ticker): position for position, ticker in enumerate(asset_names)
    }
    result: dict[str, np.ndarray] = {}
    for sector, tickers in config["universe"]["sectors"].items():
        missing = [ticker for ticker in tickers if ticker not in position_by_ticker]
        if missing:
            raise ValueError(f"Sector {sector!r} contains missing assets: {missing}")
        result[sector] = np.array(
            [position_by_ticker[ticker] for ticker in tickers],
            dtype=int,
        )
    return result


def _make_constraints(
    expected_returns: np.ndarray,
    asset_names: pd.Index,
    config: dict[str, Any],
    target_return: float | None,
) -> list[dict[str, Any]]:
    """Build sum-to-one, optional target-return, and optional sector constraints."""

    constraints: list[dict[str, Any]] = [
        {
            "type": "eq",
            "fun": lambda weights: float(np.sum(weights) - 1.0),
        }
    ]
    if target_return is not None:
        constraints.append(
            {
                "type": "eq",
                "fun": lambda weights: portfolio_return(
                    weights,
                    expected_returns,
                )
                - target_return,
            }
        )

    sector_cap = config["optimization"]["sector_cap"]
    if sector_cap is not None:
        cap = float(sector_cap)
        if not 0.0 < cap <= 1.0:
            raise ValueError("sector_cap must be in the interval (0, 1].")
        for indices in _sector_indices(asset_names, config).values():
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda weights, positions=indices: cap
                    - float(np.sum(weights[positions])),
                }
            )
    return constraints


def _validate_solution(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    asset_names: pd.Index,
    config: dict[str, Any],
    target_return: float | None,
) -> None:
    """Reject a solver result that violates any configured constraint."""

    tolerance = config["optimization"]["solver"]["constraint_tolerance"]
    lower = config["optimization"]["weight_bounds"]["minimum"]
    upper = config["optimization"]["weight_bounds"]["maximum"]

    if abs(float(weights.sum()) - 1.0) > tolerance:
        raise RuntimeError(f"Optimized weights sum to {weights.sum()}, not 1.")
    if float(weights.min()) < lower - tolerance:
        raise RuntimeError("An optimized weight is below its lower bound.")
    if float(weights.max()) > upper + tolerance:
        raise RuntimeError("An optimized weight is above its upper bound.")
    if target_return is not None:
        achieved = portfolio_return(weights, expected_returns)
        if abs(achieved - target_return) > tolerance:
            raise RuntimeError(
                f"Target return {target_return} was not achieved; got {achieved}."
            )

    sector_cap = config["optimization"]["sector_cap"]
    if sector_cap is not None:
        for sector, indices in _sector_indices(asset_names, config).items():
            allocation = float(weights[indices].sum())
            if allocation > sector_cap + tolerance:
                raise RuntimeError(
                    f"Sector {sector!r} exceeds its cap: {allocation}."
                )


def _solve(
    objective: Callable[[np.ndarray], float],
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    config: dict[str, Any],
    target_return: float | None = None,
    initial_weights: pd.Series | np.ndarray | None = None,
) -> OptimizationResult:
    """Run SLSQP and return a fully validated optimization result."""

    _validate_inputs(expected_returns, covariance)
    optimization_config = config["optimization"]
    solver_config = optimization_config["solver"]
    expected_values = expected_returns.to_numpy(dtype=float)
    covariance_values = covariance.to_numpy(dtype=float)
    initial = _make_initial_weights(
        len(expected_returns),
        initial_weights,
        optimization_config["initial_weights"],
    )
    bounds = _make_bounds(optimization_config, len(expected_returns))
    constraints = _make_constraints(
        expected_values,
        expected_returns.index,
        config,
        target_return,
    )

    solver_result = minimize(
        objective,
        initial,
        method=solver_config["method"],
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": solver_config["maximum_iterations"],
            "ftol": solver_config["function_tolerance"],
            "disp": False,
        },
    )
    if not solver_result.success:
        raise RuntimeError(
            f"{solver_config['method']} failed: {solver_result.message}"
        )

    weights = np.asarray(solver_result.x, dtype=float)
    _validate_solution(
        weights,
        expected_values,
        expected_returns.index,
        config,
        target_return,
    )
    expected_return = portfolio_return(weights, expected_values)
    volatility = portfolio_volatility(weights, covariance_values)
    sharpe_ratio = portfolio_sharpe_ratio(
        weights,
        expected_values,
        covariance_values,
        optimization_config["risk_free_rate"],
    )
    return OptimizationResult(
        weights=pd.Series(weights, index=expected_returns.index, name="weight"),
        expected_return=expected_return,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        objective_value=float(solver_result.fun),
        iterations=int(solver_result.nit),
    )


def solve_minimum_variance(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    config: dict[str, Any],
    initial_weights: pd.Series | np.ndarray | None = None,
) -> OptimizationResult:
    """Find the feasible portfolio with the lowest annual variance."""

    covariance_values = covariance.to_numpy(dtype=float)
    objective = lambda weights: portfolio_variance(weights, covariance_values)
    return _solve(
        objective,
        expected_returns,
        covariance,
        config,
        initial_weights=initial_weights,
    )


def solve_maximum_sharpe(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    config: dict[str, Any],
    initial_weights: pd.Series | np.ndarray | None = None,
) -> OptimizationResult:
    """Find the feasible portfolio with the highest configured Sharpe ratio."""

    expected_values = expected_returns.to_numpy(dtype=float)
    covariance_values = covariance.to_numpy(dtype=float)
    risk_free_rate = config["optimization"]["risk_free_rate"]
    objective = lambda weights: -portfolio_sharpe_ratio(
        weights,
        expected_values,
        covariance_values,
        risk_free_rate,
    )
    return _solve(
        objective,
        expected_returns,
        covariance,
        config,
        initial_weights=initial_weights,
    )


def solve_target_return_minimum_variance(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    target_return: float,
    config: dict[str, Any],
    initial_weights: pd.Series | np.ndarray | None = None,
) -> OptimizationResult:
    """Minimize variance while requiring one specific annual target return."""

    tolerance = config["optimization"]["solver"]["constraint_tolerance"]
    if not config["optimization"]["allow_shorting"]:
        minimum = float(expected_returns.min())
        maximum = float(expected_returns.max())
        if target_return < minimum - tolerance or target_return > maximum + tolerance:
            raise ValueError(
                f"Target return {target_return} is outside the long-only "
                f"range [{minimum}, {maximum}]."
            )

    covariance_values = covariance.to_numpy(dtype=float)
    objective = lambda weights: portfolio_variance(weights, covariance_values)
    return _solve(
        objective,
        expected_returns,
        covariance,
        config,
        target_return=target_return,
        initial_weights=initial_weights,
    )


def select_covariance_estimate(
    estimates: CovarianceEstimates,
    estimator_name: str,
) -> pd.DataFrame:
    """Return one named covariance estimate from the estimator bundle."""

    if estimator_name == "sample":
        return estimates.sample
    if estimator_name == "ledoit_wolf":
        return estimates.ledoit_wolf
    raise ValueError(f"Unsupported covariance estimator: {estimator_name!r}")


def load_optimization_inputs(
    config_path: str | Path = "config.yaml",
) -> tuple[dict[str, Any], pd.Series, pd.DataFrame]:
    """Load configured expected returns and the optimization covariance matrix."""

    config = load_config(config_path)
    _, return_summary, estimates, _ = load_covariance_data(config_path)
    expected_returns = return_summary["annualized_mean_return"]
    covariance = select_covariance_estimate(
        estimates,
        config["covariance"]["optimization_estimator"],
    )
    return config, expected_returns, covariance


def _print_result(label: str, result: OptimizationResult) -> None:
    """Print one optimization result and its constraint checks."""

    print(f"{label}:")
    print(f"  Expected return: {result.expected_return:.6f}")
    print(f"  Volatility: {result.volatility:.6f}")
    print(f"  Sharpe ratio: {result.sharpe_ratio:.6f}")
    print(f"  Weight sum: {result.weights.sum():.12f}")
    print(f"  Minimum weight: {result.weights.min():.12f}")
    print(f"  Maximum weight: {result.weights.max():.12f}")
    print(f"  Solver iterations: {result.iterations}")


def main() -> None:
    """Run and validate the minimum-variance and maximum-Sharpe portfolios."""

    parser = argparse.ArgumentParser(
        description="Solve the core constrained portfolio problems."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    arguments = parser.parse_args()

    config, expected_returns, covariance = load_optimization_inputs(
        arguments.config
    )
    minimum_variance = solve_minimum_variance(
        expected_returns,
        covariance,
        config,
    )
    maximum_sharpe = solve_maximum_sharpe(
        expected_returns,
        covariance,
        config,
    )

    print("Portfolio optimization completed successfully.")
    _print_result("Minimum-variance portfolio", minimum_variance)
    _print_result("Maximum-Sharpe portfolio", maximum_sharpe)


if __name__ == "__main__":
    main()
