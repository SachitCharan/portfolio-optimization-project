"""Generate random portfolios and verify the analytical efficient frontier."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.frontier import FrontierResult, trace_efficient_frontier
from src.optimize import (
    load_optimization_inputs,
    solve_minimum_variance,
    solve_target_return_minimum_variance,
)


@dataclass(frozen=True)
class MonteCarloDiagnostics:
    """Store checks for the generated random portfolios."""

    requested_portfolios: int
    generated_portfolios: int
    asset_count: int
    random_seed: int
    minimum_weight: float
    maximum_weight: float
    minimum_weight_sum: float
    maximum_weight_sum: float


@dataclass(frozen=True)
class FrontierComparison:
    """Store the result of comparing random portfolios with the frontier."""

    exact_checks: int
    portfolios_beating_frontier: int
    minimum_conservative_volatility_gap: float

    @property
    def analytical_frontier_holds(self) -> bool:
        """Return whether every random portfolio lies on or behind the frontier."""

        return self.portfolios_beating_frontier == 0


@dataclass(frozen=True)
class MonteCarloResult:
    """Store random portfolio statistics, weights, and validation results."""

    portfolios: pd.DataFrame
    weights: pd.DataFrame
    diagnostics: MonteCarloDiagnostics
    frontier_comparison: FrontierComparison


def _validate_inputs(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    """Validate the labels and numeric inputs used by the simulation."""

    if expected_returns.empty:
        raise ValueError("Expected returns cannot be empty.")
    if covariance.shape != (len(expected_returns), len(expected_returns)):
        raise ValueError("Covariance dimensions do not match expected returns.")
    if list(covariance.index) != list(expected_returns.index):
        raise ValueError("Covariance row labels must match expected returns.")
    if list(covariance.columns) != list(expected_returns.index):
        raise ValueError("Covariance column labels must match expected returns.")
    if not np.isfinite(expected_returns.to_numpy(dtype=float)).all():
        raise ValueError("Expected returns contain non-finite values.")
    if not np.isfinite(covariance.to_numpy(dtype=float)).all():
        raise ValueError("Covariance contains non-finite values.")


def _validate_simulation_settings(config: dict[str, Any]) -> None:
    """Reject settings that are incompatible with Dirichlet long-only weights."""

    monte_carlo_config = config["monte_carlo"]
    optimization_config = config["optimization"]
    if monte_carlo_config["number_of_portfolios"] <= 0:
        raise ValueError("number_of_portfolios must be positive.")
    if monte_carlo_config["weight_distribution"] != "dirichlet":
        raise ValueError("Only Dirichlet random weights are supported.")
    if monte_carlo_config["dirichlet_alpha"] <= 0.0:
        raise ValueError("dirichlet_alpha must be positive.")
    if optimization_config["allow_shorting"]:
        raise ValueError("Dirichlet weights cannot represent short positions.")
    if optimization_config["weight_bounds"] != {
        "minimum": 0.0,
        "maximum": 1.0,
    }:
        raise ValueError(
            "Dirichlet simulation requires weight bounds from 0 to 1."
        )
    if optimization_config["sector_cap"] is not None:
        raise ValueError(
            "Random portfolio generation with sector caps is not configured."
        )


def generate_random_weights(
    number_of_portfolios: int,
    asset_count: int,
    alpha: float,
    random_seed: int,
) -> np.ndarray:
    """Draw reproducible long-only weights that each sum to one."""

    if number_of_portfolios <= 0 or asset_count <= 0:
        raise ValueError("Portfolio and asset counts must be positive.")
    if alpha <= 0.0:
        raise ValueError("Dirichlet alpha must be positive.")

    generator = np.random.default_rng(random_seed)
    concentrations = np.full(asset_count, alpha, dtype=float)
    return generator.dirichlet(concentrations, size=number_of_portfolios)


def simulate_random_portfolios(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, MonteCarloDiagnostics]:
    """Generate random weights and calculate annual portfolio statistics."""

    _validate_inputs(expected_returns, covariance)
    _validate_simulation_settings(config)
    monte_carlo_config = config["monte_carlo"]
    number_of_portfolios = int(monte_carlo_config["number_of_portfolios"])
    random_seed = int(config["project"]["random_seed"])
    asset_count = len(expected_returns)

    weight_values = generate_random_weights(
        number_of_portfolios,
        asset_count,
        float(monte_carlo_config["dirichlet_alpha"]),
        random_seed,
    )
    expected_values = expected_returns.to_numpy(dtype=float)
    covariance_values = covariance.to_numpy(dtype=float)
    portfolio_returns = weight_values @ expected_values
    portfolio_variances = np.einsum(
        "ij,jk,ik->i",
        weight_values,
        covariance_values,
        weight_values,
    )
    if np.any(portfolio_variances < 0.0):
        raise ValueError("A random portfolio has negative variance.")
    portfolio_volatilities = np.sqrt(portfolio_variances)
    risk_free_rate = float(config["optimization"]["risk_free_rate"])
    sharpe_ratios = (
        portfolio_returns - risk_free_rate
    ) / portfolio_volatilities

    portfolios = pd.DataFrame(
        {
            "expected_return": portfolio_returns,
            "volatility": portfolio_volatilities,
            "variance": portfolio_variances,
            "sharpe_ratio": sharpe_ratios,
        }
    )
    portfolios.index.name = "portfolio"
    weights = pd.DataFrame(weight_values, columns=expected_returns.index)
    weights.index = portfolios.index
    weights.columns.name = "Ticker"

    weight_sums = weights.sum(axis=1)
    tolerance = float(config["validation"]["weight_sum_tolerance"])
    if not np.isfinite(portfolios.to_numpy(dtype=float)).all():
        raise ValueError("Random portfolio statistics contain non-finite values.")
    if float(np.max(np.abs(weight_sums.to_numpy() - 1.0))) > tolerance:
        raise RuntimeError("At least one random portfolio does not sum to one.")

    diagnostics = MonteCarloDiagnostics(
        requested_portfolios=number_of_portfolios,
        generated_portfolios=len(portfolios),
        asset_count=asset_count,
        random_seed=random_seed,
        minimum_weight=float(weights.min().min()),
        maximum_weight=float(weights.max().max()),
        minimum_weight_sum=float(weight_sums.min()),
        maximum_weight_sum=float(weight_sums.max()),
    )
    return portfolios, weights, diagnostics


def compare_with_analytical_frontier(
    portfolios: pd.DataFrame,
    weights: pd.DataFrame,
    frontier: FrontierResult,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    config: dict[str, Any],
) -> FrontierComparison:
    """Confirm that no random portfolio improves on the analytical frontier."""

    if not frontier.diagnostics.convex:
        raise RuntimeError("The analytical frontier must be convex before comparison.")
    if not frontier.diagnostics.returns_monotonic:
        raise RuntimeError("The analytical frontier returns must be monotonic.")
    if not frontier.diagnostics.volatility_monotonic:
        raise RuntimeError("The analytical frontier volatility must be monotonic.")
    if list(weights.columns) != list(expected_returns.index):
        raise ValueError("Random weight labels do not match expected returns.")

    random_returns = portfolios["expected_return"].to_numpy(dtype=float)
    random_volatilities = portfolios["volatility"].to_numpy(dtype=float)
    frontier_returns = frontier.portfolios["expected_return"].to_numpy(dtype=float)
    frontier_volatilities = frontier.portfolios["volatility"].to_numpy(dtype=float)
    tolerance = float(
        config["monte_carlo"]["frontier_comparison_tolerance"]
    )

    minimum_variance = solve_minimum_variance(
        expected_returns,
        covariance,
        config,
    )
    reference_volatilities = np.full(
        len(portfolios),
        minimum_variance.volatility,
        dtype=float,
    )
    efficient_branch = random_returns >= minimum_variance.expected_return
    reference_volatilities[efficient_branch] = np.interp(
        random_returns[efficient_branch],
        frontier_returns,
        frontier_volatilities,
    )
    conservative_gaps = random_volatilities - reference_volatilities

    lower_branch_violations = (
        (~efficient_branch)
        & (random_volatilities < minimum_variance.volatility - tolerance)
    )
    possible_upper_branch_violations = (
        efficient_branch & (conservative_gaps < tolerance)
    )
    possible_indices = np.flatnonzero(possible_upper_branch_violations)

    confirmed_violations = int(np.count_nonzero(lower_branch_violations))
    for index in possible_indices:
        exact_frontier_portfolio = solve_target_return_minimum_variance(
            expected_returns,
            covariance,
            float(random_returns[index]),
            config,
            initial_weights=weights.iloc[index],
        )
        if random_volatilities[index] < (
            exact_frontier_portfolio.volatility - tolerance
        ):
            confirmed_violations += 1

    return FrontierComparison(
        exact_checks=len(possible_indices),
        portfolios_beating_frontier=confirmed_violations,
        minimum_conservative_volatility_gap=float(conservative_gaps.min()),
    )


def run_monte_carlo(
    config_path: str | Path = "config.yaml",
) -> MonteCarloResult:
    """Load real-data estimates, simulate portfolios, and run the Phase 5 gate."""

    config, expected_returns, covariance = load_optimization_inputs(config_path)
    portfolios, weights, diagnostics = simulate_random_portfolios(
        expected_returns,
        covariance,
        config,
    )
    frontier = trace_efficient_frontier(expected_returns, covariance, config)
    comparison = compare_with_analytical_frontier(
        portfolios,
        weights,
        frontier,
        expected_returns,
        covariance,
        config,
    )
    if not comparison.analytical_frontier_holds:
        raise RuntimeError(
            f"{comparison.portfolios_beating_frontier} random portfolios "
            "beat the analytical frontier."
        )
    return MonteCarloResult(
        portfolios=portfolios,
        weights=weights,
        diagnostics=diagnostics,
        frontier_comparison=comparison,
    )


def main() -> None:
    """Run the simulation and print every Phase 5 gate check."""

    parser = argparse.ArgumentParser(
        description="Generate random portfolios and validate the frontier."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    arguments = parser.parse_args()

    result = run_monte_carlo(arguments.config)
    diagnostics = result.diagnostics
    comparison = result.frontier_comparison

    print("Monte Carlo simulation completed successfully.")
    print(f"Random seed: {diagnostics.random_seed}")
    print(f"Requested portfolios: {diagnostics.requested_portfolios}")
    print(f"Generated portfolios: {diagnostics.generated_portfolios}")
    print(f"Assets per portfolio: {diagnostics.asset_count}")
    print(f"Minimum random weight: {diagnostics.minimum_weight:.12f}")
    print(f"Maximum random weight: {diagnostics.maximum_weight:.12f}")
    print(f"Minimum weight sum: {diagnostics.minimum_weight_sum:.12f}")
    print(f"Maximum weight sum: {diagnostics.maximum_weight_sum:.12f}")
    print(f"Exact frontier checks required: {comparison.exact_checks}")
    print(
        "Minimum conservative volatility gap: "
        f"{comparison.minimum_conservative_volatility_gap:.12f}"
    )
    print(
        "Random portfolios beating analytical frontier: "
        f"{comparison.portfolios_beating_frontier}"
    )
    print(
        "Analytical frontier remains the boundary: "
        f"{comparison.analytical_frontier_holds}"
    )


if __name__ == "__main__":
    main()
