"""Automated mathematical and timing checks using real cached market data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestResult, load_backtest_data
from src.covariance import (
    CovarianceDiagnostics,
    CovarianceEstimates,
    estimate_covariances,
    load_covariance_data,
)
from src.data import get_universe_tickers, load_config, load_price_data
from src.frontier import FrontierResult, trace_efficient_frontier
from src.metrics import build_performance_summary
from src.montecarlo import MonteCarloResult, run_monte_carlo
from src.optimize import (
    load_optimization_inputs,
    select_covariance_estimate,
    solve_maximum_sharpe,
    solve_minimum_variance,
)
from src.returns import build_return_summary, compute_simple_returns


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIRECTORY / "config.yaml"


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    """Load the project configuration once for the test session."""

    return load_config(CONFIG_PATH)


@pytest.fixture(scope="session")
def covariance_bundle() -> tuple[
    CovarianceEstimates,
    dict[str, CovarianceDiagnostics],
]:
    """Estimate real-data covariance matrices once for all covariance tests."""

    _, _, estimates, diagnostics = load_covariance_data(CONFIG_PATH)
    return estimates, diagnostics


@pytest.fixture(scope="session")
def optimization_inputs() -> tuple[pd.Series, pd.DataFrame]:
    """Load full-sample expected returns and optimization covariance."""

    _, expected_returns, covariance = load_optimization_inputs(CONFIG_PATH)
    return expected_returns, covariance


@pytest.fixture(scope="session")
def frontier_result(
    config: dict[str, Any],
    optimization_inputs: tuple[pd.Series, pd.DataFrame],
) -> FrontierResult:
    """Trace the analytical frontier once for all frontier checks."""

    expected_returns, covariance = optimization_inputs
    return trace_efficient_frontier(expected_returns, covariance, config)


@pytest.fixture(scope="session")
def monte_carlo_result() -> MonteCarloResult:
    """Run the configured real-estimate Monte Carlo simulation once."""

    return run_monte_carlo(CONFIG_PATH)


@pytest.fixture(scope="session")
def backtest_result() -> BacktestResult:
    """Run the real-data out-of-sample backtest once."""

    return load_backtest_data(CONFIG_PATH)


def test_optimized_weights_respect_constraints(
    config: dict[str, Any],
    optimization_inputs: tuple[pd.Series, pd.DataFrame],
) -> None:
    """Both main optimized portfolios must satisfy long-only constraints."""

    expected_returns, covariance = optimization_inputs
    results = [
        solve_minimum_variance(expected_returns, covariance, config),
        solve_maximum_sharpe(expected_returns, covariance, config),
    ]
    tolerance = float(config["validation"]["weight_sum_tolerance"])
    for result in results:
        assert float(result.weights.sum()) == pytest.approx(1.0, abs=tolerance)
        assert float(result.weights.min()) >= -tolerance
        assert float(result.weights.max()) <= 1.0 + tolerance


def test_covariances_are_symmetric_positive_semidefinite_and_49_by_49(
    covariance_bundle: tuple[
        CovarianceEstimates,
        dict[str, CovarianceDiagnostics],
    ],
) -> None:
    """Both covariance estimators must pass every Phase 3 matrix check."""

    estimates, diagnostics = covariance_bundle
    assert estimates.sample.shape == (49, 49)
    assert estimates.ledoit_wolf.shape == (49, 49)
    for result in diagnostics.values():
        assert result.dimension == 49
        assert result.symmetric
        assert result.positive_semidefinite
    assert 0.0 <= estimates.shrinkage_intensity <= 1.0


def test_efficient_frontier_is_complete_monotonic_and_convex(
    config: dict[str, Any],
    frontier_result: FrontierResult,
) -> None:
    """The analytical sweep must solve every configured efficient point."""

    diagnostics = frontier_result.diagnostics
    assert diagnostics.requested_points == config["frontier"]["number_of_points"]
    assert diagnostics.solved_points == diagnostics.requested_points
    assert not frontier_result.failed_targets
    assert diagnostics.returns_monotonic
    assert diagnostics.volatility_monotonic
    assert diagnostics.convex
    assert np.allclose(
        frontier_result.weights.sum(axis=1).to_numpy(),
        1.0,
        atol=config["validation"]["weight_sum_tolerance"],
    )


def test_random_portfolios_do_not_beat_analytical_frontier(
    config: dict[str, Any],
    monte_carlo_result: MonteCarloResult,
) -> None:
    """The optimizer must remain the boundary of the random portfolio cloud."""

    diagnostics = monte_carlo_result.diagnostics
    comparison = monte_carlo_result.frontier_comparison
    assert diagnostics.generated_portfolios == config["monte_carlo"][
        "number_of_portfolios"
    ]
    assert np.allclose(
        monte_carlo_result.weights.sum(axis=1).to_numpy(),
        1.0,
        atol=config["validation"]["weight_sum_tolerance"],
    )
    assert comparison.portfolios_beating_frontier == 0
    assert comparison.analytical_frontier_holds


def test_backtest_uses_only_prior_training_data(
    config: dict[str, Any],
    backtest_result: BacktestResult,
) -> None:
    """Rebuild the first weights solely from dates before the holding period."""

    log = backtest_result.rebalance_log
    training_window = int(config["backtest"]["training_window_days"])
    assert backtest_result.diagnostics.no_lookahead_bias
    assert (log["training_end"] < log["holding_start"]).all()
    assert (log["training_rows"] == training_window).all()
    assert (
        log["holding_start"].iloc[1:].reset_index(drop=True)
        > log["holding_end"].iloc[:-1].reset_index(drop=True)
    ).all()

    prices, _ = load_price_data(CONFIG_PATH)
    universe_tickers = get_universe_tickers(config)
    asset_returns = compute_simple_returns(prices.loc[:, universe_tickers])
    first_rebalance = log.iloc[0]
    first_training_returns = asset_returns.loc[
        first_rebalance["training_start"] : first_rebalance["training_end"]
    ]
    assert len(first_training_returns) == training_window
    assert first_training_returns.index[-1] < first_rebalance["holding_start"]

    return_summary = build_return_summary(first_training_returns, config["returns"])
    estimates = estimate_covariances(first_training_returns, config)
    covariance = select_covariance_estimate(
        estimates,
        config["backtest"]["covariance_estimator"],
    )
    rebuilt = solve_maximum_sharpe(
        return_summary["annualized_mean_return"],
        covariance,
        config,
    )
    recorded_weights = backtest_result.target_weights.iloc[0]
    assert np.allclose(
        rebuilt.weights.reindex(recorded_weights.index).to_numpy(),
        recorded_weights.to_numpy(),
        atol=1.0e-10,
    )


def test_performance_summary_covers_strategy_and_benchmarks(
    config: dict[str, Any],
    backtest_result: BacktestResult,
) -> None:
    """Every required metric must exist and be finite for all three series."""

    summary = build_performance_summary(backtest_result, config)
    assert list(summary.index) == ["maximum_sharpe", "equal_weight", "SPY"]
    assert list(summary.columns) == config["metrics"]["include"]
    assert np.isfinite(summary.to_numpy(dtype=float)).all()
    assert (summary["annualized_volatility"] > 0.0).all()
    assert (summary["maximum_drawdown"] <= 0.0).all()
