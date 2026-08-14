"""Run a strictly out-of-sample walk-forward portfolio backtest."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.covariance import estimate_covariances
from src.data import (
    get_universe_tickers,
    load_config,
    load_price_data,
)
from src.optimize import select_covariance_estimate, solve_maximum_sharpe
from src.returns import build_return_summary, compute_simple_returns


@dataclass(frozen=True)
class BacktestDiagnostics:
    """Store the key timing and validation facts from the backtest."""

    training_window_days: int
    holding_period_days: int
    rebalance_count: int
    out_of_sample_rows: int
    out_of_sample_start: str
    out_of_sample_end: str
    minimum_training_rows: int
    maximum_training_rows: int
    no_lookahead_bias: bool


@dataclass(frozen=True)
class BacktestResult:
    """Store out-of-sample returns, allocations, turnover, and diagnostics."""

    daily_returns: pd.DataFrame
    cumulative_returns: pd.DataFrame
    target_weights: pd.DataFrame
    turnover: pd.DataFrame
    rebalance_log: pd.DataFrame
    diagnostics: BacktestDiagnostics


def _validate_backtest_settings(config: dict[str, Any]) -> None:
    """Validate the supported walk-forward configuration."""

    backtest_config = config["backtest"]
    if backtest_config["strategy"] != "maximum_sharpe":
        raise ValueError("Only the maximum_sharpe strategy is supported.")
    if backtest_config["training_window_days"] <= 1:
        raise ValueError("training_window_days must be greater than one.")
    if backtest_config["holding_period_days"] <= 0:
        raise ValueError("holding_period_days must be positive.")
    if backtest_config["rebalance_frequency"] != "quarterly":
        raise ValueError("This project expects quarterly rebalancing.")
    if backtest_config["transaction_cost_bps"] < 0.0:
        raise ValueError("transaction_cost_bps cannot be negative.")
    if set(backtest_config["benchmarks"]) != {"equal_weight", "SPY"}:
        raise ValueError("The required benchmarks are equal_weight and SPY.")


def _holding_period_returns(
    asset_returns: pd.DataFrame,
    target_weights: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Calculate buy-and-hold returns and the weights after market drift."""

    if asset_returns.empty:
        raise ValueError("A holding period cannot be empty.")
    if list(asset_returns.columns) != list(target_weights.index):
        raise ValueError("Holding-period assets and weight labels must match.")
    if asset_returns.isna().any().any():
        raise ValueError("Holding-period returns contain missing values.")

    growth_by_asset = (1.0 + asset_returns).cumprod()
    portfolio_values = growth_by_asset.mul(target_weights, axis=1).sum(axis=1)
    previous_values = portfolio_values.shift(1)
    previous_values.iloc[0] = 1.0
    portfolio_returns = portfolio_values / previous_values - 1.0

    ending_asset_values = target_weights * growth_by_asset.iloc[-1]
    ending_weights = ending_asset_values / ending_asset_values.sum()
    return portfolio_returns, ending_weights


def _one_way_turnover(
    target_weights: pd.Series,
    previous_weights: pd.Series | None,
) -> float:
    """Return half the absolute allocation change at a rebalance."""

    if previous_weights is None:
        return 0.0
    previous = previous_weights.reindex(target_weights.index)
    if previous.isna().any():
        raise ValueError("Previous weights do not match target weights.")
    return float(0.5 * np.abs(target_weights - previous).sum())


def _deduct_transaction_cost(
    daily_returns: pd.Series,
    turnover: float,
    transaction_cost_bps: float,
) -> pd.Series:
    """Deduct one proportional trading cost on the first holding day."""

    adjusted = daily_returns.copy()
    cost_fraction = turnover * transaction_cost_bps / 10_000.0
    adjusted.iloc[0] = (1.0 - cost_fraction) * (1.0 + adjusted.iloc[0]) - 1.0
    return adjusted


def run_walk_forward_backtest(
    prices: pd.DataFrame,
    config: dict[str, Any],
) -> BacktestResult:
    """Estimate on trailing data and apply weights only to future returns."""

    _validate_backtest_settings(config)
    universe_tickers = get_universe_tickers(config)
    benchmark_ticker = str(config["data"]["benchmark_ticker"]).upper()
    required_tickers = [*universe_tickers, benchmark_ticker]
    missing_tickers = sorted(set(required_tickers) - set(prices.columns))
    if missing_tickers:
        raise ValueError(f"Backtest prices are missing tickers: {missing_tickers}")

    all_returns = compute_simple_returns(prices.loc[:, required_tickers])
    asset_returns = all_returns.loc[:, universe_tickers]
    spy_returns = all_returns.loc[:, benchmark_ticker]
    backtest_config = config["backtest"]
    training_window = int(backtest_config["training_window_days"])
    holding_period = int(backtest_config["holding_period_days"])
    if len(asset_returns) <= training_window:
        raise ValueError("There is not enough data for one out-of-sample period.")

    strategy_name = str(backtest_config["strategy"])
    transaction_cost_bps = float(backtest_config["transaction_cost_bps"])
    equal_weight_target = pd.Series(
        1.0 / len(universe_tickers),
        index=universe_tickers,
        name="weight",
    )

    strategy_periods: list[pd.Series] = []
    equal_weight_periods: list[pd.Series] = []
    spy_periods: list[pd.Series] = []
    target_weight_rows: list[pd.Series] = []
    turnover_rows: list[dict[str, float]] = []
    log_rows: list[dict[str, Any]] = []
    previous_strategy_weights: pd.Series | None = None
    previous_equal_weights: pd.Series | None = None

    for holding_start_position in range(
        training_window,
        len(asset_returns),
        holding_period,
    ):
        holding_end_position = min(
            holding_start_position + holding_period,
            len(asset_returns),
        )
        training_returns = asset_returns.iloc[
            holding_start_position - training_window : holding_start_position
        ]
        holding_returns = asset_returns.iloc[
            holding_start_position:holding_end_position
        ]

        return_summary = build_return_summary(
            training_returns,
            config["returns"],
        )
        expected_returns = return_summary["annualized_mean_return"]
        covariance_estimates = estimate_covariances(training_returns, config)
        covariance = select_covariance_estimate(
            covariance_estimates,
            backtest_config["covariance_estimator"],
        )
        optimized = solve_maximum_sharpe(
            expected_returns,
            covariance,
            config,
        )
        target_weights = optimized.weights.reindex(universe_tickers)

        strategy_turnover = _one_way_turnover(
            target_weights,
            previous_strategy_weights,
        )
        equal_weight_turnover = _one_way_turnover(
            equal_weight_target,
            previous_equal_weights,
        )
        strategy_returns, previous_strategy_weights = _holding_period_returns(
            holding_returns,
            target_weights,
        )
        equal_returns, previous_equal_weights = _holding_period_returns(
            holding_returns,
            equal_weight_target,
        )
        strategy_returns = _deduct_transaction_cost(
            strategy_returns,
            strategy_turnover,
            transaction_cost_bps,
        )
        equal_returns = _deduct_transaction_cost(
            equal_returns,
            equal_weight_turnover,
            transaction_cost_bps,
        )

        holding_start_date = holding_returns.index[0]
        holding_end_date = holding_returns.index[-1]
        strategy_periods.append(strategy_returns.rename(strategy_name))
        equal_weight_periods.append(equal_returns.rename("equal_weight"))
        spy_periods.append(
            spy_returns.iloc[
                holding_start_position:holding_end_position
            ].rename("SPY")
        )
        target_weights.name = holding_start_date
        target_weight_rows.append(target_weights)
        turnover_rows.append(
            {
                strategy_name: strategy_turnover,
                "equal_weight": equal_weight_turnover,
                "SPY": 0.0,
            }
        )
        log_rows.append(
            {
                "training_start": training_returns.index[0],
                "training_end": training_returns.index[-1],
                "holding_start": holding_start_date,
                "holding_end": holding_end_date,
                "training_rows": len(training_returns),
                "holding_rows": len(holding_returns),
                "shrinkage_intensity": (
                    covariance_estimates.shrinkage_intensity
                ),
                "solver_iterations": optimized.iterations,
            }
        )

    daily_returns = pd.concat(
        {
            strategy_name: pd.concat(strategy_periods),
            "equal_weight": pd.concat(equal_weight_periods),
            "SPY": pd.concat(spy_periods),
        },
        axis=1,
    )
    daily_returns.index.name = "Date"
    if daily_returns.isna().any().any():
        raise RuntimeError("Aligned backtest returns contain missing values.")
    if not daily_returns.index.is_monotonic_increasing:
        raise RuntimeError("Backtest dates are not in chronological order.")
    if daily_returns.index.has_duplicates:
        raise RuntimeError("Backtest dates contain duplicates.")

    cumulative_returns = (1.0 + daily_returns).cumprod() - 1.0
    target_weights = pd.DataFrame(target_weight_rows)
    target_weights.index.name = "holding_start"
    target_weights.columns.name = "Ticker"
    rebalance_log = pd.DataFrame(log_rows)
    turnover = pd.DataFrame(
        turnover_rows,
        index=rebalance_log["holding_start"],
    )
    turnover.index.name = "holding_start"

    training_before_holding = (
        rebalance_log["training_end"] < rebalance_log["holding_start"]
    )
    full_training_windows = rebalance_log["training_rows"] == training_window
    no_lookahead_bias = bool(
        training_before_holding.all() and full_training_windows.all()
    )
    diagnostics = BacktestDiagnostics(
        training_window_days=training_window,
        holding_period_days=holding_period,
        rebalance_count=len(rebalance_log),
        out_of_sample_rows=len(daily_returns),
        out_of_sample_start=daily_returns.index[0].date().isoformat(),
        out_of_sample_end=daily_returns.index[-1].date().isoformat(),
        minimum_training_rows=int(rebalance_log["training_rows"].min()),
        maximum_training_rows=int(rebalance_log["training_rows"].max()),
        no_lookahead_bias=no_lookahead_bias,
    )
    if not diagnostics.no_lookahead_bias:
        raise RuntimeError("Look-ahead validation failed.")

    return BacktestResult(
        daily_returns=daily_returns,
        cumulative_returns=cumulative_returns,
        target_weights=target_weights,
        turnover=turnover,
        rebalance_log=rebalance_log,
        diagnostics=diagnostics,
    )


def load_backtest_data(
    config_path: str | Path = "config.yaml",
) -> BacktestResult:
    """Load validated real prices and run the configured backtest."""

    config = load_config(config_path)
    prices, _ = load_price_data(config_path)
    return run_walk_forward_backtest(prices, config)


def main() -> None:
    """Run the walk-forward engine and print its timing validation."""

    parser = argparse.ArgumentParser(
        description="Run the strictly out-of-sample portfolio backtest."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    arguments = parser.parse_args()

    result = load_backtest_data(arguments.config)
    diagnostics = result.diagnostics
    print("Walk-forward backtest completed successfully.")
    print(f"Training window: {diagnostics.training_window_days} trading days")
    print(f"Holding period: {diagnostics.holding_period_days} trading days")
    print(f"Rebalances: {diagnostics.rebalance_count}")
    print(f"Out-of-sample rows: {diagnostics.out_of_sample_rows}")
    print(
        "Out-of-sample range: "
        f"{diagnostics.out_of_sample_start} to {diagnostics.out_of_sample_end}"
    )
    print(
        "Training rows per rebalance: "
        f"{diagnostics.minimum_training_rows} to "
        f"{diagnostics.maximum_training_rows}"
    )
    print(f"No look-ahead bias: {diagnostics.no_lookahead_bias}")
    print("Ledoit-Wolf shrinkage intensity by rebalance:")
    for row in result.rebalance_log.itertuples(index=False):
        print(
            f"  {row.holding_start.date().isoformat()}: "
            f"{row.shrinkage_intensity:.6f}"
        )
    shrinkage = result.rebalance_log["shrinkage_intensity"]
    print(
        "Ledoit-Wolf shrinkage range: "
        f"{shrinkage.min():.6f} to {shrinkage.max():.6f}"
    )
    print("Final cumulative returns:")
    print(result.cumulative_returns.iloc[-1].to_string(float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
