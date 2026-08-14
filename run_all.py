"""Run the complete portfolio optimization pipeline from one command."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.backtest import BacktestResult, load_backtest_data
from src.covariance import CovarianceEstimates, load_covariance_data
from src.data import DataValidationReport, load_config, load_price_data
from src.frontier import FrontierResult, trace_efficient_frontier
from src.metrics import build_performance_summary, save_performance_summary
from src.montecarlo import MonteCarloResult, run_monte_carlo
from src.optimize import (
    OptimizationResult,
    load_optimization_inputs,
    solve_maximum_sharpe,
    solve_minimum_variance,
)
from src.plots import create_all_plots
from src.returns import load_return_data


def run_pipeline(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Execute every project stage and return its main validated results."""

    config = load_config(config_path)

    print("[1/8] Loading and validating real market prices...")
    prices, data_report = load_price_data(config_path)

    print("[2/8] Calculating daily and annualized returns...")
    daily_returns, return_summary = load_return_data(config_path)

    print("[3/8] Estimating and validating covariance matrices...")
    _, _, covariance_estimates, covariance_diagnostics = load_covariance_data(
        config_path
    )
    if not all(
        diagnostics.symmetric and diagnostics.positive_semidefinite
        for diagnostics in covariance_diagnostics.values()
    ):
        raise RuntimeError("At least one covariance validation failed.")

    print("[4/8] Solving analytical portfolios and efficient frontier...")
    _, expected_returns, optimization_covariance = load_optimization_inputs(
        config_path
    )
    minimum_variance = solve_minimum_variance(
        expected_returns,
        optimization_covariance,
        config,
    )
    maximum_sharpe = solve_maximum_sharpe(
        expected_returns,
        optimization_covariance,
        config,
    )
    frontier = trace_efficient_frontier(
        expected_returns,
        optimization_covariance,
        config,
    )
    if not (
        frontier.diagnostics.returns_monotonic
        and frontier.diagnostics.volatility_monotonic
        and frontier.diagnostics.convex
    ):
        raise RuntimeError("Efficient-frontier validation failed.")

    print("[5/8] Simulating and validating random portfolios...")
    monte_carlo = run_monte_carlo(config_path)

    print("[6/8] Running the strictly out-of-sample backtest...")
    backtest = load_backtest_data(config_path)

    print("[7/8] Calculating and saving performance metrics...")
    performance_summary = build_performance_summary(backtest, config)
    performance_table_path = save_performance_summary(
        performance_summary,
        config,
        config_path,
    )

    print("[8/8] Creating publication-quality figures...")
    figure_paths = create_all_plots(config_path)

    return {
        "prices": prices,
        "data_report": data_report,
        "daily_returns": daily_returns,
        "return_summary": return_summary,
        "covariance_estimates": covariance_estimates,
        "covariance_diagnostics": covariance_diagnostics,
        "minimum_variance": minimum_variance,
        "maximum_sharpe": maximum_sharpe,
        "frontier": frontier,
        "monte_carlo": monte_carlo,
        "backtest": backtest,
        "performance_summary": performance_summary,
        "performance_table_path": performance_table_path,
        "figure_paths": figure_paths,
    }


def _print_completion_summary(results: dict[str, Any]) -> None:
    """Print a concise final report for a successful complete run."""

    data_report: DataValidationReport = results["data_report"]
    covariance_estimates: CovarianceEstimates = results["covariance_estimates"]
    minimum_variance: OptimizationResult = results["minimum_variance"]
    maximum_sharpe: OptimizationResult = results["maximum_sharpe"]
    frontier: FrontierResult = results["frontier"]
    monte_carlo: MonteCarloResult = results["monte_carlo"]
    backtest: BacktestResult = results["backtest"]

    print("\nComplete pipeline finished successfully.")
    print(
        "Real price data: "
        f"{data_report.cleaned_rows} rows, "
        f"{data_report.start_date} to {data_report.end_date}"
    )
    print(
        "Ledoit-Wolf shrinkage: "
        f"{covariance_estimates.shrinkage_intensity:.6f}"
    )
    print(
        "Minimum-variance portfolio: "
        f"return {minimum_variance.expected_return:.2%}, "
        f"volatility {minimum_variance.volatility:.2%}"
    )
    print(
        "Maximum-Sharpe portfolio: "
        f"return {maximum_sharpe.expected_return:.2%}, "
        f"volatility {maximum_sharpe.volatility:.2%}, "
        f"Sharpe {maximum_sharpe.sharpe_ratio:.3f}"
    )
    print(
        "Efficient frontier: "
        f"{frontier.diagnostics.solved_points}/"
        f"{frontier.diagnostics.requested_points} points solved"
    )
    print(
        "Random portfolios beating frontier: "
        f"{monte_carlo.frontier_comparison.portfolios_beating_frontier}"
    )
    print(
        "Out-of-sample backtest: "
        f"{backtest.diagnostics.out_of_sample_start} to "
        f"{backtest.diagnostics.out_of_sample_end}, "
        f"no look-ahead bias = {backtest.diagnostics.no_lookahead_bias}"
    )
    print("\nPerformance summary:")
    print(
        results["performance_summary"].to_string(
            float_format=lambda value: f"{value:.6f}"
        )
    )
    print(f"\nSaved table: {results['performance_table_path']}")
    print("Saved figures:")
    for label, path in results["figure_paths"].items():
        print(f"  {label}: {path}")


def main() -> None:
    """Parse the config path and run the entire project."""

    parser = argparse.ArgumentParser(
        description="Run the full portfolio optimization project."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    arguments = parser.parse_args()

    results = run_pipeline(arguments.config)
    _print_completion_summary(results)


if __name__ == "__main__":
    main()
