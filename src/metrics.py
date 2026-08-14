"""Calculate and save performance statistics for backtest strategies."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest import BacktestResult, load_backtest_data
from src.data import load_config


def calculate_cagr(daily_returns: pd.Series, periods_per_year: int) -> float:
    """Return compounded annual growth over the observed daily return series."""

    if daily_returns.empty:
        raise ValueError("Cannot calculate CAGR from an empty return series.")
    total_growth = float((1.0 + daily_returns).prod())
    if total_growth <= 0.0:
        raise ValueError("Total growth must remain positive for CAGR.")
    return total_growth ** (periods_per_year / len(daily_returns)) - 1.0


def calculate_annualized_volatility(
    daily_returns: pd.Series,
    periods_per_year: int,
    standard_deviation_ddof: int,
) -> float:
    """Annualize the standard deviation of daily realized returns."""

    return float(
        daily_returns.std(ddof=standard_deviation_ddof)
        * np.sqrt(periods_per_year)
    )


def calculate_sharpe_ratio(
    daily_returns: pd.Series,
    periods_per_year: int,
    risk_free_rate: float,
    standard_deviation_ddof: int,
) -> float:
    """Return annualized arithmetic excess return per unit of volatility."""

    volatility = calculate_annualized_volatility(
        daily_returns,
        periods_per_year,
        standard_deviation_ddof,
    )
    if volatility == 0.0:
        raise ValueError("Sharpe ratio is undefined at zero volatility.")
    annualized_mean = float(daily_returns.mean() * periods_per_year)
    return (annualized_mean - risk_free_rate) / volatility


def calculate_drawdowns(daily_returns: pd.Series) -> pd.Series:
    """Return the percentage decline from each prior wealth peak."""

    if daily_returns.empty:
        raise ValueError("Cannot calculate drawdown from empty returns.")
    wealth = (1.0 + daily_returns).cumprod()
    running_peak = wealth.cummax().clip(lower=1.0)
    drawdowns = wealth / running_peak - 1.0
    drawdowns.name = daily_returns.name
    return drawdowns


def calculate_maximum_drawdown(daily_returns: pd.Series) -> float:
    """Return the most negative peak-to-trough drawdown."""

    return float(calculate_drawdowns(daily_returns).min())


def calculate_calmar_ratio(cagr: float, maximum_drawdown: float) -> float:
    """Return annual growth divided by the magnitude of maximum drawdown."""

    drawdown_magnitude = abs(maximum_drawdown)
    if drawdown_magnitude == 0.0:
        return float("nan")
    return cagr / drawdown_magnitude


def build_performance_summary(
    backtest: BacktestResult,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Calculate every configured metric for each tested strategy."""

    daily_returns = backtest.daily_returns
    if daily_returns.empty or daily_returns.isna().any().any():
        raise ValueError("Backtest returns must be complete and nonempty.")
    if not np.isfinite(daily_returns.to_numpy(dtype=float)).all():
        raise ValueError("Backtest returns contain non-finite values.")

    periods_per_year = int(config["returns"]["periods_per_year"])
    standard_deviation_ddof = int(
        config["returns"]["standard_deviation_ddof"]
    )
    risk_free_rate = float(config["optimization"]["risk_free_rate"])
    included_metrics = config["metrics"]["include"]
    required_metrics = {
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
        "calmar_ratio",
        "turnover",
    }
    if set(included_metrics) != required_metrics:
        raise ValueError("The configured performance metric set is incomplete.")

    records: list[dict[str, float | str]] = []
    for strategy in daily_returns.columns:
        strategy_returns = daily_returns[strategy]
        cagr = calculate_cagr(strategy_returns, periods_per_year)
        volatility = calculate_annualized_volatility(
            strategy_returns,
            periods_per_year,
            standard_deviation_ddof,
        )
        sharpe_ratio = calculate_sharpe_ratio(
            strategy_returns,
            periods_per_year,
            risk_free_rate,
            standard_deviation_ddof,
        )
        maximum_drawdown = calculate_maximum_drawdown(strategy_returns)
        calmar_ratio = calculate_calmar_ratio(cagr, maximum_drawdown)
        turnover = float(backtest.turnover[strategy].mean())
        records.append(
            {
                "strategy": str(strategy),
                "cagr": cagr,
                "annualized_volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "maximum_drawdown": maximum_drawdown,
                "calmar_ratio": calmar_ratio,
                "turnover": turnover,
            }
        )

    summary = pd.DataFrame(records).set_index("strategy")
    if not np.isfinite(summary.to_numpy(dtype=float)).all():
        raise ValueError("Performance summary contains non-finite values.")
    return summary


def save_performance_summary(
    summary: pd.DataFrame,
    config: dict[str, Any],
    config_path: str | Path = "config.yaml",
) -> Path:
    """Save the performance table to its configured CSV path."""

    project_directory = Path(config_path).expanduser().resolve().parent
    output_path = (project_directory / config["paths"]["performance_table"]).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, float_format="%.10f")
    return output_path


def load_performance_results(
    config_path: str | Path = "config.yaml",
) -> tuple[BacktestResult, pd.DataFrame]:
    """Run the configured backtest and return its performance summary."""

    config = load_config(config_path)
    backtest = load_backtest_data(config_path)
    summary = build_performance_summary(backtest, config)
    return backtest, summary


def main() -> None:
    """Run Phase 6, save its result table, and print the validation gate."""

    parser = argparse.ArgumentParser(
        description="Calculate backtest performance statistics."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    backtest, summary = load_performance_results(arguments.config)
    output_path = save_performance_summary(
        summary,
        config,
        arguments.config,
    )
    diagnostics = backtest.diagnostics

    print("Backtest metrics completed successfully.")
    print(
        "Out-of-sample range: "
        f"{diagnostics.out_of_sample_start} to {diagnostics.out_of_sample_end}"
    )
    print(f"Out-of-sample rows: {diagnostics.out_of_sample_rows}")
    print(f"Rebalances: {diagnostics.rebalance_count}")
    print(f"No look-ahead bias: {diagnostics.no_lookahead_bias}")
    print("Performance summary:")
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"Saved table: {output_path}")


if __name__ == "__main__":
    main()
