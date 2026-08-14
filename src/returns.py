"""Compute daily returns and annualized asset-level statistics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import get_universe_tickers, load_config, load_price_data


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert an aligned adjusted-price table into daily simple returns."""

    if prices.empty:
        raise ValueError("Cannot compute returns from an empty price table.")
    if len(prices) < 2:
        raise ValueError("At least two price observations are required.")
    if prices.isna().any().any():
        raise ValueError("Price data must not contain missing values.")
    if (prices <= 0).any().any():
        raise ValueError("Adjusted prices must be strictly positive.")

    daily_returns = prices.pct_change(fill_method=None).iloc[1:].copy()
    if daily_returns.isna().any().any():
        raise ValueError("Daily returns contain unexpected missing values.")
    if not np.isfinite(daily_returns.to_numpy()).all():
        raise ValueError("Daily returns contain non-finite values.")

    daily_returns.index.name = "Date"
    return daily_returns


def annualize_mean_returns(
    daily_returns: pd.DataFrame,
    periods_per_year: int,
) -> pd.Series:
    """Annualize arithmetic mean daily returns by multiplying by periods per year."""

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    annualized = daily_returns.mean(axis=0) * periods_per_year
    annualized.name = "annualized_mean_return"
    return annualized


def annualize_volatility(
    daily_returns: pd.DataFrame,
    periods_per_year: int,
    standard_deviation_ddof: int,
) -> pd.Series:
    """Annualize daily standard deviation using the square-root-of-time rule."""

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    if standard_deviation_ddof < 0:
        raise ValueError("standard_deviation_ddof cannot be negative.")

    annualized = daily_returns.std(
        axis=0,
        ddof=standard_deviation_ddof,
    ) * np.sqrt(periods_per_year)
    annualized.name = "annualized_volatility"
    return annualized


def build_return_summary(
    daily_returns: pd.DataFrame,
    returns_config: dict[str, Any],
) -> pd.DataFrame:
    """Build a table of annualized mean return and volatility for every asset."""

    if returns_config["method"] != "simple":
        raise ValueError(
            f"Unsupported return method: {returns_config['method']!r}."
        )
    if returns_config["expected_return_estimator"] != "historical_mean":
        raise ValueError(
            "Only the configured historical-mean return estimator is supported."
        )

    periods_per_year = returns_config["periods_per_year"]
    annualized_returns = annualize_mean_returns(
        daily_returns,
        periods_per_year,
    )
    annualized_volatility = annualize_volatility(
        daily_returns,
        periods_per_year,
        returns_config["standard_deviation_ddof"],
    )
    return pd.concat(
        [annualized_returns, annualized_volatility],
        axis=1,
    )


def load_return_data(
    config_path: str | Path = "config.yaml",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached prices and return daily returns plus annualized statistics."""

    config = load_config(config_path)
    prices, _ = load_price_data(config_path)
    universe_tickers = get_universe_tickers(config)
    universe_prices = prices.loc[:, universe_tickers]
    daily_returns = compute_simple_returns(universe_prices)
    summary = build_return_summary(daily_returns, config["returns"])
    return daily_returns, summary


def main() -> None:
    """Run the returns layer and print a concise validation summary."""

    parser = argparse.ArgumentParser(
        description="Compute daily and annualized returns from cached prices."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    arguments = parser.parse_args()

    daily_returns, summary = load_return_data(arguments.config)
    print("Return calculations completed successfully.")
    print(f"Daily return rows: {len(daily_returns)}")
    print(f"Assets: {len(daily_returns.columns)}")
    print(
        "Date range: "
        f"{daily_returns.index.min().date().isoformat()} to "
        f"{daily_returns.index.max().date().isoformat()}"
    )
    print(f"Missing values: {int(daily_returns.isna().sum().sum())}")
    print("First five annualized statistics:")
    print(summary.head().to_string(float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
