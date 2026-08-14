"""Create and save publication-quality portfolio analysis figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from src.backtest import BacktestResult, load_backtest_data
from src.data import load_config
from src.frontier import FrontierResult, trace_efficient_frontier
from src.metrics import calculate_drawdowns
from src.montecarlo import MonteCarloResult, run_monte_carlo
from src.optimize import (
    OptimizationResult,
    load_optimization_inputs,
    solve_maximum_sharpe,
    solve_minimum_variance,
)


def _figure_path(
    filename: str,
    config: dict[str, Any],
    config_path: str | Path,
) -> Path:
    """Return one configured figure path and create its directory."""

    project_directory = Path(config_path).expanduser().resolve().parent
    figure_directory = (
        project_directory / config["paths"]["figures_directory"]
    ).resolve()
    figure_directory.mkdir(parents=True, exist_ok=True)
    return figure_directory / filename


def _save_figure(
    figure: Figure,
    output_path: Path,
    config: dict[str, Any],
) -> Path:
    """Save one figure and reject an empty output file."""

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=config["plots"]["dpi"],
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Figure was not saved correctly: {output_path}")
    return output_path


def plot_efficient_frontier(
    monte_carlo: MonteCarloResult,
    frontier: FrontierResult,
    minimum_variance: OptimizationResult,
    maximum_sharpe: OptimizationResult,
    config: dict[str, Any],
    config_path: str | Path,
) -> Path:
    """Plot random portfolios beneath the analytical efficient frontier."""

    plot_config = config["plots"]
    figure, axis = plt.subplots(
        figsize=(
            plot_config["figure_width_inches"],
            plot_config["figure_height_inches"],
        )
    )
    random_portfolios = monte_carlo.portfolios
    scatter = axis.scatter(
        random_portfolios["volatility"],
        random_portfolios["expected_return"],
        c=random_portfolios["sharpe_ratio"],
        cmap=plot_config["color_map"],
        s=plot_config["monte_carlo_point_size"],
        alpha=plot_config["monte_carlo_alpha"],
        linewidths=0,
        rasterized=True,
        label="Random portfolios",
    )
    axis.plot(
        frontier.portfolios["volatility"],
        frontier.portfolios["expected_return"],
        color="#c23b22",
        linewidth=plot_config["line_width"],
        label="Analytical efficient frontier",
        zorder=4,
    )
    axis.scatter(
        minimum_variance.volatility,
        minimum_variance.expected_return,
        marker="D",
        s=70,
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.8,
        label="Minimum variance",
        zorder=5,
    )
    axis.scatter(
        maximum_sharpe.volatility,
        maximum_sharpe.expected_return,
        marker="*",
        s=150,
        color="#f2b134",
        edgecolor="#222222",
        linewidth=0.6,
        label="Maximum Sharpe",
        zorder=5,
    )
    colorbar = figure.colorbar(scatter, ax=axis, pad=0.02)
    colorbar.set_label("Sharpe ratio")
    axis.set_title("Efficient Frontier and Random Portfolios", weight="bold")
    axis.set_xlabel("Annualized volatility")
    axis.set_ylabel("Annualized expected return")
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.legend(loc="upper left", frameon=True)
    axis.grid(True, alpha=0.25)

    return _save_figure(
        figure,
        _figure_path("efficient_frontier.png", config, config_path),
        config,
    )


def plot_cumulative_returns(
    backtest: BacktestResult,
    config: dict[str, Any],
    config_path: str | Path,
) -> Path:
    """Plot growth of one dollar for the strategy and both benchmarks."""

    plot_config = config["plots"]
    figure, axis = plt.subplots(
        figsize=(
            plot_config["figure_width_inches"],
            plot_config["figure_height_inches"],
        )
    )
    wealth = backtest.cumulative_returns + 1.0
    labels = {
        "maximum_sharpe": "Maximum Sharpe",
        "equal_weight": "Equal Weight",
        "SPY": "SPY",
    }
    colors = {
        "maximum_sharpe": "#1f77b4",
        "equal_weight": "#2ca02c",
        "SPY": "#555555",
    }
    for strategy in wealth.columns:
        axis.plot(
            wealth.index,
            wealth[strategy],
            label=labels[strategy],
            color=colors[strategy],
            linewidth=plot_config["line_width"],
        )

    axis.set_title(
        "Out-of-Sample Growth of $1",
        weight="bold",
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Portfolio value ($)")
    axis.legend(loc="upper left", frameon=True)
    axis.grid(True, alpha=0.25)
    return _save_figure(
        figure,
        _figure_path("cumulative_returns.png", config, config_path),
        config,
    )


def plot_drawdowns(
    backtest: BacktestResult,
    config: dict[str, Any],
    config_path: str | Path,
) -> Path:
    """Plot each strategy's decline from its previous wealth peak."""

    plot_config = config["plots"]
    figure, axis = plt.subplots(
        figsize=(
            plot_config["figure_width_inches"],
            plot_config["figure_height_inches"],
        )
    )
    labels = {
        "maximum_sharpe": "Maximum Sharpe",
        "equal_weight": "Equal Weight",
        "SPY": "SPY",
    }
    colors = {
        "maximum_sharpe": "#1f77b4",
        "equal_weight": "#2ca02c",
        "SPY": "#555555",
    }
    for strategy in backtest.daily_returns.columns:
        drawdowns = calculate_drawdowns(backtest.daily_returns[strategy])
        axis.plot(
            drawdowns.index,
            drawdowns,
            label=labels[strategy],
            color=colors[strategy],
            linewidth=plot_config["line_width"],
        )

    axis.axhline(0.0, color="#222222", linewidth=0.8)
    axis.set_title("Out-of-Sample Drawdowns", weight="bold")
    axis.set_xlabel("Date")
    axis.set_ylabel("Drawdown from prior peak")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.legend(loc="lower left", frameon=True)
    axis.grid(True, alpha=0.25)
    return _save_figure(
        figure,
        _figure_path("drawdowns.png", config, config_path),
        config,
    )


def calculate_sector_allocations(
    weights: pd.Series,
    config: dict[str, Any],
) -> pd.Series:
    """Aggregate asset weights into the configured sector groups."""

    allocations = {
        sector: float(weights.reindex(tickers).sum())
        for sector, tickers in config["universe"]["sectors"].items()
    }
    result = pd.Series(allocations, name="allocation").sort_values(ascending=False)
    tolerance = float(config["validation"]["weight_sum_tolerance"])
    if abs(float(result.sum()) - 1.0) > tolerance:
        raise RuntimeError("Sector allocations do not sum to one.")
    return result


def plot_sector_allocation(
    maximum_sharpe: OptimizationResult,
    config: dict[str, Any],
    config_path: str | Path,
) -> Path:
    """Plot full-sample maximum-Sharpe weights grouped by sector."""

    plot_config = config["plots"]
    allocations = calculate_sector_allocations(maximum_sharpe.weights, config)
    figure, axis = plt.subplots(
        figsize=(
            plot_config["figure_width_inches"],
            plot_config["figure_height_inches"],
        )
    )
    bars = axis.bar(
        allocations.index,
        allocations.values,
        color=plt.get_cmap(plot_config["color_map"])(
            [index / max(len(allocations) - 1, 1) for index in range(len(allocations))]
        ),
    )
    axis.bar_label(
        bars,
        labels=[f"{value:.1%}" for value in allocations.values],
        padding=3,
        fontsize=9,
    )
    axis.set_title(
        "Maximum-Sharpe Portfolio Allocation by Sector",
        weight="bold",
    )
    axis.set_xlabel("Sector")
    axis.set_ylabel("Portfolio allocation")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    axis.grid(axis="x", visible=False)
    return _save_figure(
        figure,
        _figure_path("sector_allocation.png", config, config_path),
        config,
    )


def create_all_plots(
    config_path: str | Path = "config.yaml",
) -> dict[str, Path]:
    """Build every required figure from validated pipeline results."""

    config = load_config(config_path)
    plt.style.use(config["plots"]["style"])
    _, expected_returns, covariance = load_optimization_inputs(config_path)
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
    frontier = trace_efficient_frontier(expected_returns, covariance, config)
    monte_carlo = run_monte_carlo(config_path)
    backtest = load_backtest_data(config_path)

    return {
        "efficient_frontier": plot_efficient_frontier(
            monte_carlo,
            frontier,
            minimum_variance,
            maximum_sharpe,
            config,
            config_path,
        ),
        "cumulative_returns": plot_cumulative_returns(
            backtest,
            config,
            config_path,
        ),
        "drawdowns": plot_drawdowns(
            backtest,
            config,
            config_path,
        ),
        "sector_allocation": plot_sector_allocation(
            maximum_sharpe,
            config,
            config_path,
        ),
    }


def main() -> None:
    """Generate all Phase 7 figures and print their output paths."""

    parser = argparse.ArgumentParser(
        description="Create every portfolio analysis figure."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    arguments = parser.parse_args()

    paths = create_all_plots(arguments.config)
    print("Plot generation completed successfully.")
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
