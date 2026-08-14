"""Estimate and validate sample and Ledoit-Wolf covariance matrices."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.data import load_config
from src.returns import load_return_data


@dataclass(frozen=True)
class CovarianceEstimates:
    """Store annualized sample and shrinkage covariance estimates."""

    sample: pd.DataFrame
    ledoit_wolf: pd.DataFrame
    shrinkage_intensity: float


@dataclass(frozen=True)
class CovarianceDiagnostics:
    """Store numerical checks for one covariance matrix."""

    dimension: int
    symmetric: bool
    positive_semidefinite: bool
    maximum_asymmetry: float
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    condition_number: float


def estimate_sample_covariance(
    daily_returns: pd.DataFrame,
    periods_per_year: int,
    sample_ddof: int,
) -> pd.DataFrame:
    """Estimate and annualize the ordinary sample covariance matrix."""

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    if sample_ddof < 0:
        raise ValueError("sample_ddof cannot be negative.")

    covariance = daily_returns.cov(ddof=sample_ddof) * periods_per_year
    covariance.index.name = "Ticker"
    covariance.columns.name = "Ticker"
    return covariance


def estimate_ledoit_wolf_covariance(
    daily_returns: pd.DataFrame,
    periods_per_year: int,
    estimator_config: dict[str, Any],
) -> tuple[pd.DataFrame, float]:
    """Estimate annualized Ledoit-Wolf covariance and return its shrinkage level."""

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    estimator = LedoitWolf(
        store_precision=estimator_config["store_precision"],
        assume_centered=estimator_config["assume_centered"],
    )
    estimator.fit(daily_returns.to_numpy(dtype=float))

    covariance = pd.DataFrame(
        estimator.covariance_ * periods_per_year,
        index=daily_returns.columns,
        columns=daily_returns.columns,
    )
    covariance.index.name = "Ticker"
    covariance.columns.name = "Ticker"
    shrinkage_intensity = float(estimator.shrinkage_)
    if not 0.0 <= shrinkage_intensity <= 1.0:
        raise ValueError(
            "Ledoit-Wolf shrinkage intensity must lie between 0 and 1."
        )
    return covariance, shrinkage_intensity


def estimate_covariances(
    daily_returns: pd.DataFrame,
    config: dict[str, Any],
) -> CovarianceEstimates:
    """Estimate every covariance method requested by the configuration."""

    if daily_returns.empty:
        raise ValueError("Cannot estimate covariance from empty returns.")
    if daily_returns.isna().any().any():
        raise ValueError("Returns must not contain missing values.")
    if not np.isfinite(daily_returns.to_numpy()).all():
        raise ValueError("Returns must contain only finite values.")

    covariance_config = config["covariance"]
    configured_estimators = set(covariance_config["estimators"])
    required_estimators = {"sample", "ledoit_wolf"}
    if not required_estimators.issubset(configured_estimators):
        missing = sorted(required_estimators - configured_estimators)
        raise ValueError(f"Required covariance estimators are missing: {missing}")

    periods_per_year = config["returns"]["periods_per_year"]
    sample = estimate_sample_covariance(
        daily_returns,
        periods_per_year,
        covariance_config["sample_ddof"],
    )
    ledoit_wolf, shrinkage_intensity = estimate_ledoit_wolf_covariance(
        daily_returns,
        periods_per_year,
        covariance_config["ledoit_wolf"],
    )
    return CovarianceEstimates(
        sample=sample,
        ledoit_wolf=ledoit_wolf,
        shrinkage_intensity=shrinkage_intensity,
    )


def diagnose_covariance(
    covariance: pd.DataFrame,
    symmetry_tolerance: float,
    psd_tolerance: float,
) -> CovarianceDiagnostics:
    """Check dimensions, symmetry, eigenvalues, and conditioning."""

    if covariance.empty:
        raise ValueError("Covariance matrix cannot be empty.")
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"Covariance matrix is not square: {covariance.shape}")
    if list(covariance.index) != list(covariance.columns):
        raise ValueError("Covariance row and column labels must match in order.")

    values = covariance.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Covariance matrix contains non-finite values.")

    maximum_asymmetry = float(np.max(np.abs(values - values.T)))
    symmetric = maximum_asymmetry <= symmetry_tolerance
    symmetric_values = (values + values.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(symmetric_values)
    minimum_eigenvalue = float(eigenvalues.min())
    maximum_eigenvalue = float(eigenvalues.max())
    positive_semidefinite = symmetric and minimum_eigenvalue >= psd_tolerance

    return CovarianceDiagnostics(
        dimension=covariance.shape[0],
        symmetric=symmetric,
        positive_semidefinite=positive_semidefinite,
        maximum_asymmetry=maximum_asymmetry,
        minimum_eigenvalue=minimum_eigenvalue,
        maximum_eigenvalue=maximum_eigenvalue,
        condition_number=float(np.linalg.cond(values)),
    )


def load_covariance_data(
    config_path: str | Path = "config.yaml",
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    CovarianceEstimates,
    dict[str, CovarianceDiagnostics],
]:
    """Load returns, estimate covariances, and return numerical diagnostics."""

    config = load_config(config_path)
    daily_returns, return_summary = load_return_data(config_path)
    estimates = estimate_covariances(daily_returns, config)
    validation_config = config["validation"]
    diagnostics = {
        "sample": diagnose_covariance(
            estimates.sample,
            validation_config["covariance_symmetry_tolerance"],
            validation_config["covariance_psd_tolerance"],
        ),
        "ledoit_wolf": diagnose_covariance(
            estimates.ledoit_wolf,
            validation_config["covariance_symmetry_tolerance"],
            validation_config["covariance_psd_tolerance"],
        ),
    }
    return daily_returns, return_summary, estimates, diagnostics


def _print_diagnostics(
    label: str,
    diagnostics: CovarianceDiagnostics,
) -> None:
    """Print one covariance diagnostic record in a readable format."""

    print(f"{label} covariance:")
    print(f"  Dimensions: {diagnostics.dimension} x {diagnostics.dimension}")
    print(f"  Symmetric: {diagnostics.symmetric}")
    print(
        "  Positive semidefinite: "
        f"{diagnostics.positive_semidefinite}"
    )
    print(f"  Maximum asymmetry: {diagnostics.maximum_asymmetry:.12e}")
    print(f"  Minimum eigenvalue: {diagnostics.minimum_eigenvalue:.12e}")
    print(f"  Maximum eigenvalue: {diagnostics.maximum_eigenvalue:.12e}")
    print(f"  Condition number: {diagnostics.condition_number:.6f}")


def main() -> None:
    """Run covariance estimation and print the Phase 3 validation checks."""

    parser = argparse.ArgumentParser(
        description="Estimate and validate annualized covariance matrices."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    arguments = parser.parse_args()

    _, _, estimates, diagnostics = load_covariance_data(arguments.config)
    print("Covariance estimation completed successfully.")
    _print_diagnostics("Sample", diagnostics["sample"])
    _print_diagnostics("Ledoit-Wolf", diagnostics["ledoit_wolf"])
    print(f"Shrinkage intensity: {estimates.shrinkage_intensity:.12f}")


if __name__ == "__main__":
    main()
