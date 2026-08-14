"""Download, cache, validate, and load adjusted market prices."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
import yfinance as yf


@dataclass(frozen=True)
class DataValidationReport:
    """Store the key facts produced by price-data validation."""

    source: str
    original_rows: int
    cleaned_rows: int
    removed_rows: int
    column_count: int
    start_date: str
    end_date: str
    missing_values_after_cleaning: int


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load the YAML configuration file and return it as a dictionary."""

    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a top-level mapping.")
    return config


def get_universe_tickers(config: dict[str, Any]) -> list[str]:
    """Return the configured optimization tickers after validating the universe."""

    universe = config["universe"]
    sectors = universe["sectors"]
    expected_sector_count = universe["expected_sector_count"]
    tickers_per_sector = universe["tickers_per_sector"]
    expected_ticker_count = universe["expected_ticker_count"]

    if len(sectors) != expected_sector_count:
        raise ValueError(
            f"Expected {expected_sector_count} sectors, found {len(sectors)}."
        )

    tickers: list[str] = []
    for sector, sector_tickers in sectors.items():
        if len(sector_tickers) != tickers_per_sector:
            raise ValueError(
                f"Sector {sector!r} must contain {tickers_per_sector} tickers; "
                f"found {len(sector_tickers)}."
            )
        tickers.extend(str(ticker).upper() for ticker in sector_tickers)

    if len(tickers) != expected_ticker_count:
        raise ValueError(
            f"Expected {expected_ticker_count} tickers, found {len(tickers)}."
        )
    if len(set(tickers)) != len(tickers):
        duplicates = sorted(
            ticker for ticker in set(tickers) if tickers.count(ticker) > 1
        )
        raise ValueError(f"Duplicate universe tickers found: {duplicates}")
    return tickers


def get_required_tickers(config: dict[str, Any]) -> list[str]:
    """Return optimization tickers plus the configured market benchmark."""

    tickers = get_universe_tickers(config)
    benchmark = str(config["data"]["benchmark_ticker"]).upper()
    return tickers if benchmark in tickers else [*tickers, benchmark]


def _normalize_price_index(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert the price index to unique, timezone-free, ordered trading dates."""

    normalized = prices.copy()
    normalized.index = pd.to_datetime(normalized.index, errors="raise")
    if normalized.index.tz is not None:
        normalized.index = normalized.index.tz_localize(None)
    normalized.index = normalized.index.normalize()
    normalized.index.name = "Date"
    normalized = normalized.sort_index()

    if normalized.index.has_duplicates:
        duplicate_dates = normalized.index[normalized.index.duplicated()].unique()
        formatted = [date.date().isoformat() for date in duplicate_dates]
        raise ValueError(f"Duplicate trading dates found: {formatted}")
    return normalized


def _read_cached_prices(cache_path: Path) -> pd.DataFrame:
    """Read cached prices from CSV and return a normalized DataFrame."""

    cached = pd.read_csv(cache_path, index_col="Date", parse_dates=["Date"])
    return _normalize_price_index(cached)


def _cache_is_fresh(cache_path: Path, maximum_age_hours: float) -> bool:
    """Return whether the cache exists and is younger than the configured age."""

    if not cache_path.exists():
        return False
    modified_at = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
    cache_age = datetime.now(tz=timezone.utc) - modified_at
    return cache_age <= timedelta(hours=maximum_age_hours)


def _cache_covers_request(
    prices: pd.DataFrame,
    tickers: list[str],
    data_config: dict[str, Any],
) -> bool:
    """Return whether cached prices contain the requested columns and date span."""

    if prices.empty or not set(tickers).issubset(prices.columns):
        return False

    tolerance = pd.Timedelta(
        days=data_config["validation"]["start_date_tolerance_days"]
    )
    requested_start = pd.Timestamp(data_config["start_date"])
    if prices.index.min() > requested_start + tolerance:
        return False

    requested_end = data_config["end_date"]
    expected_end = (
        pd.Timestamp(requested_end) - pd.Timedelta(days=1)
        if requested_end
        else pd.Timestamp.today().normalize()
    )
    end_tolerance = pd.Timedelta(
        days=data_config["validation"]["end_date_tolerance_days"]
    )
    return prices.index.max() >= expected_end - end_tolerance


def _extract_price_field(
    downloaded: pd.DataFrame,
    price_field: str,
    tickers: list[str],
) -> pd.DataFrame:
    """Extract one configured price field from a yfinance download."""

    if downloaded.empty:
        raise ValueError("Yahoo Finance returned an empty table.")

    if isinstance(downloaded.columns, pd.MultiIndex):
        first_level = downloaded.columns.get_level_values(0)
        second_level = downloaded.columns.get_level_values(1)
        if price_field in first_level:
            prices = downloaded.xs(price_field, axis=1, level=0)
        elif price_field in second_level:
            prices = downloaded.xs(price_field, axis=1, level=1)
        else:
            raise ValueError(
                f"Price field {price_field!r} was not returned by Yahoo Finance."
            )
    else:
        prices = downloaded.copy()

    prices.columns = [str(column).upper() for column in prices.columns]
    missing_columns = sorted(set(tickers) - set(prices.columns))
    if missing_columns:
        raise ValueError(f"Yahoo Finance omitted tickers: {missing_columns}")

    prices = prices.loc[:, tickers].apply(pd.to_numeric, errors="coerce")
    prices = _normalize_price_index(prices)
    entirely_missing = prices.columns[prices.isna().all()].tolist()
    if entirely_missing:
        raise ValueError(
            "Yahoo Finance returned no usable prices for: "
            f"{entirely_missing}"
        )
    return prices


def _download_prices(
    tickers: list[str],
    data_config: dict[str, Any],
) -> pd.DataFrame:
    """Download adjusted daily closing prices with configured retry behavior."""

    download_config = data_config["download"]
    retry_attempts = download_config["retry_attempts"]
    retry_delay = download_config["retry_delay_seconds"]
    last_error: Exception | None = None

    for attempt in range(1, retry_attempts + 1):
        try:
            downloaded = yf.download(
                tickers=tickers,
                start=data_config["start_date"],
                end=data_config["end_date"],
                actions=False,
                threads=download_config["threads"],
                group_by="column",
                auto_adjust=data_config["auto_adjust"],
                repair=False,
                keepna=False,
                progress=download_config["show_progress"],
                interval="1d",
                timeout=download_config["timeout_seconds"],
                multi_level_index=True,
            )
            return _extract_price_field(
                downloaded,
                data_config["price_field"],
                tickers,
            )
        except Exception as error:  # yfinance can raise several network errors.
            last_error = error
            if attempt < retry_attempts:
                time.sleep(retry_delay)

    raise RuntimeError(
        f"Yahoo Finance download failed after {retry_attempts} attempts."
    ) from last_error


def _write_price_cache(prices: pd.DataFrame, cache_path: Path) -> None:
    """Write prices atomically so an interrupted run cannot corrupt the cache."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    prices.to_csv(temporary_path)
    temporary_path.replace(cache_path)


def _validate_and_clean_prices(
    prices: pd.DataFrame,
    data_config: dict[str, Any],
    source: str,
) -> tuple[pd.DataFrame, DataValidationReport]:
    """Validate price histories and apply the configured missing-data policy."""

    validation = data_config["validation"]
    if len(prices) < validation["minimum_observations"]:
        raise ValueError(
            f"Only {len(prices)} price rows were found; at least "
            f"{validation['minimum_observations']} are required."
        )

    entirely_missing = prices.columns[prices.isna().all()].tolist()
    if entirely_missing:
        raise ValueError(f"Tickers with no price history: {entirely_missing}")

    missing_fractions = prices.isna().mean()
    excessive_missing = missing_fractions[
        missing_fractions > validation["maximum_missing_fraction_per_ticker"]
    ]
    if not excessive_missing.empty:
        details = {
            ticker: round(float(fraction), 6)
            for ticker, fraction in excessive_missing.items()
        }
        raise ValueError(
            "Tickers exceed the allowed missing-price fraction: "
            f"{details}"
        )

    nonpositive_counts = (prices <= 0).sum()
    nonpositive = nonpositive_counts[nonpositive_counts > 0]
    if not nonpositive.empty:
        raise ValueError(
            "Nonpositive adjusted prices found: "
            f"{nonpositive.astype(int).to_dict()}"
        )

    start_limit = pd.Timestamp(data_config["start_date"]) + pd.Timedelta(
        days=validation["start_date_tolerance_days"]
    )
    late_starts = {
        ticker: prices[ticker].first_valid_index().date().isoformat()
        for ticker in prices.columns
        if prices[ticker].first_valid_index() is not None
        and prices[ticker].first_valid_index() > start_limit
    }
    if late_starts:
        raise ValueError(f"Tickers begin too late for the sample: {late_starts}")

    requested_end = data_config["end_date"]
    expected_end = (
        pd.Timestamp(requested_end) - pd.Timedelta(days=1)
        if requested_end
        else pd.Timestamp.today().normalize()
    )
    end_limit = expected_end - pd.Timedelta(
        days=validation["end_date_tolerance_days"]
    )
    early_ends = {
        ticker: prices[ticker].last_valid_index().date().isoformat()
        for ticker in prices.columns
        if prices[ticker].last_valid_index() is not None
        and prices[ticker].last_valid_index() < end_limit
    }
    if early_ends:
        raise ValueError(f"Tickers end too early for the sample: {early_ends}")

    missing_method = validation["missing_data_method"]
    if missing_method != "drop_rows":
        raise ValueError(
            f"Unsupported missing-data method: {missing_method!r}. "
            "Use 'drop_rows' to avoid inventing prices."
        )

    cleaned = prices.dropna(axis=0, how="any").copy()
    if len(cleaned) < validation["minimum_observations"]:
        raise ValueError(
            "Too few aligned observations remain after dropping missing rows: "
            f"{len(cleaned)}"
        )

    report = DataValidationReport(
        source=source,
        original_rows=len(prices),
        cleaned_rows=len(cleaned),
        removed_rows=len(prices) - len(cleaned),
        column_count=len(cleaned.columns),
        start_date=cleaned.index.min().date().isoformat(),
        end_date=cleaned.index.max().date().isoformat(),
        missing_values_after_cleaning=int(cleaned.isna().sum().sum()),
    )
    return cleaned, report


def load_price_data(
    config_path: str | Path = "config.yaml",
    force_refresh: bool | None = None,
) -> tuple[pd.DataFrame, DataValidationReport]:
    """Load validated prices from a usable cache or a fresh Yahoo download."""

    resolved_config_path = Path(config_path).expanduser().resolve()
    config = load_config(resolved_config_path)
    data_config = config["data"]
    tickers = get_required_tickers(config)
    cache_path = (
        resolved_config_path.parent / config["paths"]["price_cache"]
    ).resolve()
    should_refresh = (
        data_config["force_refresh"] if force_refresh is None else force_refresh
    )

    prices: pd.DataFrame | None = None
    source = "Yahoo Finance download"
    if not should_refresh and _cache_is_fresh(
        cache_path,
        data_config["cache_max_age_hours"],
    ):
        try:
            cached = _read_cached_prices(cache_path)
            if _cache_covers_request(cached, tickers, data_config):
                prices = cached.loc[:, tickers]
                source = "local cache"
        except (OSError, ValueError, pd.errors.ParserError):
            prices = None

    if prices is None:
        prices = _download_prices(tickers, data_config)
        _write_price_cache(prices, cache_path)

    return _validate_and_clean_prices(prices, data_config, source)


def main() -> None:
    """Run the data layer and print a concise validation summary."""

    parser = argparse.ArgumentParser(
        description="Download or load cached adjusted market prices."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore an existing cache and download fresh prices.",
    )
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    universe_count = len(get_universe_tickers(config))
    prices, report = load_price_data(arguments.config, arguments.refresh)
    benchmark = config["data"]["benchmark_ticker"]

    print("Adjusted closing prices loaded successfully.")
    print(f"Source: {report.source}")
    print(f"Optimization assets: {universe_count}")
    print(f"Benchmark: {benchmark}")
    print(f"Total columns: {report.column_count}")
    print(f"Rows before alignment: {report.original_rows}")
    print(f"Rows removed for missing prices: {report.removed_rows}")
    print(f"Final rows: {report.cleaned_rows}")
    print(f"Date range: {report.start_date} to {report.end_date}")
    print(
        "Missing values after cleaning: "
        f"{report.missing_values_after_cleaning}"
    )
    print(f"First tickers: {', '.join(prices.columns[:5])}")


if __name__ == "__main__":
    main()
