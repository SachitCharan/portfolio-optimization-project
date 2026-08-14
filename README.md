# Portfolio Optimization and Backtesting

This project asks whether a long-only maximum-Sharpe optimizer can beat simple equal weighting out of sample. Using real adjusted U.S. stock prices across 49 stocks and seven sectors, the optimizer achieves a slightly higher historical CAGR but **underperforms the 1/N equal-weight benchmark on realized Sharpe ratio**, both before and after trading costs. That result is consistent with [DeMiguel, Garlappi, and Uppal (2009)](https://doi.org/10.1093/rfs/hhm075), who found that estimation error often offsets the theoretical gains from optimized diversification out of sample. This project is not a replication of their study; it demonstrates the same practical tension in a different universe and period.

![Efficient frontier with Monte Carlo portfolios](outputs/figures/efficient_frontier.png)

*This frontier is an in-sample diagnostic built from full-sample historical estimates. It shows that the numerical optimizer finds the feasible mean-variance boundary; it is not evidence that those expected returns will be realized out of sample.*

## Results

The backtest covers **January 4, 2018 through August 13, 2026**. Each allocation uses only the previous 756 trading days, then remains invested for the next 63 trading days before rebalancing. The full-sample optimizer reports an in-sample Sharpe ratio of **1.521**; the walk-forward strategy delivers only **0.737 gross** and **0.732 after 10-bps trading costs**. Equal weighting delivers **0.837 gross** and **0.836 after costs**.

| Cost assumption | Strategy | CAGR | Annualized volatility | Sharpe ratio | Maximum drawdown | Calmar ratio | Average turnover |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 bps | Maximum Sharpe | 19.03% | 23.23% | 0.737 | -32.55% | 0.585 | 30.48% |
| 0 bps | Equal Weight | 18.56% | 18.92% | 0.837 | -36.32% | 0.511 | 4.74% |
| 0 bps | SPY | 14.83% | 19.12% | 0.663 | -33.72% | 0.440 | 0.00% |
| 10 bps | Maximum Sharpe | 18.89% | 23.23% | 0.732 | -32.71% | 0.577 | 30.48% |
| 10 bps | Equal Weight | 18.54% | 18.92% | 0.836 | -36.32% | 0.510 | 4.74% |
| 10 bps | SPY | 14.83% | 19.12% | 0.663 | -33.72% | 0.440 | 0.00% |

The optimized strategy's extra gross CAGR does not compensate for its higher volatility on a Sharpe-ratio basis. Costs also hurt it more because its average quarterly one-way turnover is over six times that of equal weight. These are historical results, not expected future returns. The machine-readable table is available at [`outputs/tables/performance_summary.csv`](outputs/tables/performance_summary.csv).

![Out-of-sample cumulative returns](outputs/figures/cumulative_returns.png)

![Out-of-sample drawdowns](outputs/figures/drawdowns.png)

![Maximum-Sharpe sector allocation](outputs/figures/sector_allocation.png)

## Methodology

### Data

- Source: Yahoo Finance through `yfinance`
- Full estimation sample: January 2, 2015 through August 13, 2026
- Assets: 49 large U.S. stocks, grouped into Technology, Financials, Healthcare, Consumer Discretionary, Energy, Industrials, and Consumer Staples
- Benchmark: SPY
- Price series: daily adjusted close (`auto_adjust=True`), which accounts for splits and distributions
- Missing values: reported and removed by dropping affected dates; prices are never filled or fabricated
- Returns: daily simple returns with 252 trading days per year

The ticker universe and every model setting are stored in [`config.yaml`](config.yaml).

### Risk and optimization

- Expected returns use annualized historical arithmetic means.
- Both sample covariance and Ledoit-Wolf shrinkage covariance are estimated and validated.
- Ledoit-Wolf covariance is used by the optimizer to reduce estimation noise.
- Full-sample Ledoit-Wolf shrinkage intensity is 0.012286. Across the 35 walk-forward rebalances, shrinkage ranges from **0.014757 to 0.053275**; `run_all.py` prints every rebalance value.
- SLSQP solves the global minimum-variance, maximum-Sharpe, and target-return problems.
- Portfolios are long-only, fully invested, and bounded between 0% and 100% per asset.
- The model uses a constant 3% annual risk-free rate and no sector cap.
- The efficient frontier contains 50 target-return portfolios from the global minimum-variance return through the highest individual estimated asset return.
- A reproducible Monte Carlo simulation generates 25,000 Dirichlet-weighted portfolios with concentration parameter `alpha = 0.1` and random seed 42. The smaller alpha deliberately samples more concentrated corner portfolios; none crosses the analytical frontier.

### Walk-forward backtest

At each rebalance, the strategy estimates expected returns and Ledoit-Wolf covariance from exactly the previous **756 trading days**. It then calculates maximum-Sharpe weights and applies them only to the following **63 trading days**. Assets drift naturally during each holding period instead of being reset to target weights every day.

This separation is what prevents look-ahead bias:

1. The training window ends on the trading day before the holding window begins.
2. No return from a holding window is used to choose that window's weights.
3. The rebalance log checks this date ordering for all 35 periods.
4. The automated test independently rebuilds the first allocation using only its recorded training dates and verifies that the weights match.

The equal-weight benchmark follows the same quarterly holding schedule across the same 49 stocks. SPY is held over the identical out-of-sample dates. The default backtest charges **10 basis points per unit of one-way turnover**, while the results table also reports a 0-bps gross scenario. Reported turnover is average one-way turnover per scheduled rebalance with the initial funding trade excluded.

## Repository structure

```text
portfolio-optimization-project/
├── config.yaml
├── run_all.py
├── requirements.txt
├── src/
│   ├── data.py
│   ├── returns.py
│   ├── covariance.py
│   ├── optimize.py
│   ├── frontier.py
│   ├── montecarlo.py
│   ├── backtest.py
│   ├── metrics.py
│   └── plots.py
├── tests/
│   └── test_math.py
└── outputs/
    ├── figures/
    └── tables/
```

## Reproduce the project

Python 3.10 or newer and Git are required.

```bash
git clone https://github.com/SachitCharan/portfolio-optimization-project.git
cd portfolio-optimization-project
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python run_all.py
```

The first run downloads real market data and stores it locally in `data/adjusted_close_prices.csv`. The data file is intentionally excluded from Git because it can be recreated from Yahoo Finance. Later runs use the cache until it reaches the configured maximum age.

Run the automated validation suite with:

```bash
python -m pytest -q
```

## Limitations

### Survivorship bias

The 49-stock universe is a fixed list of large companies selected using information available today and then carried backward through the historical sample. Companies that failed, were acquired, delisted, or simply stopped being prominent are absent. This survivorship and selection bias can make both the asset universe and its historical performance look stronger than a portfolio universe that could actually have been selected at each date.

- Historical average returns are noisy and weak predictors of future returns, so optimized weights can change sharply when the sample changes. The gap between the 1.521 in-sample Sharpe estimate and 0.737 gross out-of-sample realization is one example of that estimation risk.
- The model uses a constant 3% annual risk-free rate throughout the entire sample. Actual short-term rates changed substantially between 2015 and 2026, so the constant assumption affects both optimized weights and reported Sharpe ratios.
- The 10-bps scenario is a simplified proportional cost assumption. It still excludes taxes, time-varying bid-ask spreads, slippage, liquidity constraints, and market impact.
- No sector cap is imposed. The optimizer can produce concentrated sector or individual-stock exposures, as the sector chart demonstrates.
- Results depend on the selected dates, assets, risk-free rate, rebalance schedule, and Yahoo Finance data quality or revisions.
- Ledoit-Wolf shrinkage improves covariance stability but does not remove expected-return estimation error or guarantee strong out-of-sample performance.

This repository is an educational research project and is not investment advice.

## License

Released under the [MIT License](LICENSE).
