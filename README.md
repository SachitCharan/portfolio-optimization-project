# Portfolio Optimization and Backtesting

Portfolio theory makes optimization look clean on paper: estimate returns and risk, then choose the best mix of assets. I built this project to see how well that idea holds up when the model has to make decisions using only information that would have been available at the time.

The program downloads real market data for 49 U.S. stocks, estimates a covariance matrix, builds an efficient frontier, and runs a walk-forward backtest against equal weighting and SPY. The main result was not that the optimizer easily won. It earned a slightly higher historical return than equal weighting, but it also took more risk and finished with a lower out-of-sample Sharpe ratio. That difference between a strong in-sample model and a weaker out-of-sample test is the part of the project I found most useful.

![Efficient frontier with Monte Carlo portfolios](outputs/figures/efficient_frontier.png)

*The efficient frontier uses full-sample estimates. It checks that the optimizer finds the mean-variance boundary, but it is not an out-of-sample performance result.*

## What the project does

- Downloads adjusted daily prices from Yahoo Finance and caches them locally
- Calculates daily simple returns and annualized return and volatility estimates
- Compares the sample covariance matrix with Ledoit-Wolf shrinkage covariance
- Solves minimum-variance, maximum-Sharpe, and target-return portfolios with SLSQP
- Traces a 50-point efficient frontier and compares it with 25,000 random portfolios
- Runs a quarterly walk-forward backtest without using future data
- Compares the optimized strategy with equal weighting and SPY
- Reports performance before and after a simple transaction-cost assumption
- Saves charts and a machine-readable performance table

## Main result

The backtest runs from **January 4, 2018 through August 13, 2026**. At each rebalance, the optimizer trains on the previous 756 trading days, chooses a new portfolio, and holds it for the next 63 trading days.

| Cost assumption | Strategy | CAGR | Annualized volatility | Sharpe ratio | Maximum drawdown | Calmar ratio | Average turnover |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 bps | Maximum Sharpe | 19.03% | 23.23% | 0.737 | -32.55% | 0.585 | 30.48% |
| 0 bps | Equal Weight | 18.56% | 18.92% | 0.837 | -36.32% | 0.511 | 4.74% |
| 0 bps | SPY | 14.83% | 19.12% | 0.663 | -33.72% | 0.440 | 0.00% |
| 10 bps | Maximum Sharpe | 18.89% | 23.23% | 0.732 | -32.71% | 0.577 | 30.48% |
| 10 bps | Equal Weight | 18.54% | 18.92% | 0.836 | -36.32% | 0.510 | 4.74% |
| 10 bps | SPY | 14.83% | 19.12% | 0.663 | -33.72% | 0.440 | 0.00% |

The maximum-Sharpe strategy had the highest CAGR, but its volatility was also much higher. Equal weighting produced the best realized Sharpe ratio, meaning it earned more return per unit of risk. The optimized portfolio also traded much more: average quarterly one-way turnover was 30.48%, compared with 4.74% for equal weight.

The in-sample maximum-Sharpe portfolio had an estimated Sharpe ratio of **1.521**, while the gross walk-forward result was only **0.737**. The optimizer worked mathematically, but its inputs were noisy. In particular, historical average returns did not predict the next holding period very well. This result is consistent with the broader finding in [DeMiguel, Garlappi, and Uppal (2009)](https://doi.org/10.1093/rfs/hhm075) that estimation error can erase the theoretical advantage of optimized portfolios. This project is not a replication of their paper, but it runs into the same basic problem.

The exact results are also saved in [`outputs/tables/performance_summary.csv`](outputs/tables/performance_summary.csv).

![Out-of-sample cumulative returns](outputs/figures/cumulative_returns.png)

![Out-of-sample drawdowns](outputs/figures/drawdowns.png)

![Maximum-Sharpe sector allocation](outputs/figures/sector_allocation.png)

## How the model works

### Data and returns

The asset universe contains seven stocks from each of seven sectors: Technology, Financials, Healthcare, Consumer Discretionary, Energy, Industrials, and Consumer Staples. SPY is used as the market benchmark. The current results use adjusted prices from **January 2, 2015 through August 13, 2026**.

Prices come from Yahoo Finance through `yfinance` with `auto_adjust=True`, so splits and distributions are reflected in the series. The program reports missing values and drops dates with incomplete observations instead of filling or inventing prices. It then calculates daily simple returns. Mean returns are annualized by multiplying by 252, while volatility is annualized by multiplying daily standard deviation by the square root of 252.

The full ticker list and all model settings are in [`config.yaml`](config.yaml).

### Covariance estimation

The covariance matrix describes how the stocks move together, so it drives the optimizer's estimate of portfolio risk. The project calculates two versions:

1. **Sample covariance**, estimated directly from the historical returns.
2. **Ledoit-Wolf covariance**, which shrinks the noisy sample estimate toward a more stable target.

Shrinkage matters because optimization can react too strongly to small errors in a large covariance matrix. With 49 assets, even minor estimation noise can create extreme weights. Ledoit-Wolf does not eliminate that problem, but it usually gives the optimizer a more stable risk estimate. The full-sample shrinkage intensity was **0.012286**, and the 35 walk-forward estimates ranged from **0.014757 to 0.053275**.

### Optimization and the frontier

The project uses `scipy.optimize.minimize` with the SLSQP algorithm. It solves for:

- the global minimum-variance portfolio
- the maximum-Sharpe portfolio
- the minimum-variance portfolio for a chosen target return

The default portfolios are long-only, every weight stays between 0 and 1, and the weights must add to 1. The model uses a constant 3% annual risk-free rate and does not impose a sector cap.

The efficient frontier is built by solving 50 target-return problems. A Monte Carlo simulation then generates 25,000 random portfolios using a Dirichlet distribution with `alpha = 0.1`. The low alpha makes the simulation sample more concentrated portfolios near the edges of the feasible set. No simulated portfolio crosses the analytical frontier, which is an important check that the optimizer is finding the boundary rather than just a good random portfolio.

### Walk-forward backtest

The backtest is the most important part of the project because it separates estimation from evaluation:

```text
756 trading days of training data -> choose weights -> hold for 63 trading days -> repeat
```

The training window always ends before the holding period begins. No return from a holding period is allowed to influence the weights used during that period. The weights drift with market performance while the portfolio is being held, then reset at the next scheduled rebalance.

The equal-weight benchmark follows the same 63-day schedule across the same 49 stocks, and SPY is measured over the same out-of-sample dates. The default cost scenario charges 10 basis points per unit of one-way turnover. The results table also includes a zero-cost case so the effect of trading can be seen directly.

## Repository layout

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

Each file in `src/` handles one part of the pipeline. `run_all.py` connects those pieces but does not contain the analytical logic itself. Parameters such as dates, ticker symbols, bounds, simulation size, and rebalance timing live in `config.yaml` instead of being hidden inside the Python files.

## Run the project

Python 3.12 or newer and Git are required.

```bash
git clone https://github.com/SachitCharan/portfolio-optimization-project.git
cd portfolio-optimization-project
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python run_all.py
```

The first run downloads real market data and saves it as `data/adjusted_close_prices.csv`. That file is excluded from Git because it can be downloaded again. Later runs use the cache until it is older than the limit set in `config.yaml`.

Because Yahoo Finance continues to update, a new run may produce different end dates and slightly different results from the August 13, 2026 snapshot reported above.

Run the tests with:

```bash
python -m pytest -q
```

The tests check the portfolio constraints, covariance matrix dimensions and positive semidefiniteness, frontier shape, Monte Carlo comparison, performance outputs, and the date ordering used to prevent look-ahead bias. The backtest test also rebuilds the first portfolio from its recorded training window and checks that the reconstructed weights match.

## Limitations

This is a historical experiment, not a prediction of what the stocks will earn next.

- **Expected returns are weak inputs.** Historical averages are noisy, and the optimizer can change its weights sharply when those estimates move.
- **The universe has survivorship bias.** The 49 companies were selected using information available today and then carried backward. Failed, acquired, or less prominent companies are missing.
- **The risk-free rate is simplified.** The model uses 3% for the full period even though short-term interest rates changed over time.
- **Trading costs are simplified.** The 10-bps case does not include taxes, changing bid-ask spreads, slippage, liquidity limits, or market impact.
- **Concentration is possible.** There is no sector cap, so the optimizer may put a large share of the portfolio in a small number of stocks or sectors.
- **Results depend on the setup.** Different dates, stocks, rebalance schedules, or data revisions could change the outcome.
- **Shrinkage only addresses part of the problem.** It improves covariance stability, but it does not fix expected-return estimation error or guarantee better out-of-sample performance.

This repository is an educational research project and is not investment advice.

## License

Released under the [MIT License](LICENSE).
