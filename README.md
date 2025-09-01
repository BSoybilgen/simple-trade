# simple-trade

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![PyPI](https://img.shields.io/pypi/v/simple-trade.svg?style=flat-square)](https://pypi.org/project/simple-trade/)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![build](https://github.com/BSoybilgen/simple-trade/actions/workflows/tests.yml/badge.svg)](https://github.com/BSoybilgen/simple-trade/actions/workflows/tests.yml)
[![codecov.io](https://codecov.io/github/BSoybilgen/simple-trade/coverage.svg?branch=master)](https://codecov.io/github/BSoybilgen/simple-trade)

A Python library that allows you to compute technical indicators and build trade strategies in a simple way.

## Features

*   **Data Fetching:** Easily download historical stock data using `yfinance`.
*   **Technical Indicators:** Compute a variety of technical indicators:
    *   Trend (e.g., Moving Averages, MACD, ADX)
    *   Momentum (e.g., RSI, Stochastics)
    *   Volatility (e.g., Bollinger Bands, ATR)
    *   Volume (e.g., On-Balance Volume)
*   **Trading Strategies:** Implement and backtest common trading strategies:
    *   Cross Trade Strategies (`cross_trade`)
    *   Band Trading Strategies (`band_trade`)
*   **Backtesting:** Evaluate the performance of your trading strategies on historical data.
*   **Optimization:** Optimize strategy parameters using techniques like grid search.
*   **Plotting:** Visualize data, indicators, and backtest results using `matplotlib`.
*   **Combining:** Combine different strategies to create a more complex strategy.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url> # Replace with your repo URL
    cd simple-trade
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv myenv
    # On Windows
    myenv\Scripts\activate
    # On macOS/Linux
    source myenv/bin/activate
    ```
3.  **Install the package and dependencies:**
    ```bash
    pip install .
    ```
    Alternatively, installed with PyPI:
    ```bash
    pip install simple-trade
    ```

## Dependencies

*   Python >= 3.10
*   [yfinance](https://pypi.org/project/yfinance/)
*   [pandas](https://pandas.pydata.org/)
*   [numpy](https://numpy.org/)
*   [joblib](https://joblib.readthedocs.io/en/latest/)
*   [matplotlib](https://matplotlib.org/)

These will be installed automatically when you install `simple-trade` using `pip`.

## Basic Usage

### Calculate Indicators

Here's a quick example of how to download data and compute a technical indicator:

```python
# Load Packages and Functions
import pandas as pd
from simple_trade import compute_indicator, download_data

# Step 1: Download data
symbol = 'TSLA'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["window"] = 14
data, columns, fig = compute_indicator(
    data=data,
    indicator='adx',
    parameters=parameters
)

# Step 3: Display the plot
fig.show()
```

**Plot of Results**
<img src="https://i.imgur.com/JBpd0qo.png" alt="Figure 1" width="900" height="600">

### Backtesting Strategies

Use the `premade_backtesting` module to simulate strategies like moving average crossovers (`cross_trade`) or Bollinger Band breakouts (`band_trade`).

```python
# Load Packages and Functions
import pandas as pd
from simple_trade import download_data
from simple_trade import premade_backtest
from simple_trade import CrossTradeBacktester

# Step 1: Download data
symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2022-12-31'
interval = '1d'
data = download_data(symbol, start_date, end_date, interval=interval)

# Step 2: Set Global Parameters
global_parameters = {
    'initial_cash': 10000,
    'commission_long': 0.001,
    'commission_short': 0.001,
    'short_borrow_fee_inc_rate': 0.0,
    'long_borrow_fee_inc_rate': 0.0,
    'trading_type': 'long',
    'day1_position': 'none',
    'risk_free_rate': 0.0,
}

# Step 3: Set Strategy Parameters
strategy_name = 'sma'
specific_parameters = {
    'short_window': 25,
    'long_window': 75,
    'fig_control': 1,
}

# Step 4: Run Backtest
parameters = {**global_parameters, **specific_parameters}
results, portfolio, fig = premade_backtest(data, strategy_name, parameters)
```

**Plot of Results**
<img src="https://i.imgur.com/4qxr0dp.png" alt="Figure 2" width="900" height="600">


### Optimizing Strategies

The `premade_optimizer` module allows you to find the best parameters for your strategy (e.g., optimal moving average windows).

```python
# Load Packages and Functions
from simple_trade import download_data
from simple_trade.premade_optimizer import premade_optimizer

# Step 1: Load Data
ticker = "AAPL"
start_date = "2020-01-01"
end_date = "2023-12-31"

data = download_data(ticker, start_date, end_date)

# Step 2: Load Optimization Parameters
# Define the parameter grid to search
param_grid = {
    'short_window': [10, 20, 30],
    'long_window': [50, 100, 150],
}

# Step 3: Set Base Parameters
base_params = {
    'initial_cash': 100000.0,
    'commission_long': 0.001,         # 0.1% commission
    'commission_short': 0.001,
    'trading_type': 'long',           # Only long trades
    'day1_position': 'none',
    'risk_free_rate': 0.02,
    'metric': 'total_return_pct',     # Metric to optimize
    'maximize': True,                 # Maximize the metric
    'parallel': False,                # Sequential execution for this example
    'fig_control': 0                  # No plotting during optimization
}

# Step 4: Run Optimization
best_results, best_params, all_results = premade_optimizer(
    data=data,
    strategy_name='sma',
    parameters=base_params,
    param_grid=param_grid
)

# Show top 3 parameter combinations
print("\nTop 3 SMA Parameter Combinations:")
sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
for i, result in enumerate(sorted_results[:3]):
    print(f"  {i+1}. {result['params']} -> {result['score']:.2f}%")
```

**Output of Results**
```
Top 3 RSI Parameter Combinations:
  1. {'short_window': 10, 'long_window': 50} -> 99.87%
  2. {'short_window': 20, 'long_window': 50} -> 85.69%
  3. {'short_window': 30, 'long_window': 50} -> 67.08%
```

## Examples

For more detailed examples, please refer to the Jupyter notebooks in the `/examples` directory:

*   `/examples/indicators`: Demonstrations of various technical indicators.
*   `/examples/backtest`: Examples of backtesting different strategies.
*   `/examples/optimize`: Examples of optimizing strategy parameters.
*   `/examples/combine_trade`: Examples of combining different strategies.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue. (Further details can be added here if needed).

## License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.
