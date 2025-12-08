# Moving Average Indicators Analysis

This document categorizes the moving average indicators found in simple_trade/moving_average into Beginner, Intermediate, and Advanced levels based on their complexity, calculation method, and usage.

## Beginner Level
These indicators are fundamental, widely used, and easy to understand. They typically involve straightforward averaging calculations.

| Indicator | Classification | Full Name | Description | Key Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **SMA** | Fundamental | Simple Moving Average | The arithmetic mean of prices over a specific period. | Simplest form, equal weighting to all data points. Good for identifying general trends but lags significantly. |
| **EMA** | Fundamental | Exponential Moving Average | A weighted moving average that gives more importance to recent price data. | Reacts faster to price changes than SMA. Widely used in trend following. |
| **WMA** | Fundamental | Weighted Moving Average | Assigns linearly decreasing weights to older data points. | More responsive than SMA, less than EMA. Offers a balance between smoothing and lag. |
| **SOA** | Fundamental | Smoothed Moving Average | A very smooth moving average, similar to Wilder's MA. | Extremely smooth, slow to react. Best for very long-term trend identification. |

## Intermediate Level
These indicators introduce modifications to reduce lag or improve smoothing using more specific mathematical concepts (e.g., double smoothing, volume weighting, geometry).

| Indicator | Classification | Full Name | Description | Key Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **TMA** | Fundamental | Triangular Moving Average | A double-smoothed SMA (SMA of an SMA). | Prioritizes the middle of the data window. Very smooth but has significant lag. |
| **DEM** | Fundamental | Double Exponential Moving Average | Reduces lag by subtracting the EMA of the EMA from the doubled EMA. | Faster and more responsive than standard EMA. Good for catching trends early. |
| **TEM** | Fundamental | Triple Exponential Moving Average | Uses triple smoothing to further reduce lag compared to DEM and EMA. | Extremely responsive, almost zero lag in some conditions. Good for scalping. |
| **HMA** | Fundamental | Hull Moving Average | Uses weighted moving averages with square root period to reduce lag. | "Hugs" the price action closely. Excellent balance of smoothness and responsiveness. |
| **SWM** | Fundamental | Sine Weighted Moving Average | Uses sine wave weighting to emphasize the middle of the window. | Smooth transitions, reduces lag compared to SMA. Good for cyclical analysis. |
| **ZMA** | Non-fundamental | Zero-Lag Moving Average | Pre-adjusts input data for momentum before applying EMA. | Tracks price very closely, sensitive to reversals. |
| **VMA** | Fundamental | Volume Moving Average | Uses volume as the weighting factor for the average. | Integrates volume info. Rising VMA indicates trend supported by volume. |

## Advanced Level
These indicators are adaptive or use complex statistical/mathematical models. They adjust their parameters dynamically based on market conditions (volatility, noise, fractal dimension).

| Indicator | Classification | Full Name | Description | Key Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **AMA** | Non-fundamental | Adaptive Moving Average (Kaufman) | Adjusts smoothing based on the Efficiency Ratio (ER) (noise vs. trend). | Fast in trends, flat in choppy markets. Excellent for filtering noise. |
| **FMA** | Non-fundamental | Fractal Adaptive Moving Average | Adapts smoothing based on the fractal dimension of the price. | Uses fractal geometry to determine market state. Very sophisticated adaptation. |
| **GMA** | Non-fundamental | Guppy Multiple Moving Average | Uses two groups of EMAs (short-term and long-term) to visualize trend strength. | Visual indicator showing compression (consolidation) and expansion (trend). |
| **JMA** | Non-fundamental | Jurik Moving Average | A sophisticated smoothing algorithm designed for minimal lag and overshoot. | (Approximation in this repo). Famous for being "best in class" for low lag and smoothness. |
| **ALM** | Non-fundamental | Arnaud Legoux Moving Average | Uses Gaussian distribution for weighting. | Highly customizable (offset, sigma). Can act as a filter or reactive average. |
| **ADS** | Non-fundamental | Adaptive Deviation-Scaled MA | Adjusts smoothing based on price deviation from the average. | Reacts to volatility. Higher alpha when price moves away from average. |
| **VID** | Non-fundamental | Variable Index Dynamic Average | Adjusts smoothing based on the Chande Momentum Oscillator (CMO). | Adapts to market momentum. Volatility-adjusted smoothing. |
| **LSM** | Fundamental | Least Squares Moving Average | The endpoint of a linear regression line for the window. | Statistically fits the data. Can overshoot but identifies the "true" trend direction. |
| **TT3** | Non-fundamental | T3 Moving Average | A generalized DEMA developed by Tim Tillson that uses a volume factor to control responsiveness. | Extremely smooth, low lag. Uses multiple EMA passes. |
| **MAM** | Non-fundamental | MESA Adaptive Moving Average | Adapts to market cycles using the Hilbert Transform to distinguish trend vs cycle modes. | Uses phase rate of change to adjust alpha. Excellent for cyclical markets. |
| **EVW** | Non-fundamental | Elastic Volume Weighted MA | A volume-weighted MA that treats time as volume blocks, stretching/compressing time based on activity. | Adapts to volume flow. High volume periods have more influence. |
| **TSF** | Non-fundamental | Time Series Forecast | Projects the linear regression line one period into the future. | Forecasts the next period's price. Can act as a leading indicator. |

## Summary Statistics
- **Total Indicators:** 24
- **Beginner:** 4
- **Intermediate:** 7
- **Advanced:** 14

The library provides a comprehensive suite of moving averages ranging from the basics to highly sophisticated adaptive algorithms suitable for algorithmic trading systems.
