# Momentum Indicators Analysis

This document categorizes the momentum indicators found in `simple_trade/momentum` into Beginner, Intermediate, and Advanced levels based on their complexity, calculation method, and usage.

## Beginner Level
These indicators are fundamental, widely used, and easy to understand. They typically involve straightforward momentum calculations.

| Indicator | Classification | Full Name | Description | Key Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **RSI** | Fundamental | Relative Strength Index | The classic momentum oscillator measuring speed and change of price movements. | 0-100 range. Identifies overbought/oversold conditions (typically 70/30). |
| **ROC** | Fundamental | Rate of Change | Measures the percentage change in price between the current price and a past price. | Pure momentum. Shows velocity of price change. |
| **MAC** | Fundamental | Moving Average Convergence Divergence | Trend-following momentum indicator showing the relationship between two moving averages. | Uses signal line crossovers. Histogram shows difference between MACD and signal. |
| **STO** | Fundamental | Stochastic Oscillator | Compares a particular closing price to a range of its prices over a certain period. | Consists of %K and %D lines. Identifies turning points in ranges. |
| **WIL** | Fundamental | Williams %R | Momentum indicator measuring overbought and oversold levels. | Inverse of Fast Stochastic. Range 0 to -100. |
| **PSY** | Fundamental | Psychological Line | Ratio of rising periods to total periods. | Gauges market sentiment. Simple ratio of up days. |
| **BOP** | Fundamental | Balance of Power | Measures the strength of buyers versus sellers. | assesses ability to push price to extremes. Range -1 to 1. |
| **QST** | Non-fundamental | Qstick | Moving average of the difference between opening and closing prices. | Quantitative candlestick analysis. Captures buying/selling pressure. |

## Intermediate Level
These indicators introduce more complex calculations or combine multiple concepts to refine momentum analysis.

| Indicator | Classification | Full Name | Description | Key Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **CCI** | Fundamental | Commodity Channel Index | Measures current price level relative to an average price level over a given period. | Detects cyclic turns. Unbounded oscillator. |
| **DPO** | Fundamental | Detrended Price Oscillator | Removes trend from prices to let you identify cycles. | Peaks and troughs estimate cycle length. Shifted back in time. |
| **PPO** | Fundamental | Percentage Price Oscillator | Similar to MACD but uses percentage difference. | Allows comparison of momentum across assets with different prices. |
| **ERI** | Fundamental | Elder Ray Index | Measures the amount of buying and selling pressure (Bull/Bear Power). | Components are Bull Power and Bear Power relative to EMA. |
| **WAD** | Fundamental | Williams Accumulation/Distribution | Accumulation/Distribution based on price changes. | Uses price to proxy volume flow. Tracks market pressure. |
| **CMO** | Fundamental | Chande Momentum Oscillator | Uses momentum to calculate relative strength. | Bounded -100 to +100. Measures direct momentum. |
| **IMI** | Non-fundamental | Intraday Momentum Index | Combines candlestick analysis with RSI. | "RSI of Candlesticks". Designed for intraday trading. |
| **PGO** | Non-fundamental | Pretty Good Oscillator | Measures distance from SMA normalized by ATR. | Deviation from average in units of volatility. |
| **RMI** | Non-fundamental | Relative Momentum Index | Variation of RSI using momentum periods. | Considers momentum over a period rather than single bar change. |

## Advanced Level
These indicators are adaptive, use complex statistical/mathematical models, or combine multiple timeframes/indicators.

| Indicator | Classification | Full Name | Description | Key Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **ULT** | Fundamental | Ultimate Oscillator | Combines short, medium, and long-term periods. | Weighted sum of 3 oscillators. Reduces false signals. |
| **TSI** | Fundamental | True Strength Index | Double smoothed momentum indicator. | Low lag, filters noise. Uses double exponential smoothing. |
| **KST** | Fundamental | Know Sure Thing | Summed rate of change indicator. | Sum of 4 smoothed ROC curves. Capture long-term momentum. |
| **AWO** | Fundamental | Awesome Oscillator | Infinite impulse response filter (approx) of market momentum. | Histogram of difference between 5 and 34 SMA of median price. |
| **SRI** | Non-fundamental | Stochastic RSI | Stochastic oscillator applied to the RSI values. | "Indicator of an indicator". Highly sensitive to RSI changes. |
| **STC** | Non-fundamental | Schaff Trend Cycle | Oscillator combining MACD and Stochastic to identify trends with less lag. | Uses cycle component to faster identify trends. |
| **TTM** | Non-fundamental | TTM Squeeze | Volatility and momentum indicator identifying consolidation and breakout. | visualizes volatility compression (squeeze) and momentum. |
| **VOR** | Non-fundamental | Vortex Indicator | Captures positive and negative trend movements. | Based on water flow vortex logic. Two oscillating lines (VI+, VI-). |
| **LSI** | Non-fundamental | Laguerre RSI | Uses Laguerre filter for less lag and noise. | Uses Laguerre transform for superior smoothing/reaction. |
| **FIS** | Non-fundamental | Fisher Transform | Converts prices to a Gaussian normal distribution. | Sharpens turning points. Unbounded. |
| **COG** | Non-fundamental | Center of Gravity | Zero-lag oscillator spotting turning points. | Based on physics center of gravity. Very reactive. |
| **CRS** | Non-fundamental | Connors RSI | Composite indicator (RSI + Streak RSI + Rank). | Three components for short-term mean reversion strategies. |
| **RVG** | Non-fundamental | Relative Vigor Index | Measures conviction of recent price action. | Compares Close-Open to High-Low. Similar to Stochastics. |
| **MSI** | Non-fundamental | Momentum Strength Index | Custom indicator quantifying strength of momentum. | Quantifies magnitude of trend/momentum strength. |

## Summary Statistics
- **Total Indicators:** 31
- **Beginner:** 8
- **Intermediate:** 9
- **Advanced:** 14

The library provides a diverse set of momentum indicators, ranging from classic oscillators to modern, adaptive, and noise-reduced algorithms.
