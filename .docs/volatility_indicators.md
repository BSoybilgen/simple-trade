# Volatility Indicators Analysis

This document categorizes the volatility indicators found in `simple_trade/volatility` into Beginner, Intermediate, and Advanced levels based on their complexity, calculation method, and usage.

## Beginner Level
These indicators are fundamental, widely used, and easy to understand. They typically involve standard deviation or simple range calculations.

| Indicator | Classification | Full Name | Description | Key Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **ATR** | Fundamental | Average True Range | Measures market volatility by decomposing the entire range of an asset price for a given period. | uses Wilder's smoothing. Standard measure of volatility magnitude. |
| **BOL** | Fundamental | Bollinger Bands | Statistical chart illustrating relative high and low prices in relation to their average. | Uses standard deviation envelopes. Squeezes indicate low volatility. |
| **DON** | Fundamental | Donchian Channels | Plots the highest high and lowest low over a specified period. | definitive measure of high-low range. Breakout verification. |
| **KEL** | Fundamental | Keltner Channels | Volatility-based envelopes set above and below an exponential moving average using ATR. | Similar to Bollinger Bands but uses ATR for width. Smoother than BOL. |
| **HIV** | Fundamental | Historical Volatility | Annualized standard deviation of logarithmic returns. | Also known as realized volatility. Standard statistical risk measure. |
| **NAT** | Fundamental | Normalized ATR | ATR expressed as a percentage of the closing price. | Allows volatility comparison across different assets/prices. |
| **PCW** | Fundamental | Price Channel Width | Measures the width of a Donchian channel as a percentage of the closing price. | Simple percentage-based volatility measure. |

## Intermediate Level
These indicators use more specific mathematical concepts to measure volatility, such as trendiness efficiency, median deviation, or geometric ranges.

| Indicator | Classification | Full Name | Description | Key Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **ACB** | Fundamental | Acceleration Bands | Creates dynamic upper and lower bands using a fixed percentage multiplier of highs and lows. | "Accelerates" away from price during trends. Alternative to standard deviation bands. |
| **CHA** | Fundamental | Chaikin Volatility | Measures the rate of change of the high-low price range. | Identifies expansion/contraction of daily range. Peaks often correlate with reversals. |
| **CHO** | Non-fundamental | Choppiness Index | Determines whether the market is trending or trading sideways (choppy). | Scale 0-100. High values = consolidation, Low values = trend. |
| **EFR** | Fundamental | Efficiency Ratio | Compares net price change to the sum of absolute price changes (Kaufman). | Measures trend efficiency vs noise. Near 1 = efficient trend, Near 0 = choppy. |
| **MAD** | Fundamental | Median Absolute Deviation | Robust volatility measure using median instead of mean. | Less sensitive to outliers than standard deviation. Good for non-normal distributions. |
| **MAI** | Non-fundamental | Mass Index | Identifies trend reversals by measuring the narrowing and widening of range using EMA ratios. | "Reversal bulge" pattern detection. Does not indicate direction. |
| **VHF** | Non-fundamental | Vertical Horizontal Filter | Determines trending vs congestion phases by comparing range to sum of changes. | Similar concept to EFR but different calculation. Metric for trend strength. |

## Advanced Level
These indicators are complex, adaptive, or use sophisticated statistical/mathematical models (fractals, drift-adjustment, stochastic applications).

| Indicator | Classification | Full Name | Description | Key Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **DVI** | Non-fundamental | Dynamic Volatility Indicator | Composite indicator combining magnitude and stretch components to identify extremes. | Oscillator (0-100). Combines price position and streak analysis. |
| **FDI** | Non-fundamental | Fractal Dimension Index | Measures market complexity and choppiness based on fractal geometry. | 1.5 = Random walk, ~1.0 = Trending, ~2.0 = Mean reverting. |
| **GRV** | Non-fundamental | Garman-Klass Volatility | Efficient volatility estimator using OHLC data. | More efficient than close-to-close std dev. Assumes no drift/gaps. |
| **HAV** | Non-fundamental | Heikin-Ashi Volatility | volatility of Heikin-Ashi smoothed candles. | Filters noise before measuring volatility. Smoother signals. |
| **PAV** | Non-fundamental | Parkinson Volatility | Efficient volatility estimator using only High-Low range. | Capture intraday volatility missed by close-only metrics. Assumes continuous trading. |
| **RSV** | Non-fundamental | Rogers-Satchell Volatility | Volatility estimator that accounts for trend drift. | Better for trending markets than Garman-Klass. Uses all OHLC. |
| **RVI** | Non-fundamental | Relative Volatility Index | Applies RSI formula to standard deviation. | Measures direction of volatility. >50 means volatility is bullish. |
| **SVI** | Non-fundamental | Stochastic Volatility Indicator | Applies Stochastic Oscillator formula to ATR (or std dev). | Normalizes volatility to 0-100 range. Identifies volatility regimes. |
| **TSV** | Non-fundamental | True Strength Volatility | Applies TSI momentum formula to volatility (ATR). | Double-smoothed volatility momentum. |
| **ULI** | Non-fundamental | Ulcer Index | Measures downside risk by focusing on depth/duration of drawdowns. | Only penalizes downside volatility. severe risk measure. |
| **VRA** | Non-fundamental | Volatility Ratio | Compares short-term to long-term volatility. | Ratio > 1 indicates expanding volatility. Regime detection. |
| **VSI** | Non-fundamental | Volatility Switch Index | Binary indicator identifying high/low volatility regimes. | Switch between 0 and 1. Useful for strategy logic switching. |

## Summary Statistics
- **Total Indicators:** 29
- **Beginner:** 7
- **Intermediate:** 7
- **Advanced:** 12

The library provides a comprehensive suite of volatility indicators ranging from standard statistical measures to advanced estimators and regime detection tools.
