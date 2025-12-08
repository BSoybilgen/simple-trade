# Trend Indicators Analysis

This document categorizes the trend indicators found in `simple_trade/trend` into Beginner, Intermediate, and Advanced levels based on their complexity, calculation method, and usage.

## Beginner Level
These indicators are fundamental, widely used, and easy to understand. They typically involve standard trend measuring concepts like directional movement or trailing stops.

| Indicator | Classification | Full Name | Description | Key Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **ADX** | Fundamental | Average Directional Index | Measures the strength of a trend without regard to trend direction. | Uses +DI and -DI. Values > 25 indicate strong trend. |
| **PSA** | Fundamental | Parabolic SAR | A stop and reverse tracking system that places dots above/below price. | Trend following. Excellent for trailing stops. Acceleration factor increases speed. |

## Intermediate Level
These indicators use more specific mathematical concepts, adaptive logic, or oscillator formats to identify trends and potential reversals.

| Indicator | Classification | Full Name | Description | Key Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **ARO** | Fundamental | Aroon | Measures the time it takes for price to reach the highest highs and lowest lows. | Oscillator (Up/Down lines). 100 = New High/Low. Crossovers signal trend change. |
| **EIT** | Non-fundamental | Ehlers Instantaneous Trendline | A low-lag trendline that uses signal processing to track price closely. | Recursive filter. Smoother and faster than SMA/EMA. |
| **MGD** | Non-fundamental | McGinley Dynamic | A moving average that adjusts its speed based on market volatility. | Minimizes whipsaws. Speeds up in fast markets, slows in ranges. |
| **PRO** | Non-fundamental | Projection Oscillator | Measures the slope of the linear regression line normalized by volatility. | Oscillator. Positive = Uptrend, Negative = Downtrend. |
| **STR** | Non-fundamental | SuperTrend | A trend-following indicator based on ATR and price action. | Plots a line that flips above/below price. Used for direction and stops. |
| **TRI** | Fundamental | TRIX | A momentum oscillator showing the percent rate of change of a triple-smoothed EMA. | Filters out insignificant price movements. Zero line crossovers. |

## Advanced Level
These indicators are complex systems or use sophisticated signal processing to analyze trend components and quality.

| Indicator | Classification | Full Name | Description | Key Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **EAC** | Non-fundamental | Ehlers Adaptive CyberCycle | Separates the cycle component from the trend component of price data. | Cyclic analysis. Provides a very smooth, low-lag trendline. |
| **HTT** | Non-fundamental | Hilbert Transform Trendline | An instantaneous trendline derived using the Hilbert Transform to remove cycles. | Signal processing based. Extracts the DC phase of the signal. |
| **ICH** | Non-fundamental | Ichimoku Cloud | A comprehensive collection of indicators showing support, resistance, trend, and momentum. | Cloud (Kumo), Tenkan, Kijun, Chikou. Complete trading system. |
| **VQI** | Non-fundamental | Volatility Quality Index | Measures the quality of a trend by distinguishing between noise and genuine movement. | Uses price, volume, and volatility. Identifies sustainable trends. |

## Summary Statistics
- **Total Indicators:** 12
- **Beginner:** 2
- **Intermediate:** 6
- **Advanced:** 4

The library provides a diverse set of trend indicators ranging from classic tools to advanced signal processing algorithms.
