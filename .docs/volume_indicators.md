# Volume Indicators Analysis

This document categorizes the volume indicators found in `simple_trade/volume` into Beginner, Intermediate, and Advanced levels based on their complexity, calculation method, and usage.

## Beginner Level
These indicators are fundamental, widely used, and easy to understand. They typically involve cumulative sums or simple ratios of volume and price.

| Indicator | Classification | Full Name | Description | Key Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **ADL** | Fundamental | Accumulation/Distribution Line | Measures competitive flow of money into and out of a security. | Uses close location within high-low range to confirm trends. |
| **CMF** | Fundamental | Chaikin Money Flow | Measures the amount of Money Flow Volume over a specific period. | Sum of MFV / Sum of Volume. Oscillator for buying/selling pressure. |
| **OBV** | Fundamental | On-Balance Volume | Relates volume flow to price changes by adding/subtracting volume based on price direction. | First volume indicator. Simple confirmation of price trends. |
| **VRO** | Fundamental | Volume Rate of Change | Measures the percentage change in volume over a specified period. | Momentum indicator for volume itself. Highlights volume surges. |
| **VWA** | Fundamental | Volume Weighted Average Price | Gives the average price a security has traded at throughout the day/period. | "True" average price. Used as institutional benchmark and support. |

## Intermediate Level
These indicators use more specific mathematical concepts or combinations of price and volume to identify specific market behaviors (sentiment, smart money, efficiency).

| Indicator | Classification | Full Name | Description | Key Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **ADO** | Fundamental | Accumulation/Distribution Oscillator | Measures the momentum of the Accumulation/Distribution Line. | Rate of change of ADL. Helps identify changes in money flow pressure. |
| **BWM** | Non-fundamental | Market Facilitation Index | Measures efficiency of price movement by analyzing price change per unit of volume. | Bill Williams indicator. Identifies "Green", "Fade", "Fake", "Squat" bars. |
| **EMV** | Fundamental | Ease of Movement | Relates price change to volume to show how easily price can move. | High EMV = Price rising on low volume. Zero line crossovers. |
| **FOI** | Fundamental | Force Index | Uses price change and volume to assess the power behind a price move. | Combines direction, extent, and volume. Smoothed with EMA. |
| **MFI** | Fundamental | Money Flow Index | Momentum indicator that identifies overbought/oversold conditions using volume. | Often called volume-weighted RSI. Divergence and extremes (20/80). |
| **NVI** | Non-fundamental | Negative Volume Index | Tracks price changes on days when volume decreases from the previous day. | "Smart Money" indicator. Focuses on what informed traders do on quiet days. |
| **PVI** | Non-fundamental | Positive Volume Index | Tracks price changes on days when volume increases from the previous day. | "Crowd" indicator. Focuses on public participation. |
| **PVO** | Non-fundamental | Percentage Volume Oscillator | Momentum oscillator for volume showing relationship between two volume EMAs. | MACD applied to volume. Measures volume trend as a percentage. |
| **VOO** | Non-fundamental | Volume Oscillator | Displays the difference between two moving averages of volume. | Trend strength indicator. Simple points difference (not percentage). |
| **VPT** | Fundamental | Volume Price Trend | Relates volume to price change percentage to create aggregate pressure indicator. | Similar to OBV but weights volume by extent of price change. |

## Advanced Level
These indicators are complex, adaptive, or use sophisticated statistical/modeling techniques to analyze volume flow and trend.

| Indicator | Classification | Full Name | Description | Key Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **FVE** | Non-fundamental | Finite Volume Elements | Separates volume into bullish/bearish components based on volatility threshold. | Resolves contradictions between price and volume. Trend following. |
| **KVO** | Non-fundamental | Klinger Volume Oscillator | Compares volume flowing through securities with price movements to detect long-term trends. | Sensitive to short-term but focused on long-term money flow. Complex formula. |
| **VFI** | Non-fundamental | Volume Flow Indicator | Long-term trend indicator based on OBV but uses price ROC and coefficients to reduce noise. | Clamps volume to prevent distortion. Good for long-term efficiency. |

## Summary Statistics
- **Total Indicators:** 18
- **Beginner:** 5
- **Intermediate:** 10
- **Advanced:** 3

The library provides a comprehensive suite of volume indicators covering accumulation/distribution, money flow, and volume-weighted trend analysis.
