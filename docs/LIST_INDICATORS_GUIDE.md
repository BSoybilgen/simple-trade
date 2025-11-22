# List Indicators Guide

## Overview

The `list_indicators()` function provides a comprehensive catalog of all technical indicators available in the simple-trade library. This function helps users discover and understand the available indicators organized by category.

## Categories

Indicators are organized into four main categories:

1. **Momentum Indicators** - Measure the rate of price change
2. **Trend Indicators** - Identify market direction and strength
3. **Volatility Indicators** - Measure price fluctuation and market uncertainty
4. **Volume Indicators** - Analyze trading volume patterns

## Function Signature

```python
def list_indicators(category: str = None, return_dict: bool = False) -> dict | None:
    """List all available technical indicators with their descriptions.
    
    Args:
        category: Optional filter by category. Options: 'momentum', 'trend', 'volatility', 'volume'.
                 If None, returns all indicators.
        return_dict: If True, returns a dictionary instead of printing. Default is False.
    
    Returns:
        dict or None: If return_dict=True, returns a nested dictionary with structure:
                     {category: {indicator_name: description}}
                     Otherwise, prints the indicators and returns None.
    """
```

## Usage Examples

### Example 1: List All Indicators

```python
from simple_trade import list_indicators

# Display all available indicators
list_indicators()
```

**Output:**
```
================================================================================
AVAILABLE TECHNICAL INDICATORS
================================================================================

────────────────────────────────────────────────────────────────────────────────
MOMENTUM INDICATORS (30 total)
────────────────────────────────────────────────────────────────────────────────

  • RSI
    Calculates the Relative Strength Index (RSI), a momentum indicator used
    in technical analysis.

  • MAC
    Calculates the Moving Average Convergence Divergence (MACD), Signal
    Line, and Histogram.

  ...
```

### Example 2: Filter by Category

```python
from simple_trade import list_indicators

# List only momentum indicators
list_indicators(category='momentum')

# List only trend indicators
list_indicators(category='trend')

# List only volatility indicators
list_indicators(category='volatility')

# List only volume indicators
list_indicators(category='volume')
```

### Example 3: Get Dictionary for Programmatic Use

```python
from simple_trade import list_indicators

# Get dictionary of all indicators
indicators = list_indicators(return_dict=True)

# Access indicators by category
momentum_indicators = indicators['momentum']
trend_indicators = indicators['trend']
volatility_indicators = indicators['volatility']
volume_indicators = indicators['volume']

# Print all RSI-related indicators
for category, inds in indicators.items():
    for name, description in inds.items():
        if 'rsi' in name.lower():
            print(f"{name}: {description}")
```

### Example 4: Search for Specific Indicators

```python
from simple_trade import list_indicators

# Get all indicators
indicators = list_indicators(return_dict=True)

# Search for moving average indicators
print("Moving Average Indicators:")
for category, inds in indicators.items():
    for name, description in inds.items():
        if 'moving average' in description.lower():
            print(f"  {name.upper()} ({category}): {description}")
```

### Example 5: Count Indicators by Category

```python
from simple_trade import list_indicators

# Get indicator counts
indicators = list_indicators(return_dict=True)

print("Indicator Counts:")
for category, inds in indicators.items():
    print(f"  {category.capitalize()}: {len(inds)} indicators")

total = sum(len(inds) for inds in indicators.values())
print(f"\nTotal: {total} indicators")
```

## Integration with compute_indicator

Once you've identified an indicator using `list_indicators()`, you can use it with `compute_indicator()`:

```python
from simple_trade import list_indicators, compute_indicator, download_data

# 1. Discover available indicators
list_indicators(category='momentum')

# 2. Download data
data = download_data('AAPL', '2023-01-01', '2024-01-01')

# 3. Compute the indicator
df, columns, fig = compute_indicator(
    data, 
    indicator='rsi',
    parameters={'window': 14}
)
```

## Return Dictionary Structure

When `return_dict=True`, the function returns a nested dictionary:

```python
{
    'momentum': {
        'rsi': 'Calculates the Relative Strength Index (RSI)...',
        'mac': 'Calculates the Moving Average Convergence Divergence...',
        ...
    },
    'trend': {
        'sma': 'Calculates the Simple Moving Average (SMA)...',
        'ema': 'Calculates the Exponential Moving Average (EMA)...',
        ...
    },
    'volatility': {
        'bol': 'Calculates Bollinger Bands...',
        'atr': 'Calculates the Average True Range (ATR)...',
        ...
    },
    'volume': {
        'obv': 'Calculates the On-Balance Volume (OBV)...',
        'vma': 'Calculates the Volume Moving Average (VMA)...',
        ...
    }
}
```

## Error Handling

```python
from simple_trade import list_indicators

# Invalid category raises ValueError
try:
    list_indicators(category='invalid')
except ValueError as e:
    print(f"Error: {e}")
    # Output: Error: Invalid category 'invalid'. Valid options: momentum, trend, volatility, volume
```

## Tips

1. **Discovery**: Use `list_indicators()` without arguments to explore all available indicators
2. **Filtering**: Use the `category` parameter to focus on specific types of indicators
3. **Programmatic Access**: Use `return_dict=True` when you need to process indicator information in code
4. **Documentation**: Each indicator's description is extracted from its docstring, providing accurate and up-to-date information

## Complete Example

```python
from simple_trade import list_indicators, compute_indicator, download_data

# Step 1: Explore available indicators
print("Step 1: Exploring momentum indicators...")
list_indicators(category='momentum')

# Step 2: Get programmatic access
indicators = list_indicators(return_dict=True)
print(f"\nTotal indicators available: {sum(len(v) for v in indicators.values())}")

# Step 3: Find specific indicators
print("\nSearching for oscillators...")
for category, inds in indicators.items():
    for name, desc in inds.items():
        if 'oscillator' in desc.lower():
            print(f"  - {name.upper()} ({category})")

# Step 4: Use an indicator
print("\nDownloading data and computing RSI...")
data = download_data('SPY', '2023-01-01', '2024-01-01')
df, cols, fig = compute_indicator(data, 'rsi', parameters={'window': 14})
print(f"RSI computed successfully! Columns: {cols}")
```

## See Also

- `compute_indicator()` - Compute a specific indicator
- `download_data()` - Download historical price data
- Individual indicator documentation in the respective module folders
