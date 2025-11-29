"""
Example: Using list_indicators to explore available technical indicators

This example demonstrates how to use the list_indicators function to:
1. View all available indicators
2. Filter by category
3. Get programmatic access to indicator information
"""

from simple_trade import list_indicators

# Example 1: List all available indicators
print("=" * 80)
print("EXAMPLE 1: List all available indicators")
print("=" * 80)
list_indicators()

# Example 2: List only momentum indicators
print("\n" * 2)
print("=" * 80)
print("EXAMPLE 2: List only momentum indicators")
print("=" * 80)
list_indicators(category='momentum')

# Example 3: List only trend indicators
print("\n" * 2)
print("=" * 80)
print("EXAMPLE 3: List only trend indicators")
print("=" * 80)
list_indicators(category='trend')

# Example 4: Get dictionary of all indicators for programmatic use
print("\n" * 2)
print("=" * 80)
print("EXAMPLE 4: Get dictionary of indicators")
print("=" * 80)
indicators_dict = list_indicators(return_dict=True)

# Print summary statistics
print(f"\nTotal categories: {len(indicators_dict)}")
print(f"Total indicators: {sum(len(v) for v in indicators_dict.values())}")

# Print count by category
print("\nIndicators by category:")
for category, indicators in indicators_dict.items():
    print(f"  - {category.capitalize()}: {len(indicators)} indicators")

# Example 5: Search for specific indicators
print("\n" * 2)
print("=" * 80)
print("EXAMPLE 5: Search for RSI-related indicators")
print("=" * 80)
for category, indicators in indicators_dict.items():
    for name, description in indicators.items():
        if 'rsi' in name.lower() or 'relative strength' in description.lower():
            print(f"\n{name.upper()} ({category})")
            print(f"  {description}")

# Example 6: List all moving average indicators
print("\n" * 2)
print("=" * 80)
print("EXAMPLE 6: List all moving average indicators")
print("=" * 80)
for category, indicators in indicators_dict.items():
    for name, description in indicators.items():
        if 'moving average' in description.lower() or 'ma' in name.lower():
            print(f"\n{name.upper()} ({category})")
            print(f"  {description}")
