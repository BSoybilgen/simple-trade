"""
Example: Using list_strategies to explore available premade backtest strategies

This example demonstrates how to use the list_strategies function to:
1. View all available backtest strategies
2. Filter by category
3. Get programmatic access to strategy information
"""

from simple_trade import list_strategies

# Example 1: List all available strategies
print("=" * 80)
print("EXAMPLE 1: List all available strategies")
print("=" * 80)
list_strategies()

# Example 2: List only momentum strategies
print("\n" * 2)
print("=" * 80)
print("EXAMPLE 2: List only momentum strategies")
print("=" * 80)
list_strategies(category='momentum')

# Example 3: List only trend strategies
print("\n" * 2)
print("=" * 80)
print("EXAMPLE 3: List only trend strategies")
print("=" * 80)
list_strategies(category='trend')

# Example 4: List only volatility strategies
print("\n" * 2)
print("=" * 80)
print("EXAMPLE 4: List only volatility strategies")
print("=" * 80)
list_strategies(category='volatility')

# Example 5: List only volume strategies
print("\n" * 2)
print("=" * 80)
print("EXAMPLE 5: List only volume strategies")
print("=" * 80)
list_strategies(category='volume')

# Example 6: Get dictionary of all strategies for programmatic use
print("\n" * 2)
print("=" * 80)
print("EXAMPLE 6: Get dictionary of strategies")
print("=" * 80)
strategies_dict = list_strategies(return_dict=True)

# Print summary statistics
print(f"\nTotal categories: {len(strategies_dict)}")
print(f"Total strategies: {sum(len(v) for v in strategies_dict.values())}")

# Print count by category
print("\nStrategies by category:")
for category, strategies in strategies_dict.items():
    print(f"  - {category.capitalize()}: {len(strategies)} strategies")

# Example 7: Search for RSI-related strategies
print("\n" * 2)
print("=" * 80)
print("EXAMPLE 7: Search for RSI-related strategies")
print("=" * 80)
for category, strategies in strategies_dict.items():
    for name, description in strategies.items():
        if 'rsi' in name.lower() or 'rsi' in description.lower():
            print(f"\n{name.upper()} ({category})")
            print(f"  {description}")

# Example 8: List all mean reversion strategies
print("\n" * 2)
print("=" * 80)
print("EXAMPLE 8: List all mean reversion strategies")
print("=" * 80)
for category, strategies in strategies_dict.items():
    for name, description in strategies.items():
        if 'mean reversion' in description.lower():
            print(f"\n{name.upper()} ({category})")
            print(f"  {description}")

# Example 9: List all crossover strategies
print("\n" * 2)
print("=" * 80)
print("EXAMPLE 9: List all crossover strategies")
print("=" * 80)
for category, strategies in strategies_dict.items():
    for name, description in strategies.items():
        if 'crossover' in description.lower():
            print(f"\n{name.upper()} ({category})")
            print(f"  {description}")

# Example 10: List all band/breakout strategies
print("\n" * 2)
print("=" * 80)
print("EXAMPLE 10: List all band/breakout strategies")
print("=" * 80)
for category, strategies in strategies_dict.items():
    for name, description in strategies.items():
        if 'band' in description.lower() or 'breakout' in description.lower():
            print(f"\n{name.upper()} ({category})")
            print(f"  {description}")
