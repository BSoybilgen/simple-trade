"""
Main indicator handling module that coordinates the calculation of various technical indicators.
"""
import yfinance as yf
import pandas as pd
from .core import INDICATORS
from simple_trade.plot_ind import plot_indicator
from typing import Literal, Optional, Tuple


def compute_indicator(
    data: pd.DataFrame,
    indicator: str,
    figure: bool=True,
    plot_type: Literal['line', 'candlestick'] = 'line',
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    **indicator_kwargs
) -> tuple:
    """Computes a specified technical indicator on the provided financial data.

    Args:
        data: pandas.DataFrame containing the financial data (must include 'Close',
              and possibly 'High', 'Low' depending on the indicator).
        indicator: Technical indicator to compute (e.g., 'rsi', 'sma', 'mac', 'adx').
        title: Optional title for the figure. Defaults to '{INDICATOR} Indicator'.
        figsize: Optional tuple (width, height) in inches for the figure size passed to plot_indicator.
        **indicator_kwargs: Keyword arguments specific to the chosen indicator.

    Returns:
        pandas.DataFrame: Original DataFrame with the calculated indicator column(s) added.

    Raises:
        ValueError: If the indicator is not supported or the required columns are missing.
    """
    # Validate indicator exists
    if indicator not in INDICATORS:
        raise ValueError(f"Indicator '{indicator}' not supported. Available: {list(INDICATORS.keys())}")

    # Create a copy to avoid modifying the original DataFrame
    df = data.copy()
    indicator_func = INDICATORS[indicator]
    # print(f"Computing {indicator.upper()}...")

    try:
        # Delegate to specific handler based on indicator type
        indicator_result, columns = _calculate_indicator(df, indicator_func, **indicator_kwargs)
        
        # Add the result to the original DataFrame
        df = _add_indicator_to_dataframe(df, indicator_result, indicator_kwargs)

        if indicator in (
            'ado', 'adl', 'adx', 'aro', 'atp', 'atr', 'awo', 'bbw', 
            'bop', 'bwm', 'cci', 'cha', 'cho', 'cmf', 'cmo', 'cog', 
            'crs', 'dpo', 'dvi', 'efr', 'emv', 'eri', 'fdi', 'fis', 
            'foi', 'fve', 'grv', 'hav', 'hiv', 'htt', 'imi', 'kst', 
            'kur', 'kvo', 'lsi', 'mab', 'mac', 'mad', 'mai', 'mfi', 'msi', 
            'nat', 'nvi', 'obv', 'pav', 'pcw', 'pgo', 'ppo', 'pro', 
            'psy', 'pvi', 'pvo', 'qst', 'rmi', 'roc', 'rsi', 'rsv', 
            'rvg', 'rvi', 'skw', 'sri', 'stc', 'std', 'sto', 'svi', 'tri', 
            'tsi', 'tsv', 'ttm', 'uli', 'ult', 'var', 'vfi', 'vhf', 'voo', 
            'vor', 'vpt', 'vra', 'vqi', 'vro', 'vsi', 'wad', 'wil', 'zsc'
        ):
            plot_on_subplot=True
        else:
            plot_on_subplot=False

        if indicator in ('psa', 'str'):
            columns = [columns[0]]

        if figure:
            fig = plot_indicator(
                df,
                price_col='Close',
                column_names=columns,
                plot_on_subplot=plot_on_subplot,
                plot_type=plot_type,
                title=title if title else f"{indicator.upper()} Indicator",
                figsize=figsize
            )
        
            return df, columns, fig
        else:
            return df, columns, None
        
    except Exception as e:
        print(f"Error calculating indicator '{indicator}': {e}")
        return df, None, None  # Return the original df if calculation fails


def _calculate_indicator(df, indicator_func, **indicator_kwargs):
    """Dispatch to the appropriate handler for each indicator type."""
    # Keep parameters and columns as dictionaries
    # This is important for indicator functions that expect them as dictionaries
    return indicator_func(df, **indicator_kwargs)


def _add_indicator_to_dataframe(df, indicator_result, indicator_kwargs):
    """Add the calculated indicator to the DataFrame with appropriate naming."""
    # Handle various return types from indicator functions
    if isinstance(indicator_result, pd.Series):
        df[indicator_result.name] = indicator_result
        
    elif isinstance(indicator_result, pd.DataFrame):
        df = df.join(indicator_result)

    elif isinstance(indicator_result, tuple):
        # Expecting (data, columns)
        data_part, _ = indicator_result
        if isinstance(data_part, pd.Series):
            df[data_part.name] = data_part
        elif isinstance(data_part, pd.DataFrame):
            df = df.join(data_part)
        else:
            print(f"Warning: Unexpected tuple data part type: {type(data_part)}")
    else:
        print(f"Warning: Indicator function returned an unexpected type: {type(indicator_result)}")
    
    return df


def download_data(symbol: str, start_date: str, end_date: str = None, interval: str = '1d') -> pd.DataFrame:
    """Download historical price data for a given symbol using yfinance."""
    # Set auto_adjust=False to get raw OHLCV and prevent yfinance from potentially altering columns
    df = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data found for {symbol}.")

    # Clean up column names: remove multi-index if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        # Remove duplicate columns
        df = df.loc[:,~df.columns.duplicated()]

    # Force column names to lowercase for consistent mapping
    df.columns = df.columns.str.lower()

    # Standardize column names to Title Case
    column_map = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'adj close': 'Adj Close',
        'volume': 'Volume'
    }
    df = df.rename(columns=column_map)

    # Ensure all expected columns are present, derived if needed
    if 'Adj Close' not in df.columns and 'Close' in df.columns:
        df['Adj Close'] = df['Close']  # Use Close as Adj Close if not available

    # Add a symbol attribute to the dataframe for reference
    df.attrs['symbol'] = symbol

    return df


def list_indicators(category: str = None, return_dict: bool = False) -> dict | None:
    """List all available technical indicators with their descriptions.
    
    This function provides a comprehensive catalog of all indicators available in the
    simple_trade library, organized by category (momentum, trend, volatility, volume).
    
    Args:
        category: Optional filter by category. Options: 'momentum', 'trend', 'volatility', 'volume'.
                 If None, returns all indicators.
        return_dict: If True, returns a dictionary instead of printing. Default is False.
    
    Returns:
        dict or None: If return_dict=True, returns a nested dictionary with structure:
                     {category: {indicator_name: description}}
                     Otherwise, prints the indicators and returns None.
    
    Example:
        >>> list_indicators()  # Print all indicators
        >>> list_indicators(category='momentum')  # Print only momentum indicators
        >>> indicators = list_indicators(return_dict=True)  # Get dictionary of all indicators
    """
    from . import momentum, trend, volatility, volume, moving_average, statistics
    import inspect
    
    # Define indicator categories and their modules
    categories = {
        'momentum': momentum,
        'trend': trend,
        'volatility': volatility,
        'volume': volume,
        'moving_average': moving_average,
        'statistics': statistics,
    }
    
    # Filter by category if specified
    if category:
        if category.lower() not in categories:
            valid_categories = ', '.join(categories.keys())
            raise ValueError(f"Invalid category '{category}'. Valid options: {valid_categories}")
        categories = {category.lower(): categories[category.lower()]}
    
    # Collect all indicators with their descriptions
    all_indicators = {}
    
    for cat_name, module in categories.items():
        all_indicators[cat_name] = {}
        
        # Get all functions from the module's __all__ list
        if hasattr(module, '__all__'):
            for indicator_name in module.__all__:
                # Get the function object
                indicator_func = getattr(module, indicator_name, None)
                if indicator_func and callable(indicator_func):
                    # Extract the first line of the docstring as description
                    doc = inspect.getdoc(indicator_func)
                    if doc:
                        # Get the first meaningful line (skip empty lines)
                        lines = [line.strip() for line in doc.split('\n') if line.strip()]
                        description = lines[0] if lines else "No description available"
                    else:
                        description = "No description available"
                    
                    all_indicators[cat_name][indicator_name] = description
    
    # Return dictionary if requested
    if return_dict:
        return all_indicators
    
    # Otherwise, print formatted output
    print("\n" + "="*80)
    print("AVAILABLE TECHNICAL INDICATORS")
    print("="*80 + "\n")
    
    for cat_name, indicators in all_indicators.items():
        print(f"\n{'─'*80}")
        print(f"{cat_name.upper()} INDICATORS ({len(indicators)} total)")
        print(f"{'─'*80}\n")
        
        for indicator_name, description in sorted(indicators.items()):
            # Format the output with indicator name and description
            print(f"  • {indicator_name.upper()}")
            # Wrap long descriptions
            desc_lines = _wrap_text(description, width=72, indent=4)
            for line in desc_lines:
                print(f"    {line}")
            print()
    
    print("="*80)
    print(f"Total: {sum(len(indicators) for indicators in all_indicators.values())} indicators")
    print("="*80 + "\n")
    
    return None


def _wrap_text(text: str, width: int = 70, indent: int = 0) -> list:
    """Wrap text to a specified width with optional indentation.
    
    Args:
        text: The text to wrap
        width: Maximum line width
        indent: Number of spaces to indent wrapped lines
    
    Returns:
        list: List of wrapped text lines
    """
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        word_length = len(word)
        # +1 for the space
        if current_length + word_length + len(current_line) > width:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
        else:
            current_line.append(word)
            current_length += word_length
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines