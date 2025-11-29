"""
Backtest plotting functions.

This module provides function-based implementations for plotting backtest results,
replacing the class-based BacktestPlotter approach.
"""
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Optional


def plot_backtest_results(
    data_df: pd.DataFrame,
    history_df: pd.DataFrame,
    price_col: str = 'Close',
    indicator_cols: Optional[List[str]] = None,
    title: Optional[str] = None,
    show_indicator_panel: bool = True,
    show_extra_panel: bool = False,
    extra_panel_cols: Optional[List[str]] = None,
    extra_panel_title: Optional[str] = "Price vs Indicator"
) -> Optional[plt.Figure]:
    """
    Generates and displays the backtest results plot.

    Args:
        data_df: The original data DataFrame with prices and indicators.
        history_df: The portfolio history DataFrame from a backtest.
        price_col: The name of the column containing the price data.
        indicator_cols: List of indicator column names to plot in panel 3.
        title: Optional title for the plot.
        show_indicator_panel: Whether to show the 3rd panel (indicators).
        show_extra_panel: Whether to show a 4th panel at the bottom.
        extra_panel_cols: List of columns to plot in the 4th panel.
        extra_panel_title: Title for the 4th panel.

    Returns:
        plt.Figure: The generated matplotlib Figure object, or None if plotting fails.
        
    Example:
        >>> fig = plot_backtest_results(
        ...     data_df=data,
        ...     history_df=portfolio,
        ...     price_col='Close',
        ...     indicator_cols=['RSI_14', 'RSI_Upper', 'RSI_Lower'],
        ...     title='RSI Strategy Backtest'
        ... )
    """
    # Ensure DatetimeIndex
    if not isinstance(data_df.index, pd.DatetimeIndex):
        try:
            data_df = data_df.copy()
            data_df.index = pd.to_datetime(data_df.index)
        except Exception as e:
            raise ValueError(f"Failed to convert data_df index to DatetimeIndex: {e}")
    
    if not isinstance(history_df.index, pd.DatetimeIndex):
        try:
            history_df = history_df.copy()
            history_df.index = pd.to_datetime(history_df.index)
        except Exception as e:
            raise ValueError(f"Failed to convert history_df index to DatetimeIndex: {e}")

    # Ensure alignment (use intersection of indices)
    plot_index = data_df.index.intersection(history_df.index)
    if plot_index.empty:
        print("Warning: No overlapping dates found between data_df and history_df. Cannot plot.")
        return None

    data_df = data_df.loc[plot_index]
    history_df = history_df.loc[plot_index]

    # Determine layout
    if show_extra_panel:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), sharex=True,
                                            gridspec_kw={'height_ratios': [3, 1, 1.5, 1.5]})
    elif show_indicator_panel:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                                            gridspec_kw={'height_ratios': [3, 1, 2]})
        ax4 = None
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                        gridspec_kw={'height_ratios': [3, 1]})
        ax3 = None
        ax4 = None

    # --- Plot 1: Price and Trades ---
    ax1.plot(plot_index, data_df[price_col], label=f'{price_col} Price', color='skyblue', linewidth=1.5)
    ax1.set_ylabel('Price')
    ax1.set_title(title or f'{price_col} and Trade Signals')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot trades from history using the 'Action' column
    if 'Action' in history_df.columns:
        buys = history_df[history_df['Action'].str.contains('Buy', case=False, na=False) & 
                         ~history_df['Action'].str.contains('Cover', case=False, na=False)]
        sells = history_df[history_df['Action'].str.contains('Sell', case=False, na=False) & 
                          ~history_df['Action'].str.contains('Short', case=False, na=False)]
        shorts = history_df[history_df['Action'].str.contains('Short', case=False, na=False) & 
                           ~history_df['Action'].str.contains('Sell', case=False, na=False)]
        covers = history_df[history_df['Action'].str.contains('Cover', case=False, na=False) & 
                           ~history_df['Action'].str.contains('Buy', case=False, na=False)]
        sell_and_shorts = history_df[history_df['Action'].str.contains('Sell and Short', case=False, na=False)]
        cover_and_buys = history_df[history_df['Action'].str.contains('Cover and Buy', case=False, na=False)]

        if not buys.empty:
            ax1.plot(buys.index, data_df.loc[buys.index, price_col], '^', markersize=8, color='lime', label='Buy')
        if not sells.empty:
            ax1.plot(sells.index, data_df.loc[sells.index, price_col], 'v', markersize=8, color='red', label='Sell')
        if not shorts.empty:
            ax1.plot(shorts.index, data_df.loc[shorts.index, price_col], 'v', markersize=8, color='fuchsia', label='Short')
        if not covers.empty:
            ax1.plot(covers.index, data_df.loc[covers.index, price_col], '^', markersize=8, color='orange', label='Cover')
        if not sell_and_shorts.empty:
            ax1.plot(sell_and_shorts.index, data_df.loc[sell_and_shorts.index, price_col], 'x', markersize=8, color='darkred', label='Sell & Short')
        if not cover_and_buys.empty:
            ax1.plot(cover_and_buys.index, data_df.loc[cover_and_buys.index, price_col], 'P', markersize=8, color='darkgreen', label='Cover & Buy')

    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

    # --- Plot 2: Portfolio Value ---
    if 'PortfolioValue' in history_df.columns:
        ax2.plot(plot_index, history_df['PortfolioValue'], label='Portfolio Value', color='purple', linewidth=1.5)
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.set_title('Portfolio Value Over Time')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

    # --- Plot 3: Indicators ---
    if show_indicator_panel and ax3 is not None:
        valid_indicator_cols = []
        if indicator_cols:
            for col in indicator_cols:
                if col in data_df.columns:
                    valid_indicator_cols.append(col)
                else:
                    print(f"Warning: Indicator column '{col}' not found in data_df.")
        
        if valid_indicator_cols:
            contrast_colors = ['#FFD700', '#32CD32', '#1E90FF', '#FF8C00', '#FF69B4', '#BA55D3']
            for i, col in enumerate(valid_indicator_cols):
                color_idx = i % len(contrast_colors)
                ax3.plot(plot_index, data_df[col], label=col, linewidth=1.5, color=contrast_colors[color_idx])
            ax3.set_ylabel('Indicator Value')
            ax3.set_title('Indicator Values')
            ax3.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
            ax3.grid(True, linestyle='--', alpha=0.6)
        else:
            ax3.set_title('No Indicators Specified/Found for Plotting')
            ax3.grid(False)

    # --- Plot 4: Extra Panel ---
    if show_extra_panel and ax4 is not None:
        valid_extra_cols = []
        if extra_panel_cols:
            for col in extra_panel_cols:
                if col in data_df.columns:
                    valid_extra_cols.append(col)
                else:
                    print(f"Warning: Extra panel column '{col}' not found in data_df.")
        
        if valid_extra_cols:
            contrast_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            for i, col in enumerate(valid_extra_cols):
                color_idx = i % len(contrast_colors)
                ax4.plot(plot_index, data_df[col], label=col, linewidth=1.5, color=contrast_colors[color_idx])
            ax4.set_ylabel('Value')
            ax4.set_title(extra_panel_title)
            ax4.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
            ax4.grid(True, linestyle='--', alpha=0.6)
        else:
            ax4.set_title('No Columns Specified for Extra Panel')
            ax4.grid(False)

    # --- Final Touches ---
    plt.xlabel('Date')
    plt.tight_layout(rect=[0, 0, 0.85, 0.96])

    return fig


