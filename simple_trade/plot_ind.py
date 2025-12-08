"""
Indicator plotting functions.

This module provides functions for plotting financial data and technical indicators.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from typing import List, Optional, Literal, Tuple
import logging


def plot_indicator(
    df: pd.DataFrame,
    price_col: str = 'Close',
    column_names: Optional[List[str]] = None,
    plot_on_subplot: bool = False,
    title: Optional[str] = None,
    plot_type: Literal['line', 'candlestick'] = 'line',
    figsize: Optional[Tuple[float, float]] = None
) -> plt.Figure:
    """
    Plots the price data and specified indicator columns.

    Args:
        df: DataFrame containing price and indicator data.
        price_col: Name of the column representing the price.
        column_names: List of column names to plot. These must exist in the DataFrame.
        plot_on_subplot: Whether to plot indicators on a separate subplot below price.
                        If False, indicators will be overlaid on the price chart.
        title: Optional title for the plot.
        plot_type: Type of plot for price data. Options are 'line' (default) or 'candlestick'.
                  For candlestick, the DataFrame must contain 'Open', 'High', 'Low', and 'Close' columns.
        figsize: Optional tuple (width, height) in inches for the figure size.
                 If None, defaults to (16, 10) for 3 subplots, (16, 8) for 2 subplots or single plot.

    Returns:
        plt.Figure: The generated matplotlib Figure object.

    Example:
        >>> fig = plot_indicator(
        ...     data, price_col='Close',
        ...     column_names=['SMA_20', 'SMA_50'],
        ...     plot_on_subplot=False,
        ...     title='Price with Moving Averages'
        ... )
    """
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame.")
        
    # Check for required columns when using candlestick chart
    if plot_type == 'candlestick':
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Candlestick plot requires columns {required_cols}, but {missing_cols} are missing.")

    # Check if column_names is provided and verify which ones exist in the DataFrame
    indicator_cols = []
    if column_names:
        # Filter to include only columns that exist in the DataFrame
        indicator_cols = [col for col in column_names if col in df.columns]
        if not indicator_cols:
            logging.warning(f"None of the specified columns {column_names} exist in DataFrame. Plotting price only.")
    else:
        logging.warning("No indicator columns specified. Plotting price only.")
    
    # Create Figure and Axes with adjusted size
    column_names = [] if column_names is None else column_names
    columns_to_plot = [col for col in column_names if col in df.columns]
    missing_cols = [col for col in column_names if col not in df.columns]
    if missing_cols:
        logging.warning(f"Columns {missing_cols} specified but not found in DataFrame.")
    if not columns_to_plot:
        logging.warning("No valid indicator columns specified or found. Plotting price only.")

    # Separate squeeze columns so they can be plotted on their own panel
    squeeze_cols = [
        col for col in columns_to_plot
        if 'squeeze_on' in col.lower() or 'squeeze_off' in col.lower()
    ]
    
    # Separate +DI/-DI columns for second panel
    di_cols = [
        col for col in columns_to_plot
        if col.startswith('+DI_') or col.startswith('-DI_')
    ]
    
    # Separate ADX columns for third panel
    adx_cols = [
        col for col in columns_to_plot
        if col.upper().startswith('ADX_')
    ]
    
    # Separate ARO_UP/ARO_DOWN columns for second panel
    aro_updown_cols = [
        col for col in columns_to_plot
        if col.upper().startswith('ARO_UP_') or col.upper().startswith('ARO_DOWN_')
    ]
    
    # Separate ARO_OSCILLATOR columns for third panel
    aro_osc_cols = [
        col for col in columns_to_plot
        if col.upper().startswith('ARO_OSCILLATOR_')
    ]
    
    # Separate MACD line/signal columns for second panel
    macd_line_cols = [
        col for col in columns_to_plot
        if col.upper().startswith('MAC_') or col.upper().startswith('SIGNAL_')
    ]
    
    # Separate MACD histogram columns for third panel
    macd_hist_cols = [
        col for col in columns_to_plot
        if col.upper().startswith('HIST_')
    ]
    
    # Separate PPO line/signal columns for second panel
    ppo_line_cols = [
        col for col in columns_to_plot
        if (col.upper().startswith('PPO_') and not col.upper().startswith('PPO_HIST')) or col.upper().startswith('PPO_SIG_')
    ]
    
    # Separate PPO histogram columns for third panel
    ppo_hist_cols = [
        col for col in columns_to_plot
        if col.upper().startswith('PPO_HIST')
    ]
    
    # Separate PVO line/signal columns for second panel
    pvo_line_cols = [
        col for col in columns_to_plot
        if (col.upper().startswith('PVO_') and not col.upper().startswith('PVO_HIST')) or col.upper().startswith('PVO_SIGNAL_')
    ]
    
    # Separate PVO histogram columns for third panel
    pvo_hist_cols = [
        col for col in columns_to_plot
        if col.upper().startswith('PVO_HIST')
    ]
    
    indicator_plot_cols = [col for col in columns_to_plot if col not in squeeze_cols and col not in di_cols and col not in adx_cols and col not in aro_updown_cols and col not in aro_osc_cols and col not in macd_line_cols and col not in macd_hist_cols and col not in ppo_line_cols and col not in ppo_hist_cols and col not in pvo_line_cols and col not in pvo_hist_cols]

    need_indicator_subplot = plot_on_subplot and bool(indicator_plot_cols)
    need_squeeze_subplot = bool(squeeze_cols)
    
    # DI, ARO_UP/DOWN, MACD line/signal, PPO line/signal, and PVO line/signal go in second panel
    need_di_subplot = bool(di_cols) or bool(aro_updown_cols) or bool(macd_line_cols) or bool(ppo_line_cols) or bool(pvo_line_cols)
    # ADX, ARO_OSCILLATOR, MACD histogram, PPO histogram, and PVO histogram go in third panel
    need_oscillator_subplot = bool(adx_cols) or bool(aro_osc_cols) or bool(macd_hist_cols) or bool(ppo_hist_cols) or bool(pvo_hist_cols)

    # Initialize optional axes
    ax_di = None  # For +DI/-DI or ARO_UP/DOWN
    ax_osc = None  # For ADX or ARO_OSCILLATOR
    ax_squeeze = None

    # Determine subplot configuration based on what's needed
    num_subplots = 1 + int(need_indicator_subplot or need_di_subplot) + int(need_oscillator_subplot) + int(need_squeeze_subplot)

    if num_subplots == 4:  # price + di/indicator + oscillator + squeeze
        size = figsize if figsize else (16, 12)
        fig, axes = plt.subplots(
            4, 1, sharex=True, figsize=size,
            gridspec_kw={'height_ratios': [3, 1.2, 1.2, 0.8]}
        )
        ax_price, ax_ind, ax_osc, ax_squeeze = axes
        if need_di_subplot:
            ax_di = ax_ind
    elif num_subplots == 3:
        if (need_indicator_subplot or need_di_subplot) and need_oscillator_subplot:
            size = figsize if figsize else (16, 10)
            fig, axes = plt.subplots(
                3, 1, sharex=True, figsize=size,
                gridspec_kw={'height_ratios': [3, 1.2, 1.2]}
            )
            ax_price, ax_ind, ax_osc = axes
            if need_di_subplot:
                ax_di = ax_ind
        elif (need_indicator_subplot or need_di_subplot) and need_squeeze_subplot:
            size = figsize if figsize else (16, 10)
            fig, axes = plt.subplots(
                3, 1, sharex=True, figsize=size,
                gridspec_kw={'height_ratios': [3, 1.4, 0.8]}
            )
            ax_price, ax_ind, ax_squeeze = axes
            if need_di_subplot:
                ax_di = ax_ind
        elif need_oscillator_subplot and need_squeeze_subplot:
            size = figsize if figsize else (16, 10)
            fig, axes = plt.subplots(
                3, 1, sharex=True, figsize=size,
                gridspec_kw={'height_ratios': [3, 1.2, 0.8]}
            )
            ax_price, ax_osc, ax_squeeze = axes
            ax_ind = ax_price
    elif num_subplots == 2:
        if need_indicator_subplot or need_di_subplot:
            size = figsize if figsize else (16, 8)
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=size, gridspec_kw={'height_ratios': [3, 1]})
            ax_price, ax_ind = axes
            if need_di_subplot:
                ax_di = ax_ind
        elif need_oscillator_subplot:
            size = figsize if figsize else (16, 8)
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=size, gridspec_kw={'height_ratios': [3, 1.2]})
            ax_price, ax_osc = axes
            ax_ind = ax_price
        elif need_squeeze_subplot:
            size = figsize if figsize else (16, 8)
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=size, gridspec_kw={'height_ratios': [3, 0.8]})
            ax_price, ax_squeeze = axes
            ax_ind = ax_price
    else:
        size = figsize if figsize else (16, 8)
        fig, ax = plt.subplots(1, 1, figsize=size)
        ax_price = ax
        ax_ind = ax

    # Plot Price based on plot_type
    if plot_type == 'line':
        # Plot as a line with increased thickness and darker color
        ax_price.plot(df.index, df[price_col], label=price_col, color='#0343df', linewidth=2.5)
    elif plot_type == 'candlestick':
        # Convert datetime index to matplotlib dates for proper candlestick rendering
        dates = mdates.date2num(df.index.to_numpy())
        
        # Calculate candlestick width based on data frequency
        if len(df) > 1:
            width = 0.6 * (dates[1] - dates[0])  # Width is a fraction of the time between points
        else:
            width = 0.6  # Default width for single point
            
        # Plot candlesticks
        for i, (date, row) in enumerate(df.iterrows()):
            date_num = dates[i]
            open_price, close_price = row['Open'], row['Close']
            high, low = row['High'], row['Low']
            
            # Determine color based on price movement
            color = '#27ae60' if close_price >= open_price else '#e74c3c'  # Green for bullish, red for bearish
            
            # Plot the candle body
            rect = Rectangle(
                xy=(date_num - width/2, min(open_price, close_price)),
                width=width,
                height=abs(close_price - open_price),
                facecolor=color,
                edgecolor=color,
                alpha=0.8,
                linewidth=1
            )
            ax_price.add_patch(rect)
            
            # Plot the high/low wicks
            ax_price.plot(
                [date_num, date_num],
                [max(open_price, close_price), high],
                color=color,
                linewidth=1.5
            )
            ax_price.plot(
                [date_num, date_num],
                [min(open_price, close_price), low],
                color=color,
                linewidth=1.5
            )
        
        # Create a custom legend entry for candlesticks
        ax_price.plot([], [], label='Candlestick', color='gray', linewidth=0)  # Invisible line for legend
    else:
        # Raise error for invalid plot_type if it's not 'line' or 'candlestick'
        raise ValueError(f"Invalid plot_type: {plot_type}. Choose 'line' or 'candlestick'.")

    ax_price.set_ylabel('Price', fontweight='bold', fontsize=12)
    ax_price.grid(True, linestyle='--', alpha=0.7, color='#303030')
    
    # Plot Indicators if any are specified
    if indicator_plot_cols:
        plot_hist = False
        hist_col_name = ''
        # Special handling for MACD Histogram
        hist_col_name = next((col for col in indicator_plot_cols if 'hist' in col), None)
        if hist_col_name:
            plot_hist = True
            indicator_plot_cols.remove(hist_col_name)

        # Plot lines with high contrast colors and increased line thickness
        contrast_colors = ['#e50000', '#00b300', '#9900cc', '#ff9500', '#00c3c3']

        # Handle Ichimoku Cloud shading if both spans are present
        if 'Ichimoku_senkou_span_a' in indicator_plot_cols and 'Ichimoku_senkou_span_b' in indicator_plot_cols:
            # Fill bullish regions (green) - when Senkou A is above Senkou B
            ax_ind.fill_between(
                df.index, df['Ichimoku_senkou_span_a'], df['Ichimoku_senkou_span_b'],
                where=df['Ichimoku_senkou_span_a'] >= df['Ichimoku_senkou_span_b'],
                color='#27ae60',  # Green
                alpha=0.2,
                interpolate=True,
                label='Kumo (Bullish)'
            )
            
            # Fill bearish regions (red) - when Senkou B is above Senkou A
            ax_ind.fill_between(
                df.index, df['Ichimoku_senkou_span_a'], df['Ichimoku_senkou_span_b'],
                where=df['Ichimoku_senkou_span_a'] < df['Ichimoku_senkou_span_b'],
                color='#e74c3c',  # Red
                alpha=0.2,
                interpolate=True,
                label='Kumo (Bearish)'
            )
        
        for i, col in enumerate(indicator_plot_cols):
            # Special handling for PSAR - display as dots instead of lines
            if 'PSA' in col or 'STR' in col:
                # Determine whether dots should be above or below the price (above in downtrend, below in uptrend)
                # PSAR dots are above price in a downtrend (bearish) and below price in an uptrend (bullish)
                above_price = df[col] > df[price_col]
                
                # Use different colors for bullish vs bearish indicators
                ax_ind.scatter(
                    df.index[above_price], df[col][above_price],
                    label=f'{col} (Bearish)',
                    color='#e50000',  # Red for bearish
                    marker='v',       # Down-pointing triangle for bearish
                    s=30,             # Marker size
                    alpha=0.8
                )
                
                ax_ind.scatter(
                    df.index[~above_price], df[col][~above_price],
                    label=f'{col} (Bullish)',
                    color='#00b300',  # Green for bullish
                    marker='^',       # Up-pointing triangle for bullish
                    s=30,             # Marker size
                    alpha=0.8
                )
            elif col == 'Ichimoku_tenkan_sen':
                ax_ind.plot(df.index, df[col], 
                           label='Tenkan-sen (Conversion Line)', 
                           color='#3498db',  # Blue
                           linewidth=1.5, 
                           alpha=1.0)
            elif col == 'Ichimoku_kijun_sen':
                ax_ind.plot(df.index, df[col], 
                           label='Kijun-sen (Base Line)', 
                           color='#9b59b6',  # Purple
                           linewidth=1.5, 
                           alpha=1.0)
            elif col == 'Ichimoku_senkou_span_a':
                ax_ind.plot(df.index, df[col], 
                           label='Senkou Span A (Leading Span A)', 
                           color='#e74c3c',  # Red
                           linewidth=1.5, 
                           alpha=0.8)
            elif col == 'Ichimoku_senkou_span_b':
                ax_ind.plot(df.index, df[col], 
                           label='Senkou Span B (Leading Span B)', 
                           color='#27ae60',  # Green
                           linewidth=1.5, 
                           alpha=0.8)
            elif col == 'Ichimoku_chikou_span':
                ax_ind.plot(df.index, df[col], 
                           label='Chikou Span (Lagging Span)', 
                           color='#d4cd11',  # Yellow
                           linewidth=1.5, 
                           alpha=0.8)
            else:
                # Use the correct axes with a rotating color scheme for other indicators
                color_idx = i % len(contrast_colors)
                
                ax_ind.plot(df.index, df[col], 
                           label=col, 
                           color=contrast_colors[color_idx],
                           linewidth=2.0, 
                           alpha=1.0)

        # Plot MACD Histogram if applicable with more vivid colors
        if plot_hist:
            colors = ['#00cc00' if x >= 0 else '#e60000' for x in df[hist_col_name]]
            ax_ind.bar(df.index, df[hist_col_name], label=hist_col_name, color=colors, alpha=0.8, width=0.7)

    # Plot +DI/-DI columns on second panel
    if di_cols and ax_di is not None:
        di_colors = {'plus_di': '#00b300', 'minus_di': '#9900cc'}
        for col in di_cols:
            if col.startswith('+DI_'):
                ax_di.plot(df.index, df[col], label=col, color=di_colors['plus_di'], linewidth=1.8)
            elif col.startswith('-DI_'):
                ax_di.plot(df.index, df[col], label=col, color=di_colors['minus_di'], linewidth=1.8)
        ax_di.set_ylabel('+DI / -DI', fontweight='bold', fontsize=11)
        ax_di.grid(True, linestyle='--', alpha=0.7, color='#303030')
        ax_di.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Plot ARO_UP/ARO_DOWN columns on second panel
    if aro_updown_cols and ax_di is not None:
        aro_colors = {'up': '#00b300', 'down': '#e50000'}
        for col in aro_updown_cols:
            col_upper = col.upper()
            if col_upper.startswith('ARO_UP_'):
                ax_di.plot(df.index, df[col], label=col, color=aro_colors['up'], linewidth=1.8)
            elif col_upper.startswith('ARO_DOWN_'):
                ax_di.plot(df.index, df[col], label=col, color=aro_colors['down'], linewidth=1.8)
        ax_di.set_ylabel('Aroon Up/Down', fontweight='bold', fontsize=11)
        ax_di.grid(True, linestyle='--', alpha=0.7, color='#303030')
        ax_di.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Plot ADX columns on third panel
    if adx_cols and ax_osc is not None:
        for col in adx_cols:
            ax_osc.plot(df.index, df[col], label=col, color='#e50000', linewidth=2.0)
        ax_osc.set_ylabel('ADX', fontweight='bold', fontsize=11)
        ax_osc.grid(True, linestyle='--', alpha=0.7, color='#303030')
        ax_osc.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Plot ARO_OSCILLATOR columns on third panel
    if aro_osc_cols and ax_osc is not None:
        for col in aro_osc_cols:
            ax_osc.plot(df.index, df[col], label=col, color='#9900cc', linewidth=2.0)
        ax_osc.set_ylabel('Aroon Oscillator', fontweight='bold', fontsize=11)
        ax_osc.grid(True, linestyle='--', alpha=0.7, color='#303030')
        ax_osc.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Plot MACD line/signal columns on second panel
    if macd_line_cols and ax_di is not None:
        macd_colors = {'mac': '#e50000', 'signal': '#00b300'}
        for col in macd_line_cols:
            col_upper = col.upper()
            if col_upper.startswith('MAC_'):
                ax_di.plot(df.index, df[col], label=col, color=macd_colors['mac'], linewidth=1.8)
            elif col_upper.startswith('SIGNAL_'):
                ax_di.plot(df.index, df[col], label=col, color=macd_colors['signal'], linewidth=1.8)
        ax_di.set_ylabel('MACD', fontweight='bold', fontsize=11)
        ax_di.grid(True, linestyle='--', alpha=0.7, color='#303030')
        ax_di.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Plot MACD histogram columns on third panel
    if macd_hist_cols and ax_osc is not None:
        for col in macd_hist_cols:
            colors = ['#00cc00' if x >= 0 else '#e60000' for x in df[col]]
            ax_osc.bar(df.index, df[col], label=col, color=colors, alpha=0.8, width=0.7)
        ax_osc.set_ylabel('MACD Histogram', fontweight='bold', fontsize=11)
        ax_osc.grid(True, linestyle='--', alpha=0.7, color='#303030')
        ax_osc.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Plot PPO line/signal columns on second panel
    if ppo_line_cols and ax_di is not None:
        ppo_colors = {'ppo': '#e50000', 'signal': '#00b300'}
        for col in ppo_line_cols:
            col_upper = col.upper()
            if col_upper.startswith('PPO_SIG_'):
                ax_di.plot(df.index, df[col], label=col, color=ppo_colors['signal'], linewidth=1.8)
            elif col_upper.startswith('PPO_'):
                ax_di.plot(df.index, df[col], label=col, color=ppo_colors['ppo'], linewidth=1.8)
        ax_di.set_ylabel('PPO', fontweight='bold', fontsize=11)
        ax_di.grid(True, linestyle='--', alpha=0.7, color='#303030')
        ax_di.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Plot PPO histogram columns on third panel
    if ppo_hist_cols and ax_osc is not None:
        for col in ppo_hist_cols:
            colors = ['#00cc00' if x >= 0 else '#e60000' for x in df[col]]
            ax_osc.bar(df.index, df[col], label=col, color=colors, alpha=0.8, width=0.7)
        ax_osc.set_ylabel('PPO Histogram', fontweight='bold', fontsize=11)
        ax_osc.grid(True, linestyle='--', alpha=0.7, color='#303030')
        ax_osc.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Plot PVO line/signal columns on second panel
    if pvo_line_cols and ax_di is not None:
        pvo_colors = {'pvo': '#e50000', 'signal': '#00b300'}
        for col in pvo_line_cols:
            col_upper = col.upper()
            if col_upper.startswith('PVO_SIGNAL_'):
                ax_di.plot(df.index, df[col], label=col, color=pvo_colors['signal'], linewidth=1.8)
            elif col_upper.startswith('PVO_'):
                ax_di.plot(df.index, df[col], label=col, color=pvo_colors['pvo'], linewidth=1.8)
        ax_di.set_ylabel('PVO', fontweight='bold', fontsize=11)
        ax_di.grid(True, linestyle='--', alpha=0.7, color='#303030')
        ax_di.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Plot PVO histogram columns on third panel
    if pvo_hist_cols and ax_osc is not None:
        for col in pvo_hist_cols:
            colors = ['#00cc00' if x >= 0 else '#e60000' for x in df[col]]
            ax_osc.bar(df.index, df[col], label=col, color=colors, alpha=0.8, width=0.7)
        ax_osc.set_ylabel('PVO Histogram', fontweight='bold', fontsize=11)
        ax_osc.grid(True, linestyle='--', alpha=0.7, color='#303030')
        ax_osc.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Plot squeeze columns on dedicated axis when present
    if squeeze_cols:
        squeeze_colors = ['#00b300', '#c20078']
        if not ax_squeeze:
            ax_squeeze = ax_price
        for i, col in enumerate(squeeze_cols):
            color = squeeze_colors[i % len(squeeze_colors)]
            ax_squeeze.step(
                df.index,
                df[col].astype(int),
                where='post',
                label=col,
                color=color,
                linewidth=1.8
            )
        ax_squeeze.set_ylim(-0.2, 1.2)
        ax_squeeze.set_ylabel('Squeeze', fontweight='bold', fontsize=11)
        ax_squeeze.set_yticks([0, 1])
        ax_squeeze.grid(True, linestyle='--', alpha=0.5, color='#505050')

    # Final Touches
    if title:
        fig.suptitle(title)
    else:
        base_title = f"{df.attrs.get('symbol', 'Data')} {price_col}"
        if columns_to_plot:
            base_title += f" and {', '.join(columns_to_plot)}"
        fig.suptitle(base_title)

    ax_price.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if need_indicator_subplot:
        ax_ind.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if need_squeeze_subplot and ax_squeeze is not None:
        ax_squeeze.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Apply grid to the indicator axis regardless of subplot mode
    if ax_ind is not ax_price or plot_on_subplot:
        ax_ind.grid(True, linestyle='--', alpha=0.7, color='#303030')

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to prevent title overlap
    return fig


