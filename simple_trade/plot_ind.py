import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from typing import List, Optional, Union, Literal
import logging

class IndicatorPlotter:
    """Handles plotting financial data and technical indicators."""

    def __init__(self):
        """Initializes the plotter."""
        # Initialization logic can be added here if needed
        pass

    def plot_results(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close',
        column_names: List[str] = None,
        plot_on_subplot: bool = False,
        title: Optional[str] = None,
        plot_type: Literal['line', 'candlestick'] = 'line'
    ):
        """Plots the price data and specified indicator columns.

        Args:
            df: DataFrame containing price and indicator data.
            price_col: Name of the column representing the price.
            column_names: List of column names to plot. These must exist in the DataFrame.
            plot_on_subplot: Whether to plot indicators on a separate subplot below price.
                            If False, indicators will be overlaid on the price chart.
            title: Optional title for the plot.
            plot_type: Type of plot for price data. Options are 'line' (default) or 'candlestick'.
                      For candlestick, the DataFrame must contain 'Open', 'High', 'Low', and 'Close' columns.
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

        # Determine layout based on plot_on_subplot
        if plot_on_subplot and columns_to_plot:
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(16, 8), gridspec_kw={'height_ratios': [3, 1]})
            ax_price, ax_ind = axes
        else:
            # Always create 1x1 subplot even for overlay to maintain consistency
            fig, ax = plt.subplots(1, 1, figsize=(16, 8))
            ax_price = ax # Price axis is the only axis
            ax_ind = ax   # Indicators will plot on the same axis

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
        if indicator_cols:
            plot_hist = False
            hist_col_name = ''
            # Special handling for MACD Histogram
            hist_col_name = next((col for col in indicator_cols if 'hist' in col), None)
            if hist_col_name:
                plot_hist = True
                indicator_cols.remove(hist_col_name)

            # Plot lines with high contrast colors and increased line thickness
            contrast_colors = ['#e50000', '#00b300', '#9900cc', '#ff9500', '#00c3c3']

            # Handle Ichimoku Cloud shading if both spans are present
            if 'Ichimoku_senkou_span_a' in indicator_cols and 'Ichimoku_senkou_span_b' in indicator_cols:
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
            
            for i, col in enumerate(indicator_cols):
                # Special handling for PSAR - display as dots instead of lines
                if 'PSAR' in col or 'Supertrend' in col:
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

        # Final Touches
        if title:
            fig.suptitle(title)
        else:
             base_title = f"{df.attrs.get('symbol', 'Data')} {price_col}"
             if indicator_cols:
                 base_title += f" and {', '.join(indicator_cols)}"
             fig.suptitle(base_title)

        ax_price.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if plot_on_subplot:
            ax_ind.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Apply grid to the indicator axis regardless of subplot mode
        ax_ind.grid(True, linestyle='--', alpha=0.7, color='#303030')

        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to prevent title overlap
        # Return the figure object instead of showing it directly
        # This allows the caller to control when and how to display the plot
        return fig