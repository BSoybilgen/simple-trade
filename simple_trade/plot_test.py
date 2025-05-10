import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Optional

# Define which indicators typically overlay the price chart
# Copied from plot_ind.py for consistency in finding columns
# OVERLAY_INDICATORS = {'sma', 'ema', 'wma', 'hma', 'bollinger'} # No longer needed for column finding

class BacktestPlotter:
    """Handles plotting of backtesting results."""

    def __init__(self):
        """
        Initializes the plotter, primarily storing the results dictionary.

        Args:
            results (dict): The results dictionary returned by the backtester.
                          This is used for metadata like strategy name.
        """

    def plot_results(
        self,
        data_df: pd.DataFrame,
        history_df: pd.DataFrame,
        price_col: str = 'Close',
        indicator_cols: Optional[List[str]] = None, # Accept list of columns directly
        title: Optional[str] = None,
        # **indicator_kwargs # Remove kwargs
    ) -> plt.Figure:
        """
        Generates and displays the backtest results plot.

        Args:
            data_df (pd.DataFrame): The original data DataFrame with prices and indicators.
            history_df (pd.DataFrame): The portfolio history DataFrame.
            price_col (str): The name of the column containing the price data.
            indicator_cols (Optional[List[str]]): List of indicator column names to plot.
            title (Optional[str]): Optional title for the plot.
            # indicator (Optional[str]): The name of the primary indicator used in the strategy. # Removed
            # **indicator_kwargs: Keyword arguments passed to identify the indicator columns. # Removed

        Returns:
            plt.Figure: The generated matplotlib Figure object.
        """
        if not isinstance(data_df.index, pd.DatetimeIndex):
            try:
                data_df.index = pd.to_datetime(data_df.index)
            except Exception as e:
                raise ValueError(f"Failed to convert data_df index to DatetimeIndex: {e}")
        if not isinstance(history_df.index, pd.DatetimeIndex):
            try:
                history_df.index = pd.to_datetime(history_df.index)
            except Exception as e:
                raise ValueError(f"Failed to convert history_df index to DatetimeIndex: {e}")

        # Ensure alignment (use intersection of indices)
        plot_index = data_df.index.intersection(history_df.index)
        if plot_index.empty:
            print("Warning: No overlapping dates found between data_df and history_df. Cannot plot.")
            return None # Or raise error?

        data_df = data_df.loc[plot_index]
        history_df = history_df.loc[plot_index]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                                            gridspec_kw={'height_ratios': [3, 1, 2]})

        # --- Plot 1: Price and Trades --- 
        ax1.plot(plot_index, data_df[price_col], label=f'{price_col} Price', color='skyblue', linewidth=1.5)
        ax1.set_ylabel('Price')
        ax1.set_title(f'{price_col} and Trade Signals')
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Plot trades from history using the 'Action' column - use case=False for case-insensitive matching
        buys = history_df[history_df['Action'].str.contains('Buy', case=False, na=False) & ~history_df['Action'].str.contains('Cover', case=False, na=False)] # Exclude 'Cover and Buy'
        sells = history_df[history_df['Action'].str.contains('Sell', case=False, na=False) & ~history_df['Action'].str.contains('Short', case=False, na=False)] # Exclude 'Sell and Short'
        shorts = history_df[history_df['Action'].str.contains('Short', case=False, na=False) & ~history_df['Action'].str.contains('Sell', case=False, na=False)] # Exclude 'Sell and Short'
        covers = history_df[history_df['Action'].str.contains('Cover', case=False, na=False) & ~history_df['Action'].str.contains('Buy', case=False, na=False)] # Exclude 'Cover and Buy'

        # Combined actions
        sell_and_shorts = history_df[history_df['Action'].str.contains('Sell and Short', case=False, na=False)]
        cover_and_buys = history_df[history_df['Action'].str.contains('Cover and Buy', case=False, na=False)]

        ax1.plot(buys.index, data_df.loc[buys.index, price_col], '^', markersize=8, color='lime', label='Buy')
        ax1.plot(sells.index, data_df.loc[sells.index, price_col], 'v', markersize=8, color='red', label='Sell')
        ax1.plot(shorts.index, data_df.loc[shorts.index, price_col], 'v', markersize=8, color='fuchsia', label='Short')
        ax1.plot(covers.index, data_df.loc[covers.index, price_col], '^', markersize=8, color='orange', label='Cover')
        ax1.plot(sell_and_shorts.index, data_df.loc[sell_and_shorts.index, price_col], 'x', markersize=8, color='darkred', label='Sell & Short') # New
        ax1.plot(cover_and_buys.index, data_df.loc[cover_and_buys.index, price_col], 'P', markersize=8, color='darkgreen', label='Cover & Buy') # New

        # Move legend outside plot to the right
        ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

        # --- Plot 2: Portfolio Value --- 
        ax2.plot(plot_index, history_df['PortfolioValue'], label='Portfolio Value', color='purple', linewidth=1.5)
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.set_title('Portfolio Value Over Time')
        ax2.grid(True, linestyle='--', alpha=0.6)
        # Move legend outside plot to the right
        ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

        # --- Plot 3: Indicators --- 
        # Use the directly provided indicator_cols list
        valid_indicator_cols = []
        if indicator_cols:
            for col in indicator_cols:
                if col in data_df.columns:
                    valid_indicator_cols.append(col)
                else:
                    print(f"Warning: Indicator column '{col}' not found in data_df.")
        
        if valid_indicator_cols:
            # Define contrasting colors for indicator lines
            contrast_colors = ['#FFD700', '#32CD32', '#1E90FF', '#FF8C00', '#FF69B4', '#BA55D3'] 
            for i, col in enumerate(valid_indicator_cols):
                color_idx = i % len(contrast_colors)
                ax3.plot(plot_index, data_df[col], label=col, linewidth=1.5, color=contrast_colors[color_idx])
            ax3.set_ylabel('Indicator Value')
            ax3.set_title('Indicator Values') # More generic title
            # Move legend outside plot to the right
            ax3.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
            ax3.grid(True, linestyle='--', alpha=0.6)
        else:
            ax3.set_title('No Indicators Specified/Found for Plotting')
            ax3.grid(False)

        # --- Final Touches ---        
        plt.xlabel('Date')
        # Adjust layout to make space for legends on the right and suptitle
        plt.tight_layout(rect=[0, 0, 0.85, 0.96])

        return fig


if __name__ == '__main__':
    # Example Usage (Placeholder - needs actual data)
    print("BacktestPlotter class defined. Create an instance and call plot_results with required DataFrames and arguments.")
    # Example:
    # Assuming 'results', 'history_df', 'data_df' are populated from a backtest:
    # plotter = BacktestPlotter(results) # Only pass results dict
    # indicator_cols_to_plot = ['SMA_50', 'SMA_200'] # Explicitly define columns
    # fig = plotter.plot_results(data_df, history_df, price_col='Adj Close', indicator_cols=indicator_cols_to_plot, title="My SMA Backtest")
    # if fig:
    #    plt.show()
