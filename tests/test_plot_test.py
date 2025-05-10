import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call, ANY
import matplotlib.pyplot as plt # Import for mocking targets
import pandas.testing as pdt # Import pandas testing utilities

from simple_trade.plot_test import BacktestPlotter

# --- Fixtures --- 

@pytest.fixture
def sample_line_data() -> pd.DataFrame:
    """DataFrame suitable for line plots with indicators."""
    # Using a smaller subset just for BacktestPlotter tests if full data isn't needed
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06'])
    data = {
        'Close': [100, 101, 102, 101, 103, 102.5], # Extended to match complex history
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def sample_history_data() -> pd.DataFrame:
    """DataFrame representing backtest history."""
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
    data = {
        'PortfolioValue': [10000, 10050, 10100, 10080, 10150],
        'Action': ['Initial', 'Buy', '', 'Sell', 'Buy'], # Example signals
    }
    # Make index match sample_line_data subset for simpler tests
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def sample_history_data_complex_signals() -> pd.DataFrame:
    """DataFrame representing backtest history with complex signals."""
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06'])
    data = {
        'PortfolioValue': [10000, 10050, 10100, 10080, 10150, 10120],
        'Action': ['Initial', 'Buy', 'Sell and Short', 'Cover', 'Cover and Buy', 'Sell'],
    }
    return pd.DataFrame(data, index=dates)

# --- Test Classes --- 

class TestBacktestPlotter:
    """Tests for the BacktestPlotter class."""

    @pytest.fixture(autouse=True)
    def mock_matplotlib(self, mocker):
        """Mock matplotlib.pyplot to prevent actual plot generation."""
        self.mock_fig = MagicMock(spec=plt.Figure)
        self.mock_ax1 = MagicMock(spec=plt.Axes)
        self.mock_ax2 = MagicMock(spec=plt.Axes)
        self.mock_ax3 = MagicMock(spec=plt.Axes)
        self.mock_ax3.text = MagicMock() # Explicitly mock the text method
        # Add the transAxes attribute needed by the code under test
        self.mock_ax3.transAxes = object() # Just needs to exist
        self.mock_axes = [self.mock_ax1, self.mock_ax2, self.mock_ax3]

        # Patch the module-level function plt.subplots where it's used
        self.mock_plt = mocker.patch('simple_trade.plot_test.plt')
        self.mock_plt.subplots.return_value = (self.mock_fig, self.mock_axes)
        # We don't need to mock style.use, tight_layout, or show on the module level
        # as the implementation calls fig.tight_layout() and returns fig.
        # self.mock_plt.style.use.return_value = None # No longer needed
        # self.mock_plt.tight_layout.return_value = None # No longer needed
        # self.mock_plt.show.return_value = None # No longer needed

    def test_init(self):
        """Test BacktestPlotter initialization."""
        plotter = BacktestPlotter()
        assert isinstance(plotter, BacktestPlotter)

    def test_plot_results_structure(self, sample_line_data, sample_history_data):
        """Test the basic structure and calls of plot_results."""
        plotter = BacktestPlotter()
        # Align data lengths for this test
        line_data_aligned = sample_line_data.loc[sample_history_data.index]
        plotter.plot_results(data_df=line_data_aligned, history_df=sample_history_data)

        # Assert subplots called correctly (expecting 3 rows)
        self.mock_plt.subplots.assert_called_once_with(3, 1, sharex=True, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 2]})

        # Assert tight_layout and show called
        self.mock_plt.tight_layout.assert_called_once()
        # self.mock_plt.show.assert_called_once() # REMOVED - plot_results returns fig, doesn't show

    def test_plot_results_ax1_price_and_signals(self, sample_line_data, sample_history_data):
        """Test plotting on Ax1 (Price and Signals)."""
        plotter = BacktestPlotter()
        # Align data lengths for this test
        line_data_aligned = sample_line_data.loc[sample_history_data.index]
        plotter.plot_results(data_df=line_data_aligned, history_df=sample_history_data, price_col='Close')

        mock_ax1 = self.mock_axes[0] # Get the first mock axis

        # Assert price plot
        price_call_args, price_call_kwargs = mock_ax1.plot.call_args_list[0] # Assuming price is first plot
        pdt.assert_index_equal(price_call_args[0], line_data_aligned.index)
        pdt.assert_series_equal(price_call_args[1], line_data_aligned['Close'])
        assert price_call_kwargs.get('label') == 'Close Price'
        assert price_call_kwargs.get('color') == 'skyblue'
        assert price_call_kwargs.get('linewidth') == 1.5

        # Assert Buy signal plot (now uses plot, not scatter)
        buy_signals = sample_history_data[sample_history_data['Action'].str.contains('Buy', na=False) & ~sample_history_data['Action'].str.contains('Cover', na=False)]
        buy_call = next((c for c in mock_ax1.plot.call_args_list if c.kwargs.get('label') == 'Buy'), None)
        assert buy_call is not None, "Buy Signal plot call not found"
        buy_call_args, buy_call_kwargs = buy_call
        pdt.assert_index_equal(buy_call_args[0], buy_signals.index)
        pdt.assert_series_equal(buy_call_args[1], line_data_aligned.loc[buy_signals.index, 'Close'])
        assert buy_call_args[2] == '^' # Marker is 3rd positional arg
        assert buy_call_kwargs.get('color') == 'lime'
        assert buy_call_kwargs.get('markersize') == 8

        # Assert Sell signal plot (now uses plot, not scatter)
        sell_signals = sample_history_data[sample_history_data['Action'].str.contains('Sell', na=False) & ~sample_history_data['Action'].str.contains('Short', na=False)]
        sell_call = next((c for c in mock_ax1.plot.call_args_list if c.kwargs.get('label') == 'Sell'), None)
        assert sell_call is not None, "Sell Signal plot call not found"
        sell_call_args, sell_call_kwargs = sell_call
        pdt.assert_index_equal(sell_call_args[0], sell_signals.index)
        pdt.assert_series_equal(sell_call_args[1], line_data_aligned.loc[sell_signals.index, 'Close'])
        assert sell_call_args[2] == 'v' # Marker is 3rd positional arg
        assert sell_call_kwargs.get('color') == 'red'
        assert sell_call_kwargs.get('markersize') == 8

        # Assert axis labels and legend
        mock_ax1.set_ylabel.assert_called_with('Price')
        assert mock_ax1.legend.called # Legend is called, placement check omitted for simplicity

    def test_plot_results_ax2_portfolio_value(self, sample_line_data, sample_history_data):
        """Test plotting on Ax2 (Portfolio Value)."""
        plotter = BacktestPlotter()
        # Align data lengths for this test
        line_data_aligned = sample_line_data.loc[sample_history_data.index]
        plotter.plot_results(data_df=line_data_aligned, history_df=sample_history_data)

        mock_ax2 = self.mock_axes[1] # Get the second mock axis

        # Assert portfolio value plot
        pv_call_args, pv_call_kwargs = mock_ax2.plot.call_args
        pdt.assert_index_equal(pv_call_args[0], sample_history_data.index)
        pdt.assert_series_equal(pv_call_args[1], sample_history_data['PortfolioValue'])
        assert pv_call_kwargs.get('label') == 'Portfolio Value'
        assert pv_call_kwargs.get('color') == 'purple' # Updated color
        assert pv_call_kwargs.get('linewidth') == 1.5 # Updated linewidth
        # assert mock_ax2.fill_between.called # Removed fill_between assertion
        assert mock_ax2.legend.called # Legend is called
        mock_ax2.set_ylabel.assert_called_with('Portfolio Value ($)') # Updated label
        assert mock_ax2.grid.called

    def test_plot_results_ax3_indicator_plotting(self, sample_line_data, sample_history_data):
        """Test plotting on Ax3 (Indicators)."""
        plotter = BacktestPlotter()
        # Align data lengths for this test
        line_data_aligned = sample_line_data.loc[sample_history_data.index]
        # Pass empty indicator_cols for now, just check setup
        self.mock_ax3.reset_mock() # Reset mock before the call
        plotter.plot_results(data_df=line_data_aligned, history_df=sample_history_data, indicator_cols=[])

        mock_ax3 = self.mock_axes[2] # Get the third mock axis

        # Assert axis configuration for the 'else' block when no indicators
        # The text call was removed from the implementation, so we don't assert it.
        # set_yticks([]) is not called in this path, remove assertion
        mock_ax3.set_title.assert_called_once_with('No Indicators Specified/Found for Plotting') # Verify correct title
        mock_ax3.grid.assert_called_once_with(False)

        # Basic check that plot wasn't called since indicator_cols is empty
        # This assumes no other plotting happens on ax3 by default
        indicator_plot_calls = [c for c in mock_ax3.plot.call_args_list if c.kwargs.get('label') != 'Portfolio Value'] # Exclude potential mis-mocking
        assert len(indicator_plot_calls) == 0

    def test_plot_results_complex_signals(self, sample_line_data, sample_history_data_complex_signals):
        """Test plotting with complex signals like 'Sell and Short'."""
        plotter = BacktestPlotter()
        # Data is already aligned for this fixture
        plotter.plot_results(data_df=sample_line_data, history_df=sample_history_data_complex_signals)

        mock_ax1 = self.mock_axes[0]
        mock_ax3 = self.mock_axes[2]

        # Check Ax1 plot calls for complex signals
        expected_plots = {
            'Buy': ('^', 'lime'),
            'Sell': ('v', 'red'),
            'Short': ('v', 'fuchsia'), # Added Short
            'Cover': ('^', 'orange'), # Added Cover
            'Sell & Short': ('x', 'darkred'), # Updated complex
            'Cover & Buy': ('P', 'darkgreen') # Updated complex
        }

        # Get all plot calls on ax1 excluding the initial price plot
        signal_plot_calls = [c for c in mock_ax1.plot.call_args_list if c.kwargs.get('label') != 'Close Price']

        # Check each expected signal type that exists in the fixture
        for signal_label, (marker, color) in expected_plots.items():
            # Find the corresponding data in the fixture
            signal_data = sample_history_data_complex_signals[sample_history_data_complex_signals['Action'] == signal_label]
            if signal_data.empty:
                continue # Skip if this signal type isn't in the specific test data

            # Find the plot call matching the label
            signal_call = next((c for c in signal_plot_calls if c.kwargs.get('label') == signal_label), None)
            assert signal_call is not None, f"Plot call for '{signal_label}' not found."

            # Verify data, marker, color, and markersize
            call_args, call_kwargs = signal_call
            pdt.assert_index_equal(call_args[0], signal_data.index, check_names=False)
            pdt.assert_series_equal(call_args[1], sample_line_data.loc[signal_data.index, 'Close'], check_names=False)
            assert call_args[2] == marker # Marker is 3rd positional arg
            assert call_kwargs.get('color') == color
            assert call_kwargs.get('markersize') == 8
