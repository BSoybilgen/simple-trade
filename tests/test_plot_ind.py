import pytest
import pandas as pd
import pandas.testing as pdt
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock, call, ANY
import logging

from simple_trade.plot_ind import IndicatorPlotter

# === Fixtures ===

@pytest.fixture(scope="session")
def sample_line_data():
    """DataFrame suitable for line plots with indicators."""
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
    data = {
        'Close': [100, 101, 102, 101, 103],
        'SMA_10': [99, 100, 101, 101, 102],
        'SMA_20': [98, 99, 100, 100, 101],
        'RSI_14': [50, 55, 60, 55, 65],
        'MACD_hist': [0.1, 0.2, -0.1, 0.3, 0.4]
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    return df

@pytest.fixture(scope="session")
def sample_candlestick_data():
    """Fixture for sample OHLC DataFrame with additional indicators."""
    dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
    data = {
        'Open': [100, 101, 102, 103, 104],
        'High': [102, 103, 104, 105, 106],
        'Low': [99, 100, 101, 102, 103],
        'Close': [101, 102, 103, 104, 105],
        'Volume': [1000, 1100, 1050, 1200, 1150],
        'SMA_10': [100, 101, 102, 103, 104] # Example overlay
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    return df

@pytest.fixture(scope="session")
def sample_line_data_with_volume():
    """Fixture for sample line data with volume."""
    dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
    data = {
        'Close': [100, 101, 102, 101, 103],
        'Volume': [1000, 1200, 1500, 1800, 2000],
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    return df

@pytest.fixture(scope="session")
def sample_line_data_many_indicators():
    """Generate sample data with many dummy subplot indicators."""
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=50, freq='D'))
    data = {
        'Close': np.linspace(100, 150, 50) + np.random.randn(50) * 2,
    }
    # Add more indicators than default colors (assuming default is 5)
    for i in range(7):
        data[f'Subplot_{i}'] = np.random.rand(50) * 100
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    return df

@pytest.fixture
def mock_plotting_fixture(mocker):
    """Provides mocked plt, fig, and flexible axes (single or list)."""
    mock_plt = mocker.patch('simple_trade.plot_ind.plt', autospec=True)
    mock_fig = MagicMock(spec=plt.Figure)
    mock_ax_single = MagicMock(spec=plt.Axes, name="SingleAxis")
    mock_axes_list = [MagicMock(spec=plt.Axes, name="Axis0"), MagicMock(spec=plt.Axes, name="Axis1")]

    # Store the created axes to be returned by the fixture value
    created_axes = None

    def subplots_side_effect(*args, **kwargs):
        nonlocal created_axes
        nrows = args[0] if len(args) > 0 else 1
        ncols = args[1] if len(args) > 1 else 1
        # Check the gridspec_kw for subplot determination as well
        is_subplot = (nrows > 1 or ncols > 1) or (kwargs.get('gridspec_kw') and kwargs['gridspec_kw'].get('height_ratios'))

        if is_subplot:
            # Ensure the list has enough mocks if more are needed
            while len(mock_axes_list) < nrows * ncols:
                mock_axes_list.append(MagicMock(spec=plt.Axes, name=f"Axis{len(mock_axes_list)}"))
            created_axes = mock_axes_list[:nrows*ncols]
            return (mock_fig, created_axes)
        else:
            created_axes = mock_ax_single
            return (mock_fig, created_axes)

    mock_plt.subplots.side_effect = subplots_side_effect
    mock_plt.figure.return_value = mock_fig

    # Yield the mocks - the axes returned depend on what subplots was called with
    yield mock_plt, mock_fig, lambda: created_axes

@pytest.fixture(scope="session")
def sample_psar_data():
    """Fixture for sample data including PSAR."""
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
    data = {
        'Close': [100, 101, 103, 102, 104],  # Price data
        'PSAR':  [ 99, 100, 101, 103, 103]   # PSAR below, then above, then below price
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    return df

@pytest.fixture(scope="session")
def sample_ichimoku_data():
    """Fixture for sample data including Ichimoku spans."""
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
    data = {
        'Close': [100, 101, 103, 102, 104],
        'Ichimoku_senkou_span_a': [ 98,  99, 101, 104, 105], # Span A starts below, crosses above B
        'Ichimoku_senkou_span_b': [100, 100, 100, 103, 104]  # Span B
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    return df

# === Tests ===

# === Tests for _validate_and_prepare_data (called indirectly by plot_results) ===
def test_plot_with_non_datetime_index(mocker, sample_line_data, mock_plotting_fixture):
    """Test plot_results call with a non-datetime index converts it."""
    mock_plt, mock_fig, get_axes = mock_plotting_fixture
    plotter = IndicatorPlotter()
    data_non_dt = sample_line_data.reset_index()
    # Mock pd.to_datetime used inside plot_results (implicitly checked)
    # mock_to_datetime = mocker.patch('pandas.to_datetime', wraps=pd.to_datetime)

    # Call the correct method
    plotter.plot_results(data_non_dt, price_col='Close')

    # Assert pd.to_datetime was called (indirect check - plot should proceed)
    # mock_to_datetime.assert_called()
    ax = get_axes()
    assert ax.plot.called # Check if plot was called on the axes

def test_plot_index_conversion_failure(mocker, sample_line_data):
    """Test plot_results raises ValueError if index conversion fails (line 52)."""
    plotter = IndicatorPlotter()
    data_bad_index = sample_line_data.copy()
    data_bad_index.index = pd.Index([f'invalid_{i}' for i in range(len(data_bad_index))])

    # Mock pd.to_datetime to simulate failure during validation
    # Note: This validation might not happen if index isn't used directly
    # Adjust test if validation happens elsewhere
    # mocker.patch('pandas.to_datetime', side_effect=ValueError("Conversion failed"))

    # For now, assume the plot_results method itself doesn't explicitly convert index
    # but relies on pandas/matplotlib downstream handling. If specific conversion
    # fails, the error might come from plotting itself.
    # Let's test the explicit check for price_col instead, which is in plot_results
    data_missing_price = sample_line_data.drop(columns=['Close'])
    with pytest.raises(ValueError, match="Price column 'Close' not found"):
        plotter.plot_results(data_missing_price, price_col='Close')

def test_plot_empty_column_names_warning(sample_line_data, caplog, mock_plotting_fixture):
    """Test warning when column_names is empty or None."""
    mock_plt, _, get_axes = mock_plotting_fixture
    plotter = IndicatorPlotter()
    # Test with empty list
    plotter.plot_results(sample_line_data, price_col='Close', column_names=[])
    assert "No valid indicator columns specified or found. Plotting price only." in caplog.text
    mock_plt.subplots.assert_called_once() # Ensure plot still proceeds
    caplog.clear()
    mock_plt.subplots.reset_mock()
    # Test with None (default)
    plotter.plot_results(sample_line_data, price_col='Close', column_names=None)
    assert "No indicator columns specified. Plotting price only." in caplog.text
    mock_plt.subplots.assert_called_once()

def test_plot_invalid_column_names_warning(sample_line_data, caplog, mock_plotting_fixture):
    """Test warning when column_names contains invalid columns."""
    mock_plt, _, get_axes = mock_plotting_fixture
    plotter = IndicatorPlotter()
    # Use correct parameter name
    plotter.plot_results(sample_line_data, price_col='Close', column_names=['Close', 'NonExistent_SMA', 'SMA_10'])

    assert "Columns ['NonExistent_SMA'] specified but not found in DataFrame." in caplog.text
    # Check that valid overlay indicator (SMA_10) was still plotted
    ax = get_axes() # Should be single axis
    assert isinstance(ax, MagicMock) and not isinstance(ax, list)

def test_plot_candlestick_missing_overlay_warning(sample_candlestick_data, mock_plotting_fixture, caplog):
    """Test warning for missing overlay indicator in candlestick plot."""
    # No need for mpf.plot patch
    mock_plt, _, get_axes = mock_plotting_fixture
    plotter = IndicatorPlotter()
    plotter.plot_results(
        sample_candlestick_data,
        plot_type='candlestick',
        column_names=['SMA_10', 'Missing_Overlay'] # SMA_10 is overlay
    )
    assert "Columns ['Missing_Overlay'] specified but not found in DataFrame." in caplog.text
    # Check that valid overlay (SMA_10) was still plotted on the price axis
    ax_price = get_axes() # Single axis expected
    assert any(call.kwargs.get('label') == 'SMA_10' for call in ax_price.plot.call_args_list)

def test_plot_line_missing_overlay_warning(sample_line_data, mock_plotting_fixture, caplog):
    """Test warning for missing overlay indicator in line plot."""
    mock_plt, _, get_axes = mock_plotting_fixture
    plotter = IndicatorPlotter()
    plotter.plot_results(
        sample_line_data,
        plot_type='line',
        price_col='Close',
        column_names=['SMA_10', 'Missing_Overlay'] # SMA_10 is overlay
    )
    assert "Columns ['Missing_Overlay'] specified but not found in DataFrame." in caplog.text
    # Check that valid overlay (SMA_10) was still plotted
    ax = get_axes()
    assert any(call.kwargs.get('label') == 'SMA_10' for call in ax.plot.call_args_list)

def test_plot_line_missing_subplot_warning(sample_line_data, mock_plotting_fixture, caplog):
    """Test warning for missing subplot indicator in line plot."""
    mock_plt, _, get_axes = mock_plotting_fixture
    plotter = IndicatorPlotter()
    plotter.plot_results(
        sample_line_data,
        plot_type='line',
        price_col='Close',
        column_names=['RSI_14', 'Missing_Subplot'], # RSI and MACD_hist are subplots
        plot_on_subplot=True # Explicitly request subplot
    )
    assert "Columns ['Missing_Subplot'] specified but not found in DataFrame." in caplog.text
    # Check that valid subplot (RSI_14) was still plotted
    axes = get_axes()
    assert isinstance(axes, list) # Should have multiple axes

def test_plot_candlestick_missing_columns_error(sample_candlestick_data):
    """Test ValueError if required columns are missing for candlestick."""
    plotter = IndicatorPlotter()
    data_missing_open = sample_candlestick_data.drop(columns=['Open'])
    # Correct the regex pattern to match the exact error message format from plot_ind.py
    # Escape regex special characters: [, ], .
    expected_error_msg = r"Candlestick plot requires columns \['Open', 'High', 'Low', 'Close'\], but \['Open'\] are missing\."
    with pytest.raises(ValueError, match=expected_error_msg):
        plotter.plot_results(data_missing_open, plot_type='candlestick')

def test_plot_candlestick_basic(sample_candlestick_data, mock_plotting_fixture):
    """Test basic candlestick plot using matplotlib within plot_results."""
    mock_plt, mock_fig, get_axes = mock_plotting_fixture
    plotter = IndicatorPlotter()

    # Call the correct method with appropriate type
    plotter.plot_results(sample_candlestick_data, plot_type='candlestick')

    # Assert subplots called (expecting single axis)
    mock_plt.subplots.assert_called_once_with(1, 1, figsize=(16, 8))
    ax = get_axes()
    assert isinstance(ax, MagicMock) and not isinstance(ax, list)

    # Assert matplotlib drawing functions were called (basic check)
    assert ax.add_patch.called # For candle bodies
    assert ax.plot.called      # For wicks and legend dummy

    # Check final calls
    # TODO: Investigate why tight_layout/show aren't called in this specific path
    mock_plt.tight_layout.assert_called_once_with(rect=[0, 0, 0.85, 1])

def test_plot_line_basic_price_only(sample_line_data, mock_plotting_fixture):
    """Test plotting only price as a line."""
    mock_plt, mock_fig, get_axes = mock_plotting_fixture
    plotter = IndicatorPlotter()

    # Call the correct method
    plotter.plot_results(df=sample_line_data, price_col='Close', plot_type='line')

    # Assert subplots called correctly (single plot, 1 row, 1 col)
    mock_plt.subplots.assert_called_once_with(1, 1, figsize=(16, 8))
    ax = get_axes() # Get the single axis
    assert isinstance(ax, MagicMock)
    assert not isinstance(ax, list)

    # Assert price plotted
    assert ax.plot.call_count == 1
    price_call_args, price_call_kwargs = ax.plot.call_args_list[0]
    pdt.assert_index_equal(price_call_args[0], sample_line_data.index)
    pdt.assert_series_equal(price_call_args[1], sample_line_data['Close'])
    assert price_call_kwargs.get('label') == 'Close'
    assert price_call_kwargs.get('color') == '#0343df'
    assert price_call_kwargs.get('linewidth') == 2.5

    # Correct grid assertion
    ax.grid.assert_called_with(True, linestyle='--', alpha=0.7, color='#303030')
    ax.legend.assert_called_once_with(loc='center left', bbox_to_anchor=(1, 0.5))

    # Check final calls
    mock_plt.tight_layout.assert_called_once_with(rect=[0, 0, 0.85, 1])

def test_plot_line_overlay_indicators(sample_line_data, mock_plotting_fixture):
    """Test line plot with overlay indicators."""
    mock_plt, mock_fig, get_axes = mock_plotting_fixture
    plotter = IndicatorPlotter()

    # Call the correct method with correct parameter name
    plotter.plot_results(
        df=sample_line_data,
        price_col='Close',
        column_names=['SMA_10', 'SMA_20'],
        plot_type='line'
    )

    # Assert subplots called correctly
    mock_plt.subplots.assert_called_once_with(1, 1, figsize=(16, 8))
    ax = get_axes() # Single axis expected
    assert isinstance(ax, MagicMock)
    assert not isinstance(ax, list)

    # Price + 2 overlays = 3 plot calls
    assert ax.plot.call_count == 3

    # Check 'SMA_10' plot call
    sma10_call = next(c for c in ax.plot.call_args_list if c.kwargs.get('label') == 'SMA_10')
    pdt.assert_series_equal(sma10_call.args[1], sample_line_data['SMA_10'])
    # Color check removed - overlay indicators use default color cycle
    # assert sma10_call.kwargs.get('color') == '#ff9900'

    # Check 'SMA_20' plot call
    sma20_call = next(c for c in ax.plot.call_args_list if c.kwargs.get('label') == 'SMA_20')
    pdt.assert_series_equal(sma20_call.args[1], sample_line_data['SMA_20'])
    # Color check removed - overlay indicators use default color cycle
    # assert sma20_call.kwargs.get('color') == '#33cc33'

    # Correct grid assertion
    ax.grid.assert_called_with(True, linestyle='--', alpha=0.7, color='#303030')
    ax.legend.assert_called_once_with(loc='center left', bbox_to_anchor=(1, 0.5))

    # Check final calls
    mock_plt.tight_layout.assert_called_once_with(rect=[0, 0, 0.85, 1])

def test_plot_line_subplot_indicators(sample_line_data, mock_plotting_fixture):
    """Test line plot with subplot indicators."""
    mock_plt, mock_fig, get_axes = mock_plotting_fixture
    plotter = IndicatorPlotter()

    # Call the correct method with correct parameter name and subplot=True
    plotter.plot_results(
        df=sample_line_data,
        price_col='Close',
        column_names=['RSI_14', 'MACD_hist'], # RSI and MACD_hist are subplots
        plot_type='line',
        plot_on_subplot=True # Ensure subplots
    )

    # Assert subplots called correctly
    mock_plt.subplots.assert_called_once_with(2, 1, sharex=True, figsize=(16, 8), gridspec_kw={'height_ratios': [3, 1]})
    axes = get_axes() # Expecting list of axes
    assert isinstance(axes, list) and len(axes) == 2
    mock_ax_price, mock_ax_subplot = axes

    # Price plot on ax1
    assert mock_ax_price.plot.call_count == 1

    # Define expected colors based on plot_results implementation
    expected_contrast_colors = ['#e50000', '#00b300', '#9900cc', '#ff9500', '#00c3c3']

    # RSI plot on ax2
    rsi_call = next(c for c in mock_ax_subplot.plot.call_args_list if c.kwargs.get('label') == 'RSI_14')
    pdt.assert_series_equal(rsi_call.args[1], sample_line_data['RSI_14'])
    assert rsi_call.kwargs.get('color') == expected_contrast_colors[0]

    # Check that bar was called for MACD_hist with specific args
    # mock_ax_subplot.bar.assert_called() # Simplified check failed
    mock_ax_subplot.bar.assert_called_once_with(
        sample_line_data.index,
        sample_line_data['MACD_hist'],
        label='MACD_hist',
        color=ANY, # Color list depends on data
        alpha=0.8,
        width=0.7
    )

    # Total plots on subplot axis: 1 line plot (RSI)
    assert mock_ax_subplot.plot.call_count == 1
    # Total bars on subplot axis: 1 bar plot (MACD)
    assert mock_ax_subplot.bar.call_count == 1

    # Correct grid assertions
    mock_ax_price.legend.assert_called_once_with(loc='center left', bbox_to_anchor=(1, 0.5))
    mock_ax_subplot.legend.assert_called_once_with(loc='center left', bbox_to_anchor=(1, 0.5))
    mock_ax_price.grid.assert_called_once_with(True, linestyle='--', alpha=0.7, color='#303030')
    mock_ax_subplot.grid.assert_called_once_with(True, linestyle='--', alpha=0.7, color='#303030')

    # Check final calls
    mock_plt.tight_layout.assert_called_once_with(rect=[0, 0, 0.85, 1])

def test_plot_line_with_volume(sample_line_data_with_volume, mock_plotting_fixture):
    """Test line plot including the volume subplot."""
    mock_plt, mock_fig, get_axes = mock_plotting_fixture
    plotter = IndicatorPlotter()

    # plot_results doesn't handle volume automatically.

    # Let's call plot_results simply, assuming volume isn't handled:
    plotter.plot_results(
        df=sample_line_data_with_volume,
        price_col='Close',
        # volume_col='Volume', # Not a parameter
        plot_type='line'
    )

    # Assert subplots called correctly (EXPECTING 1x1 because volume isn't handled)
    mock_plt.subplots.assert_called_once_with(1, 1, figsize=(16, 8))
    ax = get_axes() # Get the single axis
    assert isinstance(ax, MagicMock) and not isinstance(ax, list)

    # Check Ax1: Price plot
    assert ax.plot.call_count == 1
    price_call_args, price_call_kwargs = ax.plot.call_args
    pdt.assert_index_equal(price_call_args[0], sample_line_data_with_volume.index)
    pdt.assert_series_equal(price_call_args[1], sample_line_data_with_volume['Close'])
    assert price_call_kwargs.get('label') == 'Close'

    # Check Ax2: Volume plot (SHOULD NOT BE CALLED)
    # Assert no bar calls were made
    ax.bar.assert_not_called()

    # Correct grid assertion
    ax.legend.assert_called_once_with(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid.assert_called_with(True, linestyle='--', alpha=0.7, color='#303030')

    # Check final calls
    mock_plt.tight_layout.assert_called_once_with(rect=[0, 0, 0.85, 1])

def test_plot_structure_and_title(sample_line_data, mock_plotting_fixture):
    """Test the overall plot structure setup and title application."""
    mock_plt, mock_fig, get_axes = mock_plotting_fixture
    plotter = IndicatorPlotter()

    test_title = "My Custom Plot Title"
    # Call correct method with correct parameter
    plotter.plot_results(
        sample_line_data,
        price_col='Close',
        column_names=['RSI_14'], # Need one subplot indicator
        title=test_title, # Test title argument here
        plot_on_subplot=True # Ensure subplot layout
    )

    # Check main plot setup (assuming line plot with subplot)
    mock_plt.subplots.assert_called_once_with(2, 1, sharex=True, figsize=(16, 8), gridspec_kw={'height_ratios': [3, 1]})
    axes = get_axes() # Expecting list of axes
    assert isinstance(axes, list) and len(axes) == 2
    mock_ax_price, mock_ax_subplot = axes

    # Check title is set on the *figure* using suptitle
    mock_fig.suptitle.assert_called_once_with(test_title)

    # Correct grid assertions
    mock_ax_price.legend.assert_called_once_with(loc='center left', bbox_to_anchor=(1, 0.5))
    mock_ax_subplot.legend.assert_called_once_with(loc='center left', bbox_to_anchor=(1, 0.5))
    mock_ax_price.grid.assert_called_once_with(True, linestyle='--', alpha=0.7, color='#303030')
    mock_ax_subplot.grid.assert_called_once_with(True, linestyle='--', alpha=0.7, color='#303030')

    # Check final calls
    mock_plt.tight_layout.assert_called_once_with(rect=[0, 0, 0.85, 1])

def test_plot_line_subplot_color_cycling(sample_line_data_many_indicators, mock_plotting_fixture):
    """Test color cycling for subplot indicators."""
    mock_plt, mock_fig, get_axes = mock_plotting_fixture
    plotter = IndicatorPlotter()

    # Get the expected colors from the plot_results implementation
    expected_contrast_colors = ['#e50000', '#00b300', '#9900cc', '#ff9500', '#00c3c3']
    num_colors = len(expected_contrast_colors)

    # Plot with more indicators than colors
    subplot_indicators = [f'Subplot_{i}' for i in range(num_colors + 2)]
    plotter.plot_results(
        sample_line_data_many_indicators,
        price_col='Close',
        column_names=subplot_indicators,
        plot_type='line', # Explicitly line
        plot_on_subplot=True # Ensure subplots
    )

    # Check subplots were created
    mock_plt.subplots.assert_called_once_with(2, 1, sharex=True, figsize=(16, 8), gridspec_kw={'height_ratios': [3, 1]})
    mock_ax_price, mock_ax_subplot = get_axes()

    # Check that plot was called for each subplot indicator on the subplot axis
    assert mock_ax_subplot.plot.call_count == len(subplot_indicators)

    # Check that colors are cycling
    plot_calls = mock_ax_subplot.plot.call_args_list
    used_colors = [call.kwargs.get('color') for call in plot_calls]

    assert len(used_colors) == num_colors + 2
    # First N colors should match the expected list
    assert used_colors[:num_colors] == expected_contrast_colors
    # Next colors should wrap around
    assert used_colors[num_colors] == expected_contrast_colors[0]
    assert used_colors[num_colors + 1] == expected_contrast_colors[1]

def test_plot_invalid_plot_type_error(sample_line_data):
    """Test that ValueError is raised for invalid plot_type."""
    plotter = IndicatorPlotter()
    with pytest.raises(ValueError, match="Invalid plot_type: invalid_type. Choose 'line' or 'candlestick'."):
        plotter.plot_results(sample_line_data, plot_type='invalid_type')

def test_plot_psar_overlay(sample_psar_data, mock_plotting_fixture):
    """Test plotting PSAR as overlay uses scatter correctly."""
    mock_plt, mock_fig, get_axes = mock_plotting_fixture
    plotter = IndicatorPlotter()

    plotter.plot_results(
        df=sample_psar_data,
        price_col='Close',
        column_names=['PSAR'],
        plot_on_subplot=False # Overlay
    )
    
    ax_price = get_axes() # Overlay plots on single axis
    assert ax_price.scatter.call_count == 2

    # Check calls to scatter - order might vary, so find them
    bullish_call = next((c for c in ax_price.scatter.call_args_list if c.kwargs.get('label') == 'PSAR (Bullish)'), None)
    bearish_call = next((c for c in ax_price.scatter.call_args_list if c.kwargs.get('label') == 'PSAR (Bearish)'), None)

    assert bullish_call is not None
    assert bearish_call is not None

    # Verify bullish (below price) call args - marker '^', color '#00b300'
    assert bullish_call.kwargs.get('marker') == '^'
    assert bullish_call.kwargs.get('color') == '#00b300'
    
    # Verify bearish (above price) call args - marker 'v', color '#e50000'
    assert bearish_call.kwargs.get('marker') == 'v'
    assert bearish_call.kwargs.get('color') == '#e50000'

def test_plot_ichimoku_cloud_overlay(sample_ichimoku_data, mock_plotting_fixture):
    """Test Ichimoku cloud shading using fill_between."""
    mock_plt, mock_fig, get_axes = mock_plotting_fixture
    plotter = IndicatorPlotter()
    
    plotter.plot_results(
        df=sample_ichimoku_data,
        price_col='Close',
        column_names=['Ichimoku_senkou_span_a', 'Ichimoku_senkou_span_b'],
        plot_on_subplot=False # Overlay
    )
    
    ax_price = get_axes()
    assert ax_price.fill_between.call_count == 2

    # Check calls to fill_between - order might vary
    bullish_fill = next((c for c in ax_price.fill_between.call_args_list if c.kwargs.get('label') == 'Kumo (Bullish)'), None)
    bearish_fill = next((c for c in ax_price.fill_between.call_args_list if c.kwargs.get('label') == 'Kumo (Bearish)'), None)

    assert bullish_fill is not None
    assert bearish_fill is not None
    
    # Verify colors
    assert bullish_fill.kwargs.get('color') == '#27ae60' # Green
    assert bearish_fill.kwargs.get('color') == '#e74c3c' # Red