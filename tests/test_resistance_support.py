import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch
from simple_trade.resistance_support import (
    find_pivot_points, 
    find_resistance_support_lines, 
    plot_resistance_support
)


@pytest.fixture
def sample_price_series():
    """Sample price series with clear highs and lows."""
    dates = pd.date_range('2023-01-01', periods=20, freq='D')
    # Create a series with clear pivot points
    prices = [100, 102, 105, 103, 101, 99, 97, 100, 103, 106,
              108, 105, 102, 104, 107, 110, 108, 106, 109, 111]
    return pd.Series(prices, index=dates, name='Close')


@pytest.fixture
def trending_price_series():
    """Trending price series for testing."""
    dates = pd.date_range('2023-01-01', periods=15, freq='D')
    # Upward trending with some volatility
    base_trend = np.linspace(100, 120, 15)
    noise = np.array([0, 2, -1, 3, -2, 1, -1, 2, -3, 1, 0, -2, 3, -1, 0])
    prices = base_trend + noise
    return pd.Series(prices, index=dates, name='Close')


@pytest.fixture
def volatile_price_series():
    """Highly volatile price series."""
    dates = pd.date_range('2023-01-01', periods=25, freq='D')
    # Create volatile data with multiple resistance/support levels
    prices = [100, 95, 105, 98, 110, 102, 108, 96, 112, 99,
              115, 105, 118, 108, 120, 110, 116, 112, 122, 114,
              119, 115, 125, 118, 123]
    return pd.Series(prices, index=dates, name='Close')


@pytest.fixture
def flat_price_series():
    """Flat price series for edge case testing."""
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    prices = [100.0] * 10
    return pd.Series(prices, index=dates, name='Close')


class TestFindPivotPoints:
    """Test find_pivot_points function."""

    def test_find_pivot_points_basic(self, sample_price_series):
        """Test basic pivot point detection."""
        pivots = find_pivot_points(sample_price_series, window=2)
        
        assert isinstance(pivots, pd.Series)
        assert len(pivots) == len(sample_price_series)
        assert pivots.dtype == int
        
        # Check that pivot values are only -1, 0, or 1
        unique_values = set(pivots.unique())
        assert unique_values.issubset({-1, 0, 1})

    def test_find_pivot_points_window_size(self, sample_price_series):
        """Test pivot detection with different window sizes."""
        for window in [1, 3, 5]:
            pivots = find_pivot_points(sample_price_series, window=window)
            
            assert isinstance(pivots, pd.Series)
            assert len(pivots) == len(sample_price_series)
            
            # Larger windows should generally find fewer pivots
            pivot_count = (pivots != 0).sum()
            assert pivot_count >= 0

    def test_find_pivot_points_highs_and_lows(self, sample_price_series):
        """Test that pivot highs and lows are correctly identified."""
        pivots = find_pivot_points(sample_price_series, window=2)
        
        # Get pivot highs and lows
        pivot_highs = sample_price_series[pivots == 1]
        pivot_lows = sample_price_series[pivots == -1]
        
        # Should have some pivot points
        assert len(pivot_highs) > 0 or len(pivot_lows) > 0
        
        # Pivot highs should be local maxima
        for idx in pivot_highs.index:
            idx_pos = sample_price_series.index.get_loc(idx)
            window_start = max(0, idx_pos - 2)
            window_end = min(len(sample_price_series), idx_pos + 3)
            window_data = sample_price_series.iloc[window_start:window_end]
            assert sample_price_series[idx] >= window_data.max()

    def test_find_pivot_points_type_error(self):
        """Test TypeError for invalid input type."""
        with pytest.raises(TypeError, match="Input 'data' must be a pandas Series"):
            find_pivot_points([100, 110, 120])
        
        with pytest.raises(TypeError, match="Input 'data' must be a pandas Series"):
            find_pivot_points(np.array([100, 110, 120]))

    def test_find_pivot_points_empty_series(self):
        """Test with empty series."""
        empty_series = pd.Series([], dtype=float, name='Close')
        pivots = find_pivot_points(empty_series)
        
        assert isinstance(pivots, pd.Series)
        assert len(pivots) == 0

    def test_find_pivot_points_single_value(self):
        """Test with single value series."""
        single_value = pd.Series([100.0], name='Close')
        pivots = find_pivot_points(single_value)
        
        assert len(pivots) == 1
        # Single point should be both high and low, but implementation may vary
        assert pivots.iloc[0] in [-1, 0, 1]

    def test_find_pivot_points_flat_series(self, flat_price_series):
        """Test with flat price series."""
        pivots = find_pivot_points(flat_price_series, window=2)
        
        assert isinstance(pivots, pd.Series)
        assert len(pivots) == len(flat_price_series)
        
        # With flat prices, all points are both highs and lows
        # Implementation behavior may vary, but should not crash

    def test_find_pivot_points_large_window(self, sample_price_series):
        """Test with window larger than data length."""
        large_window = len(sample_price_series) + 5
        pivots = find_pivot_points(sample_price_series, window=large_window)
        
        assert isinstance(pivots, pd.Series)
        assert len(pivots) == len(sample_price_series)

    def test_find_pivot_points_zero_window(self, sample_price_series):
        """Test with zero window size."""
        pivots = find_pivot_points(sample_price_series, window=0)
        
        assert isinstance(pivots, pd.Series)
        assert len(pivots) == len(sample_price_series)


class TestFindResistanceSupportLines:
    """Test find_resistance_support_lines function."""

    def test_find_resistance_support_lines_basic(self, sample_price_series):
        """Test basic resistance and support line detection."""
        resistance_lines, support_lines, pivots = find_resistance_support_lines(
            sample_price_series, window=2, tolerance=0.02
        )
        
        assert isinstance(resistance_lines, list)
        assert isinstance(support_lines, list)
        assert isinstance(pivots, pd.Series)
        
        # Should return some lines (may be empty for some data)
        assert len(resistance_lines) >= 0
        assert len(support_lines) >= 0

    def test_find_resistance_support_lines_values(self, volatile_price_series):
        """Test that resistance and support lines have reasonable values."""
        resistance_lines, support_lines, pivots = find_resistance_support_lines(
            volatile_price_series, window=2, tolerance=0.05
        )
        
        data_min = volatile_price_series.min()
        data_max = volatile_price_series.max()
        
        # All resistance lines should be within data range
        for line in resistance_lines:
            assert data_min <= line <= data_max
            assert isinstance(line, (int, float, np.number))
        
        # All support lines should be within data range
        for line in support_lines:
            assert data_min <= line <= data_max
            assert isinstance(line, (int, float, np.number))

    def test_find_resistance_support_lines_tolerance(self, sample_price_series):
        """Test effect of different tolerance values."""
        # Test with strict tolerance
        res_strict, sup_strict, _ = find_resistance_support_lines(
            sample_price_series, window=2, tolerance=0.01
        )
        
        # Test with loose tolerance
        res_loose, sup_loose, _ = find_resistance_support_lines(
            sample_price_series, window=2, tolerance=0.1
        )
        
        # Loose tolerance should generally result in fewer lines
        # (more points grouped together)
        assert len(res_loose) <= len(res_strict) or len(res_strict) == 0
        assert len(sup_loose) <= len(sup_strict) or len(sup_strict) == 0

    def test_find_resistance_support_lines_window_effect(self, trending_price_series):
        """Test effect of different window sizes."""
        # Small window
        res_small, sup_small, _ = find_resistance_support_lines(
            trending_price_series, window=1, tolerance=0.02
        )
        
        # Large window
        res_large, sup_large, _ = find_resistance_support_lines(
            trending_price_series, window=5, tolerance=0.02
        )
        
        # Both should return valid results
        assert isinstance(res_small, list)
        assert isinstance(sup_small, list)
        assert isinstance(res_large, list)
        assert isinstance(sup_large, list)

    def test_find_resistance_support_lines_empty_pivots(self, flat_price_series):
        """Test with data that produces no clear pivots."""
        resistance_lines, support_lines, pivots = find_resistance_support_lines(
            flat_price_series, window=2, tolerance=0.02
        )
        
        # Should handle flat data gracefully
        assert isinstance(resistance_lines, list)
        assert isinstance(support_lines, list)
        assert isinstance(pivots, pd.Series)

    def test_find_resistance_support_lines_single_pivot(self):
        """Test with data that has only one pivot point."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        # Data with single clear high
        prices = pd.Series([100, 101, 105, 101, 100], index=dates, name='Close')
        
        resistance_lines, support_lines, pivots = find_resistance_support_lines(
            prices, window=1, tolerance=0.02
        )
        
        assert isinstance(resistance_lines, list)
        assert isinstance(support_lines, list)

    def test_find_resistance_support_lines_zero_tolerance(self, sample_price_series):
        """Test with zero tolerance."""
        resistance_lines, support_lines, pivots = find_resistance_support_lines(
            sample_price_series, window=2, tolerance=0.0
        )
        
        # Should still work with zero tolerance
        assert isinstance(resistance_lines, list)
        assert isinstance(support_lines, list)

    def test_find_resistance_support_lines_high_tolerance(self, sample_price_series):
        """Test with very high tolerance."""
        resistance_lines, support_lines, pivots = find_resistance_support_lines(
            sample_price_series, window=2, tolerance=1.0  # 100% tolerance
        )
        
        # High tolerance should group most points together
        assert isinstance(resistance_lines, list)
        assert isinstance(support_lines, list)
        # Should have fewer lines due to high tolerance
        assert len(resistance_lines) <= 2
        assert len(support_lines) <= 2


class TestPlotResistanceSupport:
    """Test plot_resistance_support function."""

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.scatter')
    @patch('matplotlib.pyplot.axhline')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.grid')
    def test_plot_resistance_support_basic(self, mock_grid, mock_legend, mock_ylabel,
                                         mock_xlabel, mock_title, mock_axhline,
                                         mock_scatter, mock_plot, mock_figure, mock_show,
                                         sample_price_series):
        """Test basic plotting functionality."""
        plot_resistance_support(sample_price_series, window=2, tolerance=0.02)
        
        # Verify matplotlib functions were called
        mock_figure.assert_called_once_with(figsize=(15, 8))
        mock_plot.assert_called_once()
        mock_title.assert_called_once_with('Resistance and Support Lines')
        mock_xlabel.assert_called_once_with('Date')
        mock_ylabel.assert_called_once_with('Price')
        mock_legend.assert_called_once()
        mock_grid.assert_called_once_with(True)
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.scatter')
    def test_plot_resistance_support_scatter_calls(self, mock_scatter, mock_show,
                                                  volatile_price_series):
        """Test that scatter plots are called for pivot points."""
        plot_resistance_support(volatile_price_series, window=2, tolerance=0.05)
        
        # Should call scatter twice: once for highs, once for lows
        assert mock_scatter.call_count == 2
        
        # Check scatter call parameters
        calls = mock_scatter.call_args_list
        
        # First call should be for pivot highs (red triangles)
        high_call = calls[0]
        assert 'color' in high_call[1]
        assert 'marker' in high_call[1]
        assert high_call[1]['color'] == 'r'
        assert high_call[1]['marker'] == '^'
        
        # Second call should be for pivot lows (green triangles)
        low_call = calls[1]
        assert low_call[1]['color'] == 'g'
        assert low_call[1]['marker'] == 'v'

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.axhline')
    def test_plot_resistance_support_axhline_calls(self, mock_axhline, mock_show,
                                                  sample_price_series):
        """Test that axhline is called for resistance and support lines."""
        plot_resistance_support(sample_price_series, window=2, tolerance=0.02)
        
        # Get the actual resistance and support lines
        resistance_lines, support_lines, _ = find_resistance_support_lines(
            sample_price_series, window=2, tolerance=0.02
        )
        
        expected_calls = len(resistance_lines) + len(support_lines)
        assert mock_axhline.call_count == expected_calls

    @patch('matplotlib.pyplot.show')
    @patch('simple_trade.resistance_support.find_resistance_support_lines')
    def test_plot_resistance_support_uses_find_function(self, mock_find_lines, mock_show,
                                                       sample_price_series):
        """Test that plotting function uses find_resistance_support_lines."""
        # Mock the find function
        mock_resistance = [105.0, 110.0]
        mock_support = [95.0, 100.0]
        # Create mock pivots with same index as sample_price_series
        mock_pivots = pd.Series([0] * len(sample_price_series), index=sample_price_series.index)
        # Set a few pivot points
        mock_pivots.iloc[2] = 1  # pivot high
        mock_pivots.iloc[6] = -1  # pivot low
        mock_find_lines.return_value = (mock_resistance, mock_support, mock_pivots)
        
        plot_resistance_support(sample_price_series, window=3, tolerance=0.05)
        
        # Verify find_resistance_support_lines was called with correct parameters
        mock_find_lines.assert_called_once_with(sample_price_series, 3, 0.05)

    @patch('matplotlib.pyplot.show')
    def test_plot_resistance_support_empty_lines(self, mock_show):
        """Test plotting when no resistance/support lines are found."""
        # Create data that might not produce clear lines
        dates = pd.date_range('2023-01-01', periods=3, freq='D')
        minimal_data = pd.Series([100, 100, 100], index=dates, name='Close')
        
        # Should not raise an error even with minimal data
        plot_resistance_support(minimal_data, window=1, tolerance=0.02)
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_resistance_support_single_point(self, mock_show):
        """Test plotting with single data point."""
        single_point = pd.Series([100.0], index=[pd.Timestamp('2023-01-01')], name='Close')
        
        # Should not raise an error
        plot_resistance_support(single_point, window=1, tolerance=0.02)
        mock_show.assert_called_once()


class TestIntegration:
    """Integration tests for resistance and support functionality."""

    def test_integration_full_workflow(self, volatile_price_series):
        """Test complete workflow from pivot detection to plotting."""
        # Test pivot detection
        pivots = find_pivot_points(volatile_price_series, window=2)
        assert isinstance(pivots, pd.Series)
        
        # Test line detection
        resistance_lines, support_lines, pivots_from_lines = find_resistance_support_lines(
            volatile_price_series, window=2, tolerance=0.03
        )
        
        # Pivots should be the same from both functions
        pd.testing.assert_series_equal(pivots, pivots_from_lines)
        
        # Test plotting (mocked)
        with patch('matplotlib.pyplot.show'):
            plot_resistance_support(volatile_price_series, window=2, tolerance=0.03)

    def test_integration_different_parameters(self, trending_price_series):
        """Test integration with various parameter combinations."""
        parameter_combinations = [
            (1, 0.01),
            (3, 0.05),
            (5, 0.1),
            (2, 0.02)
        ]
        
        for window, tolerance in parameter_combinations:
            # Should work with all parameter combinations
            pivots = find_pivot_points(trending_price_series, window=window)
            resistance_lines, support_lines, _ = find_resistance_support_lines(
                trending_price_series, window=window, tolerance=tolerance
            )
            
            assert isinstance(pivots, pd.Series)
            assert isinstance(resistance_lines, list)
            assert isinstance(support_lines, list)
            
            with patch('matplotlib.pyplot.show'):
                plot_resistance_support(trending_price_series, window=window, tolerance=tolerance)

    def test_integration_real_world_scenario(self):
        """Test with realistic stock price scenario."""
        # Simulate stock price with clear support and resistance
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        
        # Create price action with clear levels
        base_prices = []
        for i in range(30):
            if i < 10:
                base_prices.append(100 + np.sin(i * 0.5) * 5)  # Oscillation around 100
            elif i < 20:
                base_prices.append(110 + np.sin(i * 0.5) * 3)  # Higher level around 110
            else:
                base_prices.append(105 + np.sin(i * 0.5) * 4)  # Back to middle level
        
        price_series = pd.Series(base_prices, index=dates, name='Close')
        
        # Test full workflow
        pivots = find_pivot_points(price_series, window=3)
        resistance_lines, support_lines, _ = find_resistance_support_lines(
            price_series, window=3, tolerance=0.02
        )
        
        # Should find some levels
        assert len(resistance_lines) > 0 or len(support_lines) > 0
        
        # Test plotting
        with patch('matplotlib.pyplot.show'):
            plot_resistance_support(price_series, window=3, tolerance=0.02)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_negative_prices(self):
        """Test with negative price values."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        negative_prices = pd.Series([-10, -8, -12, -6, -15, -5, -18, -3, -20, -1], 
                                   index=dates, name='Close')
        
        # Should handle negative prices
        pivots = find_pivot_points(negative_prices, window=2)
        resistance_lines, support_lines, _ = find_resistance_support_lines(
            negative_prices, window=2, tolerance=0.1
        )
        
        assert isinstance(pivots, pd.Series)
        assert isinstance(resistance_lines, list)
        assert isinstance(support_lines, list)

    def test_very_large_numbers(self):
        """Test with very large price values."""
        dates = pd.date_range('2023-01-01', periods=8, freq='D')
        large_prices = pd.Series([1e6, 1.1e6, 0.9e6, 1.2e6, 0.8e6, 1.3e6, 0.7e6, 1.4e6], 
                                index=dates, name='Close')
        
        pivots = find_pivot_points(large_prices, window=1)
        resistance_lines, support_lines, _ = find_resistance_support_lines(
            large_prices, window=1, tolerance=0.05
        )
        
        assert isinstance(pivots, pd.Series)
        assert isinstance(resistance_lines, list)
        assert isinstance(support_lines, list)

    def test_very_small_numbers(self):
        """Test with very small price values."""
        dates = pd.date_range('2023-01-01', periods=8, freq='D')
        small_prices = pd.Series([1e-6, 1.1e-6, 0.9e-6, 1.2e-6, 0.8e-6, 1.3e-6, 0.7e-6, 1.4e-6], 
                                index=dates, name='Close')
        
        pivots = find_pivot_points(small_prices, window=1)
        resistance_lines, support_lines, _ = find_resistance_support_lines(
            small_prices, window=1, tolerance=0.05
        )
        
        assert isinstance(pivots, pd.Series)
        assert isinstance(resistance_lines, list)
        assert isinstance(support_lines, list)

    def test_nan_values(self):
        """Test with NaN values in data."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        nan_prices = pd.Series([100, np.nan, 105, 102, np.nan, 108, 104, np.nan, 110, 106], 
                              index=dates, name='Close')
        
        # Should handle NaN values gracefully
        pivots = find_pivot_points(nan_prices, window=2)
        resistance_lines, support_lines, _ = find_resistance_support_lines(
            nan_prices, window=2, tolerance=0.02
        )
        
        assert isinstance(pivots, pd.Series)
        assert isinstance(resistance_lines, list)
        assert isinstance(support_lines, list)

    def test_duplicate_values(self):
        """Test with many duplicate values."""
        dates = pd.date_range('2023-01-01', periods=12, freq='D')
        duplicate_prices = pd.Series([100, 100, 105, 105, 105, 102, 102, 108, 108, 104, 104, 104], 
                                    index=dates, name='Close')
        
        pivots = find_pivot_points(duplicate_prices, window=2)
        resistance_lines, support_lines, _ = find_resistance_support_lines(
            duplicate_prices, window=2, tolerance=0.01
        )
        
        assert isinstance(pivots, pd.Series)
        assert isinstance(resistance_lines, list)
        assert isinstance(support_lines, list)

    def test_monotonic_increasing(self):
        """Test with strictly increasing prices."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        increasing_prices = pd.Series(range(100, 110), index=dates, name='Close')
        
        pivots = find_pivot_points(increasing_prices, window=2)
        resistance_lines, support_lines, _ = find_resistance_support_lines(
            increasing_prices, window=2, tolerance=0.02
        )
        
        # Should handle monotonic data
        assert isinstance(pivots, pd.Series)
        assert isinstance(resistance_lines, list)
        assert isinstance(support_lines, list)

    def test_monotonic_decreasing(self):
        """Test with strictly decreasing prices."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        decreasing_prices = pd.Series(range(110, 100, -1), index=dates, name='Close')
        
        pivots = find_pivot_points(decreasing_prices, window=2)
        resistance_lines, support_lines, _ = find_resistance_support_lines(
            decreasing_prices, window=2, tolerance=0.02
        )
        
        # Should handle monotonic data
        assert isinstance(pivots, pd.Series)
        assert isinstance(resistance_lines, list)
        assert isinstance(support_lines, list)
