import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from simple_trade.compute_fibonacci_retracement import calculate_fibonacci_levels, plot_fibonacci_retracement


@pytest.fixture
def sample_price_series():
    """Sample price series for testing."""
    dates = pd.date_range('2023-01-01', periods=20, freq='D')
    # Create a trending price series from 100 to 120
    prices = np.linspace(100, 120, 20)
    return pd.Series(prices, index=dates, name='Close')


@pytest.fixture
def volatile_price_series():
    """Volatile price series for testing."""
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    np.random.seed(42)
    # Create volatile price data with clear high and low
    base_prices = [100, 105, 110, 115, 120, 118, 115, 110, 105, 100,
                   95, 90, 85, 80, 75, 78, 82, 87, 92, 97,
                   102, 107, 112, 117, 122, 119, 114, 109, 104, 99]
    return pd.Series(base_prices, index=dates, name='Close')


@pytest.fixture
def flat_price_series():
    """Flat price series for testing edge cases."""
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    prices = [100.0] * 10  # All same price
    return pd.Series(prices, index=dates, name='Close')


class TestCalculateFibonacciLevels:
    """Test calculate_fibonacci_levels function."""

    def test_calculate_fibonacci_levels_basic(self, sample_price_series):
        """Test basic fibonacci level calculation."""
        levels = calculate_fibonacci_levels(sample_price_series)
        
        # Check that all expected levels are present
        expected_levels = ['level_0', 'level_236', 'level_382', 'level_500', 
                          'level_618', 'level_786', 'level_100']
        
        assert isinstance(levels, dict)
        for level in expected_levels:
            assert level in levels
            assert isinstance(levels[level], (int, float))

    def test_calculate_fibonacci_levels_values(self, sample_price_series):
        """Test that fibonacci levels are calculated correctly."""
        levels = calculate_fibonacci_levels(sample_price_series)
        
        high = sample_price_series.max()
        low = sample_price_series.min()
        price_range = high - low
        
        # Verify level calculations
        assert levels['level_0'] == high
        assert levels['level_100'] == low
        assert abs(levels['level_236'] - (high - 0.236 * price_range)) < 1e-10
        assert abs(levels['level_382'] - (high - 0.382 * price_range)) < 1e-10
        assert abs(levels['level_500'] - (high - 0.5 * price_range)) < 1e-10
        assert abs(levels['level_618'] - (high - 0.618 * price_range)) < 1e-10
        assert abs(levels['level_786'] - (high - 0.786 * price_range)) < 1e-10

    def test_calculate_fibonacci_levels_ordering(self, volatile_price_series):
        """Test that fibonacci levels are properly ordered."""
        levels = calculate_fibonacci_levels(volatile_price_series)
        
        # Levels should be in descending order
        level_values = [
            levels['level_0'],
            levels['level_236'],
            levels['level_382'],
            levels['level_500'],
            levels['level_618'],
            levels['level_786'],
            levels['level_100']
        ]
        
        # Check that levels are in descending order
        for i in range(len(level_values) - 1):
            assert level_values[i] >= level_values[i + 1]

    def test_calculate_fibonacci_levels_flat_prices(self, flat_price_series):
        """Test fibonacci levels with flat price data (no range)."""
        levels = calculate_fibonacci_levels(flat_price_series)
        
        # All levels should be the same when there's no price range
        expected_price = flat_price_series.iloc[0]
        for level_name, level_value in levels.items():
            assert level_value == expected_price

    def test_calculate_fibonacci_levels_single_value(self):
        """Test fibonacci levels with single price point."""
        single_price = pd.Series([100.0], name='Close')
        levels = calculate_fibonacci_levels(single_price)
        
        # All levels should equal the single price
        for level_name, level_value in levels.items():
            assert level_value == 100.0

    def test_calculate_fibonacci_levels_negative_prices(self):
        """Test fibonacci levels with negative prices."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        negative_prices = pd.Series([-10, -5, 0, 5, 10], index=dates, name='Close')
        
        levels = calculate_fibonacci_levels(negative_prices)
        
        high = 10
        low = -10
        price_range = 20
        
        assert levels['level_0'] == high
        assert levels['level_100'] == low
        assert abs(levels['level_500'] - 0.0) < 1e-10  # Should be exactly 0

    def test_calculate_fibonacci_levels_type_error(self):
        """Test that TypeError is raised for non-Series input."""
        with pytest.raises(TypeError, match="Input 'data' must be a pandas Series"):
            calculate_fibonacci_levels([100, 110, 120])
        
        with pytest.raises(TypeError, match="Input 'data' must be a pandas Series"):
            calculate_fibonacci_levels(np.array([100, 110, 120]))
        
        with pytest.raises(TypeError, match="Input 'data' must be a pandas Series"):
            calculate_fibonacci_levels(pd.DataFrame({'Close': [100, 110, 120]}))

    def test_calculate_fibonacci_levels_empty_series(self):
        """Test fibonacci levels with empty series."""
        empty_series = pd.Series([], dtype=float, name='Close')
        
        # Empty series should work but return NaN values
        levels = calculate_fibonacci_levels(empty_series)
        
        # All levels should be NaN for empty series
        for level_name, level_value in levels.items():
            assert np.isnan(level_value)

    def test_calculate_fibonacci_levels_nan_values(self):
        """Test fibonacci levels with NaN values."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        nan_prices = pd.Series([100, np.nan, 110, 120, np.nan], index=dates, name='Close')
        
        levels = calculate_fibonacci_levels(nan_prices)
        
        # Should work with NaN values (pandas handles them in min/max)
        assert not np.isnan(levels['level_0'])
        assert not np.isnan(levels['level_100'])

    def test_calculate_fibonacci_levels_precision(self):
        """Test fibonacci level calculation precision."""
        dates = pd.date_range('2023-01-01', periods=3, freq='D')
        precise_prices = pd.Series([100.123456789, 105.987654321, 110.555555555], 
                                 index=dates, name='Close')
        
        levels = calculate_fibonacci_levels(precise_prices)
        
        # Check that precision is maintained
        high = precise_prices.max()
        low = precise_prices.min()
        price_range = high - low
        
        expected_236 = high - 0.236 * price_range
        assert abs(levels['level_236'] - expected_236) < 1e-10


class TestPlotFibonacciRetracement:
    """Test plot_fibonacci_retracement function."""

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.axhline')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.grid')
    def test_plot_fibonacci_retracement_basic(self, mock_grid, mock_legend, mock_xlabel, 
                                            mock_ylabel, mock_title, mock_axhline, 
                                            mock_plot, mock_figure, mock_show, 
                                            sample_price_series):
        """Test basic plotting functionality."""
        plot_fibonacci_retracement(sample_price_series)
        
        # Verify that matplotlib functions were called
        mock_figure.assert_called_once_with(figsize=(14, 7))
        mock_plot.assert_called_once()
        mock_title.assert_called_once_with('Fibonacci Retracement')
        mock_ylabel.assert_called_once_with('Price')
        mock_xlabel.assert_called_once_with('Date')
        mock_legend.assert_called_once()
        mock_grid.assert_called_once_with(True)
        mock_show.assert_called_once()
        
        # Should call axhline for each fibonacci level (7 levels)
        assert mock_axhline.call_count == 7

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.axhline')
    def test_plot_fibonacci_retracement_axhline_calls(self, mock_axhline, mock_plot, 
                                                    mock_figure, mock_show, 
                                                    volatile_price_series):
        """Test that axhline is called with correct parameters for each level."""
        plot_fibonacci_retracement(volatile_price_series)
        
        # Get the fibonacci levels to verify axhline calls
        levels = calculate_fibonacci_levels(volatile_price_series)
        
        # Verify axhline was called for each level
        assert mock_axhline.call_count == len(levels)
        
        # Check that axhline was called with the correct y values
        called_y_values = [call[1]['y'] for call in mock_axhline.call_args_list]
        expected_y_values = list(levels.values())
        
        for expected_y in expected_y_values:
            assert expected_y in called_y_values

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    def test_plot_fibonacci_retracement_plot_call(self, mock_plot, mock_figure, 
                                                mock_show, sample_price_series):
        """Test that plot is called with correct data."""
        plot_fibonacci_retracement(sample_price_series)
        
        # Verify plot was called with the series index and values
        mock_plot.assert_called_once_with(sample_price_series.index, 
                                        sample_price_series, label='Price')

    @patch('matplotlib.pyplot.show')
    @patch('simple_trade.compute_fibonacci_retracement.calculate_fibonacci_levels')
    def test_plot_fibonacci_retracement_uses_calculate_function(self, mock_calc_levels, 
                                                              mock_show, sample_price_series):
        """Test that plotting function uses calculate_fibonacci_levels."""
        # Mock the calculate function to return known values
        mock_levels = {
            'level_0': 120.0,
            'level_236': 115.28,
            'level_382': 112.36,
            'level_500': 110.0,
            'level_618': 107.64,
            'level_786': 104.28,
            'level_100': 100.0
        }
        mock_calc_levels.return_value = mock_levels
        
        plot_fibonacci_retracement(sample_price_series)
        
        # Verify calculate_fibonacci_levels was called with the input data
        mock_calc_levels.assert_called_once_with(sample_price_series)

    @patch('matplotlib.pyplot.show')
    def test_plot_fibonacci_retracement_flat_prices(self, mock_show, flat_price_series):
        """Test plotting with flat price data."""
        # Should not raise an error even with flat prices
        plot_fibonacci_retracement(flat_price_series)
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_fibonacci_retracement_single_point(self, mock_show):
        """Test plotting with single data point."""
        single_point = pd.Series([100.0], index=[pd.Timestamp('2023-01-01')], name='Close')
        
        # Should not raise an error even with single point
        plot_fibonacci_retracement(single_point)
        mock_show.assert_called_once()


class TestIntegration:
    """Integration tests for fibonacci retracement functionality."""

    def test_integration_calculate_and_plot(self, sample_price_series):
        """Test integration between calculate and plot functions."""
        # Calculate levels manually
        levels = calculate_fibonacci_levels(sample_price_series)
        
        # Verify levels are reasonable
        high = sample_price_series.max()
        low = sample_price_series.min()
        
        assert levels['level_0'] == high
        assert levels['level_100'] == low
        assert low <= levels['level_500'] <= high
        
        # Test that plotting doesn't raise errors
        with patch('matplotlib.pyplot.show'):
            plot_fibonacci_retracement(sample_price_series)

    def test_integration_real_world_scenario(self):
        """Test with realistic stock price scenario."""
        # Simulate a stock that goes up then retraces
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        
        # Uptrend followed by retracement
        uptrend = np.linspace(100, 150, 25)
        retracement = np.linspace(150, 125, 25)
        prices = np.concatenate([uptrend, retracement])
        
        price_series = pd.Series(prices, index=dates, name='Close')
        
        levels = calculate_fibonacci_levels(price_series)
        
        # Verify levels make sense for this scenario
        assert levels['level_0'] == 150.0  # High
        assert levels['level_100'] == 100.0  # Low
        assert 125.0 <= levels['level_500'] <= 125.0  # 50% level should be 125
        
        # Test plotting
        with patch('matplotlib.pyplot.show'):
            plot_fibonacci_retracement(price_series)

    def test_integration_multiple_series_types(self):
        """Test with different types of price series."""
        test_cases = [
            # Trending up
            pd.Series(np.linspace(50, 100, 20), name='Uptrend'),
            # Trending down  
            pd.Series(np.linspace(100, 50, 20), name='Downtrend'),
            # Volatile
            pd.Series([100, 90, 110, 85, 115, 80, 120, 75, 125], name='Volatile'),
            # Small range
            pd.Series(np.linspace(99.9, 100.1, 10), name='SmallRange')
        ]
        
        for series in test_cases:
            # Should not raise errors for any series type
            levels = calculate_fibonacci_levels(series)
            assert isinstance(levels, dict)
            assert len(levels) == 7
            
            with patch('matplotlib.pyplot.show'):
                plot_fibonacci_retracement(series)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_large_numbers(self):
        """Test with very large price values."""
        large_prices = pd.Series([1e6, 1.1e6, 1.2e6, 1.3e6, 1.4e6], name='Large')
        
        levels = calculate_fibonacci_levels(large_prices)
        
        # Should handle large numbers correctly
        assert levels['level_0'] == 1.4e6
        assert levels['level_100'] == 1e6
        assert isinstance(levels['level_500'], float)

    def test_very_small_numbers(self):
        """Test with very small price values."""
        small_prices = pd.Series([1e-6, 1.1e-6, 1.2e-6, 1.3e-6, 1.4e-6], name='Small')
        
        levels = calculate_fibonacci_levels(small_prices)
        
        # Should handle small numbers correctly
        assert levels['level_0'] == 1.4e-6
        assert levels['level_100'] == 1e-6
        assert isinstance(levels['level_500'], float)

    def test_integer_prices(self):
        """Test with integer price values."""
        int_prices = pd.Series([100, 105, 110, 115, 120], name='Integer')
        
        levels = calculate_fibonacci_levels(int_prices)
        
        # Should work with integer inputs - numpy int64 is also valid
        assert isinstance(levels['level_0'], (int, float, np.integer))
        assert isinstance(levels['level_500'], (int, float, np.integer))

    def test_mixed_data_types(self):
        """Test with mixed numeric data types."""
        mixed_prices = pd.Series([100, 105.5, 110, 115.7, 120], name='Mixed')
        
        levels = calculate_fibonacci_levels(mixed_prices)
        
        # Should handle mixed int/float correctly
        for level_value in levels.values():
            assert isinstance(level_value, (int, float))
            assert not np.isnan(level_value)

    def test_extreme_volatility(self):
        """Test with extremely volatile price data."""
        volatile_prices = pd.Series([100, 200, 50, 300, 25, 400, 10], name='Extreme')
        
        levels = calculate_fibonacci_levels(volatile_prices)
        
        # Should handle extreme volatility
        assert levels['level_0'] == 400
        assert levels['level_100'] == 10
        assert 10 <= levels['level_500'] <= 400

    def test_duplicate_high_low_values(self):
        """Test when multiple values equal the high or low."""
        duplicate_prices = pd.Series([100, 100, 110, 120, 120, 110, 100], name='Duplicates')
        
        levels = calculate_fibonacci_levels(duplicate_prices)
        
        # Should handle duplicates correctly
        assert levels['level_0'] == 120
        assert levels['level_100'] == 100
        assert 100 <= levels['level_500'] <= 120
