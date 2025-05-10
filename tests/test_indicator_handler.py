import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from simple_trade.data.indicator_handler import (
    compute_indicator,
    _calculate_indicator,
    _add_indicator_to_dataframe,
    download_data,
)

# --- Fixtures ---

@pytest.fixture
def sample_price_data():
    """Fixture providing a basic OHLCV DataFrame for testing indicators."""
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
    data = pd.DataFrame(index=dates)
    data['Open'] = [100, 102, 104, 103, 105, 107, 108, 109, 110, 111, 112, 111, 110, 112, 114, 115, 116, 115, 113, 114]
    data['High'] = [103, 105, 107, 106, 108, 110, 111, 112, 113, 114, 115, 114, 113, 115, 117, 118, 119, 118, 116, 117]
    data['Low'] = [98, 100, 102, 101, 103, 105, 106, 107, 108, 109, 110, 109, 108, 110, 112, 113, 114, 113, 111, 112]
    data['Close'] = [102, 104, 106, 105, 107, 109, 110, 111, 112, 113, 114, 113, 112, 114, 116, 117, 118, 117, 115, 116]
    data['Volume'] = [1000, 1200, 1300, 1100, 1400, 1500, 1600, 1500, 1400, 1600, 1700, 1500, 1400, 1600, 1800, 1900, 2000, 1800, 1700, 1900]
    return data

@pytest.fixture
def mock_indicators():
    """Fixture providing mocked indicator functions for predictable testing."""
    with patch('simple_trade.data.indicator_handler.INDICATORS') as mock_indicators:
        # Create mock functions for each indicator type that return predictable values
        def sma_mock(series, **kwargs):
            # Return series with same index as input but all values set to 105.0
            return pd.Series([105.0] * len(series), index=series.index)
        
        def rsi_mock(series, **kwargs):
            # Return series with same index as input but all values set to 60.0
            return pd.Series([60.0] * len(series), index=series.index)
        
        # Create mock for Bollinger Bands (returns DataFrame)
        def bollinger_mock(series, **kwargs):
            # Return DataFrame with same index as input
            return pd.DataFrame({
                'BOLLIN_UPPER': [115.0] * len(series),
                'BOLLIN_MIDDLE': [105.0] * len(series),
                'BOLLIN_LOWER': [95.0] * len(series)
            }, index=series.index)
        
        # Create mock for MACD (returns DataFrame)
        def macd_mock(series, **kwargs):
            # Return DataFrame with same index as input
            return pd.DataFrame({
                'MACD': [2.0] * len(series),
                'MACD_Signal': [1.5] * len(series),
                'MACD_Hist': [0.5] * len(series)
            }, index=series.index)
        
        # Create mock for Ichimoku (returns dict)
        def ichimoku_mock(df, **kwargs):
            # Return dict of series with same index as input
            return {
                'tenkan_sen': pd.Series([110.0] * len(df), index=df.index),
                'kijun_sen': pd.Series([105.0] * len(df), index=df.index),
                'senkou_span_a': pd.Series([108.0] * len(df), index=df.index),
                'senkou_span_b': pd.Series([102.0] * len(df), index=df.index),
                'chikou_span': pd.Series([112.0] * len(df), index=df.index)
            }
        
        # Create mock for Aroon (returns tuple of 3 series)
        def aroon_mock(df, **kwargs):
            # Return tuple of series with same index as input
            return (
                pd.Series([70.0] * len(df), index=df.index),  # aroon_up
                pd.Series([30.0] * len(df), index=df.index),  # aroon_down
                pd.Series([40.0] * len(df), index=df.index)   # aroon_oscillator
            )
        
        # Set up mock indicators
        mock_indicators.__getitem__.side_effect = lambda key: {
            'sma': sma_mock,
            'ema': sma_mock,  # reuse sma_mock for simplicity
            'rsi': rsi_mock,
            'bollin': bollinger_mock,
            'macd': macd_mock,
            'ichimoku': ichimoku_mock,
            'aroon': aroon_mock
        }.get(key, MagicMock(return_value=pd.Series([100.0] * 20)))
        
        mock_indicators.__contains__.side_effect = lambda key: key in [
            'sma', 'ema', 'rsi', 'bollin', 'macd', 'ichimoku', 'aroon',
            'strend', 'adx', 'psar', 'wma', 'hma', 'trix',
            'stoch', 'cci', 'roc',
            'atr', 'kelt', 'donch', 'chaik',
            'obv', 'vma', 'adline', 'cmf', 'vpt'
        ]
        
        yield mock_indicators

# --- Test Classes ---

class TestComputeIndicator:
    """Tests for the compute_indicator function."""
    
    def test_invalid_indicator(self, sample_price_data):
        """Test that ValueError is raised for invalid indicator name."""
        with pytest.raises(ValueError, match="Indicator 'invalid_indicator' not supported"):
            compute_indicator(sample_price_data, 'invalid_indicator')
    
    @patch('simple_trade.data.indicator_handler._calculate_indicator')
    def test_compute_simple_indicator(self, mock_calculate, sample_price_data, mock_indicators):
        """Test computation of a simple indicator like SMA."""
        # Create mock return value for _calculate_indicator
        mock_series = pd.Series([105.0] * len(sample_price_data), index=sample_price_data.index, name='SMA_10')
        mock_calculate.return_value = mock_series
        
        # Patch the necessary functions
        with patch('simple_trade.data.indicator_handler.INDICATORS', mock_indicators):
            result = compute_indicator(sample_price_data, 'sma', window=10)
            
            # Verify _calculate_indicator was called with correct args
            mock_calculate.assert_called_once()
            
            # Verify the indicator has been added
            assert 'SMA_10' in result.columns
            # Check a value from the dataframe
            assert result['SMA_10'].iloc[0] == 105.0
    
    @patch('simple_trade.data.indicator_handler._calculate_indicator')
    def test_compute_bollinger_bands(self, mock_calculate, sample_price_data, mock_indicators):
        """Test computation of Bollinger Bands (returns DataFrame)."""
        # Create mock return value for _calculate_indicator
        mock_df = pd.DataFrame({
            'BOLLIN_UPPER': [115.0] * len(sample_price_data),
            'BOLLIN_MIDDLE': [105.0] * len(sample_price_data),
            'BOLLIN_LOWER': [95.0] * len(sample_price_data)
        }, index=sample_price_data.index)
        mock_calculate.return_value = mock_df
        
        with patch('simple_trade.data.indicator_handler.INDICATORS', mock_indicators):
            result = compute_indicator(sample_price_data, 'bollin', window=20, window_dev=2)
            
            # Verify the calculation was called
            mock_calculate.assert_called_once()
            
            # Verify all bands have been added
            assert 'BOLLIN_UPPER' in result.columns
            assert 'BOLLIN_MIDDLE' in result.columns
            assert 'BOLLIN_LOWER' in result.columns
            
            # Check values
            assert result['BOLLIN_UPPER'].iloc[0] == 115.0
            assert result['BOLLIN_MIDDLE'].iloc[0] == 105.0
            assert result['BOLLIN_LOWER'].iloc[0] == 95.0
    
    @patch('simple_trade.data.indicator_handler._calculate_indicator')
    def test_compute_macd(self, mock_calculate, sample_price_data, mock_indicators):
        """Test computation of MACD (returns DataFrame with multiple components)."""
        # Create mock return value for _calculate_indicator
        mock_df = pd.DataFrame({
            'MACD': [2.0] * len(sample_price_data),
            'MACD_Signal': [1.5] * len(sample_price_data),
            'MACD_Hist': [0.5] * len(sample_price_data)
        }, index=sample_price_data.index)
        mock_calculate.return_value = mock_df
        
        with patch('simple_trade.data.indicator_handler.INDICATORS', mock_indicators):
            result = compute_indicator(sample_price_data, 'macd', fast=12, slow=26, signal=9)
            
            # Verify the calculation was called
            mock_calculate.assert_called_once()
            
            # Verify all components have been added
            assert 'MACD' in result.columns
            assert 'MACD_Signal' in result.columns
            assert 'MACD_Hist' in result.columns
            
            # Check values
            assert result['MACD'].iloc[0] == 2.0
            assert result['MACD_Signal'].iloc[0] == 1.5
            assert result['MACD_Hist'].iloc[0] == 0.5
    
    @patch('simple_trade.data.indicator_handler._calculate_indicator')
    def test_compute_ichimoku(self, mock_calculate, sample_price_data, mock_indicators):
        """Test computation of Ichimoku Cloud (returns dict of components)."""
        # Create mock return value for _calculate_indicator
        initial_columns = set(sample_price_data.columns)
        mock_dict = {
            'tenkan_sen': pd.Series([110.0] * len(sample_price_data), index=sample_price_data.index),
            'kijun_sen': pd.Series([105.0] * len(sample_price_data), index=sample_price_data.index),
            'senkou_span_a': pd.Series([108.0] * len(sample_price_data), index=sample_price_data.index),
            'senkou_span_b': pd.Series([102.0] * len(sample_price_data), index=sample_price_data.index),
            'chikou_span': pd.Series([112.0] * len(sample_price_data), index=sample_price_data.index)
        }
        mock_calculate.return_value = mock_dict
        
        with patch('simple_trade.data.indicator_handler.INDICATORS', mock_indicators):
            result = compute_indicator(sample_price_data, 'ichimoku')
            
            # Verify the calculation was called
            mock_calculate.assert_called_once()
            
            # Verify no new columns have been added as dict is not handled for column creation
            assert set(result.columns) == initial_columns
    
    @patch('simple_trade.data.indicator_handler._calculate_indicator')
    def test_compute_aroon(self, mock_calculate, sample_price_data, mock_indicators):
        """Test computation of Aroon (returns tuple of components)."""
        # Create mock return value for _calculate_indicator
        initial_columns = set(sample_price_data.columns)
        mock_tuple = (
            pd.Series([70.0] * len(sample_price_data), index=sample_price_data.index),  # aroon_up
            pd.Series([30.0] * len(sample_price_data), index=sample_price_data.index),  # aroon_down
            pd.Series([40.0] * len(sample_price_data), index=sample_price_data.index)   # aroon_oscillator
        )
        mock_calculate.return_value = mock_tuple
        
        with patch('simple_trade.data.indicator_handler.INDICATORS', mock_indicators):
            result = compute_indicator(sample_price_data, 'aroon', window=14)
            
            # Verify the calculation was called
            mock_calculate.assert_called_once()
            
            # Verify no new columns have been added as tuple is not handled for column creation
            assert set(result.columns) == initial_columns
    
    def test_missing_column(self, sample_price_data, mock_indicators):
        """Test error handling when required column is missing."""
        with patch('simple_trade.data.indicator_handler.INDICATORS', mock_indicators):
            # Remove Close column
            data_no_close = sample_price_data.drop(columns=['Close'])
            
            # Test _calculate_indicator directly using an indicator that validates for 'Close' column
            with pytest.raises(ValueError, match="DataFrame must contain 'Close' column."):
                _calculate_indicator(data_no_close, 'rsi', mock_indicators['rsi'], window=14)

class TestCalculateIndicator:
    """Tests for the _calculate_indicator function."""
    
    def test_trend_indicators(self, sample_price_data, mock_indicators):
        """Test calculation of trend indicators."""
        with patch('simple_trade.data.indicator_handler.INDICATORS', mock_indicators):
            # Test SMA (simple trend indicator)
            result = _calculate_indicator(sample_price_data, 'sma', mock_indicators['sma'], window=10)
            assert isinstance(result, pd.Series)
            assert result.iloc[0] == 105.0
    
    def test_momentum_indicators_calculation(self, sample_price_data, mock_indicators):
        """Test that _calculate_indicator correctly handles momentum indicators (lines 101-126)."""
        with patch('simple_trade.data.indicator_handler.INDICATORS', mock_indicators):
            # Specifically targeting lines 101-126 in indicator_handler.py
            
            # --- Test Stochastic ---
            # Mock the handle_stochastic function for direct _calculate_indicator testing
            with patch('simple_trade.data.indicator_handler.handle_stochastic') as mock_stoch:
                mock_stoch.return_value = pd.Series([80.0] * len(sample_price_data), index=sample_price_data.index)
                result = _calculate_indicator(sample_price_data, 'stoch', mock_indicators['stoch'], window=14)
                # Verify the mock was called correctly
                mock_stoch.assert_called_once()
                # Verify correct return value
                assert isinstance(result, pd.Series)
                assert result.iloc[0] == 80.0
            
            # --- Test CCI ---
            with patch('simple_trade.data.indicator_handler.handle_cci') as mock_cci:
                mock_cci.return_value = pd.Series([120.0] * len(sample_price_data), index=sample_price_data.index)
                result = _calculate_indicator(sample_price_data, 'cci', mock_indicators['cci'], window=20)
                # Verify the mock was called correctly
                mock_cci.assert_called_once()
                # Verify correct return value
                assert isinstance(result, pd.Series)
                assert result.iloc[0] == 120.0
            
            # --- Test ROC ---
            with patch('simple_trade.data.indicator_handler.handle_roc') as mock_roc:
                mock_roc.return_value = pd.Series([5.0] * len(sample_price_data), index=sample_price_data.index)
                result = _calculate_indicator(sample_price_data, 'roc', mock_indicators['roc'], window=10)
                # Verify the mock was called correctly
                mock_roc.assert_called_once()
                # Verify correct return value
                assert isinstance(result, pd.Series)
                assert result.iloc[0] == 5.0
            
            # --- Test MACD ---
            with patch('simple_trade.data.indicator_handler.handle_macd') as mock_macd:
                mock_macd_df = pd.DataFrame({
                    'MACD': [2.0] * len(sample_price_data),
                    'MACD_Signal': [1.5] * len(sample_price_data),
                    'MACD_Hist': [0.5] * len(sample_price_data)
                }, index=sample_price_data.index)
                mock_macd.return_value = mock_macd_df
                
                # Call the function we're testing
                result = _calculate_indicator(sample_price_data, 'macd', mock_indicators['macd'], fast=12, slow=26, signal=9)
                
                # Verify mock was called
                mock_macd.assert_called_once()
                
                # Verify result
                assert isinstance(result, pd.DataFrame)
                assert 'MACD' in result.columns
                assert result['MACD'].iloc[0] == 2.0
            
            # --- Test RSI ---
            with patch('simple_trade.data.indicator_handler.handle_rsi') as mock_rsi:
                mock_rsi.return_value = pd.Series([60.0] * len(sample_price_data), index=sample_price_data.index)
                
                # Call the function we're testing
                result = _calculate_indicator(sample_price_data, 'rsi', mock_indicators['rsi'], window=14)
                
                # Verify mock was called
                mock_rsi.assert_called_once()
                
                # Verify result
                assert isinstance(result, pd.Series)
                assert result.iloc[0] == 60.0
    
    def test_momentum_indicators_error_handling(self, sample_price_data, mock_indicators):
        """Test error handling for momentum indicators when required columns are missing."""
        # Remove required columns
        df_missing_high_low = sample_price_data.drop(columns=['High', 'Low'])
        
        # Test stochastic oscillator requires High, Low, Close
        with patch('simple_trade.data.indicator_handler.INDICATORS', mock_indicators):
            with pytest.raises(ValueError, match="DataFrame must contain"):
                _calculate_indicator(df_missing_high_low, 'stoch', mock_indicators['stoch'])
        
        # Test CCI requires High, Low, Close
        with patch('simple_trade.data.indicator_handler.INDICATORS', mock_indicators):
            with pytest.raises(ValueError, match="DataFrame must contain"):
                _calculate_indicator(df_missing_high_low, 'cci', mock_indicators['cci'])
        
        # Test that ROC, MACD, RSI require Close only
        df_close_only = pd.DataFrame({'Close': sample_price_data['Close']}, index=sample_price_data.index)
        
        # These should not raise errors
        with patch('simple_trade.data.indicator_handler.INDICATORS', mock_indicators):
            with patch('simple_trade.data.indicator_handler.handle_roc') as mock_roc:
                mock_roc.return_value = pd.Series([5.0] * len(df_close_only), index=df_close_only.index)
                _calculate_indicator(df_close_only, 'roc', mock_indicators['roc'])
                mock_roc.assert_called_once()
            
            with patch('simple_trade.data.indicator_handler.handle_macd') as mock_macd:
                mock_macd.return_value = pd.DataFrame({'MACD': [2.0] * len(df_close_only)}, index=df_close_only.index)
                _calculate_indicator(df_close_only, 'macd', mock_indicators['macd'])
                mock_macd.assert_called_once()
            
            with patch('simple_trade.data.indicator_handler.handle_rsi') as mock_rsi:
                mock_rsi.return_value = pd.Series([60.0] * len(df_close_only), index=df_close_only.index)
                _calculate_indicator(df_close_only, 'rsi', mock_indicators['rsi'])
                mock_rsi.assert_called_once()

    def test_direct_momentum_indicators(self, sample_price_data):
        """Test _calculate_indicator with direct calls to handler functions to cover lines 101-126."""
        # Mock INDICATORS dictionary with real functions
        from simple_trade.data.momentum_handlers import (
            handle_stochastic, handle_cci, handle_roc, handle_macd, handle_rsi
        )
        
        mock_indicators = {
            'stoch': lambda *args, **kwargs: pd.Series([80.0] * len(sample_price_data), index=sample_price_data.index),
            'cci': lambda *args, **kwargs: pd.Series([120.0] * len(sample_price_data), index=sample_price_data.index),
            'roc': lambda *args, **kwargs: pd.Series([5.0] * len(sample_price_data), index=sample_price_data.index),
            'macd': lambda *args, **kwargs: pd.DataFrame({
                'MACD': [2.0] * len(sample_price_data),
                'MACD_Signal': [1.5] * len(sample_price_data),
                'MACD_Hist': [0.5] * len(sample_price_data)
            }, index=sample_price_data.index),
            'rsi': lambda *args, **kwargs: pd.Series([60.0] * len(sample_price_data), index=sample_price_data.index)
        }
        
        with patch('simple_trade.data.indicator_handler.handle_stochastic', wraps=handle_stochastic), \
             patch('simple_trade.data.indicator_handler.handle_cci', wraps=handle_cci), \
             patch('simple_trade.data.indicator_handler.handle_roc', wraps=handle_roc), \
             patch('simple_trade.data.indicator_handler.handle_macd', wraps=handle_macd), \
             patch('simple_trade.data.indicator_handler.handle_rsi', wraps=handle_rsi):
            
            # Test with Stochastic Oscillator
            with patch('simple_trade.data.indicator_handler.INDICATORS', mock_indicators):
                result = _calculate_indicator(sample_price_data, 'stoch', mock_indicators['stoch'], window=14)
                assert isinstance(result, pd.Series)
            
            # Test with CCI
            with patch('simple_trade.data.indicator_handler.INDICATORS', mock_indicators):
                result = _calculate_indicator(sample_price_data, 'cci', mock_indicators['cci'], window=20)
                assert isinstance(result, pd.Series)
            
            # Test with ROC
            with patch('simple_trade.data.indicator_handler.INDICATORS', mock_indicators):
                result = _calculate_indicator(sample_price_data, 'roc', mock_indicators['roc'], window=10)
                assert isinstance(result, pd.Series)
            
            # Test with MACD
            with patch('simple_trade.data.indicator_handler.INDICATORS', mock_indicators):
                result = _calculate_indicator(sample_price_data, 'macd', mock_indicators['macd'], fast=12, slow=26, signal=9)
                assert isinstance(result, pd.DataFrame)
            
            # Test with RSI
            with patch('simple_trade.data.indicator_handler.INDICATORS', mock_indicators):
                result = _calculate_indicator(sample_price_data, 'rsi', mock_indicators['rsi'], window=14)
                assert isinstance(result, pd.Series)

class TestAddIndicatorToDataFrame:
    """Tests for the _add_indicator_to_dataframe function."""
    
    def test_add_series_indicator(self, sample_price_data):
        """Test adding a Series indicator to DataFrame."""
        # Create a Series indicator (like SMA)
        series_name = 'CUSTOM_SMA_10'
        indicator_series = pd.Series([105.0] * len(sample_price_data), index=sample_price_data.index, name=series_name)
        
        # Add it to the DataFrame
        result = _add_indicator_to_dataframe(sample_price_data.copy(), 'sma', indicator_series, {'window': 10}) # Use .copy() to avoid modifying fixture
        
        # Check it was added
        assert series_name in result.columns
        assert result[series_name].iloc[0] == 105.0
    
    def test_add_dataframe_indicator(self, sample_price_data):
        """Test adding a DataFrame indicator to DataFrame."""
        # Create a DataFrame indicator (like Bollinger Bands)
        indicator_df = pd.DataFrame({
            'BOLLIN_UPPER': [115.0] * len(sample_price_data),
            'BOLLIN_MIDDLE': [105.0] * len(sample_price_data),
            'BOLLIN_LOWER': [95.0] * len(sample_price_data)
        }, index=sample_price_data.index)
        
        # Add it to the DataFrame
        result = _add_indicator_to_dataframe(sample_price_data.copy(), 'bollin', indicator_df, {'window': 20}) # Use .copy()
        
        # Check all columns were added
        assert 'BOLLIN_UPPER' in result.columns
        assert 'BOLLIN_MIDDLE' in result.columns
        assert 'BOLLIN_LOWER' in result.columns
    
    def test_add_ichimoku_indicator(self, sample_price_data):
        """Test adding an Ichimoku Cloud indicator (dict) to DataFrame."""
        # Create a dict indicator for Ichimoku Cloud
        indicator_dict = {
            'tenkan_sen': pd.Series([110.0] * len(sample_price_data), index=sample_price_data.index),
            'kijun_sen': pd.Series([105.0] * len(sample_price_data), index=sample_price_data.index),
            'senkou_span_a': pd.Series([108.0] * len(sample_price_data), index=sample_price_data.index),
            'senkou_span_b': pd.Series([102.0] * len(sample_price_data), index=sample_price_data.index),
            'chikou_span': pd.Series([112.0] * len(sample_price_data), index=sample_price_data.index)
        }
        
        # Add it to the DataFrame
        initial_columns = set(sample_price_data.columns)
        result = _add_indicator_to_dataframe(sample_price_data.copy(), 'ichimoku', indicator_dict, {}) # Use .copy()
        
        # Check components were added with proper naming
        assert set(result.columns) == initial_columns # Dicts are not processed to add columns
    
    def test_add_aroon_indicator(self, sample_price_data):
        """Test adding an Aroon indicator (tuple) to DataFrame."""
        # Create indicator tuple for Aroon
        initial_columns = set(sample_price_data.columns)
        indicator_tuple = (
            pd.Series([70.0] * len(sample_price_data), index=sample_price_data.index),  # aroon_up
            pd.Series([30.0] * len(sample_price_data), index=sample_price_data.index),  # aroon_down
            pd.Series([40.0] * len(sample_price_data), index=sample_price_data.index)   # aroon_oscillator
        )
        
        # Add it to the DataFrame
        result = _add_indicator_to_dataframe(sample_price_data.copy(), 'aroon', indicator_tuple, {'window': 14}) # Use .copy()
            
        # Check components were added with proper naming
        assert set(result.columns) == initial_columns # Tuples are not processed to add columns

class TestMomentumIndicatorHandlers:
    """Tests for momentum indicator handlers (direct calls to improve coverage)."""
    
    def test_handle_stochastic(self, sample_price_data):
        """Test the handle_stochastic function."""
        from simple_trade.data.momentum_handlers import handle_stochastic
        
        # Create a mock stochastic function
        mock_stoch_func = MagicMock(return_value=pd.Series([80.0] * len(sample_price_data), index=sample_price_data.index))
        
        # Call the handler
        result = handle_stochastic(sample_price_data, mock_stoch_func, window=14)
        
        # Verify result
        assert isinstance(result, pd.Series)
        assert result.iloc[0] == 80.0
        
        # Verify the mock was called with correct arguments
        mock_stoch_func.assert_called_once()
        
        # Test error handling for missing columns
        df_missing_cols = sample_price_data.drop(columns=['High', 'Low'])
        with pytest.raises(ValueError, match="DataFrame must contain"):
            handle_stochastic(df_missing_cols, mock_stoch_func)
    
    def test_handle_cci(self, sample_price_data):
        """Test the handle_cci function."""
        from simple_trade.data.momentum_handlers import handle_cci
        
        # Create a mock CCI function
        mock_cci_func = MagicMock(return_value=pd.Series([120.0] * len(sample_price_data), index=sample_price_data.index))
        
        # Call the handler
        result = handle_cci(sample_price_data, mock_cci_func, window=20)
        
        # Verify result
        assert isinstance(result, pd.Series)
        assert result.iloc[0] == 120.0
        
        # Verify the mock was called with correct arguments
        mock_cci_func.assert_called_once()
        
        # Test error handling for missing columns
        df_missing_cols = sample_price_data.drop(columns=['High', 'Low'])
        with pytest.raises(ValueError, match="DataFrame must contain"):
            handle_cci(df_missing_cols, mock_cci_func)
    
    def test_handle_roc(self, sample_price_data):
        """Test the handle_roc function."""
        from simple_trade.data.momentum_handlers import handle_roc
        
        # Create a mock ROC function
        mock_roc_func = MagicMock(return_value=pd.Series([5.0] * len(sample_price_data), index=sample_price_data.index))
        
        # Call the handler
        result = handle_roc(sample_price_data, mock_roc_func, window=10)
        
        # Verify result
        assert isinstance(result, pd.Series)
        assert result.iloc[0] == 5.0
        
        # Verify the mock was called with correct arguments
        mock_roc_func.assert_called_once()
        
        # Test error handling for missing columns
        df_missing_cols = sample_price_data.drop(columns=['Close'])
        with pytest.raises(ValueError, match="DataFrame must contain"):
            handle_roc(df_missing_cols, mock_roc_func)
    
    def test_handle_macd(self, sample_price_data):
        """Test the handle_macd function."""
        from simple_trade.data.momentum_handlers import handle_macd
        
        # Create a mock MACD function
        mock_macd_result = pd.DataFrame({
            'MACD': [2.0] * len(sample_price_data),
            'MACD_Signal': [1.5] * len(sample_price_data),
            'MACD_Hist': [0.5] * len(sample_price_data)
        }, index=sample_price_data.index)
        mock_macd_func = MagicMock(return_value=mock_macd_result)
        
        # Call the handler
        result = handle_macd(sample_price_data, mock_macd_func, fast=12, slow=26, signal=9)
        
        # Verify result
        assert isinstance(result, pd.DataFrame)
        assert 'MACD' in result.columns
        assert result['MACD'].iloc[0] == 2.0
        
        # Verify the mock was called with correct arguments
        mock_macd_func.assert_called_once()
        args, kwargs = mock_macd_func.call_args
        assert kwargs.get('window_fast') == 12
        assert kwargs.get('window_slow') == 26
        assert kwargs.get('window_signal') == 9
        assert kwargs.get('close_col') == 'Close' # Assuming default close column
        
        # Test error handling for missing columns
        df_missing_cols = sample_price_data.drop(columns=['Close'])
        with pytest.raises(ValueError, match="DataFrame must contain"):
            handle_macd(df_missing_cols, mock_macd_func)
    
    def test_handle_rsi(self, sample_price_data):
        """Test the handle_rsi function."""
        from simple_trade.data.momentum_handlers import handle_rsi
        
        # Create a mock RSI function
        mock_rsi_func = MagicMock(return_value=pd.Series([60.0] * len(sample_price_data), index=sample_price_data.index))
        
        # Call the handler
        result = handle_rsi(sample_price_data, mock_rsi_func, window=14)
        
        # Verify result
        assert isinstance(result, pd.Series)
        assert result.iloc[0] == 60.0
        
        # Verify the mock was called with correct arguments
        mock_rsi_func.assert_called_once()
        args, kwargs = mock_rsi_func.call_args
        assert kwargs.get('window') == 14
        
        # Test error handling for missing columns
        df_missing_cols = sample_price_data.drop(columns=['Close'])
        with pytest.raises(ValueError, match="DataFrame must contain"):
            handle_rsi(df_missing_cols, mock_rsi_func)