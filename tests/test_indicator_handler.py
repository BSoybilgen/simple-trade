import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from simple_trade.data.indicator_handler import (
    compute_indicator,
    _calculate_indicator,
    _add_indicator_to_dataframe,
    _format_indicator_name,
    download_data,
    download_and_compute_indicator
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
        mock_series = pd.Series([105.0] * len(sample_price_data), index=sample_price_data.index)
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
            
            # Verify all components have been added with proper naming
            assert 'Ichimoku_tenkan_sen' in result.columns
            assert 'Ichimoku_kijun_sen' in result.columns
            assert 'Ichimoku_senkou_span_a' in result.columns
            assert 'Ichimoku_senkou_span_b' in result.columns
            assert 'Ichimoku_chikou_span' in result.columns
            
            # Check values
            assert result['Ichimoku_tenkan_sen'].iloc[0] == 110.0
            assert result['Ichimoku_kijun_sen'].iloc[0] == 105.0
    
    @patch('simple_trade.data.indicator_handler._calculate_indicator')
    def test_compute_aroon(self, mock_calculate, sample_price_data, mock_indicators):
        """Test computation of Aroon (returns tuple of components)."""
        # Create mock return value for _calculate_indicator
        mock_tuple = (
            pd.Series([70.0] * len(sample_price_data), index=sample_price_data.index),
            pd.Series([30.0] * len(sample_price_data), index=sample_price_data.index),
            pd.Series([40.0] * len(sample_price_data), index=sample_price_data.index)
        )
        mock_calculate.return_value = mock_tuple
        
        with patch('simple_trade.data.indicator_handler.INDICATORS', mock_indicators):
            with patch('simple_trade.data.indicator_handler._format_indicator_name', return_value='_14'):
                result = compute_indicator(sample_price_data, 'aroon', window=14)
                
                # Verify the calculation was called
                mock_calculate.assert_called_once()
                
                # Verify all components have been added
                assert 'AROON_UP_14' in result.columns
                assert 'AROON_DOWN_14' in result.columns
                assert 'AROON_OSC_14' in result.columns
                
                # Check values
                assert result['AROON_UP_14'].iloc[0] == 70.0
                assert result['AROON_DOWN_14'].iloc[0] == 30.0
                assert result['AROON_OSC_14'].iloc[0] == 40.0
    
    def test_missing_column(self, sample_price_data, mock_indicators):
        """Test error handling when required column is missing."""
        with patch('simple_trade.data.indicator_handler.INDICATORS', mock_indicators):
            # Remove Close column
            data_no_close = sample_price_data.drop(columns=['Close'])
            
            # Test _calculate_indicator directly since it will raise the exception
            with pytest.raises(ValueError, match="DataFrame must contain a 'Close' column"):
                _calculate_indicator(data_no_close, 'sma', mock_indicators['sma'], window=10)

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
        # Create a simple Series indicator
        indicator_series = pd.Series([105.0] * len(sample_price_data), index=sample_price_data.index)
        
        # Add it to the DataFrame
        with patch('simple_trade.data.indicator_handler._format_indicator_name', return_value='_10'):
            result = _add_indicator_to_dataframe(sample_price_data, 'sma', indicator_series, {'window': 10})
            
            # Check it was added with proper name
            assert 'SMA_10' in result.columns
            assert result['SMA_10'].iloc[0] == 105.0
    
    def test_add_dataframe_indicator(self, sample_price_data):
        """Test adding a DataFrame indicator to DataFrame."""
        # Create a DataFrame indicator (like Bollinger Bands)
        indicator_df = pd.DataFrame({
            'BOLLIN_UPPER': [115.0] * len(sample_price_data),
            'BOLLIN_MIDDLE': [105.0] * len(sample_price_data),
            'BOLLIN_LOWER': [95.0] * len(sample_price_data)
        }, index=sample_price_data.index)
        
        # Add it to the DataFrame
        result = _add_indicator_to_dataframe(sample_price_data, 'bollin', indicator_df, {'window': 20})
        
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
        result = _add_indicator_to_dataframe(sample_price_data, 'ichimoku', indicator_dict, {})
        
        # Check components were added with proper naming
        assert 'Ichimoku_tenkan_sen' in result.columns
        assert 'Ichimoku_kijun_sen' in result.columns
        assert 'Ichimoku_senkou_span_a' in result.columns
        assert 'Ichimoku_senkou_span_b' in result.columns
        assert 'Ichimoku_chikou_span' in result.columns
    
    def test_add_aroon_indicator(self, sample_price_data):
        """Test adding an Aroon indicator (tuple) to DataFrame."""
        # Create indicator tuple for Aroon
        indicator_tuple = (
            pd.Series([70.0] * len(sample_price_data), index=sample_price_data.index),  # aroon_up
            pd.Series([30.0] * len(sample_price_data), index=sample_price_data.index),  # aroon_down
            pd.Series([40.0] * len(sample_price_data), index=sample_price_data.index)   # aroon_oscillator
        )
        
        # Add it to the DataFrame
        with patch('simple_trade.data.indicator_handler._format_indicator_name', return_value='_14'):
            result = _add_indicator_to_dataframe(sample_price_data, 'aroon', indicator_tuple, {'window': 14})
            
            # Check components were added with proper naming
            assert 'AROON_UP_14' in result.columns
            assert 'AROON_DOWN_14' in result.columns
            assert 'AROON_OSC_14' in result.columns

class TestFormatIndicatorName:
    """Tests for the _format_indicator_name function."""
    
    def test_format_trend_indicator(self):
        """Test formatting for trend indicators."""
        with patch('simple_trade.data.indicator_handler.format_trend_indicator_name', return_value='_10'):
            name = _format_indicator_name('sma', {'window': 10})
            assert name == '_10'  # Standard format for trend indicators
    
    def test_format_momentum_indicator(self):
        """Test formatting for momentum indicators."""
        with patch('simple_trade.data.indicator_handler.format_momentum_indicator_name', return_value='_14'):
            name = _format_indicator_name('rsi', {'window': 14})
            assert isinstance(name, str)
    
    def test_format_volatility_indicator(self):
        """Test formatting for volatility indicators."""
        with patch('simple_trade.data.indicator_handler.format_volatility_indicator_name', return_value='_20_2'):
            name = _format_indicator_name('bollin', {'window': 20, 'window_dev': 2})
            assert isinstance(name, str)
    
    def test_format_volume_indicator(self):
        """Test formatting for volume indicators."""
        with patch('simple_trade.data.indicator_handler.format_volume_indicator_name', return_value='_20'):
            name = _format_indicator_name('vma', {'window': 20})
            assert isinstance(name, str)
    
    def test_default_format(self):
        """Test default formatting for unknown indicator types."""
        name = _format_indicator_name('unknown', {'window': 10})
        assert name == ""  # Default is empty string

    def test_format_momentum_indicator_name(self):
        """Test the format_momentum_indicator_name function."""
        from simple_trade.data.momentum_handlers import format_momentum_indicator_name
        
        # Test with RSI
        result = format_momentum_indicator_name('rsi', {'window': 14})
        assert result == '_14'
        
        # Test with MACD (currently returns empty string)
        result = format_momentum_indicator_name('macd', {'fast': 12, 'slow': 26, 'signal': 9})
        assert result == ""
        
        # Test with stochastic
        result = format_momentum_indicator_name('stoch', {'k_period': 14, 'smooth_window': 3})
        assert result == '_14'
        
        # Test with CCI
        result = format_momentum_indicator_name('cci', {'window': 20})
        assert result == '_20'
        
        # Test with ROC
        result = format_momentum_indicator_name('roc', {'window': 10})
        assert result == '_10'

class TestDownloadFunctions:
    """Tests for the download_data and download_and_compute_indicator functions."""
    
    @patch('simple_trade.data.indicator_handler.yf.download')
    def test_download_data(self, mock_download):
        """Test downloading data with yfinance."""
        # Setup mock return value
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [98, 99, 100],
            'Close': [101, 102, 103],
            'Adj Close': [101, 102, 103],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range(start='2023-01-01', periods=3))
        
        mock_download.return_value = mock_data
        
        # Call the function
        result = download_data('AAPL', '2023-01-01', '2023-01-03')
        
        # Check that download was called with correct parameters
        mock_download.assert_called_once_with(
            'AAPL', 
            start='2023-01-01', 
            end='2023-01-03', 
            interval='1d', 
            progress=False, 
            auto_adjust=False
        )
        
        # Check that the result has the expected columns
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
            assert col in result.columns
        
        # Check symbol was added to attrs
        assert result.attrs['symbol'] == 'AAPL'
    
    @patch('simple_trade.data.indicator_handler.download_data')
    @patch('simple_trade.data.indicator_handler.compute_indicator')
    def test_download_and_compute_indicator(self, mock_compute, mock_download):
        """Test downloading data and computing indicator."""
        # Setup mock returns
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [98, 99, 100],
            'Close': [101, 102, 103],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range(start='2023-01-01', periods=3))
        
        mock_result = mock_data.copy()
        mock_result['SMA_10'] = [105, 105, 105]
        
        mock_download.return_value = mock_data
        mock_compute.return_value = mock_result
        
        # Call the function
        result = download_and_compute_indicator(
            'AAPL', 
            '2023-01-01', 
            'sma', 
            end_date='2023-01-03', 
            window=10
        )
        
        # Verify download_data was called
        mock_download.assert_called_once_with('AAPL', '2023-01-01', '2023-01-03')
        
        # Verify compute_indicator was called
        mock_compute.assert_called_once_with(mock_data, 'sma', window=10)
        
        # Check result
        assert 'SMA_10' in result.columns

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
        assert kwargs.get('fast') == 12
        assert kwargs.get('slow') == 26
        assert kwargs.get('signal') == 9
        
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
    
    def test_format_momentum_indicator_name(self):
        """Test the format_momentum_indicator_name function."""
        from simple_trade.data.momentum_handlers import format_momentum_indicator_name
        
        # Test with RSI
        result = format_momentum_indicator_name('rsi', {'window': 14})
        assert result == '_14'
        
        # Test with MACD (currently returns empty string)
        result = format_momentum_indicator_name('macd', {'fast': 12, 'slow': 26, 'signal': 9})
        assert result == ""
        
        # Test with stochastic
        result = format_momentum_indicator_name('stoch', {'k_period': 14, 'smooth_window': 3})
        assert result == '_14'
        
        # Test with CCI
        result = format_momentum_indicator_name('cci', {'window': 20})
        assert result == '_20'
        
        # Test with ROC
        result = format_momentum_indicator_name('roc', {'window': 10})
        assert result == '_10' 