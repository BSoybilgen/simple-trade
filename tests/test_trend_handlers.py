import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from simple_trade.data.trend_handlers import (
    handle_strend,
    handle_adx,
    handle_psar,
    handle_ichimoku,
    handle_aroon,
)

# --- Fixtures ---

@pytest.fixture
def sample_price_data():
    """Fixture providing a basic OHLCV DataFrame for testing trend indicators."""
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
    data = pd.DataFrame(index=dates)
    data['Open'] = [100, 102, 104, 103, 105, 107, 108, 109, 110, 111, 112, 111, 110, 112, 114, 115, 116, 115, 113, 114]
    data['High'] = [103, 105, 107, 106, 108, 110, 111, 112, 113, 114, 115, 114, 113, 115, 117, 118, 119, 118, 116, 117]
    data['Low'] = [98, 100, 102, 101, 103, 105, 106, 107, 108, 109, 110, 109, 108, 110, 112, 113, 114, 113, 111, 112]
    data['Close'] = [102, 104, 106, 105, 107, 109, 110, 111, 112, 113, 114, 113, 112, 114, 116, 117, 118, 117, 115, 116]
    data['Volume'] = [1000, 1200, 1300, 1100, 1400, 1500, 1600, 1500, 1400, 1600, 1700, 1500, 1400, 1600, 1800, 1900, 2000, 1800, 1700, 1900]
    return data

# --- Test Classes ---

class TestHandleStrend:
    """Tests for the handle_strend function."""
    
    def test_valid_input(self, sample_price_data):
        """Test with valid input data."""
        # Create a mock indicator function that returns a predictable Series
        mock_strend_func = MagicMock(return_value=pd.Series([100.0] * len(sample_price_data), index=sample_price_data.index))
        
        # Call the handler
        result = handle_strend(sample_price_data, mock_strend_func, period=10, multiplier=3.0)
        
        # Check the result
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        
        # Verify the mock was called with the correct arguments
        mock_strend_func.assert_called_once()
        args, kwargs = mock_strend_func.call_args
        assert kwargs['period'] == 10
        assert kwargs['multiplier'] == 3.0
    
    def test_missing_columns(self, sample_price_data):
        """Test that ValueError is raised when required columns are missing."""
        # Create a DataFrame missing the 'High' column
        df_no_high = sample_price_data.drop(columns=['High'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High', 'Low', and 'Close' columns"):
            handle_strend(df_no_high, MagicMock(), period=10)
        
        # Create a DataFrame missing the 'Low' column
        df_no_low = sample_price_data.drop(columns=['Low'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High', 'Low', and 'Close' columns"):
            handle_strend(df_no_low, MagicMock(), period=10)
        
        # Create a DataFrame missing the 'Close' column
        df_no_close = sample_price_data.drop(columns=['Close'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High', 'Low', and 'Close' columns"):
            handle_strend(df_no_close, MagicMock(), period=10)
    
    def test_default_parameters(self, sample_price_data):
        """Test that default parameters are correctly applied."""
        # Create a mock indicator function
        mock_strend_func = MagicMock(return_value=pd.Series([100.0] * len(sample_price_data), index=sample_price_data.index))
        
        # Call without specifying period and multiplier
        result = handle_strend(sample_price_data, mock_strend_func)
        
        # Verify the mock was called with default parameters
        args, kwargs = mock_strend_func.call_args
        assert kwargs['period'] == 7  # Default period
        assert kwargs['multiplier'] == 3.0  # Default multiplier
    
    def test_remove_hlc_kwargs(self, sample_price_data):
        """Test that high, low, close kwargs are removed if passed."""
        # Create a mock indicator function
        mock_strend_func = MagicMock(return_value=pd.Series([100.0] * len(sample_price_data), index=sample_price_data.index))
        
        # Call with high, low, close in kwargs (should be removed)
        result = handle_strend(sample_price_data, mock_strend_func, high="High", low="Low", close="Close")
        
        # Verify the mock was called without high, low, close in kwargs
        args, kwargs = mock_strend_func.call_args
        assert 'high' not in kwargs
        assert 'low' not in kwargs
        assert 'close' not in kwargs


class TestHandleADX:
    """Tests for the handle_adx function."""
    
    def test_valid_input(self, sample_price_data):
        """Test with valid input data."""
        # Create a mock indicator function that returns a predictable Series
        mock_adx_func = MagicMock(return_value=pd.Series([50.0] * len(sample_price_data), index=sample_price_data.index))
        
        # Call the handler
        result = handle_adx(sample_price_data, mock_adx_func, window=14)
        
        # Check the result
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        
        # Verify the mock was called with the correct arguments
        mock_adx_func.assert_called_once()
        args, kwargs = mock_adx_func.call_args
        assert kwargs['window'] == 14
    
    def test_missing_columns(self, sample_price_data):
        """Test that ValueError is raised when required columns are missing."""
        # Create a DataFrame missing the 'High' column
        df_no_high = sample_price_data.drop(columns=['High'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High', 'Low', and 'Close' columns"):
            handle_adx(df_no_high, MagicMock(), window=14)
        
        # Create a DataFrame missing the 'Low' column
        df_no_low = sample_price_data.drop(columns=['Low'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High', 'Low', and 'Close' columns"):
            handle_adx(df_no_low, MagicMock(), window=14)
        
        # Create a DataFrame missing the 'Close' column
        df_no_close = sample_price_data.drop(columns=['Close'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High', 'Low', and 'Close' columns"):
            handle_adx(df_no_close, MagicMock(), window=14)
    
    def test_default_parameters(self, sample_price_data):
        """Test that default parameters are correctly applied."""
        # Create a mock indicator function
        mock_adx_func = MagicMock(return_value=pd.Series([50.0] * len(sample_price_data), index=sample_price_data.index))
        
        # Call without specifying window
        result = handle_adx(sample_price_data, mock_adx_func)
        
        # Verify the mock was called with default parameters
        args, kwargs = mock_adx_func.call_args
        assert kwargs['window'] == 14  # Default window
    
    def test_remove_hlc_kwargs(self, sample_price_data):
        """Test that high, low, close kwargs are removed if passed."""
        # Create a mock indicator function
        mock_adx_func = MagicMock(return_value=pd.Series([50.0] * len(sample_price_data), index=sample_price_data.index))
        
        # Call with high, low, close in kwargs (should be removed)
        result = handle_adx(sample_price_data, mock_adx_func, high="High", low="Low", close="Close")
        
        # Verify the mock was called without high, low, close in kwargs
        args, kwargs = mock_adx_func.call_args
        assert 'high' not in kwargs
        assert 'low' not in kwargs
        assert 'close' not in kwargs


class TestHandlePSAR:
    """Tests for the handle_psar function."""
    
    def test_valid_input(self, sample_price_data):
        """Test with valid input data."""
        # Create a mock indicator function that returns a predictable Series
        mock_psar_func = MagicMock(return_value=pd.Series([105.0] * len(sample_price_data), index=sample_price_data.index))
        
        # Call the handler
        result = handle_psar(sample_price_data, mock_psar_func, af_initial=0.02, af_step=0.02, af_max=0.2)
        
        # Check the result
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        
        # Verify the mock was called with the correct arguments
        mock_psar_func.assert_called_once()
        args, kwargs = mock_psar_func.call_args
        assert kwargs['af_initial'] == 0.02
        assert kwargs['af_step'] == 0.02
        assert kwargs['af_max'] == 0.2
    
    def test_missing_columns(self, sample_price_data):
        """Test that ValueError is raised when required columns are missing."""
        # Create a DataFrame missing the 'High' column
        df_no_high = sample_price_data.drop(columns=['High'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High', 'Low', and 'Close' columns"):
            handle_psar(df_no_high, MagicMock())
        
        # Create a DataFrame missing the 'Low' column
        df_no_low = sample_price_data.drop(columns=['Low'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High', 'Low', and 'Close' columns"):
            handle_psar(df_no_low, MagicMock())

    def test_missing_close_column(self, sample_price_data):
        """Test that ValueError is raised when Close column is missing."""
        # Create a DataFrame missing the 'Close' column but with 'High' and 'Low'
        df_no_close = sample_price_data.drop(columns=['Close'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High', 'Low', and 'Close' columns"):
            handle_psar(df_no_close, MagicMock())

    def test_default_parameters(self, sample_price_data):
        """Test that default parameters are correctly applied."""
        # Create a mock indicator function
        mock_psar_func = MagicMock(return_value=pd.Series([105.0] * len(sample_price_data), index=sample_price_data.index))
        
        # Call without specifying acceleration factors
        result = handle_psar(sample_price_data, mock_psar_func)
        
        # Verify the mock was called with default parameters
        args, kwargs = mock_psar_func.call_args
        assert kwargs['af_initial'] == 0.02  # Default af_initial
        assert kwargs['af_step'] == 0.02     # Default af_step
        assert kwargs['af_max'] == 0.2       # Default af_max


class TestHandleIchimoku:
    """Tests for the handle_ichimoku function."""
    
    def test_full_ichimoku(self, sample_price_data):
        """Test calculation of full Ichimoku Cloud."""
        # Create a mock indicator function that returns a dict of components
        mock_ichimoku_dict = {
            'tenkan_sen': pd.Series([110.0] * len(sample_price_data), index=sample_price_data.index),
            'kijun_sen': pd.Series([105.0] * len(sample_price_data), index=sample_price_data.index),
            'senkou_span_a': pd.Series([108.0] * len(sample_price_data), index=sample_price_data.index),
            'senkou_span_b': pd.Series([102.0] * len(sample_price_data), index=sample_price_data.index),
            'chikou_span': pd.Series([112.0] * len(sample_price_data), index=sample_price_data.index)
        }
        mock_ichimoku_func = MagicMock(return_value=mock_ichimoku_dict)
        
        # Call the handler for full Ichimoku
        result = handle_ichimoku(sample_price_data, mock_ichimoku_func, 'ichimoku',
                               tenkan_period=9, kijun_period=26, senkou_b_period=52, displacement=26)
        
        # Check the result is a dict with all components
        assert isinstance(result, dict)
        assert 'tenkan_sen' in result
        assert 'kijun_sen' in result
        assert 'senkou_span_a' in result
        assert 'senkou_span_b' in result
        assert 'chikou_span' in result
        
        # Verify the mock was called with the correct arguments
        mock_ichimoku_func.assert_called_once()
        args, kwargs = mock_ichimoku_func.call_args
        assert kwargs['tenkan_period'] == 9
        assert kwargs['kijun_period'] == 26
        assert kwargs['senkou_b_period'] == 52
        assert kwargs['displacement'] == 26
    
    def test_individual_components(self, sample_price_data):
        """Test calculation of individual Ichimoku components."""
        # Create a mock component function
        mock_component_func = MagicMock(return_value=pd.Series([110.0] * len(sample_price_data), index=sample_price_data.index))
        
        # Test for tenkan_sen
        result = handle_ichimoku(sample_price_data, mock_component_func, 'tenkan_sen')
        mock_component_func.assert_called()
        
        # Test for chikou_span (uses different parameters)
        mock_component_func.reset_mock()
        result = handle_ichimoku(sample_price_data, mock_component_func, 'chikou_span')
        args, kwargs = mock_component_func.call_args
        # Should be called with Close and displacement
        assert len(args) == 1  # Just Close for chikou_span
    
    def test_missing_columns(self, sample_price_data):
        """Test that ValueError is raised when required columns are missing."""
        # Create a DataFrame missing the 'High' column
        df_no_high = sample_price_data.drop(columns=['High'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High', 'Low', and 'Close' columns"):
            handle_ichimoku(df_no_high, MagicMock(), 'ichimoku')
        
        # Create a DataFrame missing the 'Low' column
        df_no_low = sample_price_data.drop(columns=['Low'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High', 'Low', and 'Close' columns"):
            handle_ichimoku(df_no_low, MagicMock(), 'ichimoku')
        
        # Create a DataFrame missing the 'Close' column
        df_no_close = sample_price_data.drop(columns=['Close'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High', 'Low', and 'Close' columns"):
            handle_ichimoku(df_no_close, MagicMock(), 'ichimoku')
    
    def test_default_parameters(self, sample_price_data):
        """Test that default parameters are correctly applied."""
        # Create a mock indicator function
        mock_ichimoku_func = MagicMock(return_value={})
        
        # Call without specifying parameters
        result = handle_ichimoku(sample_price_data, mock_ichimoku_func, 'ichimoku')
        
        # Verify the mock was called with default parameters
        args, kwargs = mock_ichimoku_func.call_args
        assert kwargs['tenkan_period'] == 9        # Default tenkan_period
        assert kwargs['kijun_period'] == 26        # Default kijun_period
        assert kwargs['senkou_b_period'] == 52     # Default senkou_b_period
        assert kwargs['displacement'] == 26        # Default displacement


class TestHandleAroon:
    """Tests for the handle_aroon function."""
    
    def test_valid_input(self, sample_price_data):
        """Test with valid input data."""
        # Create a mock indicator function that returns a tuple of Series for up, down, oscillator
        mock_aroon_result = (
            pd.Series([70.0] * len(sample_price_data), index=sample_price_data.index),  # aroon_up
            pd.Series([30.0] * len(sample_price_data), index=sample_price_data.index),  # aroon_down
            pd.Series([40.0] * len(sample_price_data), index=sample_price_data.index)   # aroon_oscillator
        )
        mock_aroon_func = MagicMock(return_value=mock_aroon_result)
        
        # Call the handler
        result = handle_aroon(sample_price_data, mock_aroon_func, period=14)
        
        # Check the result is a tuple of 3 Series
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(series, pd.Series) for series in result)
        
        # Verify the mock was called with the correct arguments
        mock_aroon_func.assert_called_once()
        args, kwargs = mock_aroon_func.call_args
        assert kwargs['period'] == 14
    
    def test_missing_columns(self, sample_price_data):
        """Test that ValueError is raised when required columns are missing."""
        # Create a DataFrame missing the 'High' column
        df_no_high = sample_price_data.drop(columns=['High'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High' and 'Low' columns"):
            handle_aroon(df_no_high, MagicMock(), period=14)
        
        # Create a DataFrame missing the 'Low' column
        df_no_low = sample_price_data.drop(columns=['Low'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High' and 'Low' columns"):
            handle_aroon(df_no_low, MagicMock(), period=14)
    
    def test_default_parameters(self, sample_price_data):
        """Test that default parameters are correctly applied."""
        # Create a mock indicator function
        mock_aroon_func = MagicMock(return_value=(None, None, None))
        
        # Call without specifying period
        result = handle_aroon(sample_price_data, mock_aroon_func)
        
        # Verify the mock was called with default parameters
        args, kwargs = mock_aroon_func.call_args
        assert kwargs['period'] == 14  # Default period
    
    def test_remove_hl_kwargs(self, sample_price_data):
        """Test that high, low kwargs are removed if passed."""
        # Create a mock indicator function
        mock_aroon_func = MagicMock(return_value=(None, None, None))
        
        # Call with high, low in kwargs (should be removed)
        result = handle_aroon(sample_price_data, mock_aroon_func, high="High", low="Low")
        
        # Verify the mock was called without high, low in kwargs
        args, kwargs = mock_aroon_func.call_args
        assert 'high' not in kwargs
        assert 'low' not in kwargs