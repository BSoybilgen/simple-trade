import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from simple_trade.data.volatility_handlers import (
    handle_bollin,
    handle_atr,
    handle_kelt,
    handle_donch,
    handle_chaik,
)

# --- Fixtures ---

@pytest.fixture
def sample_price_data():
    """Fixture providing a basic OHLCV DataFrame for testing volatility indicators."""
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
    data = pd.DataFrame(index=dates)
    data['Open'] = [100, 102, 104, 103, 105, 107, 108, 109, 110, 111, 112, 111, 110, 112, 114, 115, 116, 115, 113, 114]
    data['High'] = [103, 105, 107, 106, 108, 110, 111, 112, 113, 114, 115, 114, 113, 115, 117, 118, 119, 118, 116, 117]
    data['Low'] = [98, 100, 102, 101, 103, 105, 106, 107, 108, 109, 110, 109, 108, 110, 112, 113, 114, 113, 111, 112]
    data['Close'] = [102, 104, 106, 105, 107, 109, 110, 111, 112, 113, 114, 113, 112, 114, 116, 117, 118, 117, 115, 116]
    data['Volume'] = [1000, 1200, 1300, 1100, 1400, 1500, 1600, 1500, 1400, 1600, 1700, 1500, 1400, 1600, 1800, 1900, 2000, 1800, 1700, 1900]
    return data

# --- Test Classes ---

class TestHandleBollingerBands:
    """Tests for the handle_bollin function."""
    
    def test_valid_input(self, sample_price_data):
        """Test with valid input data."""
        # Create a mock result with upper, middle, and lower bands
        mock_bollin_result = pd.DataFrame({
            'BOLLIN_UPPER': [115.0] * len(sample_price_data),
            'BOLLIN_MIDDLE': [105.0] * len(sample_price_data),
            'BOLLIN_LOWER': [95.0] * len(sample_price_data)
        }, index=sample_price_data.index)
        
        # Create a mock indicator function
        mock_bollin_func = MagicMock(return_value=mock_bollin_result)
        
        # Call the handler
        result = handle_bollin(sample_price_data, mock_bollin_func, window=20, num_std=2)
        
        # Check the result
        assert isinstance(result, pd.DataFrame)
        assert 'BOLLIN_UPPER' in result.columns
        assert 'BOLLIN_MIDDLE' in result.columns
        assert 'BOLLIN_LOWER' in result.columns
        
        # Verify the mock was called with the correct arguments
        mock_bollin_func.assert_called_once()
        args, kwargs = mock_bollin_func.call_args
        assert kwargs['window'] == 20
        assert kwargs['num_std'] == 2
    
    def test_missing_close_column(self, sample_price_data):
        """Test that ValueError is raised when Close column is missing."""
        # Create a DataFrame missing the 'Close' column
        df_no_close = sample_price_data.drop(columns=['Close'])
        
        with pytest.raises(ValueError, match=r"DataFrame must contain 'Close' columns\."):
            handle_bollin(df_no_close, MagicMock())
    
    def test_default_parameters(self, sample_price_data):
        """Test that default parameters are correctly applied."""
        # Create a mock indicator function
        mock_bollin_func = MagicMock(return_value=pd.DataFrame())
        
        # Call without specifying parameters
        result = handle_bollin(sample_price_data, mock_bollin_func)
        
        # Verify the mock was called with default parameters
        args, kwargs = mock_bollin_func.call_args
        assert kwargs['window'] == 20  # Default window
        assert kwargs['num_std'] == 2  # Default standard deviation
    
    def test_std_dev_parameter(self, sample_price_data):
        """Test handling of 'std_dev' parameter (renamed to num_std)."""
        # Create a mock indicator function
        mock_bollin_func = MagicMock(return_value=pd.DataFrame())
        
        # Call with std_dev parameter instead of num_std
        result = handle_bollin(sample_price_data, mock_bollin_func, std_dev=2.5)
        
        # Verify param was renamed to num_std
        args, kwargs = mock_bollin_func.call_args
        assert 'std_dev' not in kwargs
        assert kwargs['num_std'] == 2.5


class TestHandleATR:
    """Tests for the handle_atr function."""
    
    def test_valid_input(self, sample_price_data):
        """Test with valid input data."""
        # Create a mock indicator function that returns a predictable Series
        mock_atr_series = pd.Series([5.0] * len(sample_price_data), index=sample_price_data.index)
        mock_atr_func = MagicMock(return_value=mock_atr_series)
        
        # Call the handler
        result = handle_atr(sample_price_data, mock_atr_func, window=14)
        
        # Check the result
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        assert result.iloc[0] == 5.0
        
        # Verify the mock was called with the correct arguments
        mock_atr_func.assert_called_once()
        args, kwargs = mock_atr_func.call_args
        assert kwargs['window'] == 14
    
    def test_missing_columns(self, sample_price_data):
        """Test that ValueError is raised when required columns are missing."""
        # Create a DataFrame missing the 'High' column
        df_no_high = sample_price_data.drop(columns=['High'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High', 'Low', and 'Close' columns"):
            handle_atr(df_no_high, MagicMock())
        
        # Create a DataFrame missing the 'Low' column
        df_no_low = sample_price_data.drop(columns=['Low'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High', 'Low', and 'Close' columns"):
            handle_atr(df_no_low, MagicMock())
        
        # Create a DataFrame missing the 'Close' column
        df_no_close = sample_price_data.drop(columns=['Close'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High', 'Low', and 'Close' columns"):
            handle_atr(df_no_close, MagicMock())
    
    def test_default_parameters(self, sample_price_data):
        """Test that default parameters are correctly applied."""
        # Create a mock indicator function
        mock_atr_func = MagicMock(return_value=pd.Series())
        
        # Call without specifying window
        result = handle_atr(sample_price_data, mock_atr_func)
        
        # Verify the mock was called with default parameters
        args, kwargs = mock_atr_func.call_args
        assert kwargs['window'] == 14  # Default window
    
    def test_remove_hlc_kwargs(self, sample_price_data):
        """Test that high, low, close kwargs are removed if passed."""
        # Create a mock indicator function
        mock_atr_func = MagicMock(return_value=pd.Series())
        
        # Call with high, low, close in kwargs (should be removed)
        result = handle_atr(sample_price_data, mock_atr_func, high="High", low="Low", close="Close")
        
        # Verify the mock was called without high, low, close in kwargs
        args, kwargs = mock_atr_func.call_args
        assert 'high' not in kwargs
        assert 'low' not in kwargs
        assert 'close' not in kwargs


class TestHandleKeltnerChannels:
    """Tests for the handle_kelt function."""
    
    def test_valid_input(self, sample_price_data):
        """Test with valid input data."""
        # Create a mock result for Keltner Channels
        mock_kelt_result = pd.DataFrame({
            'KELT_UPPER': [115.0] * len(sample_price_data),
            'KELT_MIDDLE': [105.0] * len(sample_price_data),
            'KELT_LOWER': [95.0] * len(sample_price_data)
        }, index=sample_price_data.index)
        
        # Create a mock indicator function
        mock_kelt_func = MagicMock(return_value=mock_kelt_result)
        
        # Call the handler
        result = handle_kelt(sample_price_data, mock_kelt_func, ema_window=20, atr_window=10, atr_multiplier=2.0)
        
        # Check the result
        assert isinstance(result, pd.DataFrame)
        assert 'KELT_UPPER' in result.columns
        assert 'KELT_MIDDLE' in result.columns
        assert 'KELT_LOWER' in result.columns
        
        # Verify the mock was called with the correct arguments
        mock_kelt_func.assert_called_once()
        args, kwargs = mock_kelt_func.call_args
        assert kwargs['ema_window'] == 20
        assert kwargs['atr_window'] == 10
        assert kwargs['atr_multiplier'] == 2.0
    
    def test_missing_columns(self, sample_price_data):
        """Test that ValueError is raised when required columns are missing."""
        # Create a DataFrame missing the 'High' column
        df_no_high = sample_price_data.drop(columns=['High'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High', 'Low', and 'Close' columns"):
            handle_kelt(df_no_high, MagicMock())
        
        # Create a DataFrame missing the 'Low' column
        df_no_low = sample_price_data.drop(columns=['Low'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High', 'Low', and 'Close' columns"):
            handle_kelt(df_no_low, MagicMock())
        
        # Create a DataFrame missing the 'Close' column
        df_no_close = sample_price_data.drop(columns=['Close'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High', 'Low', and 'Close' columns"):
            handle_kelt(df_no_close, MagicMock())
    
    def test_default_parameters(self, sample_price_data):
        """Test that default parameters are correctly applied."""
        # Create a mock indicator function
        mock_kelt_func = MagicMock(return_value=pd.DataFrame())
        
        # Call without specifying parameters
        result = handle_kelt(sample_price_data, mock_kelt_func)
        
        # Verify the mock was called with default parameters
        args, kwargs = mock_kelt_func.call_args
        assert kwargs['ema_window'] == 20       # Default EMA window
        assert kwargs['atr_window'] == 10       # Default ATR window
        assert kwargs['atr_multiplier'] == 2.0  # Default multiplier
    
    def test_remove_hlc_kwargs(self, sample_price_data):
        """Test that high, low, close kwargs are removed if passed."""
        # Create a mock indicator function
        mock_kelt_func = MagicMock(return_value=pd.DataFrame())
        
        # Call with high, low, close in kwargs (should be removed)
        result = handle_kelt(sample_price_data, mock_kelt_func, high="High", low="Low", close="Close")
        
        # Verify the mock was called without high, low, close in kwargs
        args, kwargs = mock_kelt_func.call_args
        assert 'high' not in kwargs
        assert 'low' not in kwargs
        assert 'close' not in kwargs


class TestHandleDonchianChannels:
    """Tests for the handle_donch function."""
    
    def test_valid_input(self, sample_price_data):
        """Test with valid input data."""
        # Create a mock result for Donchian Channels
        mock_donch_result = pd.DataFrame({
            'DONCH_UPPER': [115.0] * len(sample_price_data),
            'DONCH_MIDDLE': [105.0] * len(sample_price_data),
            'DONCH_LOWER': [95.0] * len(sample_price_data)
        }, index=sample_price_data.index)
        
        # Create a mock indicator function
        mock_donch_func = MagicMock(return_value=mock_donch_result)
        
        # Call the handler
        result = handle_donch(sample_price_data, mock_donch_func, window=20)
        
        # Check the result
        assert isinstance(result, pd.DataFrame)
        assert 'DONCH_UPPER' in result.columns
        assert 'DONCH_MIDDLE' in result.columns
        assert 'DONCH_LOWER' in result.columns
        
        # Verify the mock was called with the correct arguments
        mock_donch_func.assert_called_once()
        args, kwargs = mock_donch_func.call_args
        assert kwargs['window'] == 20
    
    def test_missing_columns(self, sample_price_data):
        """Test that ValueError is raised when required columns are missing."""
        # Create a DataFrame missing the 'High' column
        df_no_high = sample_price_data.drop(columns=['High'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High' and 'Low' columns"):
            handle_donch(df_no_high, MagicMock())
        
        # Create a DataFrame missing the 'Low' column
        df_no_low = sample_price_data.drop(columns=['Low'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High' and 'Low' columns"):
            handle_donch(df_no_low, MagicMock())
    
    def test_default_parameters(self, sample_price_data):
        """Test that default parameters are correctly applied."""
        # Create a mock indicator function
        mock_donch_func = MagicMock(return_value=pd.DataFrame())
        
        # Call without specifying parameters
        result = handle_donch(sample_price_data, mock_donch_func)
        
        # Verify the mock was called with default parameters
        args, kwargs = mock_donch_func.call_args
        assert kwargs['window'] == 20  # Default window
    
    def test_remove_hl_kwargs(self, sample_price_data):
        """Test that high, low kwargs are removed if passed."""
        # Create a mock indicator function
        mock_donch_func = MagicMock(return_value=pd.DataFrame())
        
        # Call with high, low in kwargs (should be removed)
        result = handle_donch(sample_price_data, mock_donch_func, high="High", low="Low")
        
        # Verify the mock was called without high, low in kwargs
        args, kwargs = mock_donch_func.call_args
        assert 'high' not in kwargs
        assert 'low' not in kwargs


class TestHandleChaikinVolatility:
    """Tests for the handle_chaik function."""
    
    def test_valid_input(self, sample_price_data):
        """Test with valid input data."""
        # Create a mock result for Chaikin Volatility
        mock_chaik_series = pd.Series([3.5] * len(sample_price_data), index=sample_price_data.index)
        mock_chaik_func = MagicMock(return_value=mock_chaik_series)
        
        # Call the handler
        result = handle_chaik(sample_price_data, mock_chaik_func, ema_window=10, roc_window=10)
        
        # Check the result
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        assert result.iloc[0] == 3.5
        
        # Verify the mock was called with the correct arguments
        mock_chaik_func.assert_called_once()
        args, kwargs = mock_chaik_func.call_args
        assert kwargs['ema_window'] == 10
        assert kwargs['roc_window'] == 10
    
    def test_missing_columns(self, sample_price_data):
        """Test that ValueError is raised when required columns are missing."""
        # Create a DataFrame missing the 'High' column
        df_no_high = sample_price_data.drop(columns=['High'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High' and 'Low' columns"):
            handle_chaik(df_no_high, MagicMock())
        
        # Create a DataFrame missing the 'Low' column
        df_no_low = sample_price_data.drop(columns=['Low'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High' and 'Low' columns"):
            handle_chaik(df_no_low, MagicMock())
    
    def test_default_parameters(self, sample_price_data):
        """Test that default parameters are correctly applied."""
        # Create a mock indicator function
        mock_chaik_func = MagicMock(return_value=pd.Series())
        
        # Call without specifying parameters
        result = handle_chaik(sample_price_data, mock_chaik_func)
        
        # Verify the mock was called with default parameters
        args, kwargs = mock_chaik_func.call_args
        assert kwargs['ema_window'] == 10  # Default EMA window
        assert kwargs['roc_window'] == 10  # Default ROC window
    
    def test_remove_hl_kwargs(self, sample_price_data):
        """Test that high, low kwargs are removed if passed."""
        # Create a mock indicator function
        mock_chaik_func = MagicMock(return_value=pd.Series())
        
        # Call with high, low in kwargs (should be removed)
        result = handle_chaik(sample_price_data, mock_chaik_func, high="High", low="Low")
        
        # Verify the mock was called without high, low in kwargs
        args, kwargs = mock_chaik_func.call_args
        assert 'high' not in kwargs
        assert 'low' not in kwargs