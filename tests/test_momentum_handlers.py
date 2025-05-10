import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from simple_trade.data.momentum_handlers import (
    handle_stochastic,
    handle_cci,
    handle_roc,
    handle_macd,
    handle_rsi,
)

# --- Fixtures ---

@pytest.fixture
def sample_price_data():
    """Fixture providing a basic OHLCV DataFrame for testing momentum indicators."""
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
    data = pd.DataFrame(index=dates)
    data['Open'] = [100, 102, 104, 103, 105, 107, 108, 109, 110, 111, 112, 111, 110, 112, 114, 115, 116, 115, 113, 114]
    data['High'] = [103, 105, 107, 106, 108, 110, 111, 112, 113, 114, 115, 114, 113, 115, 117, 118, 119, 118, 116, 117]
    data['Low'] = [98, 100, 102, 101, 103, 105, 106, 107, 108, 109, 110, 109, 108, 110, 112, 113, 114, 113, 111, 112]
    data['Close'] = [102, 104, 106, 105, 107, 109, 110, 111, 112, 113, 114, 113, 112, 114, 116, 117, 118, 117, 115, 116]
    data['Volume'] = [1000, 1200, 1300, 1100, 1400, 1500, 1600, 1500, 1400, 1600, 1700, 1500, 1400, 1600, 1800, 1900, 2000, 1800, 1700, 1900]
    return data

# --- Test Classes ---

class TestHandleStochastic:
    """Tests for the handle_stochastic function."""
    
    def test_valid_input(self, sample_price_data):
        """Test with valid input data."""
        # Create a mock result for Stochastic Oscillator
        mock_stoch_result = pd.DataFrame({
            'STOCHk': [80.0] * len(sample_price_data),
            'STOCHd': [70.0] * len(sample_price_data)
        }, index=sample_price_data.index)
        
        # Create a mock indicator function
        mock_stoch_func = MagicMock(return_value=mock_stoch_result)
        
        # Call the handler
        result = handle_stochastic(sample_price_data, mock_stoch_func, k_period=14, d_period=3, smooth_k=3)
        
        # Check the result
        assert isinstance(result, pd.DataFrame)
        assert 'STOCHk' in result.columns
        assert 'STOCHd' in result.columns
        
        # Verify the mock was called with the correct arguments
        mock_stoch_func.assert_called_once()
        args, kwargs = mock_stoch_func.call_args
        assert kwargs['k_period'] == 14
        assert kwargs['d_period'] == 3
        assert kwargs['smooth_k'] == 3
    
    def test_missing_columns(self, sample_price_data):
        """Test that ValueError is raised when required columns are missing."""
        # Create a DataFrame missing the 'High' column
        df_no_high = sample_price_data.drop(columns=['High'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High', 'Low', and 'Close' columns."):
            handle_stochastic(df_no_high, MagicMock())
        
        # Create a DataFrame missing the 'Low' column
        df_no_low = sample_price_data.drop(columns=['Low'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High', 'Low', and 'Close' columns."):
            handle_stochastic(df_no_low, MagicMock())
        
        # Create a DataFrame missing the 'Close' column
        df_no_close = sample_price_data.drop(columns=['Close'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High', 'Low', and 'Close' columns."):
            handle_stochastic(df_no_close, MagicMock())
    
    def test_default_parameters(self, sample_price_data):
        """Test that default parameters are correctly applied."""
        # Create a mock indicator function
        mock_stoch_func = MagicMock(return_value=pd.DataFrame())
        
        # Call without specifying parameters
        result = handle_stochastic(sample_price_data, mock_stoch_func)
        
        # Verify the mock was called with default parameters
        args, kwargs = mock_stoch_func.call_args
        assert kwargs['k_period'] == 14  # Default k_period
        assert kwargs['d_period'] == 3   # Default d_period
        assert kwargs['smooth_k'] == 3   # Default smooth_k
    
    def test_remove_hlc_kwargs(self, sample_price_data):
        """Test that high, low, close kwargs are removed if passed."""
        # Create a mock indicator function
        mock_stoch_func = MagicMock(return_value=pd.DataFrame())
        
        # Call with high, low, close in kwargs (should be removed)
        result = handle_stochastic(sample_price_data, mock_stoch_func, high="High", low="Low", close="Close")
        
        # Verify the mock was called without high, low, close in kwargs
        args, kwargs = mock_stoch_func.call_args
        assert 'high' not in kwargs
        assert 'low' not in kwargs
        assert 'close' not in kwargs


class TestHandleCCI:
    """Tests for the handle_cci function."""
    
    def test_valid_input(self, sample_price_data):
        """Test with valid input data."""
        # Create a mock result for CCI
        mock_cci_series = pd.Series([120.0] * len(sample_price_data), index=sample_price_data.index)
        mock_cci_func = MagicMock(return_value=mock_cci_series)
        
        # Call the handler
        result = handle_cci(sample_price_data, mock_cci_func, window=20, constant=0.015)
        
        # Check the result
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        assert result.iloc[0] == 120.0
        
        # Verify the mock was called with the correct arguments
        mock_cci_func.assert_called_once()
        args, kwargs = mock_cci_func.call_args
        assert kwargs['window'] == 20
        assert kwargs['constant'] == 0.015
    
    def test_missing_columns(self, sample_price_data):
        """Test that ValueError is raised when required columns are missing."""
        # Create a DataFrame missing the 'High' column
        df_no_high = sample_price_data.drop(columns=['High'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High', 'Low', and 'Close' columns."):
            handle_cci(df_no_high, MagicMock())
        
        # Create a DataFrame missing the 'Low' column
        df_no_low = sample_price_data.drop(columns=['Low'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High', 'Low', and 'Close' columns."):
            handle_cci(df_no_low, MagicMock())
        
        # Create a DataFrame missing the 'Close' column
        df_no_close = sample_price_data.drop(columns=['Close'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'High', 'Low', and 'Close' columns."):
            handle_cci(df_no_close, MagicMock())
    
    def test_default_parameters(self, sample_price_data):
        """Test that default parameters are correctly applied."""
        # Create a mock indicator function
        mock_cci_func = MagicMock(return_value=pd.Series())
        
        # Call without specifying parameters
        result = handle_cci(sample_price_data, mock_cci_func)
        
        # Verify the mock was called with default parameters
        args, kwargs = mock_cci_func.call_args
        assert kwargs['window'] == 20       # Default window
        assert kwargs['constant'] == 0.015  # Default constant
    
    def test_remove_hlc_kwargs(self, sample_price_data):
        """Test that high, low, close kwargs are removed if passed."""
        # Create a mock indicator function
        mock_cci_func = MagicMock(return_value=pd.Series())
        
        # Call with high, low, close in kwargs (should be removed)
        result = handle_cci(sample_price_data, mock_cci_func, high="High", low="Low", close="Close")
        
        # Verify the mock was called without high, low, close in kwargs
        args, kwargs = mock_cci_func.call_args
        assert 'high' not in kwargs
        assert 'low' not in kwargs
        assert 'close' not in kwargs


class TestHandleROC:
    """Tests for the handle_roc function."""
    
    def test_valid_input(self, sample_price_data):
        """Test with valid input data."""
        # Create a mock result for ROC
        mock_roc_series = pd.Series([5.0] * len(sample_price_data), index=sample_price_data.index)
        mock_roc_func = MagicMock(return_value=mock_roc_series)
        
        # Call the handler
        result = handle_roc(sample_price_data, mock_roc_func, window=12)
        
        # Check the result
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        assert result.iloc[0] == 5.0
        
        # Verify the mock was called with the correct arguments
        mock_roc_func.assert_called_once()
        args, kwargs = mock_roc_func.call_args
        assert kwargs['window'] == 12
    
    def test_missing_close_column(self, sample_price_data):
        """Test that ValueError is raised when Close column is missing."""
        # Create a DataFrame missing the 'Close' column
        df_no_close = sample_price_data.drop(columns=['Close'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'Close' column."):
            handle_roc(df_no_close, MagicMock())
    
    def test_default_parameters(self, sample_price_data):
        """Test that default parameters are correctly applied."""
        # Create a mock indicator function
        mock_roc_func = MagicMock(return_value=pd.Series())
        
        # Call without specifying parameters
        result = handle_roc(sample_price_data, mock_roc_func)
        
        # Verify the mock was called with default parameters
        args, kwargs = mock_roc_func.call_args
        assert kwargs['window'] == 12  # Default window


class TestHandleMACD:
    """Tests for the handle_macd function."""
    
    def test_valid_input(self, sample_price_data):
        """Test with valid input data."""
        # Create a mock result for MACD (DataFrame with multiple components)
        mock_macd_result = pd.DataFrame({
            'MACD': [2.0] * len(sample_price_data),
            'MACD_Signal': [1.5] * len(sample_price_data),
            'MACD_Hist': [0.5] * len(sample_price_data)
        }, index=sample_price_data.index)
        
        # Create a mock indicator function
        mock_macd_func = MagicMock(return_value=mock_macd_result)
        
        # Call the handler
        result = handle_macd(sample_price_data, mock_macd_func, window_fast=12, window_slow=26, window_signal=9)
        
        # Check the result
        assert isinstance(result, pd.DataFrame)
        assert 'MACD' in result.columns
        assert 'MACD_Signal' in result.columns
        assert 'MACD_Hist' in result.columns
        
        # Verify the mock was called with the correct arguments
        mock_macd_func.assert_called_once()
        args, kwargs = mock_macd_func.call_args
        assert kwargs['window_fast'] == 12
        assert kwargs['window_slow'] == 26
        assert kwargs['window_signal'] == 9
    
    def test_missing_close_column(self, sample_price_data):
        """Test that ValueError is raised when Close column is missing."""
        # Create a DataFrame missing the 'Close' column
        df_no_close = sample_price_data.drop(columns=['Close'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'Close' column."):
            handle_macd(df_no_close, MagicMock())
    
    def test_default_parameters(self, sample_price_data):
        """Test that default parameters are correctly applied."""
        # Create a mock indicator function
        mock_macd_func = MagicMock(return_value=pd.DataFrame())
        
        # Call without specifying parameters
        result = handle_macd(sample_price_data, mock_macd_func)
        
        # Verify the mock was called with default parameters
        args, kwargs = mock_macd_func.call_args
        assert kwargs['window_fast'] == 12     # Default fast window
        assert kwargs['window_slow'] == 26     # Default slow window
        assert kwargs['window_signal'] == 9    # Default signal window


class TestHandleRSI:
    """Tests for the handle_rsi function."""
    
    def test_valid_input(self, sample_price_data):
        """Test with valid input data."""
        # Create a mock result for RSI
        mock_rsi_series = pd.Series([65.0] * len(sample_price_data), index=sample_price_data.index)
        mock_rsi_func = MagicMock(return_value=mock_rsi_series)
        
        # Call the handler
        result = handle_rsi(sample_price_data, mock_rsi_func, window=14)
        
        # Check the result
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        assert result.iloc[0] == 65.0
        
        # Verify the mock was called with the correct arguments
        mock_rsi_func.assert_called_once()
        args, kwargs = mock_rsi_func.call_args
        assert kwargs['window'] == 14
    
    def test_missing_close_column(self, sample_price_data):
        """Test that ValueError is raised when Close column is missing."""
        # Create a DataFrame missing the 'Close' column
        df_no_close = sample_price_data.drop(columns=['Close'])
        
        with pytest.raises(ValueError, match="DataFrame must contain 'Close' column."):
            handle_rsi(df_no_close, MagicMock())
    
    def test_default_parameters(self, sample_price_data):
        """Test that default parameters are correctly applied."""
        # Create a mock indicator function
        mock_rsi_func = MagicMock(return_value=pd.Series())
        
        # Call without specifying parameters
        result = handle_rsi(sample_price_data, mock_rsi_func)
        
        # Verify the mock was called with default parameters
        args, kwargs = mock_rsi_func.call_args
        assert kwargs['window'] == 14  # Default window