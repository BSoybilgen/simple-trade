import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from simple_trade.data.volume_handlers import (
    handle_obv,
    handle_vma,
    handle_adline,
    handle_cmf,
    handle_vpt,
)

# --- Fixtures ---

@pytest.fixture
def sample_price_data():
    """Fixture providing a basic OHLCV DataFrame for testing volume indicators."""
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
    data = pd.DataFrame(index=dates)
    data['Open'] = [100, 102, 104, 103, 105, 107, 108, 109, 110, 111, 112, 111, 110, 112, 114, 115, 116, 115, 113, 114]
    data['High'] = [103, 105, 107, 106, 108, 110, 111, 112, 113, 114, 115, 114, 113, 115, 117, 118, 119, 118, 116, 117]
    data['Low'] = [98, 100, 102, 101, 103, 105, 106, 107, 108, 109, 110, 109, 108, 110, 112, 113, 114, 113, 111, 112]
    data['Close'] = [102, 104, 106, 105, 107, 109, 110, 111, 112, 113, 114, 113, 112, 114, 116, 117, 118, 117, 115, 116]
    data['Volume'] = [1000, 1200, 1300, 1100, 1400, 1500, 1600, 1500, 1400, 1600, 1700, 1500, 1400, 1600, 1800, 1900, 2000, 1800, 1700, 1900]
    return data

# --- Test Classes ---

class TestHandleOBV:
    """Tests for the handle_obv function."""
    
    def test_valid_input(self, sample_price_data):
        """Test with valid input data."""
        # Create a mock result for OBV
        mock_obv_series = pd.Series([5000.0] * len(sample_price_data), index=sample_price_data.index)
        mock_obv_func = MagicMock(return_value=mock_obv_series)
        
        # Call the handler
        result = handle_obv(sample_price_data, mock_obv_func)
        
        # Check the result
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        assert result.iloc[0] == 5000.0
        
        # Verify the mock was called with the correct arguments
        mock_obv_func.assert_called_once()
        args, kwargs = mock_obv_func.call_args
        assert args[0].equals(sample_price_data) # Check DataFrame is passed
        assert kwargs.get('close_col') == 'Close'
        assert kwargs.get('volume_col') == 'Volume'
    
    def test_missing_columns(self, sample_price_data):
        """Test that ValueError is raised when required columns are missing."""
        # Create a DataFrame missing the 'Close' column
        df_no_close = sample_price_data.drop(columns=['Close'])
        
        with pytest.raises(ValueError, match=r"DataFrame must contain 'Close' and 'Volume' columns\."):
            handle_obv(df_no_close, MagicMock())
        
        # Create a DataFrame missing the 'Volume' column
        df_no_volume = sample_price_data.drop(columns=['Volume'])
        
        with pytest.raises(ValueError, match=r"DataFrame must contain 'Close' and 'Volume' columns\."):
            handle_obv(df_no_volume, MagicMock())
    
    def test_remove_column_kwargs(self, sample_price_data):
        """Test that close and volume kwargs are removed if passed."""
        # Create a mock indicator function
        mock_obv_func = MagicMock(return_value=pd.Series())
        
        # Call with close, volume in kwargs (should be removed)
        result = handle_obv(sample_price_data, mock_obv_func, close="Close", volume="Volume")
        
        # Verify the mock was called without close, volume in kwargs
        args, kwargs = mock_obv_func.call_args
        assert 'close' not in kwargs
        assert 'volume' not in kwargs


class TestHandleVMA:
    """Tests for the handle_vma function."""
    
    def test_valid_input(self, sample_price_data):
        """Test with valid input data."""
        # Create a mock result for VMA
        mock_vma_series = pd.Series([1500.0] * len(sample_price_data), index=sample_price_data.index)
        mock_vma_func = MagicMock(return_value=mock_vma_series)
        
        # Call the handler
        result = handle_vma(sample_price_data, mock_vma_func, window=14)
        
        # Check the result
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        assert result.iloc[0] == 1500.0
        
        # Verify the mock was called with the correct arguments
        mock_vma_func.assert_called_once()
        args, kwargs = mock_vma_func.call_args
        assert args[0].equals(sample_price_data)
        assert kwargs.get('window') == 14 # Default window
        assert kwargs.get('close_col') == 'Close'
        assert kwargs.get('volume_col') == 'Volume'

    def test_missing_columns(self, sample_price_data):
        """Test that ValueError is raised when required columns are missing."""
        # Create a DataFrame missing the 'Close' column
        df_no_close = sample_price_data.drop(columns=['Close'])
        
        with pytest.raises(ValueError, match=r"DataFrame must contain 'Close' and 'Volume' columns\."):
            handle_vma(df_no_close, MagicMock())
        
        # Create a DataFrame missing the 'Volume' column
        df_no_volume = sample_price_data.drop(columns=['Volume'])
        
        with pytest.raises(ValueError, match=r"DataFrame must contain 'Close' and 'Volume' columns\."):
            handle_vma(df_no_volume, MagicMock())
    
    def test_default_parameters(self, sample_price_data):
        """Test that default parameters are correctly applied."""
        # Create a mock indicator function
        mock_vma_func = MagicMock(return_value=pd.Series())
        
        # Call without specifying window
        result = handle_vma(sample_price_data, mock_vma_func)
        
        # Verify the mock was called with default parameters
        args, kwargs = mock_vma_func.call_args
        assert kwargs['window'] == 14  # Default window
    
    def test_remove_column_kwargs(self, sample_price_data):
        """Test that close and volume kwargs are removed if passed."""
        # Create a mock indicator function
        mock_vma_func = MagicMock(return_value=pd.Series())
        
        # Call with close, volume in kwargs (should be removed)
        result = handle_vma(sample_price_data, mock_vma_func, close="Close", volume="Volume")
        
        # Verify the mock was called without close, volume in kwargs
        args, kwargs = mock_vma_func.call_args
        assert 'close' not in kwargs
        assert 'volume' not in kwargs


class TestHandleADLine:
    """Tests for the handle_adline function."""
    
    def test_valid_input(self, sample_price_data):
        """Test with valid input data."""
        # Create a mock result for A/D Line
        mock_adline_series = pd.Series([8000.0] * len(sample_price_data), index=sample_price_data.index)
        mock_adline_func = MagicMock(return_value=mock_adline_series)
        
        # Call the handler
        result = handle_adline(sample_price_data, mock_adline_func)
        
        # Check the result
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        assert result.iloc[0] == 8000.0
        
        # Verify the mock was called with the correct arguments
        mock_adline_func.assert_called_once()
        args, kwargs = mock_adline_func.call_args
        assert args[0].equals(sample_price_data)
        assert kwargs.get('high_col') == 'High'
        assert kwargs.get('low_col') == 'Low'
        assert kwargs.get('close_col') == 'Close'
        assert kwargs.get('volume_col') == 'Volume'

    def test_missing_columns(self, sample_price_data):
        """Test that ValueError is raised when required columns are missing."""
        # Test missing 'High' column
        df_no_high = sample_price_data.drop(columns=['High'])
        with pytest.raises(ValueError, match=r"DataFrame must contain 'High', 'Low', 'Close' and 'Volume' columns\."):
            handle_adline(df_no_high, MagicMock())

        # Test missing 'Volume' column
        df_no_volume = sample_price_data.drop(columns=['Volume'])
        with pytest.raises(ValueError, match=r"DataFrame must contain 'High', 'Low', 'Close' and 'Volume' columns\."):
            handle_adline(df_no_volume, MagicMock())
        
        # Test missing 'Low' column
        df_no_low = sample_price_data.drop(columns=['Low'])
        with pytest.raises(ValueError, match=r"DataFrame must contain 'High', 'Low', 'Close' and 'Volume' columns\."):
            handle_adline(df_no_low, MagicMock())
        
        # Test missing 'Close' column
        df_no_close = sample_price_data.drop(columns=['Close'])
        with pytest.raises(ValueError, match=r"DataFrame must contain 'High', 'Low', 'Close' and 'Volume' columns\."):
            handle_adline(df_no_close, MagicMock())
    
    def test_remove_column_kwargs(self, sample_price_data):
        """Test that high, low, close, volume kwargs are removed if passed."""
        # Create a mock indicator function
        mock_adline_func = MagicMock(return_value=pd.Series())
        
        # Call with column names in kwargs (should be removed)
        result = handle_adline(sample_price_data, mock_adline_func, high="High", low="Low", close="Close", volume="Volume")
        
        # Verify the mock was called without column names in kwargs
        args, kwargs = mock_adline_func.call_args
        assert 'high' not in kwargs
        assert 'low' not in kwargs
        assert 'close' not in kwargs
        assert 'volume' not in kwargs


class TestHandleCMF:
    """Tests for the handle_cmf function."""
    
    def test_valid_input(self, sample_price_data):
        """Test with valid input data."""
        # Create a mock result for CMF
        mock_cmf_series = pd.Series([0.25] * len(sample_price_data), index=sample_price_data.index)
        mock_cmf_func = MagicMock(return_value=mock_cmf_series)
        
        # Call the handler
        result = handle_cmf(sample_price_data, mock_cmf_func, period=20)
        
        # Check the result
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        assert result.iloc[0] == 0.25
        
        # Verify the mock was called with the correct arguments
        mock_cmf_func.assert_called_once()
        args, kwargs = mock_cmf_func.call_args
        assert args[0].equals(sample_price_data)
        assert kwargs.get('period') == 20 # Default period
        assert kwargs.get('high_col') == 'High'
        assert kwargs.get('low_col') == 'Low'
        assert kwargs.get('close_col') == 'Close'
        assert kwargs.get('volume_col') == 'Volume'

    def test_missing_columns(self, sample_price_data):
        """Test that ValueError is raised when required columns are missing."""
        # Create a DataFrame missing the 'High' column
        df_no_high = sample_price_data.drop(columns=['High'])
        
        with pytest.raises(ValueError, match=r"DataFrame must contain 'High', 'Low', 'Close' and 'Volume' columns\."):
            handle_cmf(df_no_high, MagicMock())
        
        # Create a DataFrame missing the 'Low' column
        df_no_low = sample_price_data.drop(columns=['Low'])
        
        with pytest.raises(ValueError, match=r"DataFrame must contain 'High', 'Low', 'Close' and 'Volume' columns\."):
            handle_cmf(df_no_low, MagicMock())
        
        # Create a DataFrame missing the 'Close' column
        df_no_close = sample_price_data.drop(columns=['Close'])
        
        with pytest.raises(ValueError, match=r"DataFrame must contain 'High', 'Low', 'Close' and 'Volume' columns\."):
            handle_cmf(df_no_close, MagicMock())
        
        # Create a DataFrame missing the 'Volume' column
        df_no_volume = sample_price_data.drop(columns=['Volume'])
        
        with pytest.raises(ValueError, match=r"DataFrame must contain 'High', 'Low', 'Close' and 'Volume' columns\."):
            handle_cmf(df_no_volume, MagicMock())
    
    def test_default_parameters(self, sample_price_data):
        """Test that default parameters are correctly applied."""
        # Create a mock indicator function
        mock_cmf_func = MagicMock(return_value=pd.Series())
        
        # Call without specifying period
        result = handle_cmf(sample_price_data, mock_cmf_func)
        
        # Verify the mock was called with default parameters
        args, kwargs = mock_cmf_func.call_args
        assert kwargs['period'] == 20  # Default period
    
    def test_remove_column_kwargs(self, sample_price_data):
        """Test that high, low, close, volume kwargs are removed if passed."""
        # Create a mock indicator function
        mock_cmf_func = MagicMock(return_value=pd.Series())
        
        # Call with column names in kwargs (should be removed)
        result = handle_cmf(sample_price_data, mock_cmf_func, high="High", low="Low", close="Close", volume="Volume")
        
        # Verify the mock was called without column names in kwargs
        args, kwargs = mock_cmf_func.call_args
        assert 'high' not in kwargs
        assert 'low' not in kwargs
        assert 'close' not in kwargs
        assert 'volume' not in kwargs


class TestHandleVPT:
    """Tests for the handle_vpt function."""
    
    def test_valid_input(self, sample_price_data):
        """Test with valid input data."""
        # Create a mock result for VPT
        mock_vpt_series = pd.Series([3000.0] * len(sample_price_data), index=sample_price_data.index)
        mock_vpt_func = MagicMock(return_value=mock_vpt_series)
        
        # Call the handler
        result = handle_vpt(sample_price_data, mock_vpt_func)
        
        # Check the result
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        assert result.iloc[0] == 3000.0
        
        # Verify the mock was called with the correct arguments
        mock_vpt_func.assert_called_once()
        args, kwargs = mock_vpt_func.call_args
        assert args[0].equals(sample_price_data) # Check DataFrame
        assert kwargs.get('close_col') == 'Close'
        assert kwargs.get('volume_col') == 'Volume'

    def test_missing_columns(self, sample_price_data):
        """Test that ValueError is raised when required columns are missing."""
        # Create a DataFrame missing the 'Close' column
        df_no_close = sample_price_data.drop(columns=['Close'])
        
        with pytest.raises(ValueError, match=r"DataFrame must contain 'Close' and 'Volume' columns\."):
            handle_vpt(df_no_close, MagicMock())
        
        # Create a DataFrame missing the 'Volume' column
        df_no_volume = sample_price_data.drop(columns=['Volume'])
        
        with pytest.raises(ValueError, match=r"DataFrame must contain 'Close' and 'Volume' columns\."):
            handle_vpt(df_no_volume, MagicMock())
    
    def test_remove_column_kwargs(self, sample_price_data):
        """Test that close and volume kwargs are removed if passed."""
        # Create a mock indicator function
        mock_vpt_func = MagicMock(return_value=pd.Series())
        
        # Call with close, volume in kwargs (should be removed)
        result = handle_vpt(sample_price_data, mock_vpt_func, close="Close", volume="Volume")
        
        # Verify the mock was called without close, volume in kwargs
        args, kwargs = mock_vpt_func.call_args
        assert 'close' not in kwargs
        assert 'volume' not in kwargs