import pytest
import pandas as pd
import numpy as np
from simple_trade.volatility import (
    bollinger_bands, atr, keltner_channels, donchian_channels, chaikin_volatility
)
from simple_trade.trend import ema # Needed for Chaikin

# Fixture for sample data (consistent with other test modules)
@pytest.fixture
def sample_data():
    """Fixture to provide sample OHLC data for testing volatility indicators"""
    index = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42) # for reproducibility

    # Create a series with varying volatility
    base = np.linspace(100, 110, 50)
    # Add a period of higher volatility
    high_vol = base + np.random.normal(0, 5, 50)
    # Add a period of lower volatility
    low_vol = base + np.random.normal(0, 1, 50)
    
    close = pd.Series(np.concatenate([high_vol, low_vol]), index=index)

    # Create high and low with spread reflecting volatility
    high_vol_spread = np.random.uniform(2, 8, size=50)
    low_vol_spread = np.random.uniform(0.5, 2, size=50)
    spread = np.concatenate([high_vol_spread, low_vol_spread])
    
    high = close + spread / 2
    low = close - spread / 2

    # Ensure low is not higher than close and high is not lower than close
    low = pd.Series(np.minimum(low.values, close.values - 0.1), index=index)
    high = pd.Series(np.maximum(high.values, close.values + 0.1), index=index)

    return {
        'high': high,
        'low': low,
        'close': close
    }

class TestBollingerBands:
    """Tests for Bollinger Bands"""

    def test_bb_calculation(self, sample_data):
        """Test basic Bollinger Bands calculation structure"""
        window=20
        num_std=2
        # Create a DataFrame with 'Close' column
        df = pd.DataFrame({'Close': sample_data['close']})
        result = bollinger_bands(df, window=window, num_std=num_std)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) == len(sample_data['close'])
        assert result.index.equals(sample_data['close'].index)
        
        # Check columns
        expected_cols = [f'BB_Middle_{window}', f'BB_Upper_{window}_{num_std}', f'BB_Lower_{window}_{num_std}']
        assert all(col in result.columns for col in expected_cols)
        
        # Check initial NaNs (first window-1)
        assert result.iloc[:window-1].isna().all().all() # Check all columns are NaN initially
        assert not result.iloc[window-1:].isna().any().any() # Check no NaNs after window
        
        # Check band properties on non-NaN data
        valid_result = result.dropna()
        assert (valid_result[expected_cols[1]] >= valid_result[expected_cols[0]]).all() # Upper >= Middle
        assert (valid_result[expected_cols[0]] >= valid_result[expected_cols[2]]).all() # Middle >= Lower

    def test_bb_custom_params(self, sample_data):
        """Test Bollinger Bands with custom parameters"""
        window = 10
        num_std = 3
        # Create a DataFrame with 'Close' column
        df = pd.DataFrame({'Close': sample_data['close']})
        result = bollinger_bands(df, window=window, num_std=num_std)
        
        assert isinstance(result, pd.DataFrame)
        expected_cols = [f'BB_Middle_{window}', f'BB_Upper_{window}_{num_std}', f'BB_Lower_{window}_{num_std}']
        assert all(col in result.columns for col in expected_cols)
        assert len(result) == len(sample_data['close'])
        assert result.iloc[:window-1].isna().all().all()
        assert not result.iloc[window-1:].isna().any().any()
        
        # Check band properties on non-NaN data
        valid_result = result.dropna()
        assert (valid_result[expected_cols[1]] >= valid_result[expected_cols[0]]).all() # Upper >= Middle
        assert (valid_result[expected_cols[0]] >= valid_result[expected_cols[2]]).all() # Middle >= Lower


class TestATR:
    """Tests for Average True Range (ATR)"""

    def test_atr_calculation(self, sample_data):
        """Test basic ATR calculation structure"""
        window = 14 # Default
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result = atr(df, window=window)
        
        assert isinstance(result, pd.Series)
        assert not result.empty
        assert len(result) == len(sample_data['close'])
        assert result.index.equals(sample_data['close'].index)
        
        # Check initial NaNs (first window-1 are strictly NaN due to smoothing start)
        assert result.iloc[:window-1].isna().all()
        # First calculated value is at window-1
        assert not pd.isna(result.iloc[window-1])
        # Subsequent values should also not be NaN
        assert not result.iloc[window:].isna().any()
        
        # ATR should always be positive
        assert (result.dropna() >= 0).all()

    def test_atr_custom_window(self, sample_data):
        """Test ATR with a custom window"""
        window = 7
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result = atr(df, window=window)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data['close'])
        assert result.iloc[:window-1].isna().all()
        assert not pd.isna(result.iloc[window-1])
        assert not result.iloc[window:].isna().any()
        assert (result.dropna() >= 0).all()
        
    def test_atr_volatility_reflection(self, sample_data):
        """Test that ATR reflects changes in volatility in sample data"""
        window = 14
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result = atr(df, window=window).dropna()
        
        # Sample data has high vol first 50, low vol last 50
        high_vol_period_atr = result.iloc[window:50].mean() # Take mean ATR during high vol
        low_vol_period_atr = result.iloc[50:].mean()    # Take mean ATR during low vol
        
        assert high_vol_period_atr > low_vol_period_atr


class TestKeltnerChannels:
    """Tests for Keltner Channels"""

    def test_keltner_calculation(self, sample_data):
        """Test basic Keltner Channel calculation structure"""
        ema_window = 20
        atr_window = 10
        atr_multiplier = 2.0
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result = keltner_channels(df, ema_window=ema_window, atr_window=atr_window, atr_multiplier=atr_multiplier)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) == len(sample_data['close'])
        assert result.index.equals(sample_data['close'].index)
        
        expected_cols = [f'KELT_Middle_{ema_window}_{atr_window}_{atr_multiplier}', f'KELT_Upper_{ema_window}_{atr_window}_{atr_multiplier}', f'KELT_Lower_{ema_window}_{atr_window}_{atr_multiplier}']
        assert all(col in result.columns for col in expected_cols)
        
        # Check initial NaNs: The first row with no NaNs should be determined by ATR window
        valid_result = result.dropna()
        assert not valid_result.empty # Ensure some valid rows exist
        first_valid_index = valid_result.index[0]
        # ATR produces first value at atr_window - 1
        expected_first_valid_pos = atr_window - 1 
        expected_first_valid_idx = sample_data['close'].index[expected_first_valid_pos]
        assert first_valid_index == expected_first_valid_idx
        
        # Check band properties on non-NaN data (already have valid_result)
        assert (valid_result[expected_cols[1]] >= valid_result[expected_cols[0]]).all() # Upper >= Middle
        assert (valid_result[expected_cols[0]] >= valid_result[expected_cols[2]]).all() # Middle >= Lower

    def test_keltner_custom_params(self, sample_data):
        """Test Keltner Channels with custom parameters"""
        ema_window = 10
        atr_window = 5
        atr_multiplier = 1.5
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result = keltner_channels(df, ema_window=ema_window, atr_window=atr_window, atr_multiplier=atr_multiplier)

        assert isinstance(result, pd.DataFrame)
        expected_cols = [f'KELT_Middle_{ema_window}_{atr_window}_{atr_multiplier}', f'KELT_Upper_{ema_window}_{atr_window}_{atr_multiplier}', f'KELT_Lower_{ema_window}_{atr_window}_{atr_multiplier}']
        assert all(col in result.columns for col in expected_cols)
        assert len(result) == len(sample_data['close'])
        
        # Check initial NaNs: The first row with no NaNs should be determined by ATR window
        valid_result = result.dropna()
        assert not valid_result.empty
        first_valid_index = valid_result.index[0]
        expected_first_valid_pos = atr_window - 1
        expected_first_valid_idx = sample_data['close'].index[expected_first_valid_pos]
        assert first_valid_index == expected_first_valid_idx

        # Check band properties on non-NaN data
        assert (valid_result[expected_cols[1]] >= valid_result[expected_cols[0]]).all()
        assert (valid_result[expected_cols[0]] >= valid_result[expected_cols[2]]).all()


class TestDonchianChannels:
    """Tests for Donchian Channels"""

    def test_donchian_calculation(self, sample_data):
        """Test basic Donchian Channel calculation structure"""
        window = 20 # Default
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low']
        })
        result = donchian_channels(df, window=window)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) == len(sample_data['close'])
        assert result.index.equals(sample_data['close'].index)
        
        expected_cols = [f'DONCH_Upper_{window}', f'DONCH_Middle_{window}', f'DONCH_Lower_{window}']
        assert all(col in result.columns for col in expected_cols)
        
        # Check initial NaNs (first window-1 should be strictly NaN)
        assert result.iloc[:window-1].isna().all().all() # Check all columns are NaN initially
        assert not result.iloc[window-1:].isna().any().any() # Check no NaNs from window-1 onwards
        
        # Check band properties on non-NaN data
        valid_result = result.dropna()
        assert (valid_result[expected_cols[0]] >= valid_result[expected_cols[1]]).all() # Upper >= Middle
        assert (valid_result[expected_cols[1]] >= valid_result[expected_cols[2]]).all() # Middle >= Lower
        # Middle should be exactly halfway between upper and lower bands
        middle_calc = (valid_result[expected_cols[0]] + valid_result[expected_cols[2]]) / 2
        # Set the name to match the name of the middle band Series for comparison
        middle_calc.name = expected_cols[1]
        pd.testing.assert_series_equal(valid_result[expected_cols[1]], middle_calc)

    def test_donchian_custom_window(self, sample_data):
        """Test Donchian Channels with a custom window"""
        window = 10
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low']
        })
        result = donchian_channels(df, window=window)
        
        assert isinstance(result, pd.DataFrame)
        expected_cols = [f'DONCH_Upper_{window}', f'DONCH_Middle_{window}', f'DONCH_Lower_{window}']
        assert all(col in result.columns for col in expected_cols)
        assert len(result) == len(sample_data['close'])
        assert result.iloc[:window-1].isna().all().all() # Check all columns are NaN initially
        assert not result.iloc[window-1:].isna().any().any() # Check no NaNs from window-1 onwards properties on non-NaN data
        valid_result = result.dropna()
        assert (valid_result[expected_cols[0]] >= valid_result[expected_cols[2]]).all() # Upper >= Lower
        assert (valid_result[expected_cols[0]] >= valid_result[expected_cols[1]]).all() # Upper >= Middle
        assert (valid_result[expected_cols[1]] >= valid_result[expected_cols[2]]).all() # Middle >= Lower

class TestChaikinVolatility:
    """Tests for Chaikin Volatility"""

    def test_chaikin_calculation(self, sample_data):
        """Test basic Chaikin Volatility calculation structure"""
        ema_window = 10 # Default
        roc_window = 10 # Default
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low']
        })
        result = chaikin_volatility(df, ema_window=ema_window, roc_window=roc_window, high_col='High', low_col='Low')
        
        assert isinstance(result, pd.Series)
        assert not result.empty
        assert len(result) == len(sample_data['close'])
        assert result.index.equals(sample_data['close'].index)
        
        # Check initial NaNs (depend on EMA window + ROC window lookback)
        # EMA needs ema_window, ROC needs roc_window shift on top of EMA result.
        # Total lookback is complex due to EMA smoothing start.
        # Let's check a reasonable number based on defaults.
        nan_lookback = ema_window + roc_window
        # Check that *some* initial values are NaN, and *some* later values are not.
        assert result.iloc[:nan_lookback].isna().any() 
        assert not result.isna().all()
        assert not result.iloc[-1:].isna().any() # Last value should be valid
        

    def test_chaikin_custom_params(self, sample_data):
        """Test Chaikin Volatility with custom parameters"""
        ema_window = 5
        roc_window = 7
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low']
        })
        result = chaikin_volatility(df, ema_window=ema_window, roc_window=roc_window, high_col='High', low_col='Low')
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data['close'])
        assert not result.isna().all()
        assert not result.iloc[-1:].isna().any()
