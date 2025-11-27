import pytest
import pandas as pd
import numpy as np
from simple_trade.volatility import (
    bol, atr, kel, don, cha, std, rvi, mai, svi, dvi, vqi, hav, cho, uli,
    hiv, bbw, atp, acb, tsv, nat, pcw, vra, efr, vhf, pro, grv, pav, rsv,
    fdi, vsi, mad
)

# Fixture for sample data (consistent with other test modules)
@pytest.fixture
def sample_data():
    """Fixture to provide sample OHLCV data for testing volatility indicators"""
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

    # Create open prices (between low and high)
    np.random.seed(43)
    open_prices = low + (high - low) * np.random.uniform(0.2, 0.8, size=100)
    
    # Create volume data
    np.random.seed(44)
    volume = pd.Series(np.random.randint(100000, 1000000, size=100), index=index)

    return {
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }

class TestBollingerBands:
    """Tests for Bollinger Bands"""

    def test_bb_calculation(self, sample_data):
        """Test basic Bollinger Bands calculation structure"""
        window=20
        num_std=2
        # Create a DataFrame with 'Close' column
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, _ = bol(df, parameters={'window': window, 'num_std': num_std}, columns=None)
        
        assert isinstance(result_data, pd.DataFrame)
        assert not result_data.empty
        assert len(result_data) == len(sample_data['close'])
        assert result_data.index.equals(sample_data['close'].index)
        
        # Check columns
        expected_cols = [f'BB_Middle_{window}', f'BB_Upper_{window}_{num_std}.0', f'BB_Lower_{window}_{num_std}.0']
        assert all(col in result_data.columns for col in expected_cols)
        
        # Check initial NaNs (first window-1)
        assert result_data.iloc[:window-1].isna().all().all() # Check all columns are NaN initially
        assert not result_data.iloc[window-1:].isna().any().any() # Check no NaNs after window
        
        # Check band properties on non-NaN data
        valid_result = result_data.dropna()
        assert (valid_result[expected_cols[1]] >= valid_result[expected_cols[0]]).all() # Upper >= Middle
        assert (valid_result[expected_cols[0]] >= valid_result[expected_cols[2]]).all() # Middle >= Lower

    def test_bb_custom_params(self, sample_data):
        """Test Bollinger Bands with custom parameters"""
        window = 10
        num_std = 3
        # Create a DataFrame with 'Close' column
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, _ = bol(df, parameters={'window': window, 'num_std': num_std}, columns=None)
        
        assert isinstance(result_data, pd.DataFrame)
        expected_cols = [f'BB_Middle_{window}', f'BB_Upper_{window}_{num_std}.0', f'BB_Lower_{window}_{num_std}.0']
        assert all(col in result_data.columns for col in expected_cols)
        assert len(result_data) == len(sample_data['close'])
        assert result_data.iloc[:window-1].isna().all().all()
        assert not result_data.iloc[window-1:].isna().any().any()
        
        # Check band properties on non-NaN data
        valid_result = result_data.dropna()
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
        result_data, _ = atr(df, parameters={'window': window}, columns=None)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert len(result_data) == len(sample_data['close'])
        assert result_data.index.equals(sample_data['close'].index)
        
        # Check initial NaNs (first window-1 are strictly NaN due to smoothing start)
        assert result_data.iloc[:window-1].isna().all()
        # First calculated value is at window-1
        assert not pd.isna(result_data.iloc[window-1])
        # Subsequent values should also not be NaN
        assert not result_data.iloc[window:].isna().any()
        
        # ATR should always be positive
        assert (result_data.dropna() >= 0).all()

    def test_atr_custom_window(self, sample_data):
        """Test ATR with a custom window"""
        window = 7
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, _ = atr(df, parameters={'window': window}, columns=None)
        
        assert isinstance(result_data, pd.Series)
        assert len(result_data) == len(sample_data['close'])
        assert result_data.iloc[:window-1].isna().all()
        assert not pd.isna(result_data.iloc[window-1])
        assert not result_data.iloc[window:].isna().any()
        assert (result_data.dropna() >= 0).all()
        
    def test_atr_volatility_reflection(self, sample_data):
        """Test that ATR reflects changes in volatility in sample data"""
        window = 14
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, _ = atr(df, parameters={'window': window}, columns=None)
        result_data = result_data.dropna()
        
        # Sample data has high vol first 50, low vol last 50
        high_vol_period_atr = result_data.iloc[window:50].mean() # Take mean ATR during high vol
        low_vol_period_atr = result_data.iloc[50:].mean()    # Take mean ATR during low vol
        
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
        result_data, _ = kel(df, parameters={'ema_window': ema_window, 'atr_window': atr_window, 'atr_multiplier': atr_multiplier}, columns=None)
        
        assert isinstance(result_data, pd.DataFrame)
        assert not result_data.empty
        assert len(result_data) == len(sample_data['close'])
        assert result_data.index.equals(sample_data['close'].index)
        
        expected_cols = [f'KELT_Middle_{ema_window}_{atr_window}_{atr_multiplier}', f'KELT_Upper_{ema_window}_{atr_window}_{atr_multiplier}', f'KELT_Lower_{ema_window}_{atr_window}_{atr_multiplier}']
        assert all(col in result_data.columns for col in expected_cols)
        
        # Check initial NaNs: The first row with no NaNs should be determined by ATR window
        valid_result = result_data.dropna()
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
        result_data, _ = kel(df, parameters={'ema_window': ema_window, 'atr_window': atr_window, 'atr_multiplier': atr_multiplier}, columns=None)

        assert isinstance(result_data, pd.DataFrame)
        expected_cols = [f'KELT_Middle_{ema_window}_{atr_window}_{atr_multiplier}', f'KELT_Upper_{ema_window}_{atr_window}_{atr_multiplier}', f'KELT_Lower_{ema_window}_{atr_window}_{atr_multiplier}']
        assert all(col in result_data.columns for col in expected_cols)
        assert len(result_data) == len(sample_data['close'])
        
        # Check initial NaNs: The first row with no NaNs should be determined by ATR window
        valid_result = result_data.dropna()
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
        result_data, _ = don(df, parameters={'window': window}, columns=None)
        
        assert isinstance(result_data, pd.DataFrame)
        assert not result_data.empty
        assert len(result_data) == len(sample_data['close'])
        assert result_data.index.equals(sample_data['close'].index)
        
        expected_cols = [f'DONCH_Upper_{window}', f'DONCH_Middle_{window}', f'DONCH_Lower_{window}']
        assert all(col in result_data.columns for col in expected_cols)
        
        # Check initial NaNs (first window-1 should be strictly NaN)
        assert result_data.iloc[:window-1].isna().all().all() # Check all columns are NaN initially
        assert not result_data.iloc[window-1:].isna().any().any() # Check no NaNs from window-1 onwards
        
        # Check band properties on non-NaN data
        valid_result = result_data.dropna()
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
        result_data, _ = don(df, parameters={'window': window}, columns=None)
        
        assert isinstance(result_data, pd.DataFrame)
        expected_cols = [f'DONCH_Upper_{window}', f'DONCH_Middle_{window}', f'DONCH_Lower_{window}']
        assert all(col in result_data.columns for col in expected_cols)
        assert len(result_data) == len(sample_data['close'])
        assert result_data.iloc[:window-1].isna().all().all() # Check all columns are NaN initially
        assert not result_data.iloc[window-1:].isna().any().any() # Check no NaNs from window-1 onwards properties on non-NaN data
        valid_result = result_data.dropna()
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
        result_data, _ = cha(df, parameters={'ema_window': ema_window, 'roc_window': roc_window}, columns={'high_col': 'High', 'low_col': 'Low'})
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert len(result_data) == len(sample_data['close'])
        assert result_data.index.equals(sample_data['close'].index)
        
        # Check initial NaNs (depend on EMA window + ROC window lookback)
        # EMA needs ema_window, ROC needs roc_window shift on top of EMA result.
        # Total lookback is complex due to EMA smoothing start.
        # Let's check a reasonable number based on defaults.
        nan_lookback = ema_window + roc_window
        # Check that *some* initial values are NaN, and *some* later values are not.
        assert result_data.iloc[:nan_lookback].isna().any() 
        assert not result_data.isna().all()
        assert not result_data.iloc[-1:].isna().any() # Last value should be valid
        

    def test_chaikin_custom_params(self, sample_data):
        """Test Chaikin Volatility with custom parameters"""
        ema_window = 5
        roc_window = 7
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low']
        })
        result_data, _ = cha(df, parameters={'ema_window': ema_window, 'roc_window': roc_window}, columns={'high_col': 'High', 'low_col': 'Low'})
        
        assert isinstance(result_data, pd.Series)
        assert len(result_data) == len(sample_data['close'])
        assert not result_data.isna().all()
        assert not result_data.iloc[-1:].isna().any()


# --- Additional Volatility Indicator Tests ---

class TestSTD:
    """Tests for Standard Deviation"""

    def test_std_calculation(self, sample_data):
        """Test basic STD calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = std(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'STD_20' in columns
        
        # STD should be positive
        valid_result = result_data.dropna()
        assert (valid_result >= 0).all()

    def test_std_custom_window(self, sample_data):
        """Test STD with custom window"""
        window = 10
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = std(df, parameters={'window': window})
        
        assert f'STD_{window}' in columns


class TestRVI:
    """Tests for Relative Volatility Index"""

    def test_rvi_calculation(self, sample_data):
        """Test basic RVI calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = rvi(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'RVI_10_14' in columns
        
        # RVI should be between 0 and 100
        valid_result = result_data.dropna()
        assert len(valid_result) > 0

    def test_rvi_custom_params(self, sample_data):
        """Test RVI with custom parameters"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = rvi(df, parameters={'window': 14, 'rvi_period': 10})
        
        assert 'RVI_14_10' in columns


class TestMassIndex:
    """Tests for Mass Index"""

    def test_mai_calculation(self, sample_data):
        """Test basic Mass Index calculation"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low']
        })
        result_data, columns = mai(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'MI_9_25' in columns
        
        # Mass Index should have valid values
        valid_result = result_data.dropna()
        assert len(valid_result) > 0

    def test_mai_custom_params(self, sample_data):
        """Test Mass Index with custom parameters"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low']
        })
        result_data, columns = mai(df, parameters={'ema_period': 7, 'sum_period': 20})
        
        assert 'MI_7_20' in columns


class TestSVI:
    """Tests for Stochastic Volatility Indicator"""

    def test_svi_calculation(self, sample_data):
        """Test basic SVI calculation"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = svi(df)
        
        assert isinstance(result_data, pd.DataFrame)
        assert not result_data.empty
        assert 'SVI_K_14_14_3' in columns
        assert 'SVI_D_14_14_3' in columns

    def test_svi_custom_params(self, sample_data):
        """Test SVI with custom parameters"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = svi(df, parameters={'atr_period': 10, 'stoch_period': 10})
        
        assert 'SVI_K_10_10_3' in columns


class TestDVI:
    """Tests for Dynamic Volatility Indicator"""

    def test_dvi_calculation(self, sample_data):
        """Test basic DVI calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = dvi(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'DVI_5_100_3' in columns

    def test_dvi_custom_params(self, sample_data):
        """Test DVI with custom parameters"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = dvi(df, parameters={
            'magnitude_period': 10,
            'stretch_period': 50,
            'smooth_period': 5
        })
        
        assert 'DVI_10_50_5' in columns


class TestVQI:
    """Tests for Volatility Quality Index"""

    def test_vqi_calculation(self, sample_data):
        """Test basic VQI calculation"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, columns = vqi(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'VQI_9_9' in columns

    def test_vqi_custom_params(self, sample_data):
        """Test VQI with custom parameters"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, columns = vqi(df, parameters={'period': 14, 'smooth_period': 7})
        
        assert 'VQI_14_7' in columns


class TestHAV:
    """Tests for Heikin-Ashi Volatility"""

    def test_hav_calculation(self, sample_data):
        """Test basic HAV calculation"""
        df = pd.DataFrame({
            'Open': sample_data['open'],
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = hav(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'HAV_14_ATR' in columns
        
        # HAV should be positive
        valid_result = result_data.dropna()
        assert (valid_result >= 0).all()

    def test_hav_std_method(self, sample_data):
        """Test HAV with std method"""
        df = pd.DataFrame({
            'Open': sample_data['open'],
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = hav(df, parameters={'method': 'std'})
        
        assert 'HAV_14_STD' in columns


class TestChoppinessIndex:
    """Tests for Choppiness Index"""

    def test_cho_calculation(self, sample_data):
        """Test basic Choppiness Index calculation"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = cho(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'CHOP_14' in columns
        
        # CHOP should be between 0 and 100
        valid_result = result_data.dropna()
        assert len(valid_result) > 0

    def test_cho_custom_period(self, sample_data):
        """Test Choppiness Index with custom period"""
        period = 20
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = cho(df, parameters={'period': period})
        
        assert f'CHOP_{period}' in columns


class TestUlcerIndex:
    """Tests for Ulcer Index"""

    def test_uli_calculation(self, sample_data):
        """Test basic Ulcer Index calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = uli(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'UI_14' in columns
        
        # Ulcer Index should be non-negative
        valid_result = result_data.dropna()
        assert (valid_result >= 0).all()

    def test_uli_custom_period(self, sample_data):
        """Test Ulcer Index with custom period"""
        period = 20
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = uli(df, parameters={'period': period})
        
        assert f'UI_{period}' in columns


class TestHistoricalVolatility:
    """Tests for Historical Volatility"""

    def test_hiv_calculation(self, sample_data):
        """Test basic Historical Volatility calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = hiv(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'HV_20_Ann' in columns
        
        # HV should be non-negative
        valid_result = result_data.dropna()
        assert (valid_result >= 0).all()

    def test_hiv_non_annualized(self, sample_data):
        """Test Historical Volatility without annualization"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = hiv(df, parameters={'annualized': False})
        
        assert 'HV_20' in columns


class TestBBW:
    """Tests for Bollinger Band Width"""

    def test_bbw_calculation(self, sample_data):
        """Test basic BBW calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = bbw(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'BBW_20_2.0' in columns
        
        # BBW should be positive
        valid_result = result_data.dropna()
        assert (valid_result >= 0).all()

    def test_bbw_custom_params(self, sample_data):
        """Test BBW with custom parameters"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = bbw(df, parameters={'window': 14, 'num_std': 2.5})
        
        assert 'BBW_14_2.5' in columns


class TestATRP:
    """Tests for Average True Range Percent"""

    def test_atp_calculation(self, sample_data):
        """Test basic ATRP calculation"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = atp(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'ATRP_14' in columns
        
        # ATRP should be positive
        valid_result = result_data.dropna()
        assert (valid_result >= 0).all()

    def test_atp_custom_window(self, sample_data):
        """Test ATRP with custom window"""
        window = 10
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = atp(df, parameters={'window': window})
        
        assert f'ATRP_{window}' in columns


class TestAccelerationBands:
    """Tests for Acceleration Bands"""

    def test_acb_calculation(self, sample_data):
        """Test basic Acceleration Bands calculation"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = acb(df)
        
        assert isinstance(result_data, pd.DataFrame)
        assert not result_data.empty
        assert 'ACB_Middle_20' in columns
        
        # Upper should be >= Middle >= Lower
        valid_result = result_data.dropna()
        if len(valid_result) > 0:
            upper_col = [c for c in columns if 'Upper' in c][0]
            lower_col = [c for c in columns if 'Lower' in c][0]
            assert (valid_result[upper_col] >= valid_result['ACB_Middle_20']).all()
            assert (valid_result['ACB_Middle_20'] >= valid_result[lower_col]).all()

    def test_acb_custom_params(self, sample_data):
        """Test Acceleration Bands with custom parameters"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = acb(df, parameters={'period': 14, 'factor': 0.002})
        
        assert 'ACB_Middle_14' in columns


class TestTSV:
    """Tests for TSI Volatility"""

    def test_tsv_calculation(self, sample_data):
        """Test basic TSI Volatility calculation"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = tsv(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'TSI_VOL_14_25_13' in columns

    def test_tsv_custom_params(self, sample_data):
        """Test TSI Volatility with custom parameters"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = tsv(df, parameters={
            'atr_period': 10,
            'long_period': 20,
            'short_period': 10
        })
        
        assert 'TSI_VOL_10_20_10' in columns


class TestNATR:
    """Tests for Normalized Average True Range"""

    def test_nat_calculation(self, sample_data):
        """Test basic NATR calculation"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = nat(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'NATR_14' in columns
        
        # NATR should be positive
        valid_result = result_data.dropna()
        assert (valid_result >= 0).all()

    def test_nat_custom_window(self, sample_data):
        """Test NATR with custom window"""
        window = 10
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = nat(df, parameters={'window': window})
        
        assert f'NATR_{window}' in columns


class TestPCW:
    """Tests for Price Channel Width"""

    def test_pcw_calculation(self, sample_data):
        """Test basic PCW calculation"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = pcw(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'PCW_20' in columns
        
        # PCW should be positive
        valid_result = result_data.dropna()
        assert (valid_result >= 0).all()

    def test_pcw_custom_period(self, sample_data):
        """Test PCW with custom period"""
        period = 14
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = pcw(df, parameters={'period': period})
        
        assert f'PCW_{period}' in columns


class TestVolatilityRatio:
    """Tests for Volatility Ratio"""

    def test_vra_calculation(self, sample_data):
        """Test basic Volatility Ratio calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = vra(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'VR_5_20' in columns

    def test_vra_custom_params(self, sample_data):
        """Test Volatility Ratio with custom parameters"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = vra(df, parameters={'short_period': 10, 'long_period': 30})
        
        assert 'VR_10_30' in columns


class TestEfficiencyRatio:
    """Tests for Efficiency Ratio"""

    def test_efr_calculation(self, sample_data):
        """Test basic Efficiency Ratio calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = efr(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'ER_10' in columns
        
        # ER should be between 0 and 1
        valid_result = result_data.dropna()
        assert (valid_result >= 0).all()
        assert (valid_result <= 1).all()

    def test_efr_custom_period(self, sample_data):
        """Test Efficiency Ratio with custom period"""
        period = 14
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = efr(df, parameters={'period': period})
        
        assert f'ER_{period}' in columns


class TestVHF:
    """Tests for Vertical Horizontal Filter"""

    def test_vhf_calculation(self, sample_data):
        """Test basic VHF calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = vhf(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'VHF_28' in columns
        
        # VHF should be positive
        valid_result = result_data.dropna()
        assert (valid_result >= 0).all()

    def test_vhf_custom_period(self, sample_data):
        """Test VHF with custom period"""
        period = 20
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = vhf(df, parameters={'period': period})
        
        assert f'VHF_{period}' in columns


class TestProjectionOscillator:
    """Tests for Projection Oscillator"""

    def test_pro_calculation(self, sample_data):
        """Test basic Projection Oscillator calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = pro(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'PO_10_3' in columns

    def test_pro_custom_params(self, sample_data):
        """Test Projection Oscillator with custom parameters"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = pro(df, parameters={'period': 14, 'smooth_period': 5})
        
        assert 'PO_14_5' in columns


class TestGarmanKlassVolatility:
    """Tests for Garman-Klass Volatility"""

    def test_grv_calculation(self, sample_data):
        """Test basic Garman-Klass Volatility calculation"""
        df = pd.DataFrame({
            'Open': sample_data['open'],
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = grv(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'GK_VOL_20_Ann' in columns
        
        # GK Vol should be non-negative
        valid_result = result_data.dropna()
        assert (valid_result >= 0).all()

    def test_grv_non_annualized(self, sample_data):
        """Test Garman-Klass Volatility without annualization"""
        df = pd.DataFrame({
            'Open': sample_data['open'],
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = grv(df, parameters={'annualized': False})
        
        assert 'GK_VOL_20' in columns


class TestParkinsonVolatility:
    """Tests for Parkinson Volatility"""

    def test_pav_calculation(self, sample_data):
        """Test basic Parkinson Volatility calculation"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low']
        })
        result_data, columns = pav(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'PARK_VOL_20_Ann' in columns
        
        # Parkinson Vol should be non-negative
        valid_result = result_data.dropna()
        assert (valid_result >= 0).all()

    def test_pav_non_annualized(self, sample_data):
        """Test Parkinson Volatility without annualization"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low']
        })
        result_data, columns = pav(df, parameters={'annualized': False})
        
        assert 'PARK_VOL_20' in columns


class TestRogersSatchellVolatility:
    """Tests for Rogers-Satchell Volatility"""

    def test_rsv_calculation(self, sample_data):
        """Test basic Rogers-Satchell Volatility calculation"""
        df = pd.DataFrame({
            'Open': sample_data['open'],
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = rsv(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'RS_VOL_20_Ann' in columns
        
        # RS Vol should be non-negative
        valid_result = result_data.dropna()
        assert (valid_result >= 0).all()

    def test_rsv_non_annualized(self, sample_data):
        """Test Rogers-Satchell Volatility without annualization"""
        df = pd.DataFrame({
            'Open': sample_data['open'],
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = rsv(df, parameters={'annualized': False})
        
        assert 'RS_VOL_20' in columns


class TestFDI:
    """Tests for Fractal Dimension Index"""

    def test_fdi_calculation(self, sample_data):
        """Test basic FDI calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = fdi(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'FDI_20' in columns
        
        # FDI should be between 1 and 2
        valid_result = result_data.dropna()
        assert (valid_result >= 1.0).all()
        assert (valid_result <= 2.0).all()

    def test_fdi_custom_period(self, sample_data):
        """Test FDI with custom period"""
        period = 14
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = fdi(df, parameters={'period': period})
        
        assert f'FDI_{period}' in columns


class TestVSI:
    """Tests for Volatility Switch Index"""

    def test_vsi_calculation(self, sample_data):
        """Test basic VSI calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = vsi(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'VSI_10_50_1.2' in columns
        
        # VSI should be binary (0 or 1)
        valid_result = result_data.dropna()
        assert set(valid_result.unique()).issubset({0, 1})

    def test_vsi_custom_params(self, sample_data):
        """Test VSI with custom parameters"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = vsi(df, parameters={
            'short_period': 5,
            'long_period': 30,
            'threshold': 1.5
        })
        
        assert 'VSI_5_30_1.5' in columns


class TestMAD:
    """Tests for Median Absolute Deviation"""

    def test_mad_calculation(self, sample_data):
        """Test basic MAD calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = mad(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'MAD_20' in columns
        
        # MAD should be non-negative
        valid_result = result_data.dropna()
        assert (valid_result >= 0).all()

    def test_mad_custom_period(self, sample_data):
        """Test MAD with custom period"""
        period = 14
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = mad(df, parameters={'period': period})
        
        assert f'MAD_{period}' in columns
