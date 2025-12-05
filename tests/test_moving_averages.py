import pytest
import pandas as pd
import numpy as np
from simple_trade.moving_average import (
    ema, sma, wma, hma, soa, ama, tma, fma, gma, jma, zma,
    dem, tem, alm, lsm, swm, ads, vid, tt3, mam, evw, tsf
)

# Fixture for sample data
@pytest.fixture
def sample_data():
    """Fixture to provide sample OHLC data for testing moving average indicators"""
    index = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42) # for reproducibility

    # Create a series with more pronounced trends and volatility
    uptrend = np.linspace(100, 200, 40)
    downtrend = np.linspace(200, 100, 40)
    uptrend2 = np.linspace(100, 150, 20)
    noise = np.random.normal(0, 3, 100)
    combined = np.concatenate([uptrend, downtrend, uptrend2])
    close = pd.Series(combined + noise, index=index)

    # Create high and low with realistic spread
    high = close + np.random.uniform(1, 5, size=len(close))
    low = close - np.random.uniform(1, 5, size=len(close))

    # Ensure low is not higher than close and high is not lower than close
    low = pd.Series(np.minimum(low.values, close.values - 0.1), index=index)
    high = pd.Series(np.maximum(high.values, close.values + 0.1), index=index)

    return {
        'high': high,
        'low': low,
        'close': close
    }


class TestEMA:
    """Tests for the Exponential Moving Average (EMA)"""

    def test_ema_calculation(self, sample_data):
        """Test basic EMA calculation structure and properties"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, _ = ema(df, parameters=None, columns=None)
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert len(result_data) == len(sample_data['close'])
        assert result_data.index.equals(sample_data['close'].index)
        # First value should match the first close price
        assert result_data.iloc[0] == sample_data['close'].iloc[0]
        # Should not contain NaNs after the first value if input has no NaNs
        assert not result_data.iloc[1:].isna().any()

    def test_ema_custom_window(self, sample_data):
        """Test EMA with a custom window"""
        window = 5
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, _ = ema(df, parameters={'window': window}, columns=None)
        assert isinstance(result_data, pd.Series)
        assert len(result_data) == len(sample_data['close'])
        assert result_data.iloc[0] == sample_data['close'].iloc[0]
        assert not result_data.iloc[1:].isna().any()

    def test_ema_known_values(self):
        """Test EMA calculation against manually calculated values."""
        data = pd.Series([10, 20, 30, 40, 50])
        df = pd.DataFrame({'Close': data})
        result_data, _ = ema(df, parameters={'window': 3}, columns=None)
        # k = 2 / (3 + 1) = 0.5
        # EMA(1) = 10
        # EMA(2) = (20*0.5) + (10*0.5) = 15
        # EMA(3) = (30*0.5) + (15*0.5) = 22.5
        # EMA(4) = (40*0.5) + (22.5*0.5) = 31.25
        # EMA(5) = (50*0.5) + (31.25*0.5) = 40.625
        expected = pd.Series([10.0, 15.0, 22.5, 31.25, 40.625], index=df.index, name='EMA_3')
        pd.testing.assert_series_equal(result_data, expected, check_names=True)


class TestSMA:
    """Tests for the Simple Moving Average (SMA)"""

    def test_sma_calculation(self, sample_data):
        """Test basic SMA calculation structure and properties"""
        window = 14 # Default
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, _ = sma(df, parameters={'window': window}, columns=None)
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert len(result_data) == len(sample_data['close'])
        assert result_data.index.equals(sample_data['close'].index)
        # First window-1 values should be NaN
        assert result_data.iloc[:window-1].isna().all()
        # Values after window-1 should not be NaN (assuming input is valid)
        assert not result_data.iloc[window-1:].isna().any()

    def test_sma_custom_window(self, sample_data):
        """Test SMA with a custom window"""
        window = 5
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, _ = sma(df, parameters={'window': window}, columns=None)
        assert isinstance(result_data, pd.Series)
        assert len(result_data) == len(sample_data['close'])
        assert result_data.iloc[:window-1].isna().all()
        assert not result_data.iloc[window-1:].isna().any()

    def test_sma_known_values(self):
        """Test SMA calculation against manually calculated values."""
        data = pd.Series([10, 20, 30, 40, 50])
        df = pd.DataFrame({'Close': data})
        result_data, _ = sma(df, parameters={'window': 3}, columns=None)
        # SMA(3) = (10+20+30)/3 = 20
        # SMA(4) = (20+30+40)/3 = 30
        # SMA(5) = (30+40+50)/3 = 40
        expected = pd.Series([np.nan, np.nan, 20.0, 30.0, 40.0], index=df.index, name='SMA_3')
        pd.testing.assert_series_equal(result_data, expected, check_names=True)


class TestWMA:
    """Tests for the Weighted Moving Average (WMA)"""

    def test_wma_calculation(self, sample_data):
        """Test basic WMA calculation structure and properties"""
        window = 14 # Default
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, _ = wma(df, parameters={'window': window}, columns=None)
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert len(result_data) == len(sample_data['close'])
        assert result_data.index.equals(sample_data['close'].index)
        # First window-1 values should be NaN
        assert result_data.iloc[:window-1].isna().all()
        # Values after window-1 should not be NaN
        assert not result_data.iloc[window-1:].isna().any()

    def test_wma_custom_window(self, sample_data):
        """Test WMA with a custom window"""
        window = 5
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, _ = wma(df, parameters={'window': window}, columns=None)
        assert isinstance(result_data, pd.Series)
        assert len(result_data) == len(sample_data['close'])
        assert result_data.iloc[:window-1].isna().all()
        assert not result_data.iloc[window-1:].isna().any()

    def test_wma_known_values(self):
        """Test WMA calculation against manually calculated values."""
        data = pd.Series([10, 20, 30, 40, 50])
        df = pd.DataFrame({'Close': data})
        result_data, _ = wma(df, parameters={'window': 3}, columns=None)
        # weights = [1, 2, 3], sum = 6
        # WMA(3) = (10*1 + 20*2 + 30*3) / 6 = 140 / 6 = 23.333...
        # WMA(4) = (20*1 + 30*2 + 40*3) / 6 = 200 / 6 = 33.333...
        # WMA(5) = (30*1 + 40*2 + 50*3) / 6 = 260 / 6 = 43.333...
        expected = pd.Series([np.nan, np.nan, 23.333333, 33.333333, 43.333333], index=df.index, name='WMA_3')
        pd.testing.assert_series_equal(result_data, expected, check_names=True, rtol=1e-5)


class TestHMA:
    """Tests for the Hull Moving Average (HMA)"""

    def test_hma_calculation(self, sample_data):
        """Test basic HMA calculation structure and properties"""
        window = 14 # Default
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, _ = hma(df, parameters={'window': window}, columns=None)
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert len(result_data) == len(sample_data['close'])
        assert result_data.index.equals(sample_data['close'].index)
        # HMA introduces more NaNs than simple rolling, check last value is valid
        assert not result_data.isna().all() # Ensure not all are NaN
        assert not np.isnan(result_data.iloc[-1]) # Last value should be calculable

    def test_hma_custom_window(self, sample_data):
        """Test HMA with a custom window"""
        window = 5
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, _ = hma(df, parameters={'window': window}, columns=None)
        assert isinstance(result_data, pd.Series)
        assert len(result_data) == len(sample_data['close'])
        assert not result_data.isna().all()
        assert not np.isnan(result_data.iloc[-1])

    def test_hma_dependencies(self):
        """Test that HMA calculation steps match expectations."""
        data = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        df = pd.DataFrame({'Close': data})
        window=4
        half_length = int(window / 2)
        sqrt_length = int(np.sqrt(window))

        # Create the test implementation of HMA using the component functions
        df_half = pd.DataFrame({'Close': data})
        df_full = pd.DataFrame({'Close': data})
        
        wma_half_data, _ = wma(df_half, parameters={'window': half_length}, columns=None)
        wma_full_data, _ = wma(df_full, parameters={'window': window}, columns=None)
        
        # Create raw_hma from the Series operations
        raw_hma = 2 * wma_half_data - wma_full_data
        
        # Create a DataFrame for the raw_hma
        df_raw = pd.DataFrame({'Close': raw_hma})
        expected_hma_data, _ = wma(df_raw, parameters={'window': sqrt_length}, columns=None)

        # Get the actual HMA implementation result
        result_data, _ = hma(df, parameters={'window': window}, columns=None)
        
        # Compare the values instead of the Series objects directly
        # This handles potential differences in Series metadata
        np.testing.assert_allclose(
            result_data.dropna().values,
            expected_hma_data.dropna().values, 
            rtol=1e-5
        )
        
        # Check that the Series have the same length and indices
        assert len(result_data) == len(expected_hma_data)
        assert result_data.index.equals(expected_hma_data.index)


class TestSOA:
    """Tests for the Smoothed Moving Average (SmMA)"""

    def test_soa_calculation(self, sample_data):
        """Test basic SmMA calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = soa(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'SOA_20' in columns
        assert len(result_data) == len(sample_data['close'])

    def test_soa_custom_window(self, sample_data):
        """Test SmMA with custom window"""
        window = 10
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = soa(df, parameters={'window': window})
        
        assert f'SOA_{window}' in columns
        assert not result_data.isna().all()


class TestTMA:
    """Tests for the Triangular Moving Average"""

    def test_tma_calculation(self, sample_data):
        """Test basic TMA calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = tma(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'TMA_20' in columns
        
        # TMA should have some NaN values at the start
        assert result_data.iloc[-1:].notna().all()

    def test_tma_custom_window(self, sample_data):
        """Test TMA with custom window"""
        window = 10
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = tma(df, parameters={'window': window})
        
        assert f'TMA_{window}' in columns


class TestDEMA:
    """Tests for the Double Exponential Moving Average"""

    def test_dem_calculation(self, sample_data):
        """Test basic DEMA calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = dem(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'DEM_20' in columns
        assert len(result_data) == len(sample_data['close'])

    def test_dem_custom_window(self, sample_data):
        """Test DEMA with custom window"""
        window = 10
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = dem(df, parameters={'window': window})
        
        assert f'DEM_{window}' in columns


class TestTEMA:
    """Tests for the Triple Exponential Moving Average"""

    def test_tem_calculation(self, sample_data):
        """Test basic TEMA calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = tem(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'TEM_20' in columns
        assert len(result_data) == len(sample_data['close'])

    def test_tem_custom_window(self, sample_data):
        """Test TEMA with custom window"""
        window = 15
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = tem(df, parameters={'window': window})
        
        assert f'TEM_{window}' in columns


class TestAMA:
    """Tests for the Adaptive Moving Average (KAMA)"""

    def test_ama_calculation(self, sample_data):
        """Test basic AMA calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = ama(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'AMA_10_2_30' in columns
        assert len(result_data) == len(sample_data['close'])

    def test_ama_custom_params(self, sample_data):
        """Test AMA with custom parameters"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = ama(df, parameters={
            'window': 14,
            'fast_period': 3,
            'slow_period': 20
        })
        
        assert 'AMA_14_3_20' in columns


class TestFMA:
    """Tests for the Fractal Adaptive Moving Average"""

    def test_fma_calculation(self, sample_data):
        """Test basic FRAMA calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = fma(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'FMA_16' in columns
        assert len(result_data) == len(sample_data['close'])

    def test_fma_custom_window(self, sample_data):
        """Test FRAMA with custom window"""
        window = 20
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = fma(df, parameters={'window': window})
        
        assert f'FMA_{window}' in columns


class TestGMA:
    """Tests for the Guppy Multiple Moving Average"""

    def test_gma_calculation(self, sample_data):
        """Test basic GMMA calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = gma(df)
        
        assert isinstance(result_data, pd.DataFrame)
        assert not result_data.empty
        # Check for short-term EMAs
        assert 'GMA_short_3' in columns
        assert 'GMA_short_15' in columns
        # Check for long-term EMAs
        assert 'GMA_long_30' in columns
        assert 'GMA_long_60' in columns

    def test_gma_custom_windows(self, sample_data):
        """Test GMMA with custom windows"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = gma(df, parameters={
            'short_windows': (5, 10),
            'long_windows': (20, 40)
        })
        
        assert 'GMA_short_5' in columns
        assert 'GMA_short_10' in columns
        assert 'GMA_long_20' in columns
        assert 'GMA_long_40' in columns


class TestJMA:
    """Tests for the Jurik Moving Average"""

    def test_jma_calculation(self, sample_data):
        """Test basic JMA calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = jma(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'JMA_21' in columns
        assert len(result_data) == len(sample_data['close'])

    def test_jma_custom_params(self, sample_data):
        """Test JMA with custom parameters"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = jma(df, parameters={
            'length': 14,
            'phase': 50
        })
        
        assert 'JMA_14' in columns


class TestZMA:
    """Tests for the Zero-Lag Moving Average"""

    def test_zma_calculation(self, sample_data):
        """Test basic ZLEMA calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = zma(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'ZMA_20' in columns
        assert len(result_data) == len(sample_data['close'])

    def test_zma_custom_window(self, sample_data):
        """Test ZLEMA with custom window"""
        window = 10
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = zma(df, parameters={'window': window})
        
        assert f'ZMA_{window}' in columns


class TestALMA:
    """Tests for the Arnaud Legoux Moving Average"""

    def test_alm_calculation(self, sample_data):
        """Test basic ALMA calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = alm(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'ALM_9' in columns
        
        # ALMA should have NaN values at the start
        valid_result = result_data.dropna()
        assert len(valid_result) > 0

    def test_alm_custom_params(self, sample_data):
        """Test ALMA with custom parameters"""
        window = 14
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = alm(df, parameters={
            'window': window,
            'sigma': 5,
            'offset': 0.9
        })
        
        assert f'ALM_{window}' in columns


class TestLSM:
    """Tests for the Least Squares Moving Average"""

    def test_lsm_calculation(self, sample_data):
        """Test basic LSMA calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = lsm(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'LSM_20' in columns
        
        # LSMA should have NaN values at the start
        valid_result = result_data.dropna()
        assert len(valid_result) > 0

    def test_lsm_custom_window(self, sample_data):
        """Test LSMA with custom window"""
        window = 10
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = lsm(df, parameters={'window': window})
        
        assert f'LSM_{window}' in columns


class TestSWM:
    """Tests for the Sine Weighted Moving Average"""

    def test_swm_calculation(self, sample_data):
        """Test basic SWMA calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = swm(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'SWM_20' in columns
        
        # SWMA should have NaN values at the start
        valid_result = result_data.dropna()
        assert len(valid_result) > 0

    def test_swm_custom_window(self, sample_data):
        """Test SWMA with custom window"""
        window = 10
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = swm(df, parameters={'window': window})
        
        assert f'SWM_{window}' in columns


class TestADS:
    """Tests for the Adaptive Deviation-Scaled Moving Average"""

    def test_ads_calculation(self, sample_data):
        """Test basic ADSMA calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = ads(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'ADS_20' in columns
        assert len(result_data) == len(sample_data['close'])

    def test_ads_custom_params(self, sample_data):
        """Test ADSMA with custom parameters"""
        window = 10
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = ads(df, parameters={
            'window': window,
            'sensitivity': 0.8
        })
        
        assert f'ADS_{window}' in columns


class TestVID:
    """Tests for the Variable Index Dynamic Average (VIDYA)"""

    def test_vid_calculation(self, sample_data):
        """Test basic VIDYA calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = vid(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'VID_21_9' in columns
        
        # VIDYA should have some valid values
        valid_result = result_data.dropna()
        assert len(valid_result) > 0

    def test_vid_custom_params(self, sample_data):
        """Test VIDYA with custom parameters"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = vid(df, parameters={
            'window': 14,
            'cmo_window': 7
        })
        
        assert 'VID_14_7' in columns


class TestTT3:
    """Tests for the T3 Moving Average"""

    def test_tt3_calculation(self, sample_data):
        """Test basic TT3 calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = tt3(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'TT3_5' in columns
        assert len(result_data) == len(sample_data['close'])

    def test_tt3_custom_params(self, sample_data):
        """Test TT3 with custom parameters"""
        window = 10
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = tt3(df, parameters={
            'window': window,
            'v_factor': 0.8
        })
        
        assert f'TT3_{window}' in columns


class TestMAM:
    """Tests for the MESA Adaptive Moving Average"""

    def test_mam_calculation(self, sample_data):
        """Test basic MAM calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = mam(df)
        
        assert isinstance(result_data, pd.DataFrame)
        assert not result_data.empty
        assert 'MAM' in columns
        assert 'MAM_FAMA' in columns
        assert len(result_data) == len(sample_data['close'])

    def test_mam_custom_params(self, sample_data):
        """Test MAM with custom parameters"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = mam(df, parameters={
            'fast_limit': 0.6,
            'slow_limit': 0.1
        })
        
        assert 'MAM' in columns
        assert 'MAM_FAMA' in columns


class TestEVW:
    """Tests for the Elastic Volume Weighted Moving Average"""

    def test_evw_calculation(self, sample_data):
        """Test basic EVW calculation"""
        df = pd.DataFrame({
            'Close': sample_data['close'],
            'Volume': np.random.randint(1000, 10000, len(sample_data['close']))
        })
        result_data, columns = evw(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'EVW_20' in columns
        assert len(result_data) == len(sample_data['close'])

    def test_evw_custom_window(self, sample_data):
        """Test EVW with custom window"""
        window = 10
        df = pd.DataFrame({
            'Close': sample_data['close'],
            'Volume': np.random.randint(1000, 10000, len(sample_data['close']))
        })
        result_data, columns = evw(df, parameters={'window': window})
        
        assert f'EVW_{window}' in columns


class TestTSF:
    """Tests for the Time Series Forecast"""

    def test_tsf_calculation(self, sample_data):
        """Test basic TSF calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = tsf(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'TSF_14' in columns
        
        # TSF should have NaN values at the start
        valid_result = result_data.dropna()
        assert len(valid_result) > 0

    def test_tsf_custom_window(self, sample_data):
        """Test TSF with custom window"""
        window = 10
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = tsf(df, parameters={'window': window})
        
        assert f'TSF_{window}' in columns

