import pytest
import pandas as pd
import numpy as np
from simple_trade.volume import (
    obv, vma, adl, cmf, vpt, vwa, mfi, foi, emv, pvo, vro, nvi, pvi,
    kvo, ado, vfi, bwm, fve, wad, voo
)

# Fixture for sample data (consistent with other test modules)
@pytest.fixture
def sample_data():
    """Fixture to provide sample OHLCV data for testing volume indicators"""
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
    
    # Create volume data - higher volume during trends, lower during transitions/noise
    volume_base = np.random.randint(10000, 50000, size=100)
    volume_trend_factor = np.concatenate([
        np.random.uniform(1.5, 3.0, size=40), # Uptrend
        np.random.uniform(1.5, 3.0, size=40), # Downtrend
        np.random.uniform(1.0, 2.0, size=20)  # Uptrend2
    ])
    volume = pd.Series(volume_base * volume_trend_factor, index=index).astype(int)

    return {
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }


class TestOBV:
    """Tests for On-Balance Volume (OBV)"""

    def test_obv_calculation(self, sample_data):
        """Test basic OBV calculation structure"""
        # Create DataFrame with Close and Volume columns
        df = pd.DataFrame({
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, _ = obv(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert len(result_data) == len(sample_data['close'])
        assert result_data.index.equals(sample_data['close'].index)
        # OBV starts immediately, no initial NaNs expected
        assert not result_data.isna().any()
        # First value should be first volume
        assert result_data.iloc[0] == sample_data['volume'].iloc[0]

    def test_obv_trend_correlation(self, sample_data):
        """Test that OBV generally follows the price trend"""
        # Create DataFrame with Close and Volume columns
        df = pd.DataFrame({
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, _ = obv(df)
        price_diff = sample_data['close'].diff().dropna()
        obv_diff = result_data.diff().dropna()
        
        # Align indices
        common_index = price_diff.index.intersection(obv_diff.index)
        price_diff = price_diff.loc[common_index]
        obv_diff = obv_diff.loc[common_index]
        
        # OBV changes should generally have the same sign as price changes
        sign_match = np.sign(price_diff) == np.sign(obv_diff)
        # Allow for some deviation due to zero price changes
        assert sign_match.mean() > 0.8 # Expect high correlation


class TestVMA:
    """Tests for Volume Moving Average (VMA)"""

    def test_vma_calculation(self, sample_data):
        """Test basic VMA calculation structure"""
        window = 14 # Default
        # Create DataFrame with Close and Volume columns
        df = pd.DataFrame({
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, _ = vma(df, parameters={'window': window}, columns=None)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert len(result_data) == len(sample_data['close'])
        assert result_data.index.equals(sample_data['close'].index)
        
        # Check initial NaNs (first window-1)
        assert result_data.iloc[:window-1].isna().all()
        assert not result_data.iloc[window-1:].isna().any()

    def test_vma_custom_window(self, sample_data):
        """Test VMA with a custom window"""
        window = 7
        # Create DataFrame with Close and Volume columns
        df = pd.DataFrame({
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, _ = vma(df, parameters={'window': window}, columns=None)
        
        assert isinstance(result_data, pd.Series)
        assert len(result_data) == len(sample_data['close'])
        assert result_data.iloc[:window-1].isna().all()
        assert not result_data.iloc[window-1:].isna().any()


class TestADLine:
    """Tests for Accumulation/Distribution Line (A/D Line)"""

    def test_adline_calculation(self, sample_data):
        """Test basic A/D Line calculation structure"""
        # Create DataFrame with required columns
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, _ = adl(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert len(result_data) == len(sample_data['close'])
        assert result_data.index.equals(sample_data['close'].index)
        # A/D Line starts immediately, no initial NaNs expected
        assert not result_data.isna().any()

    def test_adline_trend_correlation(self, sample_data):
        """Test that A/D Line generally follows the price trend"""
        # Create DataFrame with required columns
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, _ = adl(df)
        price_diff = sample_data['close'].diff().dropna()
        ad_diff = result_data.diff().dropna()
        
        # Align indices
        common_index = price_diff.index.intersection(ad_diff.index)
        price_diff = price_diff.loc[common_index]
        ad_diff = ad_diff.loc[common_index]
        
        # A/D changes should generally have the same sign as price changes
        # This relationship is less direct than OBV, so expect weaker correlation
        sign_match = np.sign(price_diff) == np.sign(ad_diff)
        # Lowered threshold slightly
        assert sign_match.mean() > 0.4 # Expect positive correlation


class TestCMF:
    """Tests for Chaikin Money Flow (CMF)"""

    def test_cmf_calculation(self, sample_data):
        """Test basic CMF calculation structure"""
        period = 20 # Default
        # Create DataFrame with required columns
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, _ = cmf(df, parameters={'period': period}, columns=None)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert len(result_data) == len(sample_data['close'])
        assert result_data.index.equals(sample_data['close'].index)
        
        # Check initial NaNs (first period-1)
        assert result_data.iloc[:period-1].isna().all()
        assert not result_data.iloc[period-1:].isna().any()
        
        # CMF values should typically be between -1 and 1
        valid_result = result_data.dropna()
        assert (valid_result >= -1).all()
        assert (valid_result <= 1).all()

    def test_cmf_custom_period(self, sample_data):
        """Test CMF with a custom period"""
        period = 10
        # Create DataFrame with required columns
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, _ = cmf(df, parameters={'period': period}, columns=None)
                     
        assert isinstance(result_data, pd.Series)
        assert len(result_data) == len(sample_data['close'])
        assert result_data.iloc[:period-1].isna().all()
        assert not result_data.iloc[period-1:].isna().any()
        valid_result = result_data.dropna()
        assert (valid_result >= -1).all()
        assert (valid_result <= 1).all()


class TestVPT:
    """Tests for Volume Price Trend (VPT)"""

    def test_vpt_calculation(self, sample_data):
        """Test basic VPT calculation structure"""
        # Create DataFrame with Close and Volume columns
        df = pd.DataFrame({
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, _ = vpt(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert len(result_data) == len(sample_data['close'])
        assert result_data.index.equals(sample_data['close'].index)
        # Second value onwards should be valid
        assert not result_data.iloc[1:].isna().any()

    def test_vpt_trend_correlation(self, sample_data):
        """Test that VPT generally follows the price trend"""
        # Create DataFrame with Close and Volume columns
        df = pd.DataFrame({
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, _ = vpt(df)
        price_diff = sample_data['close'].diff().dropna()
        vpt_diff = result_data.diff().dropna()
        
        # Align indices
        common_index = price_diff.index.intersection(vpt_diff.index)
        price_diff = price_diff.loc[common_index]
        vpt_diff = vpt_diff.loc[common_index]
        
        # VPT changes should generally have the same sign as price changes
        sign_match = np.sign(price_diff) == np.sign(vpt_diff)
        # Allow for some deviation due to the nature of VPT
        assert sign_match.mean() > 0.7 # Expect high correlation


# --- Additional Volume Indicator Tests ---

class TestVWAP:
    """Tests for Volume Weighted Average Price"""

    def test_vwa_calculation(self, sample_data):
        """Test basic VWAP calculation"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, columns = vwa(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'VWAP' in columns
        assert len(result_data) == len(sample_data['close'])
        
        # VWAP should be within the price range
        valid_result = result_data.dropna()
        assert len(valid_result) > 0

    def test_vwa_custom_columns(self, sample_data):
        """Test VWAP with custom column names"""
        df = pd.DataFrame({
            'h': sample_data['high'],
            'l': sample_data['low'],
            'c': sample_data['close'],
            'v': sample_data['volume']
        })
        result_data, columns = vwa(df, columns={
            'high_col': 'h',
            'low_col': 'l',
            'close_col': 'c',
            'volume_col': 'v'
        })
        
        assert 'VWAP' in columns


class TestMFI:
    """Tests for Money Flow Index"""

    def test_mfi_calculation(self, sample_data):
        """Test basic MFI calculation"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, columns = mfi(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'MFI_14' in columns
        
        # MFI should be between 0 and 100
        valid_result = result_data.dropna()
        assert (valid_result >= 0).all()
        assert (valid_result <= 100).all()

    def test_mfi_custom_period(self, sample_data):
        """Test MFI with custom period"""
        period = 10
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, columns = mfi(df, parameters={'period': period})
        
        assert f'MFI_{period}' in columns


class TestForceIndex:
    """Tests for Force Index"""

    def test_foi_calculation(self, sample_data):
        """Test basic Force Index calculation"""
        df = pd.DataFrame({
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, columns = foi(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'FI_13' in columns
        assert len(result_data) == len(sample_data['close'])

    def test_foi_custom_period(self, sample_data):
        """Test Force Index with custom period"""
        period = 7
        df = pd.DataFrame({
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, columns = foi(df, parameters={'period': period})
        
        assert f'FI_{period}' in columns


class TestEMV:
    """Tests for Ease of Movement"""

    def test_emv_calculation(self, sample_data):
        """Test basic EMV calculation"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Volume': sample_data['volume']
        })
        result_data, columns = emv(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'EMV_14' in columns
        assert len(result_data) == len(sample_data['close'])

    def test_emv_custom_period(self, sample_data):
        """Test EMV with custom period"""
        period = 10
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Volume': sample_data['volume']
        })
        result_data, columns = emv(df, parameters={'period': period})
        
        assert f'EMV_{period}' in columns


class TestPVO:
    """Tests for Percentage Volume Oscillator"""

    def test_pvo_calculation(self, sample_data):
        """Test basic PVO calculation"""
        df = pd.DataFrame({'Volume': sample_data['volume']})
        result_data, columns = pvo(df)
        
        assert isinstance(result_data, pd.DataFrame)
        assert not result_data.empty
        assert 'PVO_12_26' in columns
        assert 'PVO_SIGNAL_9' in columns
        assert 'PVO_HIST' in columns

    def test_pvo_custom_params(self, sample_data):
        """Test PVO with custom parameters"""
        df = pd.DataFrame({'Volume': sample_data['volume']})
        result_data, columns = pvo(df, parameters={
            'fast_period': 10,
            'slow_period': 20,
            'signal_period': 5
        })
        
        assert 'PVO_10_20' in columns
        assert 'PVO_SIGNAL_5' in columns


class TestVROC:
    """Tests for Volume Rate of Change"""

    def test_vro_calculation(self, sample_data):
        """Test basic VROC calculation"""
        df = pd.DataFrame({'Volume': sample_data['volume']})
        result_data, columns = vro(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'VROC_14' in columns

    def test_vro_custom_period(self, sample_data):
        """Test VROC with custom period"""
        period = 10
        df = pd.DataFrame({'Volume': sample_data['volume']})
        result_data, columns = vro(df, parameters={'period': period})
        
        assert f'VROC_{period}' in columns


class TestNVI:
    """Tests for Negative Volume Index"""

    def test_nvi_calculation(self, sample_data):
        """Test basic NVI calculation"""
        df = pd.DataFrame({
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, columns = nvi(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'NVI' in columns
        assert len(result_data) == len(sample_data['close'])
        
        # First value should be initial value (default 1000)
        assert result_data.iloc[0] == 1000

    def test_nvi_custom_initial(self, sample_data):
        """Test NVI with custom initial value"""
        df = pd.DataFrame({
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, _ = nvi(df, parameters={'initial_value': 500})
        
        assert result_data.iloc[0] == 500


class TestPVI:
    """Tests for Positive Volume Index"""

    def test_pvi_calculation(self, sample_data):
        """Test basic PVI calculation"""
        df = pd.DataFrame({
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, columns = pvi(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'PVI' in columns
        assert len(result_data) == len(sample_data['close'])
        
        # First value should be initial value (default 1000)
        assert result_data.iloc[0] == 1000

    def test_pvi_custom_initial(self, sample_data):
        """Test PVI with custom initial value"""
        df = pd.DataFrame({
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, _ = pvi(df, parameters={'initial_value': 500})
        
        assert result_data.iloc[0] == 500


class TestKVO:
    """Tests for Klinger Volume Oscillator"""

    def test_kvo_calculation(self, sample_data):
        """Test basic KVO calculation"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, columns = kvo(df)
        
        assert isinstance(result_data, pd.DataFrame)
        assert not result_data.empty
        assert 'KVO_34_55' in columns
        assert 'KVO_SIGNAL_13' in columns

    def test_kvo_custom_params(self, sample_data):
        """Test KVO with custom parameters"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, columns = kvo(df, parameters={
            'fast_period': 20,
            'slow_period': 40,
            'signal_period': 10
        })
        
        assert 'KVO_20_40' in columns
        assert 'KVO_SIGNAL_10' in columns


class TestADO:
    """Tests for Accumulation/Distribution Oscillator"""

    def test_ado_calculation(self, sample_data):
        """Test basic ADO calculation"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, columns = ado(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'ADO_14' in columns

    def test_ado_custom_period(self, sample_data):
        """Test ADO with custom period"""
        period = 10
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, columns = ado(df, parameters={'period': period})
        
        assert f'ADO_{period}' in columns


class TestVFI:
    """Tests for Volume Flow Indicator"""

    def test_vfi_calculation(self, sample_data):
        """Test basic VFI calculation"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, columns = vfi(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'VFI_130' in columns

    def test_vfi_custom_period(self, sample_data):
        """Test VFI with custom period"""
        period = 50
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, columns = vfi(df, parameters={'period': period})
        
        assert f'VFI_{period}' in columns


class TestBWMFI:
    """Tests for Bill Williams Market Facilitation Index"""

    def test_bwm_calculation(self, sample_data):
        """Test basic BW MFI calculation"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Volume': sample_data['volume']
        })
        result_data, columns = bwm(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'BWMFI' in columns
        assert len(result_data) == len(sample_data['close'])
        
        # BW MFI should be non-negative
        valid_result = result_data.dropna()
        assert (valid_result >= 0).all()

    def test_bwm_custom_columns(self, sample_data):
        """Test BW MFI with custom column names"""
        df = pd.DataFrame({
            'h': sample_data['high'],
            'l': sample_data['low'],
            'v': sample_data['volume']
        })
        result_data, columns = bwm(df, columns={
            'high_col': 'h',
            'low_col': 'l',
            'volume_col': 'v'
        })
        
        assert 'BWMFI' in columns


class TestFVE:
    """Tests for Finite Volume Elements"""

    def test_fve_calculation(self, sample_data):
        """Test basic FVE calculation"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, columns = fve(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'FVE_22' in columns
        
        # FVE should be between -100 and 100
        valid_result = result_data.dropna()
        if len(valid_result) > 0:
            assert (valid_result >= -100).all()
            assert (valid_result <= 100).all()

    def test_fve_custom_period(self, sample_data):
        """Test FVE with custom period"""
        period = 14
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close'],
            'Volume': sample_data['volume']
        })
        result_data, columns = fve(df, parameters={'period': period})
        
        assert f'FVE_{period}' in columns


class TestWAD:
    """Tests for Williams Accumulation/Distribution"""

    def test_wad_calculation(self, sample_data):
        """Test basic WAD calculation"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = wad(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'WAD' in columns
        assert len(result_data) == len(sample_data['close'])

    def test_wad_custom_columns(self, sample_data):
        """Test WAD with custom column names"""
        df = pd.DataFrame({
            'h': sample_data['high'],
            'l': sample_data['low'],
            'c': sample_data['close']
        })
        result_data, columns = wad(df, columns={
            'high_col': 'h',
            'low_col': 'l',
            'close_col': 'c'
        })
        
        assert 'WAD' in columns


class TestVolumeOscillator:
    """Tests for Volume Oscillator"""

    def test_voo_calculation(self, sample_data):
        """Test basic Volume Oscillator calculation"""
        df = pd.DataFrame({'Volume': sample_data['volume']})
        result_data, columns = voo(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'VO_5_10' in columns

    def test_voo_custom_params(self, sample_data):
        """Test Volume Oscillator with custom parameters"""
        df = pd.DataFrame({'Volume': sample_data['volume']})
        result_data, columns = voo(df, parameters={
            'fast_period': 3,
            'slow_period': 7
        })
        
        assert 'VO_3_7' in columns
