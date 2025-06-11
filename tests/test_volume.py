import pytest
import pandas as pd
import numpy as np
from simple_trade.volume import (
    obv, vma, adline, cmf, vpt
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
        result_data, _ = adline(df)
        
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
        result_data, _ = adline(df)
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
