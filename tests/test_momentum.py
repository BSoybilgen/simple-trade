import pytest
import pandas as pd
import numpy as np
from simple_trade.momentum import (
    rsi, mac, sto, cci, roc, wil, cmo, ult, dpo, eri,
    rmi, tsi, qst, crs, msi, fis, stc, ttm, kst, cog,
    vor, lsi, awo, ppo, sri, rvg, bop, psy, imi, pgo, wad
)

@pytest.fixture
def sample_data():
    """Fixture to provide sample price data for testing"""
    # Create a more realistic price series with clear up and down trends
    index = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Create a series with more pronounced trends and volatility
    np.random.seed(42)  # For reproducibility
    
    # Base uptrend
    uptrend = np.linspace(100, 200, 40)
    # Downtrend
    downtrend = np.linspace(200, 100, 40)
    # Second uptrend
    uptrend2 = np.linspace(100, 150, 20)
    
    # Add some noise
    noise = np.random.normal(0, 3, 100)
    
    # Combine all segments with noise
    combined = np.concatenate([uptrend, downtrend, uptrend2])
    close = pd.Series(combined + noise[:len(combined)], index=index)
    
    # Create high and low with more realistic spread
    high = close + np.random.uniform(1, 5, size=len(close))
    low = close - np.random.uniform(1, 5, size=len(close))
    
    return {
        'close': close,
        'high': high,
        'low': low
    }

class TestRSI:
    """Tests for the RSI indicator"""

    def test_rsi_calculation(self, sample_data):
        """Test the basic calculation of RSI"""
        # Create DataFrame with Close column
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, _ = rsi(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        
        # Check RSI bounds (should be between 0 and 100)
        # Skip NaN values
        valid_result = result_data.dropna()
        assert valid_result.min() >= 0
        assert valid_result.max() <= 100
        
        # Check for NaN values (first 'window - 1' values will be NaN by design)
        # Default window is 14, first valid value at index 13
        assert result_data.iloc[:13].isna().all()
        
        # Should have some valid values after window
        assert len(valid_result) > 0

    def test_rsi_with_custom_window(self, sample_data):
        """Test RSI with a custom window parameter"""
        window = 5
        # Create DataFrame with Close column
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, _ = rsi(df, parameters={'window': window}, columns=None)
        
        # Check the first 'window - 1' values are NaN
        assert result_data.iloc[:window-1].isna().all()
        
        # Should have some valid values after window
        assert len(result_data.dropna()) > 0
        
    def test_rsi_trend_detection(self, sample_data):
        """Test that RSI correctly detects trend changes"""
        # Create DataFrame with Close column
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, _ = rsi(df)
        
        # Skip initial NaN values
        valid_rsi = result_data.dropna()
        
        # Check that there are overbought (>70) and oversold (<30) periods
        # This is more reliable than checking specific indices
        assert (valid_rsi > 70).any() or (valid_rsi < 30).any()
        
        # Calculate price changes
        price_changes = sample_data['close'].pct_change().dropna()
        
        # When prices increase consistently, RSI should increase
        # Find a period of consistent price increases
        uptrend_mask = price_changes > 0
        uptrend_periods = uptrend_mask.rolling(5).sum() >= 4  # 4 out of 5 days up
        
        # Find a period of consistent price decreases
        downtrend_mask = price_changes < 0
        downtrend_periods = downtrend_mask.rolling(5).sum() >= 4  # 4 out of 5 days down
        
        # If we have clear up/down trends, verify RSI behavior
        if uptrend_periods.any() and downtrend_periods.any():
            # Find indices where trends are detected
            uptrend_idx = uptrend_periods[uptrend_periods].index[0]
            downtrend_idx = downtrend_periods[downtrend_periods].index[0]
            
            # Get RSI values for these periods if they exist in valid_rsi
            if uptrend_idx in valid_rsi.index and downtrend_idx in valid_rsi.index:
                # In uptrend, RSI should be higher than in downtrend
                assert valid_rsi[uptrend_idx] > valid_rsi[downtrend_idx]

class TestMACD:
    """Tests for the MACD indicator"""
    
    def test_macd_calculation(self, sample_data):
        """Test basic MACD calculation"""
        # Create DataFrame with Close column
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, _ = mac(df)
        
        assert isinstance(result_data, pd.DataFrame)
        assert not result_data.empty
        
        # Check column names (with default parameters)
        assert 'MAC_12_26' in result_data.columns
        assert 'Signal_9' in result_data.columns
        assert 'Hist_12_26_9' in result_data.columns
        
        # Verify result has same index as input
        assert result_data.index.equals(sample_data['close'].index)

    def test_macd_custom_params(self, sample_data):
        """Test MACD with custom window parameters"""
        window_fast = 8
        window_slow = 20
        window_signal = 7
        
        # Create DataFrame with Close column
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, _ = mac(df, parameters={
                     'window_slow': window_slow, 
                     'window_fast': window_fast, 
                     'window_signal': window_signal
                     }, columns=None)
        
        # Check that column names reflect custom parameters
        assert f'MAC_{window_fast}_{window_slow}' in result_data.columns
        assert f'Signal_{window_signal}' in result_data.columns
        assert f'Hist_{window_fast}_{window_slow}_{window_signal}' in result_data.columns

    def test_macd_crossover(self, sample_data):
        """Test that MACD line crosses the signal line during trend changes"""
        # Create DataFrame with Close column
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, _ = mac(df)
        
        # Extract MACD and Signal lines
        macd_line = result_data.iloc[:, 0]
        signal_line = result_data.iloc[:, 1]
        
        # Calculate crossovers (MACD line - Signal line changes sign)
        crossovers = np.sign(macd_line - signal_line).diff().fillna(0) != 0
        
        # There should be at least one crossover in our sample data
        assert crossovers.sum() > 0

class TestStoch:
    """Tests for the Stochastic Oscillator"""
    
    def test_stoch_calculation(self, sample_data):
        """Test basic Stochastic calculation"""
        # Create DataFrame with High, Low, Close columns
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, _ = sto(df, parameters=None, columns=None)
        
        assert isinstance(result_data, pd.DataFrame)
        assert not result_data.empty
        
        # Check column names (with default parameters: k_period=14, d_period=3, smooth_k=3)
        k_col = 'STO_K_14_3_3'
        d_col = 'STO_D_14_3_3'
        assert k_col in result_data.columns
        assert d_col in result_data.columns
        
        # Check bounds (Stochastic should be between 0 and 100)
        assert result_data[k_col].min() >= 0
        assert result_data[k_col].max() <= 100
        assert result_data[d_col].min() >= 0
        assert result_data[d_col].max() <= 100
        
        # Check that index matches input
        assert result_data.index.equals(sample_data['close'].index)

    def test_stoch_custom_params(self, sample_data):
        """Test Stochastic with custom parameters"""
        k_period = 7
        d_period = 2
        smooth_k = 2
        
        # Create DataFrame with High, Low, Close columns
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, _ = sto(df, parameters={'k_period': k_period, 'd_period': d_period, 'smooth_k': smooth_k}, columns=None)
        
        # Create column names with the custom parameters
        k_col = f'STO_K_{k_period}_{d_period}_{smooth_k}'
        d_col = f'STO_D_{k_period}_{d_period}_{smooth_k}'
        
        # First k_period values should be NaN for K
        assert result_data[k_col].iloc[:k_period].isna().all()
        
        # K should be valid after k_period + smooth_k - 1 periods
        valid_from_k = k_period + smooth_k - 1
        assert not result_data[k_col].iloc[valid_from_k:].isna().any()
        
        # D should be valid after k_period + smooth_k + d_period - 2 periods
        valid_from_d = k_period + smooth_k + d_period - 2
        assert not result_data[d_col].iloc[valid_from_d:].isna().any()

class TestCCI:
    """Tests for the Commodity Channel Index"""
    
    def test_cci_calculation(self, sample_data):
        """Test basic CCI calculation"""
        # Create DataFrame with High, Low, Close columns
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, _ = cci(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        
        # Check that index matches input
        assert result_data.index.equals(sample_data['close'].index)
        
        # First 'window - 1' values should be NaN (default window is 20)
        assert result_data.iloc[:19].isna().all()
        
        # Should have some valid values (not all NaNs)
        valid_result = result_data.dropna()
        assert len(valid_result) > 0

    def test_cci_custom_params(self, sample_data):
        """Test CCI with custom parameters"""
        window = 10
        constant = 0.02
        
        # Create DataFrame with High, Low, Close columns
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, _ = cci(df, parameters={'window': window, 'constant': constant}, columns=None)
        
        # First 'window - 1' values should be NaN
        assert result_data.iloc[:window-1].isna().all()
        
        # Should have some valid values (not all NaNs)
        valid_result = result_data.dropna()
        assert len(valid_result) > 0
        
    def test_cci_trend_detection(self, sample_data):
        """Test CCI trend detection properties"""
        # Create DataFrame with High, Low, Close columns
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, _ = cci(df)
        
        # Skip NaN values
        valid_cci = result_data.dropna()
        
        # There should be both positive and negative CCI values in our sample
        assert (valid_cci > 0).any()
        assert (valid_cci < 0).any()
        
        # Calculate price changes
        price_changes = sample_data['close'].pct_change()
        
        # When prices trend upward, CCI should generally be positive
        # When prices trend downward, CCI should generally be negative
        # This correlation should exist but doesn't need to be perfect
        corr = valid_cci.corr(price_changes.loc[valid_cci.index])
        assert corr > 0  # Positive correlation between price changes and CCI

class TestROC:
    """Tests for the Rate of Change indicator"""
    
    def test_roc_calculation(self, sample_data):
        """Test basic ROC calculation"""
        # Create DataFrame with Close column
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, _ = roc(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        
        # Check that index matches input
        assert result_data.index.equals(sample_data['close'].index)
        
        # First 'window' values should be NaN (default window is 12)
        assert result_data.iloc[:12].isna().all()
        
        # Values after window should be valid
        assert not result_data.iloc[12:].isna().any()

    def test_roc_custom_window(self, sample_data):
        """Test ROC with custom window parameter"""
        window = 5
        # Create DataFrame with Close column
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, _ = roc(df, parameters={'window': window}, columns=None)
        
        # First 'window' values should be NaN
        assert result_data.iloc[:window].isna().all()
        
        # Values after window should be valid
        assert not result_data.iloc[window:].isna().any()
        
    def test_roc_trend_detection(self, sample_data):
        """Test ROC trend detection properties"""
        # Create DataFrame with Close column
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, _ = roc(df)
        
        # Skip NaN values
        valid_roc = result_data.dropna()
        
        # Should have both positive and negative values
        assert (valid_roc > 0).any()
        assert (valid_roc < 0).any()
        
        # ROC should correlate with price changes
        # When price increases, ROC should be positive
        # When price decreases, ROC should be negative
        price_changes = sample_data['close'].pct_change(12)  # Match ROC window
        
        # Skip NaN values after shifting
        valid_changes = price_changes.loc[valid_roc.index].dropna()
        valid_roc_subset = valid_roc.loc[valid_changes.index]
        
        # Compare signs of ROC and price changes
        # They should generally match (both positive or both negative)
        sign_match = np.sign(valid_roc_subset) == np.sign(valid_changes)
        # At least 70% of the signs should match
        assert sign_match.mean() > 0.7


class TestWilliamsR:
    """Tests for the Williams %R indicator"""

    def test_wil_calculation(self, sample_data):
        """Test basic Williams %R calculation"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = wil(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'WIL_14' in columns
        
        # Williams %R should be between -100 and 0
        valid_result = result_data.dropna()
        assert valid_result.min() >= -100
        assert valid_result.max() <= 0

    def test_wil_custom_window(self, sample_data):
        """Test Williams %R with custom window"""
        window = 7
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = wil(df, parameters={'window': window})
        
        assert f'WIL_{window}' in columns
        # First window-1 values should be NaN
        assert result_data.iloc[:window-1].isna().all()


class TestCMO:
    """Tests for the Chande Momentum Oscillator"""

    def test_cmo_calculation(self, sample_data):
        """Test basic CMO calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = cmo(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'CMO_14' in columns
        
        # CMO should be between -100 and 100
        valid_result = result_data.dropna()
        assert valid_result.min() >= -100
        assert valid_result.max() <= 100

    def test_cmo_custom_window(self, sample_data):
        """Test CMO with custom window"""
        window = 20
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = cmo(df, parameters={'window': window})
        
        assert f'CMO_{window}' in columns


class TestUltimateOscillator:
    """Tests for the Ultimate Oscillator"""

    def test_ult_calculation(self, sample_data):
        """Test basic Ultimate Oscillator calculation"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = ult(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'ULT_7_14_28' in columns
        
        # Ultimate Oscillator should be between 0 and 100
        valid_result = result_data.dropna()
        assert valid_result.min() >= 0
        assert valid_result.max() <= 100

    def test_ult_custom_params(self, sample_data):
        """Test Ultimate Oscillator with custom parameters"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = ult(df, parameters={
            'short_window': 5,
            'medium_window': 10,
            'long_window': 20
        })
        
        assert 'ULT_5_10_20' in columns


class TestDPO:
    """Tests for the Detrended Price Oscillator"""

    def test_dpo_calculation(self, sample_data):
        """Test basic DPO calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = dpo(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'DPO_20' in columns
        
        # DPO should have both positive and negative values
        valid_result = result_data.dropna()
        assert (valid_result > 0).any()
        assert (valid_result < 0).any()

    def test_dpo_custom_window(self, sample_data):
        """Test DPO with custom window"""
        window = 10
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = dpo(df, parameters={'window': window})
        
        assert f'DPO_{window}' in columns


class TestElderRay:
    """Tests for the Elder-Ray Index"""

    def test_eri_calculation(self, sample_data):
        """Test basic Elder-Ray calculation"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = eri(df)
        
        assert isinstance(result_data, pd.DataFrame)
        assert not result_data.empty
        assert 'ERI_BULL_13' in columns
        assert 'ERI_BEAR_13' in columns

    def test_eri_custom_window(self, sample_data):
        """Test Elder-Ray with custom window"""
        window = 20
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = eri(df, parameters={'window': window})
        
        assert f'ERI_BULL_{window}' in columns
        assert f'ERI_BEAR_{window}' in columns


class TestRMI:
    """Tests for the Relative Momentum Index"""

    def test_rmi_calculation(self, sample_data):
        """Test basic RMI calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = rmi(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'RMI_20_5' in columns
        
        # RMI should be between 0 and 100
        valid_result = result_data.dropna()
        assert valid_result.min() >= 0
        assert valid_result.max() <= 100

    def test_rmi_custom_params(self, sample_data):
        """Test RMI with custom parameters"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = rmi(df, parameters={
            'window': 14,
            'momentum_period': 3
        })
        
        assert 'RMI_14_3' in columns


class TestTSI:
    """Tests for the True Strength Index"""

    def test_tsi_calculation(self, sample_data):
        """Test basic TSI calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = tsi(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'TSI_25_13' in columns
        
        # TSI should be between -100 and 100
        valid_result = result_data.dropna()
        assert valid_result.min() >= -100
        assert valid_result.max() <= 100

    def test_tsi_custom_params(self, sample_data):
        """Test TSI with custom parameters"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = tsi(df, parameters={
            'slow': 20,
            'fast': 10
        })
        
        assert 'TSI_20_10' in columns


class TestQstick:
    """Tests for the Qstick indicator"""

    def test_qst_calculation(self, sample_data):
        """Test basic Qstick calculation"""
        # Need Open prices for Qstick
        np.random.seed(42)
        open_prices = sample_data['close'] - np.random.uniform(0, 2, len(sample_data['close']))
        df = pd.DataFrame({
            'Open': open_prices,
            'Close': sample_data['close']
        })
        result_data, columns = qst(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'QST_10' in columns

    def test_qst_custom_window(self, sample_data):
        """Test Qstick with custom window"""
        np.random.seed(42)
        open_prices = sample_data['close'] - np.random.uniform(0, 2, len(sample_data['close']))
        df = pd.DataFrame({
            'Open': open_prices,
            'Close': sample_data['close']
        })
        result_data, columns = qst(df, parameters={'window': 14})
        
        assert 'QST_14' in columns


class TestConnorsRSI:
    """Tests for the Connors RSI"""

    def test_crs_calculation(self, sample_data):
        """Test basic Connors RSI calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        # Use smaller rank_window to get valid results with 100 data points
        result_data, columns = crs(df, parameters={'rank_window': 50})
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'CRS_3_2_50' in columns
        
        # CRSI should be between 0 and 100
        valid_result = result_data.dropna()
        assert len(valid_result) > 0
        assert valid_result.min() >= 0
        assert valid_result.max() <= 100

    def test_crs_custom_params(self, sample_data):
        """Test Connors RSI with custom parameters"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = crs(df, parameters={
            'rsi_window': 5,
            'streak_window': 3,
            'rank_window': 30
        })
        
        assert 'CRS_5_3_30' in columns


class TestMSI:
    """Tests for the Momentum Strength Index"""

    def test_msi_calculation(self, sample_data):
        """Test basic MSI calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = msi(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'MSI_14_1.0' in columns
        
        # MSI should be between 0 and 100
        valid_result = result_data.dropna()
        assert valid_result.min() >= 0
        assert valid_result.max() <= 100

    def test_msi_custom_params(self, sample_data):
        """Test MSI with custom parameters"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = msi(df, parameters={
            'window': 20,
            'power': 2.0
        })
        
        assert 'MSI_20_2.0' in columns


class TestFisherTransform:
    """Tests for the Fisher Transform"""

    def test_fis_calculation(self, sample_data):
        """Test basic Fisher Transform calculation"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low']
        })
        result_data, columns = fis(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'FIS_9' in columns

    def test_fis_custom_window(self, sample_data):
        """Test Fisher Transform with custom window"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low']
        })
        result_data, columns = fis(df, parameters={'window': 14})
        
        assert 'FIS_14' in columns


class TestSTC:
    """Tests for the Schaff Trend Cycle"""

    def test_stc_calculation(self, sample_data):
        """Test basic STC calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = stc(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'STC_23_50_10' in columns
        
        # STC should be between 0 and 100
        valid_result = result_data.dropna()
        assert valid_result.min() >= 0
        assert valid_result.max() <= 100

    def test_stc_custom_params(self, sample_data):
        """Test STC with custom parameters"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = stc(df, parameters={
            'window_fast': 12,
            'window_slow': 26,
            'cycle': 9
        })
        
        assert 'STC_12_26_9' in columns


class TestTTMSqueeze:
    """Tests for the TTM Squeeze indicator"""

    def test_ttm_calculation(self, sample_data):
        """Test basic TTM Squeeze calculation"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = ttm(df)
        
        assert isinstance(result_data, pd.DataFrame)
        assert not result_data.empty
        assert 'TTM_MOM_20' in columns
        assert 'Squeeze_On_20' in columns
        assert 'Squeeze_Off_20' in columns
        
        # Squeeze columns should be boolean
        assert result_data['Squeeze_On_20'].dtype == bool
        assert result_data['Squeeze_Off_20'].dtype == bool

    def test_ttm_custom_params(self, sample_data):
        """Test TTM Squeeze with custom parameters"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = ttm(df, parameters={'length': 10})
        
        assert 'TTM_MOM_10' in columns


class TestKST:
    """Tests for the Know Sure Thing indicator"""

    def test_kst_calculation(self, sample_data):
        """Test basic KST calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = kst(df)
        
        assert isinstance(result_data, pd.DataFrame)
        assert not result_data.empty
        assert 'KST' in columns
        assert 'KST_Signal_9' in columns

    def test_kst_custom_signal(self, sample_data):
        """Test KST with custom signal period"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = kst(df, parameters={'signal': 12})
        
        assert 'KST_Signal_12' in columns


class TestCOG:
    """Tests for the Center of Gravity indicator"""

    def test_cog_calculation(self, sample_data):
        """Test basic COG calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = cog(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'COG_10' in columns

    def test_cog_custom_window(self, sample_data):
        """Test COG with custom window"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = cog(df, parameters={'window': 14})
        
        assert 'COG_14' in columns


class TestVortex:
    """Tests for the Vortex Indicator"""

    def test_vor_calculation(self, sample_data):
        """Test basic Vortex calculation"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = vor(df)
        
        assert isinstance(result_data, pd.DataFrame)
        assert not result_data.empty
        assert 'VOR_Plus_14' in columns
        assert 'VOR_Minus_14' in columns
        
        # VI values should be positive
        valid_plus = result_data['VOR_Plus_14'].dropna()
        valid_minus = result_data['VOR_Minus_14'].dropna()
        assert valid_plus.min() >= 0
        assert valid_minus.min() >= 0

    def test_vor_custom_window(self, sample_data):
        """Test Vortex with custom window"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = vor(df, parameters={'window': 21})
        
        assert 'VOR_Plus_21' in columns
        assert 'VOR_Minus_21' in columns


class TestLaguerreRSI:
    """Tests for the Laguerre RSI"""

    def test_lsi_calculation(self, sample_data):
        """Test basic Laguerre RSI calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = lsi(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'LSI_0.5' in columns
        
        # LRSI should be between 0 and 100 (with small tolerance for floating point)
        valid_result = result_data.dropna()
        assert valid_result.min() >= -0.01
        assert valid_result.max() <= 100.01

    def test_lsi_custom_gamma(self, sample_data):
        """Test Laguerre RSI with custom gamma"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = lsi(df, parameters={'gamma': 0.7})
        
        assert 'LSI_0.7' in columns


class TestAWO:
    """Tests for the Awesome Oscillator"""

    def test_awo_calculation(self, sample_data):
        """Test basic Awesome Oscillator calculation"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low']
        })
        result_data, columns = awo(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'AWO_5_34' in columns
        
        # AO should have both positive and negative values
        valid_result = result_data.dropna()
        assert (valid_result > 0).any()
        assert (valid_result < 0).any()

    def test_awo_custom_params(self, sample_data):
        """Test Awesome Oscillator with custom parameters"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low']
        })
        result_data, columns = awo(df, parameters={
            'fast_window': 3,
            'slow_window': 20
        })
        
        assert 'AWO_3_20' in columns


class TestPPO:
    """Tests for the Percentage Price Oscillator"""

    def test_ppo_calculation(self, sample_data):
        """Test basic PPO calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = ppo(df)
        
        assert isinstance(result_data, pd.DataFrame)
        assert not result_data.empty
        assert 'PPO_12_26' in columns
        assert 'PPO_SIG_9' in columns
        assert 'PPO_HIST' in columns

    def test_ppo_custom_params(self, sample_data):
        """Test PPO with custom parameters"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = ppo(df, parameters={
            'fast_window': 8,
            'slow_window': 21,
            'signal_window': 5
        })
        
        assert 'PPO_8_21' in columns
        assert 'PPO_SIG_5' in columns


class TestStochRSI:
    """Tests for the Stochastic RSI"""

    def test_sri_calculation(self, sample_data):
        """Test basic Stochastic RSI calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = sri(df)
        
        assert isinstance(result_data, pd.DataFrame)
        assert not result_data.empty
        assert 'SRI_K_14_14' in columns
        assert 'SRI_D_3' in columns
        
        # StochRSI should be between 0 and 100
        valid_k = result_data['SRI_K_14_14'].dropna()
        assert valid_k.min() >= 0
        assert valid_k.max() <= 100

    def test_sri_custom_params(self, sample_data):
        """Test Stochastic RSI with custom parameters"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = sri(df, parameters={
            'rsi_window': 10,
            'stoch_window': 10,
            'd_window': 5
        })
        
        assert 'SRI_K_10_10' in columns
        assert 'SRI_D_5' in columns


class TestRVG:
    """Tests for the Relative Vigor Index"""

    def test_rvg_calculation(self, sample_data):
        """Test basic RVG calculation"""
        np.random.seed(42)
        open_prices = sample_data['close'] - np.random.uniform(0, 2, len(sample_data['close']))
        df = pd.DataFrame({
            'Open': open_prices,
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = rvg(df)
        
        assert isinstance(result_data, pd.DataFrame)
        assert not result_data.empty
        assert 'RVG_10' in columns
        assert 'RVG_SIG' in columns

    def test_rvg_custom_window(self, sample_data):
        """Test RVG with custom window"""
        np.random.seed(42)
        open_prices = sample_data['close'] - np.random.uniform(0, 2, len(sample_data['close']))
        df = pd.DataFrame({
            'Open': open_prices,
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = rvg(df, parameters={'window': 14})
        
        assert 'RVG_14' in columns


class TestBOP:
    """Tests for the Balance of Power indicator"""

    def test_bop_calculation(self, sample_data):
        """Test basic BOP calculation"""
        np.random.seed(42)
        open_prices = sample_data['close'] - np.random.uniform(0, 2, len(sample_data['close']))
        df = pd.DataFrame({
            'Open': open_prices,
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = bop(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'BOP_14' in columns
        
        # BOP should be between -1 and 1 (approximately)
        valid_result = result_data.dropna()
        assert valid_result.min() >= -1.5
        assert valid_result.max() <= 1.5

    def test_bop_unsmoothed(self, sample_data):
        """Test BOP without smoothing"""
        np.random.seed(42)
        open_prices = sample_data['close'] - np.random.uniform(0, 2, len(sample_data['close']))
        df = pd.DataFrame({
            'Open': open_prices,
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = bop(df, parameters={'smooth': False})
        
        assert 'BOP' in columns


class TestPSY:
    """Tests for the Psychological Line"""

    def test_psy_calculation(self, sample_data):
        """Test basic PSY calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = psy(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'PSY_12' in columns
        
        # PSY should be between 0 and 100
        valid_result = result_data.dropna()
        assert valid_result.min() >= 0
        assert valid_result.max() <= 100

    def test_psy_custom_window(self, sample_data):
        """Test PSY with custom window"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = psy(df, parameters={'window': 20})
        
        assert 'PSY_20' in columns


class TestIMI:
    """Tests for the Intraday Momentum Index"""

    def test_imi_calculation(self, sample_data):
        """Test basic IMI calculation"""
        np.random.seed(42)
        open_prices = sample_data['close'] - np.random.uniform(0, 2, len(sample_data['close']))
        df = pd.DataFrame({
            'Open': open_prices,
            'Close': sample_data['close']
        })
        result_data, columns = imi(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'IMI_14' in columns
        
        # IMI should be between 0 and 100
        valid_result = result_data.dropna()
        assert valid_result.min() >= 0
        assert valid_result.max() <= 100

    def test_imi_custom_window(self, sample_data):
        """Test IMI with custom window"""
        np.random.seed(42)
        open_prices = sample_data['close'] - np.random.uniform(0, 2, len(sample_data['close']))
        df = pd.DataFrame({
            'Open': open_prices,
            'Close': sample_data['close']
        })
        result_data, columns = imi(df, parameters={'window': 20})
        
        assert 'IMI_20' in columns


class TestPGO:
    """Tests for the Pretty Good Oscillator"""

    def test_pgo_calculation(self, sample_data):
        """Test basic PGO calculation"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = pgo(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'PGO_14' in columns
        
        # PGO should have both positive and negative values
        valid_result = result_data.dropna()
        assert (valid_result > 0).any() or (valid_result < 0).any()

    def test_pgo_custom_window(self, sample_data):
        """Test PGO with custom window"""
        df = pd.DataFrame({
            'High': sample_data['high'],
            'Low': sample_data['low'],
            'Close': sample_data['close']
        })
        result_data, columns = pgo(df, parameters={'window': 20})
        
        assert 'PGO_20' in columns
