import pytest
import pandas as pd
import numpy as np
from simple_trade.statistics import (
    std, kur, mab, med, qua, skw, var, zsc
)


# Fixture for sample data (consistent with other test modules)
@pytest.fixture
def sample_data():
    """Fixture to provide sample OHLCV data for testing statistics indicators"""
    index = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)  # for reproducibility

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
        
    def test_std_ddof_parameter(self, sample_data):
        """Test STD with sample standard deviation (ddof=1)"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_pop, _ = std(df, parameters={'ddof': 0})
        result_sample, _ = std(df, parameters={'ddof': 1})
        
        # Sample std should be slightly larger than population std
        valid_pop = result_pop.dropna()
        valid_sample = result_sample.dropna()
        assert (valid_sample >= valid_pop).all()


class TestKurtosis:
    """Tests for Kurtosis"""

    def test_kur_calculation(self, sample_data):
        """Test basic Kurtosis calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = kur(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'KUR_20' in columns
        
        # Kurtosis should have valid values (can be negative or positive)
        valid_result = result_data.dropna()
        assert len(valid_result) > 0

    def test_kur_custom_window(self, sample_data):
        """Test Kurtosis with custom window"""
        window = 14
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = kur(df, parameters={'window': window})
        
        assert f'KUR_{window}' in columns


class TestMeanAbsoluteDeviation:
    """Tests for Mean Absolute Deviation"""

    def test_mab_calculation(self, sample_data):
        """Test basic MAB calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = mab(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'MAB_20' in columns
        
        # MAB should be positive
        valid_result = result_data.dropna()
        assert (valid_result >= 0).all()

    def test_mab_custom_window(self, sample_data):
        """Test MAB with custom window"""
        window = 10
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = mab(df, parameters={'window': window})
        
        assert f'MAB_{window}' in columns
        
    def test_mab_less_than_std(self, sample_data):
        """Test that MAB is generally less than or equal to STD"""
        df = pd.DataFrame({'Close': sample_data['close']})
        mab_result, _ = mab(df)
        std_result, _ = std(df)
        
        # MAB should generally be less than or equal to STD
        valid_mab = mab_result.dropna()
        valid_std = std_result.dropna()
        # This is a statistical property, not always true for every point
        assert valid_mab.mean() <= valid_std.mean()


class TestMedian:
    """Tests for Rolling Median"""

    def test_med_calculation(self, sample_data):
        """Test basic Median calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = med(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'MED_20' in columns
        
        # Median should be within the range of close prices
        valid_result = result_data.dropna()
        assert (valid_result >= sample_data['close'].min()).all()
        assert (valid_result <= sample_data['close'].max()).all()

    def test_med_custom_window(self, sample_data):
        """Test Median with custom window"""
        window = 10
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = med(df, parameters={'window': window})
        
        assert f'MED_{window}' in columns


class TestQuantile:
    """Tests for Rolling Quantile"""

    def test_qua_calculation(self, sample_data):
        """Test basic Quantile calculation (default 0.5 = median)"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = qua(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'QUA_20_50' in columns
        
        # Quantile should be within the range of close prices
        valid_result = result_data.dropna()
        assert (valid_result >= sample_data['close'].min()).all()
        assert (valid_result <= sample_data['close'].max()).all()

    def test_qua_custom_quantile(self, sample_data):
        """Test Quantile with custom quantile value"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_25, columns_25 = qua(df, parameters={'quantile': 0.25})
        result_75, columns_75 = qua(df, parameters={'quantile': 0.75})
        
        assert 'QUA_20_25' in columns_25
        assert 'QUA_20_75' in columns_75
        
        # 75th percentile should be >= 25th percentile
        valid_25 = result_25.dropna()
        valid_75 = result_75.dropna()
        assert (valid_75 >= valid_25).all()

    def test_qua_custom_window(self, sample_data):
        """Test Quantile with custom window"""
        window = 14
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = qua(df, parameters={'window': window})
        
        assert f'QUA_{window}_50' in columns


class TestSkewness:
    """Tests for Skewness"""

    def test_skw_calculation(self, sample_data):
        """Test basic Skewness calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = skw(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'SKW_20' in columns
        
        # Skewness can be negative or positive
        valid_result = result_data.dropna()
        assert len(valid_result) > 0

    def test_skw_custom_window(self, sample_data):
        """Test Skewness with custom window"""
        window = 14
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = skw(df, parameters={'window': window})
        
        assert f'SKW_{window}' in columns


class TestVariance:
    """Tests for Variance"""

    def test_var_calculation(self, sample_data):
        """Test basic Variance calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = var(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'VAR_20' in columns
        
        # Variance should be positive
        valid_result = result_data.dropna()
        assert (valid_result >= 0).all()

    def test_var_custom_window(self, sample_data):
        """Test Variance with custom window"""
        window = 10
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = var(df, parameters={'window': window})
        
        assert f'VAR_{window}' in columns
        
    def test_var_equals_std_squared(self, sample_data):
        """Test that Variance equals STD squared"""
        df = pd.DataFrame({'Close': sample_data['close']})
        var_result, _ = var(df)
        std_result, _ = std(df)
        
        valid_var = var_result.dropna()
        valid_std = std_result.dropna()
        
        # Variance should equal STD squared
        expected_var = valid_std ** 2
        pd.testing.assert_series_equal(valid_var, expected_var, check_names=False)


class TestZScore:
    """Tests for Z-Score"""

    def test_zsc_calculation(self, sample_data):
        """Test basic Z-Score calculation"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = zsc(df)
        
        assert isinstance(result_data, pd.Series)
        assert not result_data.empty
        assert 'ZSC_20' in columns
        
        # Z-Score should have valid values
        valid_result = result_data.dropna()
        assert len(valid_result) > 0

    def test_zsc_custom_window(self, sample_data):
        """Test Z-Score with custom window"""
        window = 14
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, columns = zsc(df, parameters={'window': window})
        
        assert f'ZSC_{window}' in columns
        
    def test_zsc_mean_near_zero(self, sample_data):
        """Test that Z-Score has mean close to zero over the window"""
        df = pd.DataFrame({'Close': sample_data['close']})
        result_data, _ = zsc(df)
        
        valid_result = result_data.dropna()
        # The mean of Z-scores should be close to zero
        assert abs(valid_result.mean()) < 1.0  # Reasonable tolerance


class TestStatisticsIntegration:
    """Integration tests for statistics indicators"""

    def test_all_indicators_return_correct_types(self, sample_data):
        """Test that all statistics indicators return correct types"""
        df = pd.DataFrame({'Close': sample_data['close']})
        
        indicators = [
            (std, 'STD'),
            (kur, 'KUR'),
            (mab, 'MAB'),
            (med, 'MED'),
            (qua, 'QUA'),
            (skw, 'SKW'),
            (var, 'VAR'),
            (zsc, 'ZSC'),
        ]
        
        for indicator_func, name in indicators:
            result_data, columns = indicator_func(df)
            assert isinstance(result_data, pd.Series), f"{name} should return pd.Series"
            assert isinstance(columns, list), f"{name} should return list of columns"
            assert len(columns) > 0, f"{name} should have at least one column name"
            assert not result_data.empty, f"{name} should not return empty series"

    def test_all_indicators_handle_nan(self, sample_data):
        """Test that all statistics indicators handle NaN values correctly"""
        df = pd.DataFrame({'Close': sample_data['close']})
        
        indicators = [std, kur, mab, med, qua, skw, var, zsc]
        
        for indicator_func in indicators:
            result_data, _ = indicator_func(df)
            # First window-1 values should be NaN
            assert result_data.iloc[:19].isna().all()
            # Later values should not all be NaN
            assert not result_data.iloc[19:].isna().all()

    def test_volatility_indicators_reflect_volatility_changes(self, sample_data):
        """Test that volatility-related indicators reflect changes in volatility"""
        df = pd.DataFrame({'Close': sample_data['close']})
        
        # Test STD, VAR, MAB - should be higher in high volatility period
        for indicator_func in [std, var, mab]:
            result_data, _ = indicator_func(df)
            valid_result = result_data.dropna()
            
            # Sample data has high vol first 50, low vol last 50
            high_vol_period = valid_result.iloc[:30].mean()  # High vol period
            low_vol_period = valid_result.iloc[-30:].mean()  # Low vol period
            
            assert high_vol_period > low_vol_period, f"{indicator_func.__name__} should reflect volatility changes"
