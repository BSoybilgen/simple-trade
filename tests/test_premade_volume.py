"""
Tests for volume premade strategies.
"""
import pandas as pd
from simple_trade.run_premade_strategies import run_premade_trade


class TestVolumeStrategies:
    """Test all volume trading strategies"""

    def test_adl_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ADL (Accumulation/Distribution Line) strategy"""
        params = default_parameters.copy()
        params.update({'sma_period': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'adl', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_ado_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ADO (Accumulation/Distribution Oscillator) strategy"""
        params = default_parameters.copy()
        params.update({'period': 14})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'ado', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_bwm_strategy(self, sample_ohlcv_data, default_parameters):
        """Test BWM (Bill Williams Market Facilitation Index) strategy"""
        params = default_parameters.copy()
        params.update({'upper_pct': 80, 'lower_pct': 20, 'lookback': 50})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'bwm', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_cmf_strategy(self, sample_ohlcv_data, default_parameters):
        """Test CMF (Chaikin Money Flow) strategy"""
        params = default_parameters.copy()
        params.update({'period': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'cmf', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_emv_strategy(self, sample_ohlcv_data, default_parameters):
        """Test EMV (Ease of Movement) strategy"""
        params = default_parameters.copy()
        params.update({'period': 14})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'emv', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_foi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test FOI (Force Index) strategy"""
        params = default_parameters.copy()
        params.update({'period': 13})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'foi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_fve_strategy(self, sample_ohlcv_data, default_parameters):
        """Test FVE (Finite Volume Elements) strategy"""
        params = default_parameters.copy()
        params.update({'period': 22, 'factor': 0.3})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'fve', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_kvo_strategy(self, sample_ohlcv_data, default_parameters):
        """Test KVO (Klinger Volume Oscillator) strategy"""
        params = default_parameters.copy()
        params.update({'fast_period': 34, 'slow_period': 55, 'signal_period': 13})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'kvo', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_mfi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test MFI (Money Flow Index) strategy"""
        params = default_parameters.copy()
        params.update({'period': 14, 'upper': 80, 'lower': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'mfi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_nvi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test NVI (Negative Volume Index) strategy"""
        params = default_parameters.copy()
        params.update({'sma_period': 50})  # Reduced from 255 for test data size
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'nvi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_obv_strategy(self, sample_ohlcv_data, default_parameters):
        """Test OBV (On-Balance Volume) strategy"""
        params = default_parameters.copy()
        params.update({'sma_period': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'obv', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_pvi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test PVI (Positive Volume Index) strategy"""
        params = default_parameters.copy()
        params.update({'sma_period': 50})  # Reduced from 255 for test data size
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'pvi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_pvo_strategy(self, sample_ohlcv_data, default_parameters):
        """Test PVO (Percentage Volume Oscillator) strategy"""
        params = default_parameters.copy()
        params.update({'fast_period': 12, 'slow_period': 26, 'signal_period': 9})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'pvo', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_vfi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test VFI (Volume Flow Indicator) strategy"""
        params = default_parameters.copy()
        params.update({
            'period': 50,  # Reduced from 130 for test data size
            'coef': 0.2, 'vcoef': 2.5, 'smoothing_period': 3
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'vfi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_vma_strategy(self, sample_ohlcv_data, default_parameters):
        """Test VMA (Volume Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({'window': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'vma', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_voo_strategy(self, sample_ohlcv_data, default_parameters):
        """Test VOO (Volume Oscillator) strategy"""
        params = default_parameters.copy()
        params.update({'fast_period': 5, 'slow_period': 10})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'voo', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_vpt_strategy(self, sample_ohlcv_data, default_parameters):
        """Test VPT (Volume Price Trend) strategy"""
        params = default_parameters.copy()
        params.update({'sma_period': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'vpt', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_vro_strategy(self, sample_ohlcv_data, default_parameters):
        """Test VRO (Volume Rate of Change) strategy"""
        params = default_parameters.copy()
        params.update({'period': 14})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'vro', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_vwa_strategy(self, sample_ohlcv_data, default_parameters):
        """Test VWA (Volume Weighted Average Price) strategy"""
        params = default_parameters.copy()
        params.update({'window': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'vwa', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

class TestVolumeStrategiesWithTradingTypes:
    """Test volume strategies with different trading types"""

    def test_volume_strategy_long_only(self, sample_ohlcv_data, default_parameters):
        """Test volume strategy with long-only trading"""
        params = default_parameters.copy()
        params['trading_type'] = 'long'
        params.update({'sma_period': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'obv', params)
        
        assert isinstance(results, dict)
        assert not (portfolio['PositionType'] == 'short').any()

    def test_volume_strategy_short_only(self, sample_ohlcv_data, default_parameters):
        """Test volume strategy with short-only trading"""
        params = default_parameters.copy()
        params['trading_type'] = 'short'
        params.update({'sma_period': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'obv', params)
        
        assert isinstance(results, dict)
        assert not (portfolio['PositionType'] == 'long').any()

    def test_volume_strategy_mixed(self, sample_ohlcv_data, default_parameters):
        """Test volume strategy with mixed trading"""
        params = default_parameters.copy()
        params['trading_type'] = 'mixed'
        params.update({'sma_period': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'obv', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
