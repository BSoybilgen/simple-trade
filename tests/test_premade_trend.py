"""
Tests for trend premade strategies.
"""
import pandas as pd
from simple_trade.run_premade_strategies import run_premade_trade


class TestTrendStrategies:
    """Test all trend trading strategies"""

    def test_ads_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ADS (Adaptive Moving Average - Smoothed) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 10, 'long_window': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'ads', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        assert 'total_return_pct' in results

    def test_adx_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ADX (Average Directional Index) strategy"""
        params = default_parameters.copy()
        params.update({'window': 14, 'adx_threshold': 25, 'ma_window': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'adx', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_alm_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ALM (ALMA - Arnaud Legoux Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 9, 'long_window': 27})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'alm', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_ama_strategy(self, sample_ohlcv_data, default_parameters):
        """Test AMA (Kaufman Adaptive Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({
            'short_window': 10, 'long_window': 30,
            'fast_period': 2, 'slow_period': 30
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'ama', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_aro_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ARO (Aroon) strategy"""
        params = default_parameters.copy()
        params.update({'period': 14})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'aro', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_dem_strategy(self, sample_ohlcv_data, default_parameters):
        """Test DEM (DEMA - Double Exponential Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 10, 'long_window': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'dem', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_eac_strategy(self, sample_ohlcv_data, default_parameters):
        """Test EAC (Exponential Adaptive Close) strategy"""
        params = default_parameters.copy()
        params.update({'short_alpha': 0.07, 'long_alpha': 0.14})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'eac', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_eit_strategy(self, sample_ohlcv_data, default_parameters):
        """Test EIT (Ehlers Instantaneous Trendline) strategy"""
        params = default_parameters.copy()
        params.update({'short_alpha': 0.07, 'long_alpha': 0.14})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'eit', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_fma_strategy(self, sample_ohlcv_data, default_parameters):
        """Test FMA (Fibonacci Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 8, 'long_window': 24})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'fma', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_fma_strategy_with_adx_filter(self, sample_ohlcv_data, default_parameters):
        """Test FMA strategy with ADX filter"""
        params = default_parameters.copy()
        params.update({
            'short_window': 8, 'long_window': 24,
            'use_adx_filter': True, 'adx_window': 14, 'adx_threshold': 25
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'fma', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_gma_strategy(self, sample_ohlcv_data, default_parameters):
        """Test GMA (Guppy Multiple Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({
            'short_windows': (3, 5, 8, 10, 12, 15),
            'long_windows': (30, 35, 40, 45, 50, 60)
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'gma', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_hma_strategy(self, sample_ohlcv_data, default_parameters):
        """Test HMA (Hull Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 25, 'long_window': 75})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'hma', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_htt_strategy(self, sample_ohlcv_data, default_parameters):
        """Test HTT (Hilbert Transform Trendline) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 8, 'long_window': 16})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'htt', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_ich_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ICH (Ichimoku Cloud) strategy"""
        params = default_parameters.copy()
        params.update({
            'tenkan_period': 9, 'kijun_period': 26,
            'senkou_b_period': 52, 'displacement': 26
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'ich', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_jma_strategy(self, sample_ohlcv_data, default_parameters):
        """Test JMA (Jurik Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({'short_length': 14, 'long_length': 42})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'jma', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_lsm_strategy(self, sample_ohlcv_data, default_parameters):
        """Test LSM (Least Squares Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 10, 'long_window': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'lsm', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_mgd_strategy(self, sample_ohlcv_data, default_parameters):
        """Test MGD (McGinley Dynamic) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 10, 'long_window': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'mgd', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_psa_strategy(self, sample_ohlcv_data, default_parameters):
        """Test PSA (Parabolic SAR) strategy"""
        params = default_parameters.copy()
        params.update({'af_initial': 0.03, 'af_step': 0.03, 'af_max': 0.3})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'psa', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_soa_strategy(self, sample_ohlcv_data, default_parameters):
        """Test SOA (Second Order Adaptive) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 10, 'long_window': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'soa', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_str_strategy(self, sample_ohlcv_data, default_parameters):
        """Test STR (SuperTrend) strategy"""
        params = default_parameters.copy()
        params.update({'period': 7, 'multiplier': 3.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'str', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_swm_strategy(self, sample_ohlcv_data, default_parameters):
        """Test SWM (Sine-Weighted Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 10, 'long_window': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'swm', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_tem_strategy(self, sample_ohlcv_data, default_parameters):
        """Test TEM (TEMA - Triple Exponential Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 10, 'long_window': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'tem', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_tma_strategy(self, sample_ohlcv_data, default_parameters):
        """Test TMA (Triangular Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 10, 'long_window': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'tma', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_tri_strategy(self, sample_ohlcv_data, default_parameters):
        """Test TRI (TRIX) strategy"""
        params = default_parameters.copy()
        params.update({'window': 7})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'tri', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_vid_strategy(self, sample_ohlcv_data, default_parameters):
        """Test VID (Variable Index Dynamic Average) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 14, 'long_window': 42, 'cmo_window': 9})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'vid', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_wma_strategy(self, sample_ohlcv_data, default_parameters):
        """Test WMA (Weighted Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 25, 'long_window': 75})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'wma', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_zma_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ZMA (Zero-Lag Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 10, 'long_window': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'zma', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)


class TestTrendStrategiesWithTradingTypes:
    """Test trend strategies with different trading types"""

    def test_trend_strategy_long_only(self, sample_ohlcv_data, default_parameters):
        """Test trend strategy with long-only trading"""
        params = default_parameters.copy()
        params['trading_type'] = 'long'
        params.update({'short_window': 10, 'long_window': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'sma', params)
        
        assert isinstance(results, dict)
        assert not (portfolio['PositionType'] == 'short').any()

    def test_trend_strategy_short_only(self, sample_ohlcv_data, default_parameters):
        """Test trend strategy with short-only trading"""
        params = default_parameters.copy()
        params['trading_type'] = 'short'
        params.update({'short_window': 10, 'long_window': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'sma', params)
        
        assert isinstance(results, dict)
        assert not (portfolio['PositionType'] == 'long').any()

    def test_trend_strategy_mixed(self, sample_ohlcv_data, default_parameters):
        """Test trend strategy with mixed trading"""
        params = default_parameters.copy()
        params['trading_type'] = 'mixed'
        params.update({'short_window': 10, 'long_window': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'sma', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
