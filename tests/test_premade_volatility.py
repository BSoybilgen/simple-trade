"""
Tests for volatility premade strategies.
"""
import pandas as pd
from simple_trade.run_premade_strategies import run_premade_trade


class TestVolatilityStrategies:
    """Test all volatility trading strategies"""

    def test_acb_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ACB (Acceleration Bands) strategy"""
        params = default_parameters.copy()
        params.update({'period': 20, 'factor': 0.001})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'acb', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_atp_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ATP (Average True Range Percent) strategy"""
        params = default_parameters.copy()
        params.update({'window': 14, 'upper': 5.0, 'lower': 2.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'atp', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_atr_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ATR (Average True Range) strategy"""
        params = default_parameters.copy()
        params.update({'window': 14, 'upper_pct': 80, 'lower_pct': 20, 'lookback': 50})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'atr', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_bbw_strategy(self, sample_ohlcv_data, default_parameters):
        """Test BBW (Bollinger Band Width) strategy"""
        params = default_parameters.copy()
        params.update({'window': 20, 'num_std': 2.0, 'upper': 10.0, 'lower': 4.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'bbw', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_bol_strategy(self, sample_ohlcv_data, default_parameters):
        """Test BOL (Bollinger Bands) strategy"""
        params = default_parameters.copy()
        params.update({'window': 20, 'num_std': 2})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'bol', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_cha_strategy(self, sample_ohlcv_data, default_parameters):
        """Test CHA (Chaikin Volatility) strategy"""
        params = default_parameters.copy()
        params.update({'ema_window': 10, 'roc_window': 10, 'upper': 20.0, 'lower': -20.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'cha', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_cho_strategy(self, sample_ohlcv_data, default_parameters):
        """Test CHO (Choppiness Index) strategy"""
        params = default_parameters.copy()
        params.update({'period': 14, 'upper': 61.8, 'lower': 38.2})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'cho', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_don_strategy(self, sample_ohlcv_data, default_parameters):
        """Test DON (Donchian Channels) strategy"""
        params = default_parameters.copy()
        params.update({'window': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'don', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_dvi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test DVI (Dynamic Volatility Index) strategy"""
        params = default_parameters.copy()
        params.update({
            'magnitude_period': 5, 'stretch_period': 50, 'smooth_period': 3,
            'upper': 70, 'lower': 30
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'dvi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_efr_strategy(self, sample_ohlcv_data, default_parameters):
        """Test EFR (Efficiency Ratio) strategy"""
        params = default_parameters.copy()
        params.update({'period': 10, 'upper': 0.7, 'lower': 0.3})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'efr', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_fdi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test FDI (Fractal Dimension Index) strategy"""
        params = default_parameters.copy()
        params.update({'period': 20, 'upper': 1.6, 'lower': 1.4})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'fdi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_grv_strategy(self, sample_ohlcv_data, default_parameters):
        """Test GRV (Garman-Klass Volatility) strategy"""
        params = default_parameters.copy()
        params.update({'period': 20, 'upper': 30.0, 'lower': 15.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'grv', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_hav_strategy(self, sample_ohlcv_data, default_parameters):
        """Test HAV (Heikin-Ashi Volatility) strategy"""
        params = default_parameters.copy()
        params.update({
            'period': 14, 'method': 'atr',
            'upper_pct': 80, 'lower_pct': 20, 'lookback': 50
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'hav', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_hiv_strategy(self, sample_ohlcv_data, default_parameters):
        """Test HIV (Historical Volatility) strategy"""
        params = default_parameters.copy()
        params.update({'period': 20, 'upper': 30.0, 'lower': 15.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'hiv', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_kel_strategy(self, sample_ohlcv_data, default_parameters):
        """Test KEL (Keltner Channel) strategy"""
        params = default_parameters.copy()
        params.update({'ema_window': 20, 'atr_window': 10, 'atr_multiplier': 2.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'kel', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_mad_strategy(self, sample_ohlcv_data, default_parameters):
        """Test MAD (Median Absolute Deviation) strategy"""
        params = default_parameters.copy()
        params.update({'period': 20, 'upper': 2.0, 'lower': 0.5})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'mad', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_mai_strategy(self, sample_ohlcv_data, default_parameters):
        """Test MAI (Mass Index) strategy"""
        params = default_parameters.copy()
        params.update({'ema_period': 9, 'sum_period': 25, 'upper': 27.0, 'lower': 26.5})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'mai', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_nat_strategy(self, sample_ohlcv_data, default_parameters):
        """Test NAT (Normalized ATR) strategy"""
        params = default_parameters.copy()
        params.update({'window': 14, 'upper': 5.0, 'lower': 2.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'nat', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_pav_strategy(self, sample_ohlcv_data, default_parameters):
        """Test PAV (Parkinson Volatility) strategy"""
        params = default_parameters.copy()
        params.update({'period': 20, 'upper': 30.0, 'lower': 15.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'pav', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_pcw_strategy(self, sample_ohlcv_data, default_parameters):
        """Test PCW (Price Channel Width) strategy"""
        params = default_parameters.copy()
        params.update({'period': 20, 'upper': 15.0, 'lower': 5.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'pcw', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_pro_strategy(self, sample_ohlcv_data, default_parameters):
        """Test PRO (Projection Oscillator) strategy"""
        params = default_parameters.copy()
        params.update({'period': 10, 'smooth_period': 3})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'pro', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_rsv_strategy(self, sample_ohlcv_data, default_parameters):
        """Test RSV (Rogers-Satchell Volatility) strategy"""
        params = default_parameters.copy()
        params.update({'period': 20, 'upper': 30.0, 'lower': 15.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'rsv', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_rvi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test RVI (Relative Volatility Index) strategy"""
        params = default_parameters.copy()
        params.update({'window': 10, 'rvi_period': 14, 'upper': 70, 'lower': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'rvi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_svi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test SVI (Stochastic Volatility Index) strategy"""
        params = default_parameters.copy()
        params.update({
            'atr_period': 14, 'stoch_period': 14,
            'smooth_k': 3, 'smooth_d': 3, 'upper': 80, 'lower': 20
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'svi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_tsv_strategy(self, sample_ohlcv_data, default_parameters):
        """Test TSV (TSI Volatility) strategy"""
        params = default_parameters.copy()
        params.update({'atr_period': 14, 'long_period': 25, 'short_period': 13})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'tsv', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_uli_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ULI (Ulcer Index) strategy"""
        params = default_parameters.copy()
        params.update({'period': 14, 'upper': 5.0, 'lower': 1.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'uli', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_vhf_strategy(self, sample_ohlcv_data, default_parameters):
        """Test VHF (Vertical Horizontal Filter) strategy"""
        params = default_parameters.copy()
        params.update({'period': 28, 'upper': 0.40, 'lower': 0.25})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'vhf', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_vra_strategy(self, sample_ohlcv_data, default_parameters):
        """Test VRA (Volatility Ratio) strategy"""
        params = default_parameters.copy()
        params.update({'short_period': 5, 'long_period': 20, 'upper': 1.5, 'lower': 0.8})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'vra', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_vsi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test VSI (Volatility Switch Index) strategy"""
        params = default_parameters.copy()
        params.update({
            'short_period': 10, 'long_period': 50, 'threshold': 1.2,
            'upper': 0.5, 'lower': 0.5
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'vsi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)


class TestVolatilityStrategiesWithTradingTypes:
    """Test volatility strategies with different trading types"""

    def test_volatility_strategy_long_only(self, sample_ohlcv_data, default_parameters):
        """Test volatility strategy with long-only trading"""
        params = default_parameters.copy()
        params['trading_type'] = 'long'
        params.update({'window': 20, 'num_std': 2})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'bol', params)
        
        assert isinstance(results, dict)
        assert not (portfolio['PositionType'] == 'short').any()

    def test_volatility_strategy_short_only(self, sample_ohlcv_data, default_parameters):
        """Test volatility strategy with short-only trading"""
        params = default_parameters.copy()
        params['trading_type'] = 'short'
        params.update({'window': 20, 'num_std': 2})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'bol', params)
        
        assert isinstance(results, dict)
        assert not (portfolio['PositionType'] == 'long').any()

    def test_volatility_strategy_mixed(self, sample_ohlcv_data, default_parameters):
        """Test volatility strategy with mixed trading"""
        params = default_parameters.copy()
        params['trading_type'] = 'mixed'
        params.update({'window': 20, 'num_std': 2})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'bol', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
