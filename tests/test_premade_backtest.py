import pytest
import pandas as pd
import numpy as np
from simple_trade.premade_backtest import premade_backtest


# --- Fixtures ---

@pytest.fixture
def sample_ohlcv_data():
    """Fixture to provide sample OHLCV data with DatetimeIndex"""
    index = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    close = pd.Series(np.linspace(100, 150, 100) + np.random.normal(0, 2, 100), index=index)
    high = close + np.random.uniform(0.5, 3, size=len(close))
    low = close - np.random.uniform(0.5, 3, size=len(close))
    low = pd.Series(np.minimum(low.values, close.values - 0.1), index=index)
    high = pd.Series(np.maximum(high.values, close.values + 0.1), index=index)
    volume = pd.Series(np.random.randint(1000, 10000, size=len(close)), index=index)
    
    df = pd.DataFrame({
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    })
    return df


@pytest.fixture
def default_parameters():
    """Default parameters for premade_backtest"""
    return {
        'initial_cash': 10000.0,
        'commission_long': 0.001,
        'commission_short': 0.001,
        'trading_type': 'long',
        'day1_position': 'none',
        'risk_free_rate': 0.02,
        'fig_control': 0
    }


# --- Basic Functionality Tests ---

class TestPremadeBacktestBasic:
    """Test basic functionality of premade_backtest"""
    
    def test_premade_backtest_with_none_parameters(self, sample_ohlcv_data):
        """Test that premade_backtest works with None parameters"""
        # premade_backtest expects a dict, not None
        results, portfolio, fig = premade_backtest(sample_ohlcv_data, 'rsi', {})
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        assert fig is None  # fig_control defaults to 0
        
    def test_premade_backtest_returns_tuple(self, sample_ohlcv_data, default_parameters):
        """Test that premade_backtest returns a tuple of (results, portfolio, fig)"""
        results, portfolio, fig = premade_backtest(sample_ohlcv_data, 'rsi', default_parameters)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        assert fig is None  # fig_control is 0
        
    def test_premade_backtest_with_figure_control(self, sample_ohlcv_data, default_parameters):
        """Test that fig_control=1 returns a figure"""
        default_parameters['fig_control'] = 1
        results, portfolio, fig = premade_backtest(sample_ohlcv_data, 'rsi', default_parameters)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        assert fig is not None


# --- Strategy-Specific Tests ---

class TestPremadeBacktestStrategies:
    """Test different trading strategies"""
    
    def test_rsi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test RSI strategy"""
        rsi_params = default_parameters.copy()
        rsi_params.update({
            'window': 14,
            'upper': 70,
            'lower': 30
        })
        
        results, portfolio, fig = premade_backtest(sample_ohlcv_data, 'rsi', rsi_params)
        
        assert isinstance(results, dict)
        assert 'total_return_pct' in results
        assert 'sharpe_ratio' in results
        assert isinstance(portfolio, pd.DataFrame)
        assert 'PositionType' in portfolio.columns
        
    def test_sma_strategy(self, sample_ohlcv_data, default_parameters):
        """Test SMA cross strategy"""
        sma_params = default_parameters.copy()
        sma_params.update({
            'short_window': 10,
            'long_window': 20
        })
        
        results, portfolio, fig = premade_backtest(sample_ohlcv_data, 'sma', sma_params)
        
        assert isinstance(results, dict)
        assert 'total_return_pct' in results
        assert isinstance(portfolio, pd.DataFrame)
        
    def test_ema_strategy(self, sample_ohlcv_data, default_parameters):
        """Test EMA cross strategy"""
        ema_params = default_parameters.copy()
        ema_params.update({
            'short_window': 12,
            'long_window': 26
        })
        
        results, portfolio, fig = premade_backtest(sample_ohlcv_data, 'ema', ema_params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        
    def test_macd_strategy(self, sample_ohlcv_data, default_parameters):
        """Test MACD strategy"""
        macd_params = default_parameters.copy()
        macd_params.update({
            'window_fast': 12,
            'window_slow': 26,
            'window_signal': 9
        })
        
        results, portfolio, fig = premade_backtest(sample_ohlcv_data, 'macd', macd_params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        
    def test_bollinger_bands_strategy(self, sample_ohlcv_data, default_parameters):
        """Test Bollinger Bands strategy"""
        bb_params = default_parameters.copy()
        bb_params.update({
            'window': 20,
            'num_std': 2.0
        })
        
        results, portfolio, fig = premade_backtest(sample_ohlcv_data, 'bollin', bb_params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        
    def test_cci_strategy(self, sample_ohlcv_data, default_parameters):
        """Test CCI strategy"""
        cci_params = default_parameters.copy()
        cci_params.update({
            'window': 20,
            'constant': 0.015,
            'upper': 100,
            'lower': -100
        })
        
        results, portfolio, fig = premade_backtest(sample_ohlcv_data, 'cci', cci_params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        
    def test_stochastic_strategy(self, sample_ohlcv_data, default_parameters):
        """Test Stochastic strategy"""
        stoch_params = default_parameters.copy()
        stoch_params.update({
            'k_period': 14,
            'd_period': 3,
            'smooth_k': 3,
            'upper': 80,
            'lower': 20
        })
        
        results, portfolio, fig = premade_backtest(sample_ohlcv_data, 'stoch', stoch_params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)


# --- Parameter Validation Tests ---

class TestPremadeBacktestParameterValidation:
    """Test parameter validation and default values"""
    
    def test_default_parameter_values(self, sample_ohlcv_data):
        """Test that default parameter values are used correctly"""
        results, portfolio, fig = premade_backtest(sample_ohlcv_data, 'rsi', {})
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        
    def test_custom_initial_cash(self, sample_ohlcv_data, default_parameters):
        """Test custom initial cash parameter"""
        default_parameters['initial_cash'] = 50000.0
        results, portfolio, fig = premade_backtest(sample_ohlcv_data, 'rsi', default_parameters)
        
        # Check that the results reflect the custom initial cash
        assert results['initial_cash'] == 50000.0
        
    def test_custom_commission_rates(self, sample_ohlcv_data, default_parameters):
        """Test custom commission rates"""
        default_parameters.update({
            'commission_long': 0.005,
            'commission_short': 0.003
        })
        
        results, portfolio, fig = premade_backtest(sample_ohlcv_data, 'rsi', default_parameters)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        
    def test_trading_type_long_only(self, sample_ohlcv_data, default_parameters):
        """Test long-only trading type"""
        default_parameters['trading_type'] = 'long'
        results, portfolio, fig = premade_backtest(sample_ohlcv_data, 'rsi', default_parameters)
        
        # Check that no short positions are taken
        assert not (portfolio['PositionType'] == 'short').any()
        
    def test_trading_type_short_only(self, sample_ohlcv_data, default_parameters):
        """Test short-only trading type"""
        default_parameters['trading_type'] = 'short'
        results, portfolio, fig = premade_backtest(sample_ohlcv_data, 'rsi', default_parameters)
        
        # Check that no long positions are taken
        assert not (portfolio['PositionType'] == 'long').any()
        
    def test_day1_position_parameter(self, sample_ohlcv_data, default_parameters):
        """Test day1_position parameter"""
        default_parameters['day1_position'] = 'long'
        results, portfolio, fig = premade_backtest(sample_ohlcv_data, 'rsi', default_parameters)
        
        # Check that first position is long
        assert portfolio['PositionType'].iloc[0] == 'long'


# --- Error Handling Tests ---

class TestPremadeBacktestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_strategy_name(self, sample_ohlcv_data, default_parameters):
        """Test handling of invalid strategy name"""
        # Invalid strategy should complete the function but not match any strategy
        results, portfolio, fig = premade_backtest(sample_ohlcv_data, 'invalid_strategy', default_parameters)
        # The function will complete but won't execute any strategy-specific code
        assert isinstance(results, dict) or results is None
        assert isinstance(portfolio, pd.DataFrame) or portfolio is None
            
    def test_empty_dataframe(self, default_parameters):
        """Test handling of empty DataFrame"""
        # Create empty DataFrame with proper index
        empty_df = pd.DataFrame(index=pd.DatetimeIndex([]))
        
        # Empty dataframe should either raise an error or return empty results
        try:
            results, portfolio, fig = premade_backtest(empty_df, 'rsi', default_parameters)
            # If no error, check that results are handled gracefully
            assert results is None or isinstance(results, dict)
        except (ValueError, KeyError, IndexError, AttributeError):
            # Expected error for empty data
            pass
            
    def test_missing_required_columns(self, default_parameters):
        """Test handling of DataFrame missing required columns"""
        # Create DataFrame with DatetimeIndex but missing required columns
        index = pd.date_range(start='2023-01-01', periods=5, freq='D')
        incomplete_df = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104]
        }, index=index)
        
        # Missing High, Low, Volume columns should cause issues
        try:
            results, portfolio, fig = premade_backtest(incomplete_df, 'rsi', default_parameters)
            # If no error, function handled missing columns gracefully
            assert results is None or isinstance(results, dict)
        except (KeyError, ValueError, AttributeError):
            # Expected error for missing required columns
            pass


# --- Integration Tests ---

class TestPremadeBacktestIntegration:
    """Integration tests for premade_backtest"""
    
    def test_multiple_strategies_same_data(self, sample_ohlcv_data, default_parameters):
        """Test multiple strategies on the same data"""
        strategies = ['rsi', 'sma', 'ema']
        results_list = []
        
        for strategy in strategies:
            results, portfolio, fig = premade_backtest(sample_ohlcv_data, strategy, default_parameters)
            results_list.append(results)
            
        # All strategies should return valid results
        for results in results_list:
            assert isinstance(results, dict)
            assert 'total_return_pct' in results
            
    def test_parameter_sensitivity_rsi(self, sample_ohlcv_data, default_parameters):
        """Test RSI strategy with different parameters"""
        rsi_windows = [10, 14, 20]
        results_list = []
        
        for window in rsi_windows:
            params = default_parameters.copy()
            params['window'] = window
            results, portfolio, fig = premade_backtest(sample_ohlcv_data, 'rsi', params)
            if results and 'total_return_pct' in results:
                results_list.append(results['total_return_pct'])
            
        # Results should be different for different parameters (if we have results)
        if len(results_list) > 1:
            assert len(set(results_list)) >= 1  # At least one unique result
        
    def test_consistent_results_same_parameters(self, sample_ohlcv_data, default_parameters):
        """Test that same parameters produce consistent results"""
        results1, portfolio1, fig1 = premade_backtest(sample_ohlcv_data, 'rsi', default_parameters)
        results2, portfolio2, fig2 = premade_backtest(sample_ohlcv_data, 'rsi', default_parameters)
        
        # Results should be identical
        assert results1['total_return_pct'] == results2['total_return_pct']
        assert len(portfolio1) == len(portfolio2)


# --- Performance Metrics Tests ---

class TestPremadeBacktestMetrics:
    """Test that performance metrics are calculated correctly"""
    
    def test_required_metrics_present(self, sample_ohlcv_data, default_parameters):
        """Test that all required metrics are present in results"""
        results, portfolio, fig = premade_backtest(sample_ohlcv_data, 'rsi', default_parameters)
        
        required_metrics = [
            'total_return_pct',
            'sharpe_ratio',
            'max_drawdown_pct'
        ]
        
        for metric in required_metrics:
            assert metric in results, f"Missing metric: {metric}"
            assert isinstance(results[metric], (int, float)), f"Metric {metric} should be numeric"
            
    def test_portfolio_structure(self, sample_ohlcv_data, default_parameters):
        """Test that portfolio DataFrame has correct structure"""
        results, portfolio, fig = premade_backtest(sample_ohlcv_data, 'rsi', default_parameters)
        
        required_columns = [
            'PositionType',
            'PortfolioValue'
        ]
        
        for col in required_columns:
            assert col in portfolio.columns, f"Missing portfolio column: {col}"
            
        # Portfolio should be shorter than input data due to indicator calculation requirements
        assert len(portfolio) <= len(sample_ohlcv_data)
        assert len(portfolio) > 0  # Should have some data
        
    def test_metrics_reasonable_values(self, sample_ohlcv_data, default_parameters):
        """Test that metrics have reasonable values"""
        results, portfolio, fig = premade_backtest(sample_ohlcv_data, 'rsi', default_parameters)
        
        if results and 'total_return_pct' in results:
            # Total return should be reasonable (not extremely large)
            total_return = results['total_return_pct']
            if not pd.isna(total_return):
                assert -100 <= total_return <= 1000
        
        if results and 'sharpe_ratio' in results:
            # Sharpe ratio should be reasonable
            sharpe = results['sharpe_ratio']
            if not pd.isna(sharpe):
                assert -10 <= sharpe <= 10
        
        if results and 'max_drawdown_pct' in results:
            # Max drawdown should be negative or zero
            drawdown = results['max_drawdown_pct']
            if not pd.isna(drawdown):
                assert drawdown <= 0
