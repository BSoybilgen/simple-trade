import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from simple_trade.combine_trade import CombineTradeBacktester, plot_combined_results


@pytest.fixture
def sample_price_data():
    """Sample price data with DatetimeIndex."""
    dates = pd.date_range('2023-01-01', periods=20, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(20) * 0.5)
    return pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, 20)
    }, index=dates)


@pytest.fixture
def sample_portfolio_df_long():
    """Sample portfolio DataFrame with long positions."""
    dates = pd.date_range('2023-01-01', periods=20, freq='D')
    position_types = ['none'] * 5 + ['long'] * 10 + ['none'] * 5
    return pd.DataFrame({
        'PositionType': position_types,
        'PortfolioValue': np.random.uniform(9000, 11000, 20)
    }, index=dates)


@pytest.fixture
def sample_portfolio_df_short():
    """Sample portfolio DataFrame with short positions."""
    dates = pd.date_range('2023-01-01', periods=20, freq='D')
    position_types = ['none'] * 3 + ['short'] * 8 + ['none'] * 9
    return pd.DataFrame({
        'PositionType': position_types,
        'PortfolioValue': np.random.uniform(9000, 11000, 20)
    }, index=dates)


@pytest.fixture
def sample_portfolio_df_mixed():
    """Sample portfolio DataFrame with mixed positions."""
    dates = pd.date_range('2023-01-01', periods=20, freq='D')
    position_types = ['none', 'long', 'long', 'none', 'short', 'short', 'none', 
                     'long', 'long', 'long', 'none', 'short', 'none', 'long', 
                     'long', 'none', 'none', 'short', 'short', 'none']
    return pd.DataFrame({
        'PositionType': position_types,
        'PortfolioValue': np.random.uniform(9000, 11000, 20)
    }, index=dates)


@pytest.fixture
def default_parameters():
    """Default parameters for backtester."""
    return {
        'initial_cash': 10000.0,
        'commission_long': 0.001,
        'commission_short': 0.001,
        'short_borrow_fee_inc_rate': 0.0001,
        'long_borrow_fee_inc_rate': 0.0001
    }


@pytest.fixture
def backtester(default_parameters):
    """CombineTradeBacktester instance with default parameters."""
    return CombineTradeBacktester(**default_parameters)


class TestCombineTradeBacktester:
    """Test suite for CombineTradeBacktester class."""


class TestInitialization:
    """Test CombineTradeBacktester initialization."""

    def test_initialization_default(self):
        """Test initialization with default parameters."""
        backtester = CombineTradeBacktester()
        assert hasattr(backtester, 'initial_cash')
        assert isinstance(backtester, CombineTradeBacktester)

    def test_initialization_custom_params(self, default_parameters):
        """Test initialization with custom parameters."""
        backtester = CombineTradeBacktester(**default_parameters)
        assert backtester.initial_cash == 10000.0
        assert backtester.commission_long == 0.001
        assert backtester.commission_short == 0.001


class TestCombineSignals:
    """Test _combine_signals method."""

    def test_combine_signals_unanimous_all_long(self, backtester, sample_price_data, sample_portfolio_df_long):
        """Test unanimous combination logic with all long signals."""
        portfolio_dfs = [sample_portfolio_df_long.copy(), sample_portfolio_df_long.copy()]
        
        result = backtester._combine_signals(portfolio_dfs, sample_price_data, 'Close', 'unanimous')
        
        assert not result.empty
        assert 'PositionType' in result.columns
        assert 'buy_signal' in result.columns
        assert 'sell_signal' in result.columns
        
        # Check that unanimous long positions are preserved
        long_positions = result[result['PositionType'] == 'long']
        assert len(long_positions) > 0

    def test_combine_signals_unanimous_mixed(self, backtester, sample_price_data, 
                                           sample_portfolio_df_long, sample_portfolio_df_short):
        """Test unanimous combination logic with mixed signals."""
        portfolio_dfs = [sample_portfolio_df_long, sample_portfolio_df_short]
        
        result = backtester._combine_signals(portfolio_dfs, sample_price_data, 'Close', 'unanimous')
        
        assert not result.empty
        # With mixed signals, unanimous logic should result in mostly 'none' positions
        none_positions = result[result['PositionType'] == 'none']
        assert len(none_positions) > 0

    def test_combine_signals_majority_logic(self, backtester, sample_price_data):
        """Test majority combination logic."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        
        # Create three portfolios: 2 long, 1 short
        portfolio1 = pd.DataFrame({'PositionType': ['long'] * 10}, index=dates)
        portfolio2 = pd.DataFrame({'PositionType': ['long'] * 10}, index=dates)
        portfolio3 = pd.DataFrame({'PositionType': ['short'] * 10}, index=dates)
        
        portfolio_dfs = [portfolio1, portfolio2, portfolio3]
        price_data = sample_price_data.iloc[:10]
        
        result = backtester._combine_signals(portfolio_dfs, price_data, 'Close', 'majority')
        
        assert not result.empty
        # Majority should be long (2 out of 3)
        long_positions = result[result['PositionType'] == 'long']
        assert len(long_positions) > 0

    def test_combine_signals_missing_position_type(self, backtester, sample_price_data):
        """Test error handling when PositionType column is missing."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        portfolio_df = pd.DataFrame({'SomeOtherColumn': [1] * 10}, index=dates)
        
        with pytest.raises(ValueError, match="missing 'PositionType' column"):
            backtester._combine_signals([portfolio_df], sample_price_data, 'Close', 'unanimous')

    def test_combine_signals_non_datetime_index(self, backtester, sample_price_data):
        """Test error handling when portfolio DataFrame doesn't have DatetimeIndex."""
        portfolio_df = pd.DataFrame({
            'PositionType': ['long'] * 10
        }, index=range(10))  # Non-datetime index
        
        with pytest.raises(TypeError, match="must be a DatetimeIndex"):
            backtester._combine_signals([portfolio_df], sample_price_data, 'Close', 'unanimous')

    def test_combine_signals_empty_result(self, backtester):
        """Test handling of empty combined signals."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        price_data = pd.DataFrame({'Close': [100, 101, 102, 103, 104]}, index=dates)
        
        # Portfolio with different date range (no overlap after join)
        different_dates = pd.date_range('2023-02-01', periods=5, freq='D')
        portfolio_df = pd.DataFrame({'PositionType': ['long'] * 5}, index=different_dates)
        
        result = backtester._combine_signals([portfolio_df], price_data, 'Close', 'unanimous')
        assert result.empty


class TestRunCombinedTrade:
    """Test run_combined_trade method."""

    def test_run_combined_trade_basic(self, backtester, sample_price_data, sample_portfolio_df_long):
        """Test basic run_combined_trade functionality."""
        portfolio_dfs = [sample_portfolio_df_long]
        
        results, portfolio_df, figures = backtester.run_combined_trade(
            portfolio_dfs=portfolio_dfs,
            price_data=sample_price_data,
            trading_type='long'
        )
        
        assert isinstance(results, dict)
        assert isinstance(portfolio_df, pd.DataFrame)
        assert figures is None  # fig_control=0 by default
        assert 'strategy' in results
        assert 'final_value' in results
        assert 'num_trades' in results

    def test_run_combined_trade_different_trading_types(self, backtester, sample_price_data, sample_portfolio_df_mixed):
        """Test run_combined_trade with different trading types."""
        portfolio_dfs = [sample_portfolio_df_mixed]
        
        for trading_type in ['long', 'short', 'mixed']:
            results, portfolio_df, figures = backtester.run_combined_trade(
                portfolio_dfs=portfolio_dfs,
                price_data=sample_price_data,
                trading_type=trading_type
            )
            
            assert results['strategy'] == f"Combined Strategy ({trading_type})"
            assert 'final_value' in results

    def test_run_combined_trade_custom_parameters(self, backtester, sample_price_data, sample_portfolio_df_long):
        """Test run_combined_trade with custom parameters."""
        portfolio_dfs = [sample_portfolio_df_long]
        
        results, portfolio_df, figures = backtester.run_combined_trade(
            portfolio_dfs=portfolio_dfs,
            price_data=sample_price_data,
            price_col='Close',
            long_entry_pct_cash=0.8,
            short_entry_pct_cash=0.2,
            trading_type='mixed',
            risk_free_rate=0.02,
            combination_logic='majority'
        )
        
        assert isinstance(results, dict)
        assert isinstance(portfolio_df, pd.DataFrame)
        assert figures is None  # fig_control=0 by default

    def test_run_combined_trade_input_validation(self, backtester, sample_price_data):
        """Test input validation in run_combined_trade."""
        # Test invalid combination_logic
        with pytest.raises(ValueError, match="combination_logic must be either"):
            backtester.run_combined_trade(
                portfolio_dfs=[sample_price_data],
                price_data=sample_price_data,
                combination_logic='invalid'
            )
        
        # Test empty portfolio_dfs
        with pytest.raises(ValueError, match="must be a non-empty list"):
            backtester.run_combined_trade(
                portfolio_dfs=[],
                price_data=sample_price_data
            )
        
        # Test non-DataFrame price_data
        with pytest.raises(TypeError, match="must be a DataFrame with a DatetimeIndex"):
            backtester.run_combined_trade(
                portfolio_dfs=[sample_price_data],
                price_data="not a dataframe"
            )
        
        # Test missing price column
        with pytest.raises(ValueError, match="Price column .* not found"):
            backtester.run_combined_trade(
                portfolio_dfs=[sample_price_data],
                price_data=sample_price_data,
                price_col='NonExistentColumn'
            )

    def test_run_combined_trade_empty_signals(self, backtester):
        """Test run_combined_trade with empty combined signals."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        price_data = pd.DataFrame({'Close': [100, 101, 102, 103, 104]}, index=dates)
        
        # Portfolio with no overlapping dates
        different_dates = pd.date_range('2023-02-01', periods=5, freq='D')
        portfolio_df = pd.DataFrame({'PositionType': ['long'] * 5}, index=different_dates)
        
        results, portfolio_df_result, figures = backtester.run_combined_trade(
            portfolio_dfs=[portfolio_df],
            price_data=price_data
        )
        
        assert 'error' in results
        assert portfolio_df_result.empty
        assert figures is None  # fig_control=0 by default


class TestRunBacktestLoop:
    """Test _run_backtest_loop method."""

    def test_run_backtest_loop_long_trading(self, backtester):
        """Test backtest loop with long trading only."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        signal_df = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'buy_signal': [True, False, False, False, False],
            'sell_signal': [False, False, False, False, True]
        }, index=dates)
        
        portfolio_log, end_state = backtester._run_backtest_loop(
            signal_df=signal_df,
            price_col='Close',
            trading_type='long',
            long_entry_pct_cash=0.9,
            short_entry_pct_cash=0.1
        )
        
        assert len(portfolio_log) == 5
        assert not end_state.empty
        assert 'Action' in end_state.columns
        assert 'PortfolioValue' in end_state.columns

    def test_run_backtest_loop_short_trading(self, backtester):
        """Test backtest loop with short trading only."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        signal_df = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'buy_signal': [False, False, False, False, True],
            'sell_signal': [True, False, False, False, False]
        }, index=dates)
        
        portfolio_log, end_state = backtester._run_backtest_loop(
            signal_df=signal_df,
            price_col='Close',
            trading_type='short',
            long_entry_pct_cash=0.9,
            short_entry_pct_cash=0.1
        )
        
        assert len(portfolio_log) == 5
        assert not end_state.empty

    def test_run_backtest_loop_mixed_trading(self, backtester):
        """Test backtest loop with mixed trading."""
        dates = pd.date_range('2023-01-01', periods=6, freq='D')
        signal_df = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105],
            'buy_signal': [True, False, False, False, False, False],
            'sell_signal': [False, False, False, True, False, False]
        }, index=dates)
        
        portfolio_log, end_state = backtester._run_backtest_loop(
            signal_df=signal_df,
            price_col='Close',
            trading_type='mixed',
            long_entry_pct_cash=0.9,
            short_entry_pct_cash=0.1
        )
        
        assert len(portfolio_log) == 6
        assert not end_state.empty
        
        # Check for mixed actions
        actions = [log['Action'] for log in portfolio_log]
        assert any(action in ['BUY', 'SELL', 'SHORT', 'COVER'] for action in actions)


class TestPrepareResults:
    """Test _prepare_results method."""

    def test_prepare_results_with_data(self, backtester, sample_price_data):
        """Test _prepare_results with valid portfolio log."""
        portfolio_log = [
            {
                'Date': pd.Timestamp('2023-01-01'),
                'Close': 100.0,
                'Cash': 9000.0,
                'PositionSize': 90,
                'PositionValue': 9000.0,
                'PositionType': 'long',
                'PortfolioValue': 18000.0,
                'CommissionPaid': 90.0,
                'ShortFee': 0.0,
                'LongFee': 0.0,
                'BuySignal': True,
                'SellSignal': False,
                'Action': 'BUY'
            }
        ]
        
        final_df = pd.DataFrame(portfolio_log).set_index('Date')
        
        with patch.object(backtester, 'calculate_performance_metrics') as mock_perf, \
             patch.object(backtester, 'compute_benchmark_return') as mock_bench:
            
            mock_perf.return_value = {
                'total_return_pct': 80.0,
                'sharpe_ratio': 1.5,
                'max_drawdown_pct': -5.0
            }
            mock_bench.return_value = {
                'benchmark_return_pct': 4.0,
                'alpha': 76.0
            }
            
            results, portfolio_df = backtester._prepare_results(
                portfolio_log=portfolio_log,
                final_df=final_df,
                original_data=sample_price_data,
                price_col='Close',
                risk_free_rate=0.02,
                trading_type='long'
            )
        
        assert isinstance(results, dict)
        assert isinstance(portfolio_df, pd.DataFrame)
        assert results['strategy'] == 'Combined Strategy (long)'
        assert results['num_trades'] == 1
        assert 'total_return_pct' in results
        assert 'benchmark_return_pct' in results

    def test_prepare_results_empty_log(self, backtester, sample_price_data):
        """Test _prepare_results with empty portfolio log."""
        results, portfolio_df = backtester._prepare_results(
            portfolio_log=[],
            final_df=pd.DataFrame(),
            original_data=sample_price_data,
            price_col='Close',
            risk_free_rate=0.02,
            trading_type='long'
        )
        
        assert 'error' in results
        assert portfolio_df.empty


class TestGetEmptyResults:
    """Test _get_empty_results method."""

    def test_get_empty_results(self, backtester):
        """Test _get_empty_results returns proper structure."""
        results = backtester._get_empty_results()
        
        assert isinstance(results, dict)
        assert 'error' in results
        assert 'strategy' in results
        assert 'initial_cash' in results
        assert 'final_value' in results
        assert 'total_return_pct' in results
        assert 'num_trades' in results
        
        assert results['total_return_pct'] == 0.0
        assert results['num_trades'] == 0
        assert results['final_value'] == results['initial_cash']


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_integration_unanimous_long_signals(self, backtester, sample_price_data):
        """Test complete workflow with unanimous long signals."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        
        # Two portfolios with identical long signals
        portfolio1 = pd.DataFrame({
            'PositionType': ['none', 'long', 'long', 'long', 'none', 'none', 'long', 'long', 'long', 'none'],
            'PortfolioValue': np.random.uniform(9000, 11000, 10)
        }, index=dates)
        
        portfolio2 = portfolio1.copy()
        portfolio_dfs = [portfolio1, portfolio2]
        price_data = sample_price_data.iloc[:10]
        
        results, portfolio_df, _ = backtester.run_combined_trade(
            portfolio_dfs=portfolio_dfs,
            price_data=price_data,
            trading_type='long',
            combination_logic='unanimous'
        )
        
        assert isinstance(results, dict)
        assert isinstance(portfolio_df, pd.DataFrame)
        assert results['strategy'] == 'Combined Strategy (long)'
        assert 'final_value' in results

    def test_integration_majority_mixed_signals(self, backtester, sample_price_data):
        """Test complete workflow with majority logic and mixed signals."""
        dates = pd.date_range('2023-01-01', periods=8, freq='D')
        
        # Three portfolios with different signals
        portfolio1 = pd.DataFrame({
            'PositionType': ['long', 'long', 'none', 'short', 'short', 'none', 'long', 'long'],
            'PortfolioValue': np.random.uniform(9000, 11000, 8)
        }, index=dates)
        
        portfolio2 = pd.DataFrame({
            'PositionType': ['long', 'none', 'none', 'short', 'none', 'long', 'long', 'none'],
            'PortfolioValue': np.random.uniform(9000, 11000, 8)
        }, index=dates)
        
        portfolio3 = pd.DataFrame({
            'PositionType': ['none', 'long', 'short', 'none', 'short', 'long', 'none', 'long'],
            'PortfolioValue': np.random.uniform(9000, 11000, 8)
        }, index=dates)
        
        portfolio_dfs = [portfolio1, portfolio2, portfolio3]
        price_data = sample_price_data.iloc[:8]
        
        results, portfolio_df, _ = backtester.run_combined_trade(
            portfolio_dfs=portfolio_dfs,
            price_data=price_data,
            trading_type='mixed',
            combination_logic='majority'
        )
        
        assert isinstance(results, dict)
        assert isinstance(portfolio_df, pd.DataFrame)
        assert results['strategy'] == 'Combined Strategy (mixed)'

    def test_integration_performance_metrics(self, backtester, sample_price_data):
        """Test that performance metrics are properly calculated."""
        dates = pd.date_range('2023-01-01', periods=15, freq='D')
        
        portfolio_df = pd.DataFrame({
            'PositionType': ['none'] * 3 + ['long'] * 8 + ['none'] * 4,
            'PortfolioValue': np.random.uniform(9000, 11000, 15)
        }, index=dates)
        
        portfolio_dfs = [portfolio_df]
        price_data = sample_price_data.iloc[:15]
        
        results, portfolio_df_result, _ = backtester.run_combined_trade(
            portfolio_dfs=portfolio_dfs,
            price_data=price_data,
            trading_type='long',
            risk_free_rate=0.02
        )
        
        # Check that key performance metrics are present
        expected_metrics = [
            'initial_cash', 'final_value', 'num_trades', 'strategy'
        ]
        
        for metric in expected_metrics:
            assert metric in results
        
        assert isinstance(results['final_value'], (int, float))
        assert isinstance(results['num_trades'], int)
        assert results['num_trades'] >= 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_portfolio_dataframe(self, backtester, sample_price_data, sample_portfolio_df_long):
        """Test with single portfolio DataFrame."""
        portfolio_dfs = [sample_portfolio_df_long]
        
        results, portfolio_df, _ = backtester.run_combined_trade(
            portfolio_dfs=portfolio_dfs,
            price_data=sample_price_data
        )
        
        assert isinstance(results, dict)
        assert isinstance(portfolio_df, pd.DataFrame)

    def test_multiple_portfolio_dataframes(self, backtester, sample_price_data, 
                                         sample_portfolio_df_long, sample_portfolio_df_short, sample_portfolio_df_mixed):
        """Test with multiple portfolio DataFrames."""
        portfolio_dfs = [sample_portfolio_df_long, sample_portfolio_df_short, sample_portfolio_df_mixed]
        
        results, portfolio_df, _ = backtester.run_combined_trade(
            portfolio_dfs=portfolio_dfs,
            price_data=sample_price_data,
            combination_logic='majority'
        )
        
        assert isinstance(results, dict)
        assert isinstance(portfolio_df, pd.DataFrame)

    def test_mismatched_date_ranges(self, backtester):
        """Test with mismatched date ranges between portfolios and price data."""
        price_dates = pd.date_range('2023-01-01', periods=10, freq='D')
        portfolio_dates = pd.date_range('2023-01-05', periods=10, freq='D')  # Partial overlap
        
        price_data = pd.DataFrame({
            'Close': np.random.uniform(95, 105, 10)
        }, index=price_dates)
        
        portfolio_df = pd.DataFrame({
            'PositionType': ['long'] * 10,
            'PortfolioValue': np.random.uniform(9000, 11000, 10)
        }, index=portfolio_dates)
        
        results, portfolio_df_result, _ = backtester.run_combined_trade(
            portfolio_dfs=[portfolio_df],
            price_data=price_data
        )
        
        # Should handle partial overlap gracefully
        assert isinstance(results, dict)
        assert isinstance(portfolio_df_result, pd.DataFrame)

    def test_all_none_positions(self, backtester, sample_price_data):
        """Test with portfolio containing only 'none' positions."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        portfolio_df = pd.DataFrame({
            'PositionType': ['none'] * 10,
            'PortfolioValue': [10000] * 10
        }, index=dates)
        
        price_data = sample_price_data.iloc[:10]
        
        results, portfolio_df_result, _ = backtester.run_combined_trade(
            portfolio_dfs=[portfolio_df],
            price_data=price_data
        )
        
        assert isinstance(results, dict)
        assert isinstance(portfolio_df_result, pd.DataFrame)
        # Should have no trades
        assert results['num_trades'] == 0


class TestPlotCombinedResults:
    """Test plot_combined_results function."""

    @pytest.fixture
    def sample_strategies(self):
        """Sample strategies dictionary for plotting tests."""
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        np.random.seed(42)
        
        portfolio1 = pd.DataFrame({
            'PositionType': ['none'] * 5 + ['long'] * 10 + ['none'] * 5,
            'PortfolioValue': np.linspace(10000, 11000, 20)
        }, index=dates)
        
        portfolio2 = pd.DataFrame({
            'PositionType': ['none'] * 3 + ['short'] * 8 + ['none'] * 9,
            'PortfolioValue': np.linspace(10000, 10500, 20)
        }, index=dates)
        
        return {
            'RSI': {
                'results': {
                    'total_return_pct': 10.0,
                    'sharpe_ratio': 1.5,
                    'num_trades': 5
                },
                'portfolio': portfolio1
            },
            'MACD': {
                'results': {
                    'total_return_pct': 5.0,
                    'sharpe_ratio': 0.8,
                    'num_trades': 3
                },
                'portfolio': portfolio2
            }
        }

    @pytest.fixture
    def sample_voting_results(self):
        """Sample voting results dictionary for plotting tests."""
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        np.random.seed(43)
        
        portfolio = pd.DataFrame({
            'PositionType': ['none'] * 4 + ['long'] * 8 + ['none'] * 8,
            'PortfolioValue': np.linspace(10000, 10800, 20)
        }, index=dates)
        
        return {
            'Combined': {
                'results': {
                    'total_return_pct': 8.0,
                    'sharpe_ratio': 1.2,
                    'num_trades': 4
                },
                'portfolio': portfolio
            }
        }

    def test_plot_combined_results_fig_control_0(self, sample_price_data, sample_strategies, sample_voting_results):
        """Test plot_combined_results with fig_control=0 returns None."""
        result = plot_combined_results(
            price_data=sample_price_data,
            strategies=sample_strategies,
            voting_results=sample_voting_results,
            fig_control=0
        )
        
        assert result == (None, None, None)

    @patch('simple_trade.combine_trade.plt')
    def test_plot_combined_results_fig_control_1(self, mock_plt, sample_price_data, sample_strategies, sample_voting_results):
        """Test plot_combined_results with fig_control=1 creates and shows figures."""
        # Setup mock figures and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax_twin = MagicMock()
        mock_ax.twinx.return_value = mock_ax_twin
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        fig_perf, fig_signals, fig_table = plot_combined_results(
            price_data=sample_price_data,
            strategies=sample_strategies,
            voting_results=sample_voting_results,
            fig_control=1
        )
        
        # Verify figures were created
        assert fig_perf is not None
        assert fig_signals is not None
        assert fig_table is not None
        
        # Verify plt.show() was called
        mock_plt.show.assert_called_once()
        
        # Verify subplots were created (3 figures)
        assert mock_plt.subplots.call_count == 3

    @patch('simple_trade.combine_trade.plt')
    def test_plot_combined_results_fig_control_2(self, mock_plt, sample_price_data, sample_strategies, sample_voting_results):
        """Test plot_combined_results with fig_control=2 creates but doesn't show figures."""
        # Setup mock figures and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax_twin = MagicMock()
        mock_ax.twinx.return_value = mock_ax_twin
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        fig_perf, fig_signals, fig_table = plot_combined_results(
            price_data=sample_price_data,
            strategies=sample_strategies,
            voting_results=sample_voting_results,
            fig_control=2
        )
        
        # Verify figures were created
        assert fig_perf is not None
        assert fig_signals is not None
        assert fig_table is not None
        
        # Verify plt.show() was NOT called
        mock_plt.show.assert_not_called()

    @patch('simple_trade.combine_trade.plt')
    def test_plot_combined_results_empty_strategies(self, mock_plt, sample_price_data, sample_voting_results):
        """Test plot_combined_results with empty strategies dict."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax_twin = MagicMock()
        mock_ax.twinx.return_value = mock_ax_twin
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        fig_perf, fig_signals, fig_table = plot_combined_results(
            price_data=sample_price_data,
            strategies={},
            voting_results=sample_voting_results,
            fig_control=2
        )
        
        # Should still create figures
        assert fig_perf is not None
        assert fig_signals is not None
        assert fig_table is not None

    @patch('simple_trade.combine_trade.plt')
    def test_plot_combined_results_empty_voting_results(self, mock_plt, sample_price_data, sample_strategies):
        """Test plot_combined_results with empty voting_results dict."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax_twin = MagicMock()
        mock_ax.twinx.return_value = mock_ax_twin
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        fig_perf, fig_signals, fig_table = plot_combined_results(
            price_data=sample_price_data,
            strategies=sample_strategies,
            voting_results={},
            fig_control=2
        )
        
        # Should still create figures
        assert fig_perf is not None
        assert fig_signals is not None
        assert fig_table is not None

    @patch('simple_trade.combine_trade.plt')
    def test_plot_combined_results_empty_portfolio(self, mock_plt, sample_price_data):
        """Test plot_combined_results with empty portfolio in strategy."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax_twin = MagicMock()
        mock_ax.twinx.return_value = mock_ax_twin
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        strategies = {
            'EmptyStrategy': {
                'results': {'total_return_pct': 0, 'sharpe_ratio': 0, 'num_trades': 0},
                'portfolio': pd.DataFrame()  # Empty portfolio
            }
        }
        
        voting_results = {
            'Combined': {
                'results': {'total_return_pct': 0, 'sharpe_ratio': 0, 'num_trades': 0},
                'portfolio': pd.DataFrame()  # Empty portfolio
            }
        }
        
        fig_perf, fig_signals, fig_table = plot_combined_results(
            price_data=sample_price_data,
            strategies=strategies,
            voting_results=voting_results,
            fig_control=2
        )
        
        # Should handle empty portfolios gracefully
        assert fig_perf is not None
        assert fig_signals is not None
        assert fig_table is not None

    @patch('simple_trade.combine_trade.plt')
    def test_plot_combined_results_custom_price_col(self, mock_plt, sample_price_data, sample_strategies, sample_voting_results):
        """Test plot_combined_results with custom price column."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax_twin = MagicMock()
        mock_ax.twinx.return_value = mock_ax_twin
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        # Add custom price column
        sample_price_data['AdjClose'] = sample_price_data['Close'] * 1.01
        
        fig_perf, fig_signals, fig_table = plot_combined_results(
            price_data=sample_price_data,
            strategies=sample_strategies,
            voting_results=sample_voting_results,
            price_col='AdjClose',
            fig_control=2
        )
        
        assert fig_perf is not None
        assert fig_signals is not None
        assert fig_table is not None

    @patch('simple_trade.combine_trade.plt')
    def test_plot_combined_results_multiple_strategies(self, mock_plt, sample_price_data):
        """Test plot_combined_results with many strategies to test color cycling."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax_twin = MagicMock()
        mock_ax.twinx.return_value = mock_ax_twin
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        
        # Create more strategies than available colors to test color cycling
        strategies = {}
        for i in range(10):
            strategies[f'Strategy_{i}'] = {
                'results': {
                    'total_return_pct': i * 2.0,
                    'sharpe_ratio': 0.5 + i * 0.1,
                    'num_trades': i + 1
                },
                'portfolio': pd.DataFrame({
                    'PositionType': ['long'] * 20,
                    'PortfolioValue': np.linspace(10000, 10000 + i * 100, 20)
                }, index=dates)
            }
        
        voting_results = {
            'Unanimous': {
                'results': {'total_return_pct': 15.0, 'sharpe_ratio': 1.0, 'num_trades': 8},
                'portfolio': pd.DataFrame({
                    'PositionType': ['long'] * 20,
                    'PortfolioValue': np.linspace(10000, 11500, 20)
                }, index=dates)
            }
        }
        
        fig_perf, fig_signals, fig_table = plot_combined_results(
            price_data=sample_price_data,
            strategies=strategies,
            voting_results=voting_results,
            fig_control=2
        )
        
        assert fig_perf is not None
        assert fig_signals is not None
        assert fig_table is not None

    @patch('simple_trade.combine_trade.plt')
    def test_plot_combined_results_sharpe_non_numeric(self, mock_plt, sample_price_data):
        """Test plot_combined_results handles non-numeric Sharpe ratio."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax_twin = MagicMock()
        mock_ax.twinx.return_value = mock_ax_twin
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        
        strategies = {
            'TestStrategy': {
                'results': {
                    'total_return_pct': 5.0,
                    'sharpe_ratio': 'N/A',  # Non-numeric Sharpe
                    'num_trades': 3
                },
                'portfolio': pd.DataFrame({
                    'PositionType': ['long'] * 20,
                    'PortfolioValue': np.linspace(10000, 10500, 20)
                }, index=dates)
            }
        }
        
        voting_results = {}
        
        fig_perf, fig_signals, fig_table = plot_combined_results(
            price_data=sample_price_data,
            strategies=strategies,
            voting_results=voting_results,
            fig_control=2
        )
        
        assert fig_perf is not None
        assert fig_signals is not None
        assert fig_table is not None


class TestRunCombinedTradeWithFigures:
    """Test run_combined_trade with figure generation."""

    @patch('simple_trade.combine_trade.plot_combined_results')
    def test_run_combined_trade_fig_control_1(self, mock_plot, backtester, sample_price_data, sample_portfolio_df_long):
        """Test run_combined_trade with fig_control=1."""
        mock_plot.return_value = (MagicMock(), MagicMock(), MagicMock())
        
        portfolio_dfs = [sample_portfolio_df_long]
        
        results, portfolio_df, figures = backtester.run_combined_trade(
            portfolio_dfs=portfolio_dfs,
            price_data=sample_price_data,
            trading_type='long',
            fig_control=1
        )
        
        assert isinstance(results, dict)
        assert isinstance(portfolio_df, pd.DataFrame)
        assert figures is not None
        assert len(figures) == 3
        mock_plot.assert_called_once()

    @patch('simple_trade.combine_trade.plot_combined_results')
    def test_run_combined_trade_fig_control_2(self, mock_plot, backtester, sample_price_data, sample_portfolio_df_long):
        """Test run_combined_trade with fig_control=2."""
        mock_plot.return_value = (MagicMock(), MagicMock(), MagicMock())
        
        portfolio_dfs = [sample_portfolio_df_long]
        
        results, portfolio_df, figures = backtester.run_combined_trade(
            portfolio_dfs=portfolio_dfs,
            price_data=sample_price_data,
            trading_type='long',
            fig_control=2
        )
        
        assert isinstance(results, dict)
        assert isinstance(portfolio_df, pd.DataFrame)
        assert figures is not None
        mock_plot.assert_called_once()

    @patch('simple_trade.combine_trade.plot_combined_results')
    def test_run_combined_trade_with_strategies_dict(self, mock_plot, backtester, sample_price_data, sample_portfolio_df_long):
        """Test run_combined_trade with strategies dict for plotting."""
        mock_plot.return_value = (MagicMock(), MagicMock(), MagicMock())
        
        portfolio_dfs = [sample_portfolio_df_long]
        
        strategies = {
            'RSI': {
                'results': {'total_return_pct': 10.0, 'sharpe_ratio': 1.5, 'num_trades': 5},
                'portfolio': sample_portfolio_df_long
            }
        }
        
        results, portfolio_df, figures = backtester.run_combined_trade(
            portfolio_dfs=portfolio_dfs,
            price_data=sample_price_data,
            trading_type='long',
            fig_control=2,
            strategies=strategies,
            strategy_name='MyStrategy'
        )
        
        assert isinstance(results, dict)
        assert figures is not None
        
        # Check that plot_combined_results was called with correct arguments
        call_kwargs = mock_plot.call_args[1]
        assert 'strategies' in call_kwargs
        assert call_kwargs['strategies'] == strategies
        assert 'MyStrategy' in call_kwargs['voting_results']

    def test_run_combined_trade_empty_signals_with_fig_control(self, backtester):
        """Test run_combined_trade with empty signals and fig_control > 0."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        price_data = pd.DataFrame({'Close': [100, 101, 102, 103, 104]}, index=dates)
        
        # Portfolio with no overlapping dates
        different_dates = pd.date_range('2023-02-01', periods=5, freq='D')
        portfolio_df = pd.DataFrame({'PositionType': ['long'] * 5}, index=different_dates)
        
        results, portfolio_df_result, figures = backtester.run_combined_trade(
            portfolio_dfs=[portfolio_df],
            price_data=price_data,
            fig_control=1
        )
        
        assert 'error' in results
        assert portfolio_df_result.empty
        assert figures == (None, None, None)
