import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from simple_trade.run_combined_trade_strategies import run_combined_trade, plot_combined_results
from simple_trade.config import BacktestConfig


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
def default_config(default_parameters):
    """BacktestConfig instance with default parameters."""
    return BacktestConfig(**default_parameters)


class TestCombineTrade:
    """Test suite for run_combined_trade function."""


class TestInitialization:
    """Test BacktestConfig initialization."""

    def test_initialization_default(self):
        """Test initialization with default parameters."""
        config = BacktestConfig()
        assert hasattr(config, 'initial_cash')
        assert isinstance(config, BacktestConfig)

    def test_initialization_custom_params(self, default_parameters):
        """Test initialization with custom parameters."""
        config = BacktestConfig(**default_parameters)
        assert config.initial_cash == 10000.0
        assert config.commission_long == 0.001
        assert config.commission_short == 0.001


class TestRunCombinedTrade:
    """Test run_combined_trade function."""

    def test_run_combined_trade_basic(self, default_config, sample_price_data, sample_portfolio_df_long):
        """Test basic run_combined_trade functionality."""
        portfolio_dfs = [sample_portfolio_df_long]
        
        results, portfolio_df, figures = run_combined_trade(
            portfolio_dfs=portfolio_dfs,
            price_data=sample_price_data,
            config=default_config,
            trading_type='long'
        )
        
        assert isinstance(results, dict)
        assert isinstance(portfolio_df, pd.DataFrame)
        assert figures is None  # fig_control=0 by default
        assert 'strategy' in results
        assert 'final_value' in results
        assert 'num_trades' in results

    def test_run_combined_trade_different_trading_types(self, default_config, sample_price_data, sample_portfolio_df_mixed):
        """Test run_combined_trade with different trading types."""
        portfolio_dfs = [sample_portfolio_df_mixed]
        
        for trading_type in ['long', 'short', 'mixed']:
            results, portfolio_df, figures = run_combined_trade(
                portfolio_dfs=portfolio_dfs,
                price_data=sample_price_data,
                config=default_config,
                trading_type=trading_type
            )
            
            assert 'final_value' in results

    def test_run_combined_trade_custom_parameters(self, default_config, sample_price_data, sample_portfolio_df_long):
        """Test run_combined_trade with custom parameters."""
        portfolio_dfs = [sample_portfolio_df_long]
        
        results, portfolio_df, figures = run_combined_trade(
            portfolio_dfs=portfolio_dfs,
            price_data=sample_price_data,
            config=default_config,
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

    def test_run_combined_trade_input_validation(self, default_config, sample_price_data):
        """Test input validation in run_combined_trade."""
        # Test invalid combination_logic
        with pytest.raises(ValueError, match="combination_logic must be either"):
            run_combined_trade(
                portfolio_dfs=[sample_price_data],
                price_data=sample_price_data,
                config=default_config,
                combination_logic='invalid'
            )
        
        # Test empty portfolio_dfs
        with pytest.raises(ValueError, match="must be a non-empty list"):
            run_combined_trade(
                portfolio_dfs=[],
                price_data=sample_price_data,
                config=default_config
            )
        
        # Test non-DataFrame price_data
        with pytest.raises(TypeError, match="must be a DataFrame with a DatetimeIndex"):
            run_combined_trade(
                portfolio_dfs=[sample_price_data],
                price_data="not a dataframe",
                config=default_config
            )
        
        # Test missing price column
        with pytest.raises(ValueError, match="Price column .* not found"):
            run_combined_trade(
                portfolio_dfs=[sample_price_data],
                price_data=sample_price_data,
                config=default_config,
                price_col='NonExistentColumn'
            )

    def test_run_combined_trade_empty_signals(self, default_config):
        """Test run_combined_trade with empty combined signals."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        price_data = pd.DataFrame({'Close': [100, 101, 102, 103, 104]}, index=dates)
        
        # Portfolio with no overlapping dates
        different_dates = pd.date_range('2023-02-01', periods=5, freq='D')
        portfolio_df = pd.DataFrame({'PositionType': ['long'] * 5}, index=different_dates)
        
        results, portfolio_df_result, figures = run_combined_trade(
            portfolio_dfs=[portfolio_df],
            price_data=price_data,
            config=default_config
        )
        
        assert 'error' in results
        assert portfolio_df_result.empty
        assert figures is None  # fig_control=0 by default


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_integration_unanimous_long_signals(self, default_config, sample_price_data):
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
        
        results, portfolio_df, _ = run_combined_trade(
            portfolio_dfs=portfolio_dfs,
            price_data=price_data,
            config=default_config,
            trading_type='long',
            combination_logic='unanimous'
        )
        
        assert isinstance(results, dict)
        assert isinstance(portfolio_df, pd.DataFrame)
        assert 'final_value' in results

    def test_integration_majority_mixed_signals(self, default_config, sample_price_data):
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
        
        results, portfolio_df, _ = run_combined_trade(
            portfolio_dfs=portfolio_dfs,
            price_data=price_data,
            config=default_config,
            trading_type='mixed',
            combination_logic='majority'
        )
        
        assert isinstance(results, dict)
        assert isinstance(portfolio_df, pd.DataFrame)

    def test_integration_performance_metrics(self, default_config, sample_price_data):
        """Test that performance metrics are properly calculated."""
        dates = pd.date_range('2023-01-01', periods=15, freq='D')
        
        portfolio_df = pd.DataFrame({
            'PositionType': ['none'] * 3 + ['long'] * 8 + ['none'] * 4,
            'PortfolioValue': np.random.uniform(9000, 11000, 15)
        }, index=dates)
        
        portfolio_dfs = [portfolio_df]
        price_data = sample_price_data.iloc[:15]
        
        results, portfolio_df_result, _ = run_combined_trade(
            portfolio_dfs=portfolio_dfs,
            price_data=price_data,
            config=default_config,
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

    def test_single_portfolio_dataframe(self, default_config, sample_price_data, sample_portfolio_df_long):
        """Test with single portfolio DataFrame."""
        portfolio_dfs = [sample_portfolio_df_long]
        
        results, portfolio_df, _ = run_combined_trade(
            portfolio_dfs=portfolio_dfs,
            price_data=sample_price_data,
            config=default_config
        )
        
        assert isinstance(results, dict)
        assert isinstance(portfolio_df, pd.DataFrame)

    def test_multiple_portfolio_dataframes(self, default_config, sample_price_data, 
                                         sample_portfolio_df_long, sample_portfolio_df_short, sample_portfolio_df_mixed):
        """Test with multiple portfolio DataFrames."""
        portfolio_dfs = [sample_portfolio_df_long, sample_portfolio_df_short, sample_portfolio_df_mixed]
        
        results, portfolio_df, _ = run_combined_trade(
            portfolio_dfs=portfolio_dfs,
            price_data=sample_price_data,
            config=default_config,
            combination_logic='majority'
        )
        
        assert isinstance(results, dict)
        assert isinstance(portfolio_df, pd.DataFrame)

    def test_mismatched_date_ranges(self, default_config):
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
        
        results, portfolio_df_result, _ = run_combined_trade(
            portfolio_dfs=[portfolio_df],
            price_data=price_data,
            config=default_config
        )
        
        # Should handle partial overlap gracefully
        assert isinstance(results, dict)
        assert isinstance(portfolio_df_result, pd.DataFrame)

    def test_all_none_positions(self, default_config, sample_price_data):
        """Test with portfolio containing only 'none' positions."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        portfolio_df = pd.DataFrame({
            'PositionType': ['none'] * 10,
            'PortfolioValue': [10000] * 10
        }, index=dates)
        
        price_data = sample_price_data.iloc[:10]
        
        results, portfolio_df_result, _ = run_combined_trade(
            portfolio_dfs=[portfolio_df],
            price_data=price_data,
            config=default_config
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

    @patch('simple_trade.run_combined_trade_strategies.plt')
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

    @patch('simple_trade.run_combined_trade_strategies.plt')
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

    @patch('simple_trade.run_combined_trade_strategies.plt')
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

    @patch('simple_trade.run_combined_trade_strategies.plt')
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

    @patch('simple_trade.run_combined_trade_strategies.plt')
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

    @patch('simple_trade.run_combined_trade_strategies.plt')
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

    @patch('simple_trade.run_combined_trade_strategies.plt')
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

    @patch('simple_trade.run_combined_trade_strategies.plt')
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

    @patch('simple_trade.run_combined_trade_strategies.plot_combined_results')
    def test_run_combined_trade_fig_control_1(self, mock_plot, default_config, sample_price_data, sample_portfolio_df_long):
        """Test run_combined_trade with fig_control=1."""
        mock_plot.return_value = (MagicMock(), MagicMock(), MagicMock())
        
        portfolio_dfs = [sample_portfolio_df_long]
        
        results, portfolio_df, figures = run_combined_trade(
            portfolio_dfs=portfolio_dfs,
            price_data=sample_price_data,
            config=default_config,
            trading_type='long',
            fig_control=1
        )
        
        assert isinstance(results, dict)
        assert isinstance(portfolio_df, pd.DataFrame)
        assert figures is not None
        assert len(figures) == 3
        mock_plot.assert_called_once()

    @patch('simple_trade.run_combined_trade_strategies.plot_combined_results')
    def test_run_combined_trade_fig_control_2(self, mock_plot, default_config, sample_price_data, sample_portfolio_df_long):
        """Test run_combined_trade with fig_control=2."""
        mock_plot.return_value = (MagicMock(), MagicMock(), MagicMock())
        
        portfolio_dfs = [sample_portfolio_df_long]
        
        results, portfolio_df, figures = run_combined_trade(
            portfolio_dfs=portfolio_dfs,
            price_data=sample_price_data,
            config=default_config,
            trading_type='long',
            fig_control=2
        )
        
        assert isinstance(results, dict)
        assert isinstance(portfolio_df, pd.DataFrame)
        assert figures is not None
        mock_plot.assert_called_once()

    @patch('simple_trade.run_combined_trade_strategies.plot_combined_results')
    def test_run_combined_trade_with_strategies_dict(self, mock_plot, default_config, sample_price_data, sample_portfolio_df_long):
        """Test run_combined_trade with strategies dict for plotting."""
        mock_plot.return_value = (MagicMock(), MagicMock(), MagicMock())
        
        portfolio_dfs = [sample_portfolio_df_long]
        
        strategies = {
            'RSI': {
                'results': {'total_return_pct': 10.0, 'sharpe_ratio': 1.5, 'num_trades': 5},
                'portfolio': sample_portfolio_df_long
            }
        }
        
        results, portfolio_df, figures = run_combined_trade(
            portfolio_dfs=portfolio_dfs,
            price_data=sample_price_data,
            config=default_config,
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

    def test_run_combined_trade_empty_signals_with_fig_control(self, default_config):
        """Test run_combined_trade with empty signals and fig_control > 0."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        price_data = pd.DataFrame({'Close': [100, 101, 102, 103, 104]}, index=dates)
        
        # Portfolio with no overlapping dates
        different_dates = pd.date_range('2023-02-01', periods=5, freq='D')
        portfolio_df = pd.DataFrame({'PositionType': ['long'] * 5}, index=different_dates)
        
        results, portfolio_df_result, figures = run_combined_trade(
            portfolio_dfs=[portfolio_df],
            price_data=price_data,
            config=default_config,
            fig_control=1
        )
        
        assert 'error' in results
        assert portfolio_df_result.empty
        assert figures == (None, None, None)
