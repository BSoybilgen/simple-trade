import pytest
import pandas as pd
import numpy as np
from simple_trade.run_cross_trade_strategies import run_cross_trade
from simple_trade.config import BacktestConfig
from unittest.mock import MagicMock
from unittest.mock import patch

# --- Fixtures ---

@pytest.fixture
def sample_cross_data():
    """Fixture to provide sample data for cross trade testing."""
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
    data = pd.DataFrame(index=dates)
    data['Close'] = [100, 102, 105, 103, 106, 108, 110, 109, 107, 111]
    # Create crossing indicators with clearer signals
    # Short crosses ABOVE Long on day 3 (idx 2), signal on day 4 (idx 3)
    # Short crosses BELOW Long on day 8 (idx 7), signal on day 9 (idx 8)
    data['SMA_S'] = [99, 100, 101, 104, 105, 107, 109, 108, 106, 107] # Short
    data['SMA_L'] = [101, 101, 101, 102, 103, 105, 107, 109, 109, 110] # Long
    return data

@pytest.fixture
def default_config():
    """Fixture to provide a default BacktestConfig."""
    return BacktestConfig(initial_cash=10000, commission_long=0.001, commission_short=0.001)

# --- Test Class ---

class TestCrossTrade:
    """Tests for the run_cross_trade function."""

    # --- Input Validation Tests ---

    def test_invalid_data_type(self, default_config):
        """Test that TypeError is raised for non-DataFrame input."""
        with pytest.raises(TypeError, match="data must be a pandas DataFrame"):
            run_cross_trade([1, 2, 3], 'SMA_S', 'SMA_L', config=default_config)

    def test_invalid_index_type(self, default_config):
        """Test that TypeError is raised for non-DatetimeIndex."""
        data = pd.DataFrame({'Close': [100], 'SMA_S': [99], 'SMA_L': [101]})
        with pytest.raises(TypeError, match="DataFrame index must be a DatetimeIndex"):
            run_cross_trade(data, 'SMA_S', 'SMA_L', config=default_config)

    def test_missing_price_column(self, default_config, sample_cross_data):
        """Test ValueError if price_col is missing."""
        data = sample_cross_data.drop(columns=['Close'])
        with pytest.raises(ValueError, match="Price column 'Close' not found"):
            run_cross_trade(data, 'SMA_S', 'SMA_L', config=default_config, price_col='Close')

    def test_missing_short_indicator_column(self, default_config, sample_cross_data):
        """Test ValueError if short_window_indicator is missing."""
        data = sample_cross_data.drop(columns=['SMA_S'])
        with pytest.raises(ValueError):
            run_cross_trade(data, 'SMA_S', 'SMA_L', config=default_config)

    def test_missing_long_indicator_column(self, default_config, sample_cross_data):
        """Test ValueError if long_window_indicator is missing."""
        data = sample_cross_data.drop(columns=['SMA_L'])
        with pytest.raises(ValueError):
            run_cross_trade(data, 'SMA_S', 'SMA_L', config=default_config)

    def test_invalid_trading_type(self, default_config, sample_cross_data):
        """Test ValueError for invalid trading_type."""
        with pytest.raises(ValueError, match="Invalid trading_type 'invalid_type'"):
            run_cross_trade(sample_cross_data, 'SMA_S', 'SMA_L', config=default_config, trading_type='invalid_type')

    def test_invalid_day1_position(self, default_config, sample_cross_data):
        """Test ValueError for invalid day1_position."""
        with pytest.raises(ValueError, match="Invalid day1_position 'sideways'"):
            run_cross_trade(sample_cross_data, 'SMA_S', 'SMA_L', config=default_config, day1_position='sideways')

    def test_incompatible_day1_long_trading_short(self, default_config, sample_cross_data):
        """Test ValueError for day1_position='long' with trading_type='short'."""
        with pytest.raises(ValueError, match="Cannot use day1_position='long' with trading_type='short'"):
            run_cross_trade(sample_cross_data, 'SMA_S', 'SMA_L', config=default_config, trading_type='short', day1_position='long')

    def test_incompatible_day1_short_trading_long(self, default_config, sample_cross_data):
        """Test ValueError for day1_position='short' with trading_type='long'."""
        with pytest.raises(ValueError, match="Cannot use day1_position='short' with trading_type='long'"):
            run_cross_trade(sample_cross_data, 'SMA_S', 'SMA_L', config=default_config, trading_type='long', day1_position='short')

    # --- Test Edge Cases ---

    def test_run_cross_trade_missing_columns(self, default_config):
        """Test running with missing required columns."""
        data_missing_close = pd.DataFrame({'SMA_S': [99, 100], 'SMA_L': [101, 101]}, index=pd.date_range('20230101', periods=2))
        with pytest.raises(ValueError, match="Price column 'Close' not found"):
            run_cross_trade(data_missing_close, 'SMA_S', 'SMA_L', config=default_config)
            
    # --- Test Trading Logic ---
    
    def test_long_only_strategy(self, default_config, sample_cross_data):
        """Test the long-only strategy executes buy and sell signals correctly."""
        # Run backtest with long-only trading type
        results, portfolio_df = run_cross_trade(
            sample_cross_data, 
            'SMA_S', 
            'SMA_L', 
            config=default_config,
            trading_type='long'
        )
        
        # Check that portfolio_df is not empty
        assert not portfolio_df.empty, "Portfolio DataFrame should not be empty"
        
        # Check number of trades matches expectations
        # Based on our sample data, should see 1 buy and 1 sell
        assert results['num_trades'] == 2, f"Expected 2 trades but got {results['num_trades']}"
        
        # Verify buy transaction occurred
        buy_day = portfolio_df[portfolio_df['Action'] == 'BUY']
        assert not buy_day.empty, "No BUY action found in portfolio log"
        
        # Verify sell transaction occurred
        sell_day = portfolio_df[portfolio_df['Action'] == 'SELL']
        assert not sell_day.empty, "No SELL action found in portfolio log"
        
        # Check that final value is reasonable (should be greater than initial after our trades)
        assert results['final_value'] > 9900, f"Final value {results['final_value']} is too low"  # Allowing for commissions
        
        # Verify that there was a positive position at some point
        assert (portfolio_df['PositionSize'] > 0).any(), "Should have a positive position at some point"
        
        # Verify that position is non-zero between buy and sell
        # Find the buy day index and sell day index
        buy_day_idx = buy_day.index[0]
        sell_day_idx = sell_day.index[0]
        
        # Check position between buy and one day before sell (since position becomes 0 on sell day)
        days_with_position = portfolio_df.loc[buy_day_idx:sell_day_idx].index
        if len(days_with_position) > 1:  # If there are days between buy and sell
            position_before_sell = portfolio_df.loc[days_with_position[:-1], 'PositionSize']
            assert (position_before_sell > 0).all(), "Position should be positive after buy until sell day"
    
    def test_short_only_strategy(self, default_config, sample_cross_data):
        """Test the short-only strategy executes short and cover signals correctly."""
        # Create data with reversed signals so we have clearer short entries
        # Make short indicator cross below long indicator for short entry signal
        data = sample_cross_data.copy()
        # Swap SMA_S and SMA_L to create opposite crossovers
        data['SMA_S'] = sample_cross_data['SMA_L'] 
        data['SMA_L'] = sample_cross_data['SMA_S']
        
        # Run backtest with short-only trading type
        results, portfolio_df = run_cross_trade(
            data, 
            'SMA_S', 
            'SMA_L', 
            config=default_config,
            trading_type='short',
            short_entry_pct_cash=0.5  # Use 50% of cash for short positions
        )
        
        # Check that portfolio_df is not empty
        assert not portfolio_df.empty, "Portfolio DataFrame should not be empty"
        
        # Verify short transaction occurred
        short_day = portfolio_df[portfolio_df['Action'] == 'SHORT']
        assert not short_day.empty, "No SHORT action found in portfolio log"
        
        # Verify cover transaction occurred
        cover_day = portfolio_df[portfolio_df['Action'] == 'COVER']
        assert not cover_day.empty, "No COVER action found in portfolio log"
        
        # Check number of trades matches expectations (should see 1 short and 1 cover)
        assert results['num_trades'] == 2, f"Expected 2 trades but got {results['num_trades']}"
        
        # Verify that there was a negative position at some point (short position)
        assert (portfolio_df['PositionSize'] < 0).any(), "Should have a negative position at some point"
        
        # Check position between short and cover
        if not short_day.empty and not cover_day.empty:
            short_day_idx = short_day.index[0]
            cover_day_idx = cover_day.index[0]
            
            # Check that days between short and one day before cover have negative position
            days_with_position = portfolio_df.loc[short_day_idx:cover_day_idx].index
            if len(days_with_position) > 1:  # If there are days between short and cover
                position_before_cover = portfolio_df.loc[days_with_position[:-1], 'PositionSize']
                assert (position_before_cover < 0).all(), "Position should be negative after short until cover day"
        
        # Verify that cash increases after short position is entered
        # (when shorting, we receive cash from selling borrowed shares)
        if not short_day.empty:
            short_idx = portfolio_df.index.get_loc(short_day.index[0])
            if short_idx > 0:
                cash_before_short = portfolio_df['Cash'].iloc[short_idx - 1]
                cash_after_short = portfolio_df['Cash'].iloc[short_idx]
                assert cash_after_short > cash_before_short, "Cash should increase after entering short position"
        
        # Check that final value is reasonable (should be different from initial cash)
        assert abs(results['final_value'] - default_config.initial_cash) > 1, "Final value should change from initial investment"
        
        # Verify the short process operates as expected:
        # 1. Portfolio value should typically decrease when market goes up during a short position
        if not short_day.empty and not cover_day.empty:
            short_idx = portfolio_df.index.get_loc(short_day.index[0])
            
            # Get indices for all days with short position
            short_days = portfolio_df.index[short_idx:portfolio_df.index.get_loc(cover_day.index[0])]
            
            if len(short_days) > 1:
                # Verify portfolio value decreases when price increases during short position
                for i in range(len(short_days) - 1):
                    current_idx = portfolio_df.index.get_loc(short_days[i])
                    next_idx = portfolio_df.index.get_loc(short_days[i+1])
                    
                    current_price = portfolio_df['Price'].iloc[current_idx]
                    next_price = portfolio_df['Price'].iloc[next_idx]
                    
                    current_value = portfolio_df['PortfolioValue'].iloc[current_idx]
                    next_value = portfolio_df['PortfolioValue'].iloc[next_idx]
                    
                    # If price increased, portfolio value should decrease (or at least not increase by much)
                    if next_price > current_price:
                        assert next_value <= current_value or abs(next_value - current_value) < 0.1, \
                            "Portfolio value should decrease when price increases during short position"
    
    def test_mixed_strategy_with_day1_position(self, default_config):
        """Test the mixed trading strategy with a day1 position."""
        # Create more complex data with multiple crossovers for mixed strategy testing
        dates = pd.date_range(start='2023-01-01', periods=15, freq='D')
        data = pd.DataFrame(index=dates)
        
        # Create price data with some volatility
        data['Close'] = [100, 102, 105, 103, 106, 108, 110, 109, 107, 111, 109, 105, 102, 106, 110]
        
        # Create crossing indicators with alternating crossovers
        # SMA_S crosses above SMA_L on day 3, below on day 7, above on day 12
        data['SMA_S'] = [99,  100, 104, 105, 107, 109, 108, 106, 105, 104, 103, 106, 107, 108, 109]
        data['SMA_L'] = [101, 102, 103, 104, 105, 106, 107, 108, 107, 106, 105, 104, 103, 104, 105]
        
        # Run backtest with mixed trading type and start with a long position on day 1
        results, portfolio_df = run_cross_trade(
            data, 
            'SMA_S', 
            'SMA_L', 
            config=default_config,
            trading_type='mixed',
            day1_position='long',  # Start with a long position
            long_entry_pct_cash=0.8,
            short_entry_pct_cash=0.3
        )
        
        # Check that portfolio_df is not empty
        assert not portfolio_df.empty, "Portfolio DataFrame should not be empty"
        
        # Verify we have more than 2 trades due to mixed strategy with multiple crossovers
        assert results['num_trades'] > 2, f"Expected more than 2 trades but got {results['num_trades']}"
        
        # Check for specific actions in mixed strategy
        # We should have at least one of each type of action
        actions = portfolio_df['Action'].unique()
        print(f"Unique actions: {actions}")
        
        # Verify we have a BUY action (either initial day1 or later crossover)
        assert 'BUY' in actions or portfolio_df.iloc[0]['PositionSize'] > 0, "No BUY action found"
        
        # Verify we have a positive position size at some point
        assert (portfolio_df['PositionSize'] > 0).any(), "No positive positions found"
        
        # Verify we have various trade types: day1 position should be long (from parameter)
        assert portfolio_df.iloc[0]['PositionSize'] > 0, "Day 1 position should be long"
        
        # Verify final portfolio value differs from initial
        assert abs(results['final_value'] - default_config.initial_cash) > 1, "Final value should change from initial"
        
        # Test that performance metrics are returned in results
        performance_metrics = ['sharpe_ratio', 'sortino_ratio', 'max_drawdown_pct', 
                              'annualized_return_pct', 'annualized_volatility_pct']
        for metric in performance_metrics:
            assert metric in results, f"Performance metric {metric} missing from results"
    
    def test_print_results(self, default_config, sample_cross_data, capsys):
        """Test that the print_results function works correctly."""
        from simple_trade.metrics import print_results
        
        # Run backtest with long-only trading type
        results, _ = run_cross_trade(
            sample_cross_data, 
            'SMA_S', 
            'SMA_L', 
            config=default_config,
            trading_type='long'
        )
        
        # Add some additional performance metrics to the results for coverage
        results.update({
            'start_date': pd.Timestamp('2023-01-01'),
            'end_date': pd.Timestamp('2023-01-10'),
            'duration_days': 9,
            'days_in_backtest': 10,
            'years': 0.04,
            'annualized_return_pct': 5.0,
            'annualized_volatility_pct': 15.0,
            'sharpe_ratio': 0.33,
            'sortino_ratio': 0.5,
            'calmar_ratio': 1.2,
            'max_drawdown_pct': -5.0,
            'avg_drawdown_pct': -2.0,
            'max_drawdown_duration_days': 3,
            'avg_drawdown_duration_days': 1.5,
            'benchmark_return_pct': 3.0,
            'benchmark_final_value': 10300.0,
            'total_commissions': 10.0
        })
        
        # Test with detailed=True (default)
        print_results(results)
        captured = capsys.readouterr()
        output_detailed = captured.out
        
        # Test with detailed=False
        print_results(results, detailed=False)
        captured = capsys.readouterr()
        output_simple = captured.out
        
        # Verify that both outputs contain the strategy name and basic metrics
        assert results['strategy'] in output_detailed
        assert results['strategy'] in output_simple
        assert "Initial Investment:" in output_detailed
        assert "Final Portfolio Value:" in output_detailed
        assert "Total Return:" in output_detailed
        assert "Number of Trades:" in output_detailed
        
        # Verify that detailed output contains risk metrics
        assert "RISK METRICS:" in output_detailed
        assert "Sharpe Ratio:" in output_detailed
        assert "Sortino Ratio:" in output_detailed
        assert "Maximum Drawdown:" in output_detailed
        
        # Verify the simplified output doesn't have the risk metrics section
        assert "RISK METRICS:" not in output_simple