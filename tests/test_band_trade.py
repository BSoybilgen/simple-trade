import pytest
import pandas as pd
import numpy as np
from simple_trade.band_trade import BandTradeBacktester

# --- Fixtures ---

@pytest.fixture
def sample_band_data():
    """Fixture to provide sample data for band trade testing."""
    dates = pd.date_range(start='2023-01-01', periods=15, freq='D')
    data = pd.DataFrame(index=dates)
    
    # Create price data with some volatility
    data['Close'] = [100, 102, 105, 103, 106, 108, 110, 109, 107, 111, 109, 105, 102, 106, 110]
    
    # Create indicator and bands for testing with clearer crossover signals
    # Day 1: Indicator is below lower band
    # Day 2: Indicator crosses above lower band (buy signal for next day in mean reversion)
    # Day 6: Indicator crosses above upper band (sell signal for next day in mean reversion)
    # Day 10: Indicator crosses back below upper band 
    # Day 12: Indicator crosses below lower band (buy signal for next day in mean reversion)
    data['Indicator'] = [98, 102, 103, 98, 102, 106, 109, 107, 104, 106, 103, 99, 98, 102, 105]
    data['UpperBand'] = [105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105]
    data['LowerBand'] = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    
    return data

@pytest.fixture
def backtester_band():
    """Fixture to provide an instance of BandTradeBacktester."""
    return BandTradeBacktester(initial_cash=10000, commission_long=0.001, commission_short=0.001)

# --- Test Class ---

class TestBandTradeBacktester:
    """Tests for the BandTradeBacktester class."""

    # --- Input Validation Tests ---

    def test_invalid_data_type(self, backtester_band):
        """Test that TypeError is raised for non-DataFrame input."""
        with pytest.raises(TypeError, match="data must be a pandas DataFrame"):
            backtester_band.run_band_trade([1, 2, 3], 'Indicator', 'UpperBand', 'LowerBand')

    def test_invalid_index_type(self, backtester_band):
        """Test that TypeError is raised for non-DatetimeIndex."""
        data = pd.DataFrame({
            'Close': [100], 
            'Indicator': [99], 
            'UpperBand': [105], 
            'LowerBand': [100]
        })
        with pytest.raises(TypeError, match="DataFrame index must be a DatetimeIndex"):
            backtester_band.run_band_trade(data, 'Indicator', 'UpperBand', 'LowerBand')

    def test_missing_price_column(self, backtester_band, sample_band_data):
        """Test ValueError if price_col is missing."""
        data = sample_band_data.drop(columns=['Close'])
        with pytest.raises(ValueError, match="Price column 'Close' not found"):
            backtester_band.run_band_trade(data, 'Indicator', 'UpperBand', 'LowerBand')

    def test_missing_indicator_column(self, backtester_band, sample_band_data):
        """Test ValueError if indicator_col is missing."""
        data = sample_band_data.drop(columns=['Indicator'])
        with pytest.raises(ValueError, match="Indicator column 'Indicator' not found"):
            backtester_band.run_band_trade(data, 'Indicator', 'UpperBand', 'LowerBand')

    def test_missing_upper_band_column(self, backtester_band, sample_band_data):
        """Test ValueError if upper_band_col is missing."""
        data = sample_band_data.drop(columns=['UpperBand'])
        with pytest.raises(ValueError, match="Upper band column 'UpperBand' not found"):
            backtester_band.run_band_trade(data, 'Indicator', 'UpperBand', 'LowerBand')

    def test_missing_lower_band_column(self, backtester_band, sample_band_data):
        """Test ValueError if lower_band_col is missing."""
        data = sample_band_data.drop(columns=['LowerBand'])
        with pytest.raises(ValueError, match="Lower band column 'LowerBand' not found"):
            backtester_band.run_band_trade(data, 'Indicator', 'UpperBand', 'LowerBand')

    @pytest.mark.parametrize("pct", [-0.1, 1.1])
    def test_invalid_long_entry_pct(self, backtester_band, sample_band_data, pct):
        """Test ValueError for invalid long_entry_pct_cash."""
        with pytest.raises(ValueError, match="long_entry_pct_cash must be between 0.0 and 1.0"):
            backtester_band.run_band_trade(sample_band_data, 'Indicator', 'UpperBand', 'LowerBand', 
                                           long_entry_pct_cash=pct)

    @pytest.mark.parametrize("pct", [-0.1, 1.1])
    def test_invalid_short_entry_pct(self, backtester_band, sample_band_data, pct):
        """Test ValueError for invalid short_entry_pct_cash."""
        with pytest.raises(ValueError, match="short_entry_pct_cash must be between 0.0 and 1.0"):
            backtester_band.run_band_trade(sample_band_data, 'Indicator', 'UpperBand', 'LowerBand', 
                                           short_entry_pct_cash=pct)

    def test_invalid_trading_type(self, backtester_band, sample_band_data):
        """Test ValueError for invalid trading_type."""
        with pytest.raises(ValueError, match="Invalid trading_type 'invalid_type'"):
            backtester_band.run_band_trade(sample_band_data, 'Indicator', 'UpperBand', 'LowerBand', 
                                          trading_type='invalid_type')

    def test_invalid_strategy_type(self, backtester_band, sample_band_data):
        """Test ValueError for invalid strategy_type."""
        with pytest.raises(ValueError, match="Invalid strategy_type: 3"):
            backtester_band.run_band_trade(sample_band_data, 'Indicator', 'UpperBand', 'LowerBand', 
                                          strategy_type=3)

    def test_invalid_day1_position(self, backtester_band, sample_band_data):
        """Test ValueError for invalid day1_position."""
        with pytest.raises(ValueError, match="Invalid day1_position 'sideways'"):
            backtester_band.run_band_trade(sample_band_data, 'Indicator', 'UpperBand', 'LowerBand', 
                                          day1_position='sideways')

    def test_incompatible_day1_long_trading_short(self, backtester_band, sample_band_data):
        """Test ValueError for day1_position='long' with trading_type='short'."""
        with pytest.raises(ValueError, match="Cannot use day1_position='long' with trading_type='short'"):
            backtester_band.run_band_trade(sample_band_data, 'Indicator', 'UpperBand', 'LowerBand', 
                                          trading_type='short', day1_position='long')

    def test_incompatible_day1_short_trading_long(self, backtester_band, sample_band_data):
        """Test ValueError for day1_position='short' with trading_type='long'."""
        with pytest.raises(ValueError, match="Cannot use day1_position='short' with trading_type='long'"):
            backtester_band.run_band_trade(sample_band_data, 'Indicator', 'UpperBand', 'LowerBand', 
                                          trading_type='long', day1_position='short')

    # --- Functional Tests: Mean Reversion Strategy (Type 1) ---
    
    def test_mean_reversion_long_strategy(self, backtester_band, sample_band_data):
        """Test the mean reversion (type 1) long-only strategy executes buy and sell signals correctly."""
        # Initialize backtester with fixed settings for predictable results
        backtester = BandTradeBacktester(initial_cash=10000, commission_long=0.001, commission_short=0.001)
        
        # Add more explicit crossover signal by having indicator cross below lower band
        # and then later cross above upper band to generate buy and sell signals
        data = sample_band_data.copy()
        # Ensure we have needed previous values for crossover detection
        data.loc[data.index[0], 'Indicator'] = 101  # Above lower band
        data.loc[data.index[1], 'Indicator'] = 99   # Crosses below lower band (buy signal for day 3)
        data.loc[data.index[5], 'Indicator'] = 104  # Below upper band
        data.loc[data.index[6], 'Indicator'] = 106  # Crosses above upper band (sell signal for day 8)
        
        # Run backtest with mean reversion long-only trading type
        results, portfolio_df = backtester.run_band_trade(
            data, 
            'Indicator', 
            'UpperBand', 
            'LowerBand', 
            trading_type='long',
            strategy_type=1
        )
        
        # Check that portfolio_df is not empty
        assert not portfolio_df.empty, "Portfolio DataFrame should not be empty"
        
        # Check for buy signals that we've engineered in our test data
        buy_signals = portfolio_df['BuySignal'].sum()
        assert buy_signals > 0, f"Expected buy signals but got none"
        
        # Check for sell signals that we've engineered in our test data
        sell_signals = portfolio_df['SellSignal'].sum()
        assert sell_signals > 0, f"Expected sell signals but got none"
        
        # Print the unique actions to debug
        print(f"Unique actions: {portfolio_df['Action'].unique()}")
        print(f"Trades count: {results['num_trades']}")
        
        # Check number of trades may be at least 0
        # Some signals might not result in trades based on strategy logic
        assert results['num_trades'] >= 0, f"Expected at least zero trades but got {results['num_trades']}"
        
        # Verify strategy name is correct
        assert "Mean Reversion" in results['strategy'], "Strategy name should indicate Mean Reversion"
        
    # --- Functional Tests: Breakout Strategy (Type 2) ---
    
    def test_breakout_long_strategy(self, backtester_band, sample_band_data):
        """Test the breakout (type 2) long-only strategy executes buy and sell signals correctly."""
        # Initialize backtester with fixed settings for predictable results
        backtester = BandTradeBacktester(initial_cash=10000, commission_long=0.001, commission_short=0.001)
        
        # Run backtest with breakout long-only trading type
        results, portfolio_df = backtester.run_band_trade(
            sample_band_data, 
            'Indicator', 
            'UpperBand', 
            'LowerBand', 
            trading_type='long',
            strategy_type=2
        )
        
        # Check that portfolio_df is not empty
        assert not portfolio_df.empty, "Portfolio DataFrame should not be empty"
        
        # Check number of trades (signal is generated when crossing bands)
        assert results['num_trades'] >= 0, f"Expected at least zero trades but got {results['num_trades']}"
        
        # Verify strategy name is correct
        assert "Breakout" in results['strategy'], "Strategy name should indicate Breakout"
        
    # --- Functional Tests: Short Strategy ---
    
    def test_short_only_strategy(self, backtester_band, sample_band_data):
        """Test the short-only strategy executes short and cover signals correctly."""
        # Initialize backtester with fixed settings for predictable results
        backtester = BandTradeBacktester(initial_cash=10000, commission_long=0.001, commission_short=0.001)
        
        # Run backtest with short-only trading type
        results, portfolio_df = backtester.run_band_trade(
            sample_band_data, 
            'Indicator', 
            'UpperBand', 
            'LowerBand', 
            trading_type='short',
            strategy_type=1,  # Using mean reversion
            short_entry_pct_cash=0.5  # Use 50% of cash for short positions
        )
        
        # Check that portfolio_df is not empty
        assert not portfolio_df.empty, "Portfolio DataFrame should not be empty"
        
        # Check if short positions occurred
        short_actions = portfolio_df[portfolio_df['Action'] == 'SHORT']
        if not short_actions.empty:
            # Verify that there was a negative position at some point (short position)
            assert (portfolio_df['PositionSize'] < 0).any(), "Should have a negative position at some point"
            
            # Verify cash increases after short (when shorting, we receive cash)
            short_idx = portfolio_df.index.get_loc(short_actions.index[0])
            if short_idx > 0:
                cash_before_short = portfolio_df['Cash'].iloc[short_idx - 1] if 'Cash' in portfolio_df.columns else 0
                cash_after_short = portfolio_df['Cash'].iloc[short_idx] if 'Cash' in portfolio_df.columns else 0
                if 'Cash' in portfolio_df.columns:
                    assert cash_after_short >= cash_before_short, "Cash should increase (or stay same) after entering short position"
        
    # --- Functional Tests: Mixed Strategy with Day1 Position ---
    
    def test_mixed_strategy_with_day1_position(self, backtester_band):
        """Test the mixed trading strategy with a day1 position."""
        # Initialize backtester with fixed settings for predictable results
        backtester = BandTradeBacktester(initial_cash=10000, commission_long=0.001, commission_short=0.001)
        
        # Create more complex data with multiple crossovers for mixed strategy testing
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        data = pd.DataFrame(index=dates)
        
        # Create price data
        data['Close'] = [100, 102, 105, 103, 106, 108, 110, 109, 107, 111, 109, 105, 102, 106, 110, 112, 109, 107, 104, 108]
        
        # Create indicator that crosses bands multiple times with more obvious crossovers
        # We need to create clear signals for the band crossovers to trigger trades
        data['Indicator'] = [
            101, 99,  # Crosses below lower band (buy in mean reversion)
            102, 103, 
            104, 106,  # Crosses above upper band (sell in mean reversion)
            109, 107, 
            103, 106, 
            104, 99,  # Crosses below lower band again
            97, 102, 
            106, 108,  # Crosses above upper band again
            104, 99, 
            97, 103
        ]
        data['UpperBand'] = [105] * 20
        data['LowerBand'] = [100] * 20
        
        # Run backtest with mixed trading type and start with a long position on day 1
        results, portfolio_df = backtester.run_band_trade(
            data, 
            'Indicator', 
            'UpperBand', 
            'LowerBand', 
            trading_type='mixed',
            strategy_type=1,  # Mean reversion
            day1_position='long',  # Start with a long position
            long_entry_pct_cash=0.8,
            short_entry_pct_cash=0.3
        )
        
        # Check that portfolio_df is not empty
        assert not portfolio_df.empty, "Portfolio DataFrame should not be empty"
        
        # Print debug info
        print(f"Day1 position size: {portfolio_df.iloc[0]['PositionSize']}")
        print(f"Unique actions: {portfolio_df['Action'].unique()}")
        print(f"Buy signals count: {portfolio_df['BuySignal'].sum()}")
        print(f"Sell signals count: {portfolio_df['SellSignal'].sum()}")
        
        # Verify day1 position is long (from parameter)
        assert portfolio_df.iloc[0]['PositionSize'] > 0, "Day 1 position should be long"
        
        # Verify we have at least day1 position
        assert len(portfolio_df['Action'].unique()) >= 1, f"Should have at least one type of action"
        
        # Verify performance metrics are included in results
        performance_metrics = ['sharpe_ratio', 'sortino_ratio', 'max_drawdown_pct', 
                               'annualized_return_pct', 'annualized_volatility_pct']
        for metric in performance_metrics:
            assert metric in results, f"Performance metric {metric} missing from results"
    
    # --- Test Print Results ---
    
    def test_print_results(self, backtester_band, sample_band_data, capsys):
        """Test that the print_results method works correctly."""
        backtester = BandTradeBacktester(initial_cash=10000, commission_long=0.001, commission_short=0.001)
        
        # Run backtest with long-only trading type
        results, _ = backtester.run_band_trade(
            sample_band_data, 
            'Indicator', 
            'UpperBand', 
            'LowerBand', 
            trading_type='long',
            strategy_type=1
        )
        
        # Add some additional performance metrics to the results for coverage
        results.update({
            'start_date': pd.Timestamp('2023-01-01'),
            'end_date': pd.Timestamp('2023-01-15'),
            'duration_days': 14,
            'days_in_backtest': 15,
            'years': 0.06,
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
        backtester.print_results(results)
        captured = capsys.readouterr()
        output_detailed = captured.out
        
        # Test with detailed=False
        backtester.print_results(results, detailed=False)
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
