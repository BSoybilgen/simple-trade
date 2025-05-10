import pytest
import pandas as pd
import numpy as np
from simple_trade.backtesting import Backtester

# --- Fixtures ---

@pytest.fixture
def backtester_instance():
    """Provides a default Backtester instance"""
    return Backtester(initial_cash=10000.0, commission_long=0.001, commission_short=0.001)

@pytest.fixture
def sample_ohlcv_data():
    """Fixture to provide sample OHLCV data with DatetimeIndex"""
    index = pd.date_range(start='2023-01-01', periods=50, freq='D') # Shorter for simplicity
    np.random.seed(42)
    close = pd.Series(np.linspace(100, 150, 50) + np.random.normal(0, 2, 50), index=index)
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
def sample_portfolio_data():
    """Fixture providing sample portfolio simulation results"""
    index = pd.date_range(start='2023-01-01', periods=50, freq='D')
    # Simulate portfolio value: Start at 10k, grow, dip, grow again
    base_growth = np.linspace(10000, 12000, 30)
    dip = np.linspace(12000, 11000, 10)
    recovery = np.linspace(11000, 13000, 10)
    portfolio_value = pd.Series(np.concatenate([base_growth, dip, recovery]), index=index)
    # Add some commissions
    commissions = pd.Series(np.random.choice([0, 5, 10], size=len(index), p=[0.9, 0.05, 0.05]), index=index)
    
    df = pd.DataFrame({
        'PortfolioValue': portfolio_value,
        'CommissionPaid': commissions
    })
    return df

# --- Test Class ---

class TestBacktester:
    """Tests for the Backtester class"""

    def test_initialization(self):
        """Test Backtester initialization with default and custom values"""
        bt_default = Backtester()
        assert bt_default.initial_cash == 10000.0
        assert bt_default.commission_long == 0.001
        assert bt_default.commission_short == 0.001
        assert bt_default.short_borrow_fee_inc_rate == 0.0
        assert bt_default.long_borrow_fee_inc_rate == 0.0
        
        bt_custom = Backtester(initial_cash=5000, commission_long=0.002, commission_short=0.002, short_borrow_fee_inc_rate=0.01, long_borrow_fee_inc_rate=0.005)
        assert bt_custom.initial_cash == 5000
        assert bt_custom.commission_long == 0.002
        assert bt_custom.commission_short == 0.002
        assert bt_custom.short_borrow_fee_inc_rate == 0.01
        assert bt_custom.long_borrow_fee_inc_rate == 0.005

    def test_compute_benchmark_return(self, backtester_instance, sample_ohlcv_data):
        """Test the benchmark calculation"""
        results = backtester_instance.compute_benchmark_return(sample_ohlcv_data, price_col='Close')
        
        assert isinstance(results, dict)
        assert 'benchmark_final_value' in results
        assert 'benchmark_return_pct' in results
        assert 'benchmark_shares' in results
        
        # Check logic: final value should be approx shares * last_price
        first_price = sample_ohlcv_data['Close'].iloc[0]
        last_price = sample_ohlcv_data['Close'].iloc[-1]
        expected_shares = backtester_instance.initial_cash / (first_price * (1 + backtester_instance.commission_long))
        expected_final_value = expected_shares * last_price
        
        assert results['benchmark_shares'] == pytest.approx(expected_shares)
        assert results['benchmark_final_value'] == pytest.approx(expected_final_value, abs=0.01)
        assert results['benchmark_return_pct'] == pytest.approx(((expected_final_value / backtester_instance.initial_cash) - 1) * 100, abs=0.01)

    def test_benchmark_input_validation(self, backtester_instance, sample_ohlcv_data):
        """Test input validation for compute_benchmark_return"""
        # Test wrong index type
        data_wrong_index = sample_ohlcv_data.reset_index()
        with pytest.raises(TypeError, match="DatetimeIndex"):
            backtester_instance.compute_benchmark_return(data_wrong_index)
            
        # Test missing price column
        with pytest.raises(ValueError, match="Price column 'NonExistentCol' not found"):
            backtester_instance.compute_benchmark_return(sample_ohlcv_data, price_col='NonExistentCol')

    def test_calculate_performance_metrics(self, backtester_instance, sample_portfolio_data):
        """Test the performance metrics calculation"""
        metrics = backtester_instance.calculate_performance_metrics(sample_portfolio_data)
        
        assert isinstance(metrics, dict)
        # Check existence of key metrics
        expected_keys = [
            "total_return_pct", "annualized_return_pct", "annualized_volatility_pct",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown_pct",
            "max_drawdown_duration_days", "avg_drawdown_duration_days", "total_commissions"
        ]
        assert all(key in metrics for key in expected_keys)
        
        # Basic sanity checks
        assert metrics['total_return_pct'] > 0 # Should be positive in sample data
        assert metrics['max_drawdown_pct'] < 0 # Drawdown should be negative
        assert metrics['sharpe_ratio'] != np.inf and not np.isnan(metrics['sharpe_ratio']) 
        assert metrics['sortino_ratio'] != np.inf and not np.isnan(metrics['sortino_ratio']) 
        assert metrics['total_commissions'] == sample_portfolio_data['CommissionPaid'].sum()

    def test_performance_metrics_input_validation(self, backtester_instance, sample_portfolio_data):
        """Test input validation for calculate_performance_metrics"""
        # Test missing portfolio value column
        data_missing_col = sample_portfolio_data.drop(columns=['PortfolioValue'])
        with pytest.raises(ValueError, match="must contain a 'PortfolioValue' column"):
            backtester_instance.calculate_performance_metrics(data_missing_col)
            
    def test_performance_metrics_edge_cases(self, backtester_instance):
        """Test performance metrics with edge case data"""
        # Case 1: Flat portfolio value (zero return, zero volatility)
        index = pd.date_range(start='2023-01-01', periods=50, freq='D')
        flat_data = pd.DataFrame({'PortfolioValue': 10000.0}, index=index)
        metrics_flat = backtester_instance.calculate_performance_metrics(flat_data)
        assert metrics_flat['total_return_pct'] == 0.0
        assert metrics_flat['annualized_return_pct'] == 0.0
        assert metrics_flat['annualized_volatility_pct'] == 0.0
        assert metrics_flat['max_drawdown_pct'] == 0.0
        # Sharpe/Sortino/Calmar can be NaN or Inf with zero volatility/return
        assert np.isnan(metrics_flat['sharpe_ratio']) or np.isinf(metrics_flat['sharpe_ratio'])
        assert np.isinf(metrics_flat['sortino_ratio'])
        assert np.isinf(metrics_flat['calmar_ratio'])

        # Case 2: No drawdown
        steady_growth = np.linspace(10000, 15000, 50)
        growth_data = pd.DataFrame({'PortfolioValue': steady_growth}, index=index)
        metrics_growth = backtester_instance.calculate_performance_metrics(growth_data)
        assert metrics_growth['max_drawdown_pct'] == 0.0
        assert metrics_growth['max_drawdown_duration_days'] == 0
        assert metrics_growth['avg_drawdown_duration_days'] == 0.0
        # Calmar should be inf if no drawdown
        assert np.isinf(metrics_growth['calmar_ratio']) 