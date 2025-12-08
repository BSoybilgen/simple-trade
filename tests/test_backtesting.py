import pytest
import pandas as pd
import numpy as np
from simple_trade.metrics import compute_benchmark_return, calculate_performance_metrics
from simple_trade.config import BacktestConfig

# --- Fixtures ---

@pytest.fixture
def default_config():
    """Provides a default BacktestConfig instance"""
    return BacktestConfig(initial_cash=10000.0, commission_long=0.001, commission_short=0.001)

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
    # Add some commissions (cumulative, as stored by the backtester)
    commission_per_trade = pd.Series(np.random.choice([0, 5, 10], size=len(index), p=[0.9, 0.05, 0.05]), index=index)
    commissions_cumulative = commission_per_trade.cumsum()
    
    df = pd.DataFrame({
        'PortfolioValue': portfolio_value,
        'CommissionPaid': commissions_cumulative
    })
    return df

# --- Test Class ---

class TestMetrics:
    """Tests for the metrics functions"""

    def test_config_initialization(self):
        """Test BacktestConfig initialization with default and custom values"""
        config_default = BacktestConfig()
        assert config_default.initial_cash == 10000.0
        assert config_default.commission_long == 0.001
        assert config_default.commission_short == 0.001
        assert config_default.short_borrow_fee_inc_rate == 0.0
        assert config_default.long_borrow_fee_inc_rate == 0.0
        
        config_custom = BacktestConfig(initial_cash=5000, commission_long=0.002, commission_short=0.002, short_borrow_fee_inc_rate=0.01, long_borrow_fee_inc_rate=0.005)
        assert config_custom.initial_cash == 5000
        assert config_custom.commission_long == 0.002
        assert config_custom.commission_short == 0.002
        assert config_custom.short_borrow_fee_inc_rate == 0.01
        assert config_custom.long_borrow_fee_inc_rate == 0.005

    def test_compute_benchmark_return(self, default_config, sample_ohlcv_data):
        """Test the benchmark calculation"""
        results = compute_benchmark_return(
            sample_ohlcv_data, 
            initial_cash=default_config.initial_cash,
            commission_long=default_config.commission_long,
            price_col='Close'
        )
        
        assert isinstance(results, dict)
        assert 'benchmark_final_value' in results
        assert 'benchmark_return_pct' in results
        assert 'benchmark_shares' in results
        
        # Check logic: final value should be approx shares * last_price
        first_price = sample_ohlcv_data['Close'].iloc[0]
        last_price = sample_ohlcv_data['Close'].iloc[-1]
        expected_shares = default_config.initial_cash / (first_price * (1 + default_config.commission_long))
        expected_final_value = expected_shares * last_price
        
        assert results['benchmark_shares'] == pytest.approx(expected_shares)
        assert results['benchmark_final_value'] == pytest.approx(expected_final_value, abs=0.01)
        assert results['benchmark_return_pct'] == pytest.approx(((expected_final_value / default_config.initial_cash) - 1) * 100, abs=0.01)

    def test_benchmark_input_validation(self, default_config, sample_ohlcv_data):
        """Test input validation for compute_benchmark_return"""
        # Test wrong index type
        data_wrong_index = sample_ohlcv_data.reset_index()
        with pytest.raises(TypeError, match="DatetimeIndex"):
            compute_benchmark_return(data_wrong_index, initial_cash=default_config.initial_cash, commission_long=default_config.commission_long)
            
        # Test missing price column
        with pytest.raises(ValueError, match="Price column 'NonExistentCol' not found"):
            compute_benchmark_return(sample_ohlcv_data, initial_cash=default_config.initial_cash, commission_long=default_config.commission_long, price_col='NonExistentCol')

    def test_calculate_performance_metrics(self, default_config, sample_portfolio_data):
        """Test the performance metrics calculation"""
        metrics = calculate_performance_metrics(
            sample_portfolio_data
        )
        
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
        assert metrics['total_commissions'] == sample_portfolio_data['CommissionPaid'].iloc[-1]

    def test_performance_metrics_input_validation(self, default_config, sample_portfolio_data):
        """Test input validation for calculate_performance_metrics"""
        # Test missing portfolio value column
        data_missing_col = sample_portfolio_data.drop(columns=['PortfolioValue'])
        with pytest.raises(ValueError, match="must contain a 'PortfolioValue' column"):
            calculate_performance_metrics(data_missing_col)
            
    def test_performance_metrics_edge_cases(self, default_config):
        """Test performance metrics with edge case data"""
        # Case 1: Flat portfolio value (zero return, zero volatility)
        index = pd.date_range(start='2023-01-01', periods=50, freq='D')
        flat_data = pd.DataFrame({'PortfolioValue': 10000.0}, index=index)
        metrics_flat = calculate_performance_metrics(flat_data)
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
        metrics_growth = calculate_performance_metrics(growth_data)
        assert metrics_growth['max_drawdown_pct'] == 0.0
        assert metrics_growth['max_drawdown_duration_days'] == 0
        assert metrics_growth['avg_drawdown_duration_days'] == 0.0
        # Calmar should be inf if no drawdown
        assert np.isinf(metrics_growth['calmar_ratio']) 