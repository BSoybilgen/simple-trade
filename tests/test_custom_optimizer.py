import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from simple_trade.optimize_custom_strategies import custom_optimizer
from simple_trade.config import BacktestConfig

# --- Fixtures ---

@pytest.fixture
def sample_opt_data():
    """Creates sample DataFrame for optimizer tests."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = pd.DataFrame(
        {
            'Close': np.random.lognormal(mean=0.001, sigma=0.02, size=100).cumprod() * 100,
            'Indicator1': np.random.rand(100) * 10,
            'Indicator2': np.random.rand(100) * 5
        },
        index=dates
    )
    return data

@pytest.fixture
def default_config():
    """Returns a default BacktestConfig."""
    return BacktestConfig(initial_cash=10000)

@pytest.fixture
def sample_param_grid():
    """Returns a sample parameter grid for optimization."""
    return {
        'short_window_indicator': ['Indicator1'],
        'long_window_indicator': ['Indicator2'],
        'long_entry_pct_cash': [0.8, 0.9]
    }

@pytest.fixture
def sample_constant_params():
    """Returns sample constant parameters."""
    return {
        'price_col': 'Close',
        'trading_type': 'long'
    }

# --- Test Class ---

class TestOptimize:
    """Tests for the optimize function."""

    def test_optimize_invalid_backtest_func(self, sample_opt_data, sample_param_grid):
        """Test that TypeError is raised for non-callable backtest_func."""
        with pytest.raises(TypeError, match="backtest_func must be a callable"):
            custom_optimizer(
                backtest_func="not_a_function",
                data=sample_opt_data,
                param_grid=sample_param_grid,
                metric_to_optimize='total_return_pct'
            )

    def test_optimize_serial_maximize(self, sample_opt_data, sample_param_grid, sample_constant_params):
        """Test optimize function in serial mode aiming to maximize."""
        # Mock the backtest function
        mock_backtest = MagicMock()

        # Define the side effects (results for each parameter combo)
        mock_backtest.side_effect = [
            ({'total_return_pct': 10.5, 'other_metric': 1}, pd.DataFrame({'Value': [100, 110.5]})),
            ({'total_return_pct': 15.2, 'other_metric': 2}, pd.DataFrame({'Value': [100, 115.2]}))
        ]

        # Run optimization
        best_params, best_metric, all_results = custom_optimizer(
            backtest_func=mock_backtest,
            data=sample_opt_data,
            param_grid=sample_param_grid,
            metric_to_optimize='total_return_pct',
            constant_params=sample_constant_params,
            maximize_metric=True,
            parallel=False
        )

        # Check best result
        assert best_params is not None
        assert best_metric == 15.2
        assert mock_backtest.call_count == 2
        assert len(all_results) == 2

    def test_optimize_serial_minimize(self, sample_opt_data, sample_param_grid, sample_constant_params):
        """Test optimize function in serial mode aiming to minimize."""
        metric_to_minimize = 'max_drawdown_pct'
        
        # Mock the backtest function
        mock_backtest = MagicMock()
        mock_backtest.side_effect = [
            ({metric_to_minimize: -5.5, 'other': 1}, pd.DataFrame({'Value': [100, 94.5]})),
            ({metric_to_minimize: -8.2, 'other': 2}, pd.DataFrame({'Value': [100, 91.8]}))
        ]

        # Run optimization
        best_params, best_metric, all_results = custom_optimizer(
            backtest_func=mock_backtest,
            data=sample_opt_data,
            param_grid=sample_param_grid,
            metric_to_optimize=metric_to_minimize,
            constant_params=sample_constant_params,
            maximize_metric=False,
            parallel=False
        )

        # Check best result (minimizing, so -8.2 is "better" than -5.5)
        assert best_params is not None
        assert best_metric == -8.2
        assert mock_backtest.call_count == 2
        assert len(all_results) == 2

    def test_optimize_parallel_execution(self, sample_opt_data, sample_param_grid, sample_constant_params):
        """Test the parallel execution path of the optimize function."""
        def mock_backtest_func(data, **kwargs):
            # Return different results based on parameters
            pct_cash = kwargs.get('long_entry_pct_cash', 0.8)
            if pct_cash == 0.8:
                return {'total_return_pct': 10.5}, pd.DataFrame({'Value': [100]})
            else:
                return {'total_return_pct': 15.2}, pd.DataFrame({'Value': [100]})
        
        # Run optimization with parallel=True
        best_params, best_metric, all_results = custom_optimizer(
            backtest_func=mock_backtest_func,
            data=sample_opt_data,
            param_grid=sample_param_grid,
            metric_to_optimize='total_return_pct',
            constant_params=sample_constant_params,
            maximize_metric=True,
            parallel=True,
            n_jobs=1
        )

        # Verify results
        assert best_params is not None
        assert best_metric == 15.2
        assert len(all_results) == 2

    def test_optimize_returns_none_when_no_valid_results(self, sample_opt_data, sample_param_grid):
        """Test that optimize returns None for best_params when no valid results."""
        # Mock the backtest function to raise exceptions
        mock_backtest = MagicMock()
        mock_backtest.side_effect = Exception("Backtest failed")

        # Run optimization
        best_params, best_metric, all_results = custom_optimizer(
            backtest_func=mock_backtest,
            data=sample_opt_data,
            param_grid=sample_param_grid,
            metric_to_optimize='total_return_pct',
            parallel=False
        )

        # Check that no valid results were found (best_params is None)
        # Note: all_results may still contain entries with -inf metric values
        assert best_params is None