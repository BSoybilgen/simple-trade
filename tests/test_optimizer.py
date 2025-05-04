import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from simple_trade.optimizer import Optimizer
from simple_trade.cross_trade import CrossTradeBacktester # Using CrossTrade for example

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
def cross_backtester():
    """Returns an instance of CrossTradeBacktester."""
    return CrossTradeBacktester(initial_cash=10000)

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

class TestOptimizer:
    """Tests for the Optimizer class."""

    def test_optimizer_initialization(self, sample_opt_data, cross_backtester, sample_param_grid, sample_constant_params):
        """Test successful initialization of Optimizer."""
        optimizer = Optimizer(
            backtest_func=cross_backtester.run_cross_trade,
            data=sample_opt_data,
            param_grid=sample_param_grid,
            metric_to_optimize='total_return_pct',
            constant_params=sample_constant_params,
            maximize_metric=True
        )
        assert optimizer.backtest_func == cross_backtester.run_cross_trade
        assert optimizer.data.equals(sample_opt_data)
        assert optimizer.param_grid == sample_param_grid
        assert optimizer.metric_to_optimize == 'total_return_pct'
        assert optimizer.constant_params == sample_constant_params
        assert optimizer.maximize_metric is True
        assert optimizer.best_metric_value == -np.inf

    def test_optimizer_initialization_minimize(self, sample_opt_data, cross_backtester, sample_param_grid):
        """Test initialization when minimizing the metric."""
        optimizer = Optimizer(
            backtest_func=cross_backtester.run_cross_trade,
            data=sample_opt_data,
            param_grid=sample_param_grid,
            metric_to_optimize='some_risk_metric',
            maximize_metric=False # Minimize
        )
        assert optimizer.maximize_metric is False
        assert optimizer.best_metric_value == np.inf

    def test_optimizer_initialization_invalid_backtest_func(self, sample_opt_data, sample_param_grid):
        """Test initialization fails with non-callable backtest_func."""
        with pytest.raises(TypeError, match="backtest_func must be a callable method"):
            Optimizer(
                backtest_func="not_a_function",
                data=sample_opt_data,
                param_grid=sample_param_grid,
                metric_to_optimize='total_return_pct'
            )

    def test_generate_parameter_combinations(self, sample_opt_data, cross_backtester, sample_param_grid):
        """Test the generation of parameter combinations."""
        optimizer = Optimizer(
            backtest_func=cross_backtester.run_cross_trade,
            data=sample_opt_data,
            param_grid=sample_param_grid,
            metric_to_optimize='total_return_pct'
        )
        expected_combinations = [
            {'short_window_indicator': 'Indicator1', 'long_window_indicator': 'Indicator2', 'long_entry_pct_cash': 0.8},
            {'short_window_indicator': 'Indicator1', 'long_window_indicator': 'Indicator2', 'long_entry_pct_cash': 0.9}
        ]
        # Convert to sets of tuples for order-independent comparison
        expected_set = set(tuple(sorted(d.items())) for d in expected_combinations)
        actual_set = set(tuple(sorted(d.items())) for d in optimizer.parameter_combinations)
        assert actual_set == expected_set
        assert len(optimizer.parameter_combinations) == 2

    def test_optimize_serial_maximize(self, sample_opt_data, cross_backtester, sample_param_grid, sample_constant_params):
        """Test optimize method in serial mode aiming to maximize."""
        optimizer = Optimizer(
            backtest_func=cross_backtester.run_cross_trade,
            data=sample_opt_data,
            param_grid=sample_param_grid,
            metric_to_optimize='total_return_pct', # Metric to maximize
            constant_params=sample_constant_params,
            maximize_metric=True
        )

        # Mock the backtest function
        mock_backtest = MagicMock()

        # Define the side effects (results for each parameter combo)
        # Combo 1: {'short_window_indicator': 'Indicator1', 'long_window_indicator': 'Indicator2', 'long_entry_pct_cash': 0.8}
        # Combo 2: {'short_window_indicator': 'Indicator1', 'long_window_indicator': 'Indicator2', 'long_entry_pct_cash': 0.9}
        # Let Combo 2 have the higher 'total_return_pct'
        mock_backtest.side_effect = [
            ({'total_return_pct': 10.5, 'other_metric': 1}, pd.DataFrame({'Value': [100, 110.5]})), # Result for Combo 1
            ({'total_return_pct': 15.2, 'other_metric': 2}, pd.DataFrame({'Value': [100, 115.2]}))  # Result for Combo 2
        ]
        optimizer.backtest_func = mock_backtest

        # Run optimization
        best_params, best_metric, all_results = optimizer.optimize(parallel=False)

        # --- Assertions ---
        # Check best result
        expected_best_params = {'short_window_indicator': 'Indicator1', 'long_window_indicator': 'Indicator2', 'long_entry_pct_cash': 0.9}
        assert best_params == expected_best_params
        assert best_metric == 15.2
        assert optimizer.best_params == expected_best_params
        assert optimizer.best_metric_value == 15.2

        # Check number of calls to the mock
        assert mock_backtest.call_count == 2

        # Check arguments passed to the mock (order might vary based on dict creation)
        call_args_list = mock_backtest.call_args_list
        # Extract the keyword arguments from each call
        called_params = []
        for call in call_args_list:
            args, kwargs = call
            # kwargs contains the merged variable + constant params AND the 'data' kwarg
            params_only = kwargs.copy() # Create a copy to modify
            params_only.pop('data', None) # Remove the data DataFrame before hashing
            called_params.append(params_only)

        # Define expected parameters passed (merging variable and constant)
        expected_call_params = [
            {'price_col': 'Close', 'trading_type': 'long', 'short_window_indicator': 'Indicator1', 'long_window_indicator': 'Indicator2', 'long_entry_pct_cash': 0.8},
            {'price_col': 'Close', 'trading_type': 'long', 'short_window_indicator': 'Indicator1', 'long_window_indicator': 'Indicator2', 'long_entry_pct_cash': 0.9}
        ]

        # Compare sets of sorted items for order independence
        called_params_set = set(tuple(sorted(p.items())) for p in called_params)
        expected_call_params_set = set(tuple(sorted(p.items())) for p in expected_call_params)
        assert called_params_set == expected_call_params_set

        # Check the structure and content of optimizer.results
        assert len(optimizer.results) == 2
        # optimizer.results contains tuples: (params_dict, metric_value)
        result_params_set = set(tuple(sorted(r[0].items())) for r in optimizer.results) # Use r[0] for params dict
        expected_params_set = set(tuple(sorted(p.items())) for p in optimizer.parameter_combinations)
        assert result_params_set == expected_params_set
        assert optimizer.results[0][1] == 10.5 or optimizer.results[1][1] == 10.5 # Use r[1] for metric
        assert optimizer.results[0][1] == 15.2 or optimizer.results[1][1] == 15.2 # Use r[1] for metric

    def test_optimize_serial_minimize(self, sample_opt_data, cross_backtester, sample_param_grid, sample_constant_params):
        """Test optimize method in serial mode aiming to minimize."""
        metric_to_minimize = 'max_drawdown_pct' # Example metric to minimize
        optimizer = Optimizer(
            backtest_func=cross_backtester.run_cross_trade,
            data=sample_opt_data,
            param_grid=sample_param_grid,
            metric_to_optimize=metric_to_minimize,
            constant_params=sample_constant_params,
            maximize_metric=False # Minimize!
        )

        # Mock the backtest function
        mock_backtest = MagicMock()

        # Define the side effects
        # Combo 1: {'short_window_indicator': 'Indicator1', 'long_window_indicator': 'Indicator2', 'long_entry_pct_cash': 0.8}
        # Combo 2: {'short_window_indicator': 'Indicator1', 'long_window_indicator': 'Indicator2', 'long_entry_pct_cash': 0.9}
        # Let Combo 2 have the lower (better) 'max_drawdown_pct'
        mock_backtest.side_effect = [
            ({metric_to_minimize: -5.5, 'other': 1}, pd.DataFrame({'Value': [100, 94.5]})), # Result for Combo 1
            ({metric_to_minimize: -8.2, 'other': 2}, pd.DataFrame({'Value': [100, 91.8]}))  # Result for Combo 2
        ]
        optimizer.backtest_func = mock_backtest

        # Run optimization
        best_params, best_metric, all_results = optimizer.optimize(parallel=False)

        # --- Assertions ---
        # Check best result (Combo 2 has the numerically smallest metric: -8.2)
        expected_best_params = {'short_window_indicator': 'Indicator1', 'long_window_indicator': 'Indicator2', 'long_entry_pct_cash': 0.9}
        assert best_params == expected_best_params
        assert best_metric == -8.2
        assert optimizer.best_params == expected_best_params
        assert optimizer.best_metric_value == -8.2

        # Check number of calls
        assert mock_backtest.call_count == 2

        # Check results structure
        assert len(optimizer.results) == 2
        assert optimizer.results[0][1] == -5.5 or optimizer.results[1][1] == -5.5
        assert optimizer.results[0][1] == -8.2 or optimizer.results[1][1] == -8.2

    # --- TODO: Add more tests for Optimizer ---
    # - test_optimize_parallel_maximize
    # - test_optimize_parallel_minimize
    # - test_optimize_metric_not_found
    # - test_optimize_backtest_exception_handling
    # - test_optimize_empty_grid
