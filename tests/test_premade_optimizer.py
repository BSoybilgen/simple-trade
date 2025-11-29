import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from simple_trade.optimize_premade_strategies import (
    premade_optimizer,
    _generate_parameter_combinations,
    _run_backtest_worker
)


# --- Fixtures ---

@pytest.fixture
def sample_ohlcv_data():
    """Fixture to provide sample OHLCV data with DatetimeIndex"""
    index = pd.date_range(start='2023-01-01', periods=50, freq='D')
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
def base_parameters():
    """Base parameters for optimization"""
    return {
        'initial_cash': 10000.0,
        'commission_long': 0.001,
        'commission_short': 0.001,
        'trading_type': 'long',
        'day1_position': 'none',
        'risk_free_rate': 0.02,
        'fig_control': 0
    }


@pytest.fixture
def simple_param_grid():
    """Simple parameter grid for testing"""
    return {
        'window': [10, 14],
        'upper': [70, 80],
        'lower': [20, 30]
    }


@pytest.fixture
def optimization_parameters():
    """Parameters for optimization including settings"""
    return {
        'initial_cash': 10000.0,
        'commission_long': 0.001,
        'commission_short': 0.001,
        'trading_type': 'long',
        'day1_position': 'none',
        'risk_free_rate': 0.02,
        'fig_control': 0,
        'metric': 'total_return_pct',
        'maximize': True,
        'parallel': False
    }


# --- Test Parameter Grid Generation ---

class TestParameterGridGeneration:
    """Test parameter grid generation functionality"""
    
    def test_generate_parameter_combinations_simple(self, simple_param_grid):
        """Test parameter combination generation with simple grid"""
        combinations = _generate_parameter_combinations(simple_param_grid)
        
        # Should generate 2 * 2 * 2 = 8 combinations
        assert len(combinations) == 8
        
        # Check that all combinations are dictionaries
        for combo in combinations:
            assert isinstance(combo, dict)
            assert 'window' in combo
            assert 'upper' in combo
            assert 'lower' in combo
            
    def test_generate_parameter_combinations_single_param(self):
        """Test parameter combination generation with single parameter"""
        param_grid = {'window': [10, 14, 20]}
        combinations = _generate_parameter_combinations(param_grid)
        
        assert len(combinations) == 3
        expected_combos = [
            {'window': 10},
            {'window': 14},
            {'window': 20}
        ]
        assert combinations == expected_combos
        
    def test_generate_parameter_combinations_empty_grid(self):
        """Test parameter combination generation with empty grid"""
        param_grid = {}
        combinations = _generate_parameter_combinations(param_grid)
        
        assert len(combinations) == 1
        assert combinations == [{}]
        
    def test_generate_parameter_combinations_single_values(self):
        """Test parameter combination generation with single values"""
        param_grid = {
            'window': [14],
            'upper': [70],
            'lower': [30]
        }
        combinations = _generate_parameter_combinations(param_grid)
        
        assert len(combinations) == 1
        assert combinations == [{'window': 14, 'upper': 70, 'lower': 30}]


# --- Test Backtest Worker ---

class TestBacktestWorker:
    """Test the backtest worker function"""
    
    @patch('simple_trade.optimize_premade_strategies.run_premade_trade')
    def test_run_backtest_worker_success(self, mock_premade_backtest, sample_ohlcv_data, base_parameters):
        """Test successful backtest worker execution"""
        # Mock successful backtest result
        mock_results = {
            'total_return_pct': 15.5,
            'sharpe_ratio': 1.2,
            'max_drawdown_pct': -5.0
        }
        mock_portfolio = pd.DataFrame({'PortfolioValue': [10000, 11000, 11550]})
        mock_premade_backtest.return_value = (mock_results, mock_portfolio, None)
        
        params = {'window': 14, 'upper': 70, 'lower': 30}
        result = _run_backtest_worker(
            params, sample_ohlcv_data, 'rsi', base_parameters, 'total_return_pct'
        )
        
        assert result['params'] == params
        assert result['score'] == 15.5
        assert result['results_df'] == mock_results
        
    @patch('simple_trade.optimize_premade_strategies.run_premade_trade')
    def test_run_backtest_worker_missing_metric(self, mock_premade_backtest, sample_ohlcv_data, base_parameters):
        """Test backtest worker with missing metric"""
        # Mock backtest result without the requested metric
        mock_results = {
            'sharpe_ratio': 1.2,
            'max_drawdown_pct': -5.0
        }
        mock_portfolio = pd.DataFrame({'PortfolioValue': [10000, 11000]})
        mock_premade_backtest.return_value = (mock_results, mock_portfolio, None)
        
        params = {'window': 14}
        
        # This should handle the KeyError gracefully
        try:
            result = _run_backtest_worker(
                params, sample_ohlcv_data, 'rsi', base_parameters, 'total_return_pct'
            )
            # If no error, check that it handled missing metric
            assert result['params'] == params
            assert result['score'] is None
            assert result['results_df'] is None
        except KeyError:
            # Expected behavior when metric is missing
            pass
        
    @patch('simple_trade.optimize_premade_strategies.run_premade_trade')
    def test_run_backtest_worker_parameter_combination(self, mock_premade_backtest, sample_ohlcv_data, base_parameters):
        """Test that parameters are correctly combined"""
        mock_results = {'total_return_pct': 10.0}
        mock_portfolio = pd.DataFrame({'PortfolioValue': [10000, 11000]})
        mock_premade_backtest.return_value = (mock_results, mock_portfolio, None)
        
        params = {'window': 20, 'upper': 80}
        _run_backtest_worker(
            params, sample_ohlcv_data, 'rsi', base_parameters, 'total_return_pct'
        )
        
        # Check that premade_backtest was called with combined parameters
        call_args = mock_premade_backtest.call_args
        combined_params = call_args[0][2]  # Third argument is parameters
        
        assert combined_params['window'] == 20
        assert combined_params['upper'] == 80
        assert combined_params['initial_cash'] == 10000.0
        assert combined_params['commission_long'] == 0.001


# --- Test Main Optimizer Function ---

class TestPremadeOptimizer:
    """Test the main premade_optimizer function"""
    
    @patch('simple_trade.optimize_premade_strategies._run_backtest_worker')
    def test_premade_optimizer_sequential(self, mock_worker, sample_ohlcv_data, optimization_parameters, simple_param_grid):
        """Test sequential optimization"""
        # Mock worker results
        mock_worker.side_effect = [
            {'params': {'window': 10, 'upper': 70, 'lower': 20}, 'score': 12.0, 'results_df': {'total_return_pct': 12.0}},
            {'params': {'window': 10, 'upper': 70, 'lower': 30}, 'score': 8.0, 'results_df': {'total_return_pct': 8.0}},
            {'params': {'window': 10, 'upper': 80, 'lower': 20}, 'score': 15.0, 'results_df': {'total_return_pct': 15.0}},
            {'params': {'window': 10, 'upper': 80, 'lower': 30}, 'score': 10.0, 'results_df': {'total_return_pct': 10.0}},
            {'params': {'window': 14, 'upper': 70, 'lower': 20}, 'score': 18.0, 'results_df': {'total_return_pct': 18.0}},
            {'params': {'window': 14, 'upper': 70, 'lower': 30}, 'score': 14.0, 'results_df': {'total_return_pct': 14.0}},
            {'params': {'window': 14, 'upper': 80, 'lower': 20}, 'score': 16.0, 'results_df': {'total_return_pct': 16.0}},
            {'params': {'window': 14, 'upper': 80, 'lower': 30}, 'score': 11.0, 'results_df': {'total_return_pct': 11.0}}
        ]
        
        best_results, best_params, all_results = premade_optimizer(
            sample_ohlcv_data, 'rsi', optimization_parameters, simple_param_grid
        )
        
        # Should find the best result (highest score)
        assert best_params == {'window': 14, 'upper': 70, 'lower': 20}
        assert best_results == {'total_return_pct': 18.0}
        assert len(all_results) == 8
        
        # Check that worker was called 8 times (2*2*2 combinations)
        assert mock_worker.call_count == 8
        
    @patch('simple_trade.optimize_premade_strategies._run_backtest_worker')
    def test_premade_optimizer_maximize_false(self, mock_worker, sample_ohlcv_data, optimization_parameters, simple_param_grid):
        """Test optimization with maximize=False"""
        optimization_parameters['maximize'] = False
        optimization_parameters['metric'] = 'max_drawdown_pct'
        
        # Use smaller grid for simplicity
        small_grid = {'window': [10, 14]}
        
        # Mock worker results (lower is better for drawdown) - need enough for all combinations
        mock_worker.side_effect = [
            {'params': {'window': 10}, 'score': -5.0, 'results_df': {'max_drawdown_pct': -5.0}},
            {'params': {'window': 14}, 'score': -2.0, 'results_df': {'max_drawdown_pct': -2.0}}
        ]
        
        best_results, best_params, all_results = premade_optimizer(
            sample_ohlcv_data, 'rsi', optimization_parameters, small_grid
        )
        
        # Should find the minimum (best) drawdown
        assert best_params == {'window': 10}
        assert best_results == {'max_drawdown_pct': -5.0}
        
    @patch('simple_trade.optimize_premade_strategies.Parallel')
    @patch('simple_trade.optimize_premade_strategies._run_backtest_worker')
    def test_premade_optimizer_parallel(self, mock_worker, mock_parallel, sample_ohlcv_data, optimization_parameters, simple_param_grid):
        """Test parallel optimization"""
        optimization_parameters['parallel'] = True
        optimization_parameters['n_jobs'] = 2
        
        # Mock parallel execution
        mock_parallel_instance = MagicMock()
        mock_parallel.return_value = mock_parallel_instance
        mock_parallel_instance.return_value = [
            {'params': {'window': 10, 'upper': 70, 'lower': 20}, 'score': 12.0, 'results_df': {'total_return_pct': 12.0}},
            {'params': {'window': 14, 'upper': 80, 'lower': 30}, 'score': 15.0, 'results_df': {'total_return_pct': 15.0}}
        ]
        
        best_results, best_params, all_results = premade_optimizer(
            sample_ohlcv_data, 'rsi', optimization_parameters, simple_param_grid
        )
        
        # Check that Parallel was called with correct parameters
        mock_parallel.assert_called_once_with(n_jobs=2, verbose=10)
        
        # Check results
        assert best_params == {'window': 14, 'upper': 80, 'lower': 30}
        assert best_results == {'total_return_pct': 15.0}
        
    def test_premade_optimizer_no_valid_results(self, sample_ohlcv_data, optimization_parameters, simple_param_grid):
        """Test optimizer when no valid results are generated"""
        with patch('simple_trade.optimize_premade_strategies._run_backtest_worker') as mock_worker:
            # Mock worker to return None scores
            mock_worker.return_value = {'params': {}, 'score': None, 'results_df': None}
            
            best_results, best_params, all_results = premade_optimizer(
                sample_ohlcv_data, 'rsi', optimization_parameters, simple_param_grid
            )
            
            assert best_results is None
            assert best_params is None
            assert all_results == []
            
    def test_premade_optimizer_default_parameters(self, sample_ohlcv_data, simple_param_grid):
        """Test optimizer with minimal parameters (using defaults)"""
        minimal_params = {
            'initial_cash': 10000.0,
            'fig_control': 0
        }
        
        with patch('simple_trade.optimize_premade_strategies._run_backtest_worker') as mock_worker:
            mock_worker.return_value = {
                'params': {'window': 14}, 
                'score': 10.0, 
                'results_df': {'total_return_pct': 10.0}
            }
            
            best_results, best_params, all_results = premade_optimizer(
                sample_ohlcv_data, 'rsi', minimal_params, {'window': [14]}
            )
            
            assert best_results is not None
            assert best_params == {'window': 14}


# --- Test Edge Cases and Error Handling ---

class TestPremadeOptimizerEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_parameter_grid(self, sample_ohlcv_data, optimization_parameters):
        """Test optimizer with empty parameter grid"""
        empty_grid = {}
        
        with patch('simple_trade.optimize_premade_strategies._run_backtest_worker') as mock_worker:
            mock_worker.return_value = {
                'params': {}, 
                'score': 5.0, 
                'results_df': {'total_return_pct': 5.0}
            }
            
            best_results, best_params, all_results = premade_optimizer(
                sample_ohlcv_data, 'rsi', optimization_parameters, empty_grid
            )
            
            assert best_params == {}
            assert len(all_results) == 1
            
    def test_single_parameter_combination(self, sample_ohlcv_data, optimization_parameters):
        """Test optimizer with single parameter combination"""
        single_grid = {'window': [14]}
        
        with patch('simple_trade.optimize_premade_strategies._run_backtest_worker') as mock_worker:
            mock_worker.return_value = {
                'params': {'window': 14}, 
                'score': 8.0, 
                'results_df': {'total_return_pct': 8.0}
            }
            
            best_results, best_params, all_results = premade_optimizer(
                sample_ohlcv_data, 'rsi', optimization_parameters, single_grid
            )
            
            assert best_params == {'window': 14}
            assert len(all_results) == 1
            
    def test_mixed_valid_invalid_results(self, sample_ohlcv_data, optimization_parameters):
        """Test optimizer with mix of valid and invalid results"""
        param_grid = {'window': [10, 14, 20]}
        
        with patch('simple_trade.optimize_premade_strategies._run_backtest_worker') as mock_worker:
            mock_worker.side_effect = [
                {'params': {'window': 10}, 'score': 12.0, 'results_df': {'total_return_pct': 12.0}},
                {'params': {'window': 14}, 'score': None, 'results_df': None},  # Invalid
                {'params': {'window': 20}, 'score': 8.0, 'results_df': {'total_return_pct': 8.0}}
            ]
            
            best_results, best_params, all_results = premade_optimizer(
                sample_ohlcv_data, 'rsi', optimization_parameters, param_grid
            )
            
            # Should only include valid results
            assert len(all_results) == 2
            assert best_params == {'window': 10}  # Higher score
            assert best_results == {'total_return_pct': 12.0}


# --- Test Integration ---

class TestPremadeOptimizerIntegration:
    """Integration tests for premade_optimizer"""
    
    def test_optimizer_with_real_backtest(self, sample_ohlcv_data):
        """Test optimizer with actual premade_backtest (not mocked)"""
        # Use very small parameter grid to keep test fast
        small_grid = {'window': [10, 14]}
        params = {
            'initial_cash': 10000.0,
            'commission_long': 0.001,
            'commission_short': 0.001,
            'trading_type': 'long',
            'fig_control': 0,
            'metric': 'total_return_pct',
            'maximize': True,
            'parallel': False
        }
        
        best_results, best_params, all_results = premade_optimizer(
            sample_ohlcv_data, 'rsi', params, small_grid
        )
        
        # Should get some results (even if not great with small dataset)
        if best_results is not None:
            assert isinstance(best_results, dict)
            assert 'window' in best_params
            assert best_params['window'] in [10, 14]
            assert len(all_results) <= 2  # At most 2 valid results
            
    def test_parameter_extraction(self, sample_ohlcv_data, simple_param_grid):
        """Test that optimization parameters are correctly extracted"""
        params = {
            'initial_cash': 50000.0,
            'commission_long': 0.002,
            'metric': 'sharpe_ratio',
            'maximize': False,
            'parallel': False,  # Use sequential to avoid parallel execution issues
            'n_jobs': 4,
            'extra_param': 'should_be_in_base'
        }
        
        with patch('simple_trade.optimize_premade_strategies._run_backtest_worker') as mock_worker:
            mock_worker.return_value = {
                'params': {'window': 14}, 
                'score': 1.5, 
                'results_df': {'sharpe_ratio': 1.5}
            }
            
            premade_optimizer(sample_ohlcv_data, 'rsi', params, {'window': [14]})
            
            # Check that worker was called and parameters were passed correctly
            assert mock_worker.called
            call_args = mock_worker.call_args[0]
            base_params = call_args[3]  # Fourth argument is base_parameters
            
            assert base_params['initial_cash'] == 50000.0
            assert base_params['commission_long'] == 0.002
            assert base_params['extra_param'] == 'should_be_in_base'
            # Optimization settings should not be in base parameters
            assert 'metric' not in base_params
            assert 'maximize' not in base_params
            assert 'parallel' not in base_params
            assert 'n_jobs' not in base_params
