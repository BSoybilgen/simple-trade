#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Provides functionality to optimize trading strategy parameters.
"""

import itertools
import time
from typing import Callable, Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import os

from .backtesting import Backtester


def _run_single_backtest_job(
    params: Dict[str, Any],
    backtest_func: Callable, 
    data: pd.DataFrame,
    metric_to_optimize: str,
    constant_params: Dict[str, Any]
) -> Tuple[Dict[str, Any], float]:
    """
    Worker function to run a single backtest instance for optimization.

    Args:
        params (Dict[str, Any]): Dictionary of parameters specific to this run.
        backtest_func (Callable): The function to call for backtesting.
        data (pd.DataFrame): The input data for the backtest.
        metric_to_optimize (str): The key in the results dictionary to use as the optimization metric.
        constant_params (Dict[str, Any]): Dictionary of parameters constant across all runs.

    Returns:
        Tuple[Dict[str, Any], float]: A tuple containing the parameter dictionary and the resulting metric value.
                                       Returns -np.inf if the backtest fails or metric is not found.
    """
    current_params = {**constant_params, **params} # Combine constant and variable params
    try:
        # Check if backtest_func is a method of a class or a standalone function
        if hasattr(backtest_func, '__self__') and backtest_func.__self__ is not None:
            # This is a bound method (e.g., instance_of_backtester.run_cross_trade)
            results, _ = backtest_func(data=data, **current_params)
        elif hasattr(backtest_func, '__name__') and backtest_func.__name__ in ['run_cross_trade', 'run_band_trade']:
            # This is an unbound method (e.g., Backtester.run_cross_trade)
            # Separate Backtester init args from strategy args
            bt_init_args = {
                'initial_cash': current_params.pop('initial_cash', 10000.0),
                'commission': current_params.pop('commission', 0.001),
                'short_borrow_fee_inc_rate': current_params.pop('short_borrow_fee_inc_rate', 0.0),
                'long_borrow_fee_inc_rate': current_params.pop('long_borrow_fee_inc_rate', 0.0),
            }

            # Instantiate backtester for this job
            backtester_class = backtest_func.__self__ if hasattr(backtest_func, '__self__') else Backtester
            backtester = backtester_class(**bt_init_args)

            # Get the actual backtesting method bound to the instance
            actual_backtest_method = getattr(backtester, backtest_func.__name__)
            
            # Call the backtest method
            results, _ = actual_backtest_method(data=data, **current_params)
        else:
            # This is a regular function (e.g., run_cross_trade_with_windows)
            results, _ = backtest_func(data=data, **current_params)
        
        # Get the metric value
        metric_value = results.get(metric_to_optimize)
        if metric_value is None:
            print(f"Warning: Metric '{metric_to_optimize}' not found in results for params {params}. Returning -inf.")
            return params, -np.inf # Use original params dict for reporting

        # Handle potential non-numeric or NaN metrics
        if not isinstance(metric_value, (int, float)) or np.isnan(metric_value):
             return params, -np.inf # Use original params dict for reporting

        return params, float(metric_value) # Use original params dict for reporting

    except Exception as e:
        # Log the error for debugging
        print(f"Error during backtest with params {params}: {e}")
        print(f"Current params passed to backtest: {current_params}")
        return params, -np.inf # Use original params dict for reporting


class Optimizer:
    """
    Optimizes trading strategy parameters by iterating through combinations 
    and evaluating performance based on a specified metric.
    """
    def __init__(self,
                 backtest_func: Callable, 
                 data: pd.DataFrame,
                 param_grid: Dict[str, List[Any]],
                 metric_to_optimize: str,
                 constant_params: Dict[str, Any] = None,
                 maximize_metric: bool = True):
        """
        Initializes the Optimizer.

        Args:
            backtest_func (Callable): The function to optimize.
            data (pd.DataFrame): The historical data for backtesting.
            param_grid (Dict[str, List[Any]]): Dictionary where keys are parameter names 
                                              and values are lists of values to test.
            metric_to_optimize (str): The key in the backtest results dictionary to optimize 
                                      (e.g., 'total_return_pct', 'sharpe_ratio').
            constant_params (Dict[str, Any], optional): Dictionary of parameters that remain 
                                                       constant during optimization. Defaults to {}.
            maximize_metric (bool, optional): Whether to maximize (True) or minimize (False) 
                                           the metric. Defaults to True.
        """
        if not callable(backtest_func):
            raise TypeError("backtest_func must be a callable method of the Backtester class")
            
        self.backtest_func = backtest_func
        self.data = data
        self.param_grid = param_grid
        self.metric_to_optimize = metric_to_optimize
        self.constant_params = constant_params if constant_params is not None else {}
        self.maximize_metric = maximize_metric
        
        self.parameter_combinations = self._generate_parameter_combinations()
        self.results = []
        self.best_params = None
        self.best_metric_value = -np.inf if maximize_metric else np.inf

    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generates all possible parameter combinations from the grid."""
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = list(itertools.product(*values))
        
        param_dicts = [dict(zip(keys, combo)) for combo in combinations]
        print(f"Generated {len(param_dicts)} parameter combinations.")
        return param_dicts

    def optimize(self, parallel: bool = True, n_jobs: int = -1) -> tuple:
        """
        Runs the optimization process.

        Args:
            parallel (bool, optional): If True, run backtests in parallel using joblib. 
                                     Defaults to True.
            n_jobs (int, optional): Number of CPU cores to use for parallel processing.
                                  -1 means using all available cores. Defaults to -1.
                                  Only used if parallel is True.
                                  
        Returns:
            tuple: A tuple containing (best_params, best_metric_value, all_results)
                  Returns None if no valid results were found.
        """
        start_time = time.time()
        num_combinations = len(self.parameter_combinations)
        print(f"Starting optimization for {num_combinations} combinations...")
        print(f"Metric: {self.metric_to_optimize} ({'Maximize' if self.maximize_metric else 'Minimize'}) | Parallel: {parallel}{f' (n_jobs={n_jobs})' if parallel else ''}")

        if parallel:
            # Ensure n_jobs is reasonable
            max_cores = os.cpu_count() or 1
            if n_jobs == -1 or n_jobs > max_cores:
                n_jobs = max_cores
            print(f"Using {n_jobs} parallel jobs.")

            # Run jobs in parallel
            # Pass only necessary data to each job
            results_list = Parallel(n_jobs=n_jobs, verbose=5)( # verbose=5 shows progress
                delayed(_run_single_backtest_job)(
                    params=param_combo,
                    backtest_func=self.backtest_func,
                    data=self.data, # Data might be large, consider alternatives if memory becomes an issue
                    metric_to_optimize=self.metric_to_optimize,
                    constant_params=self.constant_params
                )
                for param_combo in self.parameter_combinations
            )
            self.results = results_list
        else:
            # Run sequentially
            self.results = []
            for i, param_combo in enumerate(self.parameter_combinations):
                if (i + 1) % 10 == 0:
                    print(f"Processing combination {i+1}/{num_combinations}...")
                params, metric_value = _run_single_backtest_job(
                    params=param_combo,
                    backtest_func=self.backtest_func,
                    data=self.data,
                    metric_to_optimize=self.metric_to_optimize,
                    constant_params=self.constant_params
                )
                self.results.append((params, metric_value))

        # Find the best result
        for params, metric_value in self.results:
            is_better = False
            if self.maximize_metric:
                if metric_value > self.best_metric_value:
                    is_better = True
            else: # Minimize
                if metric_value < self.best_metric_value:
                    is_better = True
            
            if is_better:
                self.best_metric_value = metric_value
                self.best_params = params

        end_time = time.time()
        print(f"Optimization finished in {end_time - start_time:.2f} seconds.")
        if self.best_params:
            print(f"Best Parameters found: {self.best_params}")
            print(f"Best Metric Value ({self.metric_to_optimize}): {self.best_metric_value:.4f}")
        else:
            print("No valid results found during optimization.")
            return None
            
        # Return the optimization results as a tuple
        return self.best_params, self.best_metric_value, self.results

    def get_best_result(self) -> Tuple[Dict[str, Any], float]:
        """
        Returns the best parameters found and their corresponding metric value.

        Returns:
            Tuple[Dict[str, Any], float]: Best parameters and metric value, or (None, -inf/inf).
        """
        return self.best_params, self.best_metric_value

    def get_all_results(self) -> List[Tuple[Dict[str, Any], float]]:
        """
        Returns all parameter combinations tested and their corresponding metric values.

        Returns:
            List[Tuple[Dict[str, Any], float]]: List of (parameters, metric_value) tuples.
        """
        return self.results
