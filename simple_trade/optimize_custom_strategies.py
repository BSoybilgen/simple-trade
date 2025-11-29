"""
Optimizer functions for trading strategy parameter optimization.

This module provides function-based implementations for optimizing trading
strategy parameters, replacing the class-based Optimizer approach.
"""
import itertools
import time
import os
from typing import Callable, Dict, List, Any, Tuple, Optional

import pandas as pd
import numpy as np
from joblib import Parallel, delayed


def custom_optimizer(
    backtest_func: Callable,
    data: pd.DataFrame,
    param_grid: Dict[str, List[Any]],
    metric_to_optimize: str,
    constant_params: Optional[Dict[str, Any]] = None,
    maximize_metric: bool = True,
    parallel: bool = True,
    n_jobs: int = -1,
) -> Tuple[Optional[Dict[str, Any]], float, List[Tuple[Dict[str, Any], float]]]:
    """
    Optimizes trading strategy parameters by iterating through combinations
    and evaluating performance based on a specified metric.

    Args:
        backtest_func: The backtesting function to optimize. Should accept
                      `data` as first argument and return (results_dict, portfolio_df).
        data: The historical data for backtesting.
        param_grid: Dictionary where keys are parameter names and values are
                   lists of values to test.
        metric_to_optimize: The key in the backtest results dictionary to optimize
                           (e.g., 'total_return_pct', 'sharpe_ratio').
        constant_params: Dictionary of parameters that remain constant during
                        optimization. These are passed to every backtest call.
        maximize_metric: Whether to maximize (True) or minimize (False) the metric.
        parallel: If True, run backtests in parallel using joblib.
        n_jobs: Number of CPU cores to use for parallel processing.
               -1 means using all available cores.

    Returns:
        tuple: A tuple containing:
            - best_params: Dictionary of the best parameters found, or None if no valid results.
            - best_metric_value: The best metric value achieved.
            - all_results: List of (params, metric_value) tuples for all combinations tested.

    Example:
        >>> param_grid = {
        ...     'short_window': [10, 20, 30],
        ...     'long_window': [50, 100, 150]
        ... }
        >>> best_params, best_value, all_results = optimize(
        ...     backtest_func=run_cross_trade,
        ...     data=data,
        ...     param_grid=param_grid,
        ...     metric_to_optimize='total_return_pct',
        ...     constant_params={'initial_cash': 10000},
        ...     maximize_metric=True,
        ...     parallel=True
        ... )
    """
    if not callable(backtest_func):
        raise TypeError("backtest_func must be a callable")
    
    if constant_params is None:
        constant_params = {}
    
    # Generate all parameter combinations
    parameter_combinations = _generate_parameter_combinations(param_grid)
    num_combinations = len(parameter_combinations)
    
    start_time = time.time()
    print(f"Starting optimization for {num_combinations} combinations...")
    print(f"Metric: {metric_to_optimize} ({'Maximize' if maximize_metric else 'Minimize'}) | Parallel: {parallel}{f' (n_jobs={n_jobs})' if parallel else ''}")

    if parallel:
        max_cores = os.cpu_count() or 1
        if n_jobs == -1 or n_jobs > max_cores:
            n_jobs = max_cores
        print(f"Using {n_jobs} parallel jobs.")

        results_list = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_run_single_backtest)(
                params=param_combo,
                backtest_func=backtest_func,
                data=data,
                metric_to_optimize=metric_to_optimize,
                constant_params=constant_params
            )
            for param_combo in parameter_combinations
        )
    else:
        results_list = []
        for i, param_combo in enumerate(parameter_combinations):
            if (i + 1) % 10 == 0:
                print(f"Processing combination {i+1}/{num_combinations}...")
            params, metric_value = _run_single_backtest(
                params=param_combo,
                backtest_func=backtest_func,
                data=data,
                metric_to_optimize=metric_to_optimize,
                constant_params=constant_params
            )
            results_list.append((params, metric_value))

    # Find the best result
    best_params = None
    best_metric_value = -np.inf if maximize_metric else np.inf
    
    for params, metric_value in results_list:
        is_better = False
        if maximize_metric:
            if metric_value > best_metric_value:
                is_better = True
        else:
            if metric_value < best_metric_value:
                is_better = True
        
        if is_better:
            best_metric_value = metric_value
            best_params = params

    end_time = time.time()
    print(f"Optimization finished in {end_time - start_time:.2f} seconds.")
    
    if best_params:
        print(f"Best Parameters found: {best_params}")
        print(f"Best Metric Value ({metric_to_optimize}): {best_metric_value:.4f}")
    else:
        print("No valid results found during optimization.")
        return None, best_metric_value, results_list
        
    return best_params, best_metric_value, results_list


def _generate_parameter_combinations(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generates all possible parameter combinations from the grid."""
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    param_dicts = [dict(zip(keys, combo)) for combo in combinations]
    print(f"Generated {len(param_dicts)} parameter combinations.")
    return param_dicts


def _run_single_backtest(
    params: Dict[str, Any],
    backtest_func: Callable,
    data: pd.DataFrame,
    metric_to_optimize: str,
    constant_params: Dict[str, Any]
) -> Tuple[Dict[str, Any], float]:
    """
    Worker function to run a single backtest instance for optimization.

    Args:
        params: Dictionary of parameters specific to this run.
        backtest_func: The function to call for backtesting.
        data: The input data for the backtest.
        metric_to_optimize: The key in the results dictionary to use as the optimization metric.
        constant_params: Dictionary of parameters constant across all runs.

    Returns:
        Tuple containing the parameter dictionary and the resulting metric value.
        Returns -np.inf if the backtest fails or metric is not found.
    """
    current_params = {**constant_params, **params}
    
    try:
        # Call the backtest function
        result = backtest_func(data=data, **current_params)
        
        # Handle different return types
        if isinstance(result, tuple):
            results = result[0]  # First element should be the results dict
        else:
            results = result
        
        # Get the metric value
        metric_value = results.get(metric_to_optimize)
        if metric_value is None:
            print(f"Warning: Metric '{metric_to_optimize}' not found in results for params {params}. Returning -inf.")
            return params, -np.inf

        # Handle potential non-numeric or NaN metrics
        if not isinstance(metric_value, (int, float)) or np.isnan(metric_value):
            return params, -np.inf

        return params, float(metric_value)

    except Exception as e:
        print(f"Error during backtest with params {params}: {e}")
        return params, -np.inf


def get_top_results(
    all_results: List[Tuple[Dict[str, Any], float]],
    n: int = 10,
    maximize: bool = True
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Get the top N results from optimization.
    
    Args:
        all_results: List of (params, metric_value) tuples from optimization.
        n: Number of top results to return.
        maximize: If True, return highest values; if False, return lowest.
        
    Returns:
        List of top N (params, metric_value) tuples, sorted by metric value.
    """
    # Filter out invalid results
    valid_results = [(p, v) for p, v in all_results if v != -np.inf and v != np.inf and not np.isnan(v)]
    
    # Sort by metric value
    sorted_results = sorted(valid_results, key=lambda x: x[1], reverse=maximize)
    
    return sorted_results[:n]


def results_to_dataframe(
    all_results: List[Tuple[Dict[str, Any], float]],
    metric_name: str = 'metric_value'
) -> pd.DataFrame:
    """
    Convert optimization results to a pandas DataFrame.
    
    Args:
        all_results: List of (params, metric_value) tuples from optimization.
        metric_name: Name for the metric column in the DataFrame.
        
    Returns:
        DataFrame with one row per parameter combination, including the metric value.
    """
    rows = []
    for params, metric_value in all_results:
        row = params.copy()
        row[metric_name] = metric_value
        rows.append(row)
    
    return pd.DataFrame(rows)


