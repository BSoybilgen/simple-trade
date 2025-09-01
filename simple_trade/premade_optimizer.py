import pandas as pd
import itertools
from typing import Dict, List, Any
from joblib import Parallel, delayed

from .premade_backtest import premade_backtest

def _generate_parameter_combinations(param_grid) -> List[Dict[str, Any]]:
    """Generates all possible parameter combinations from the grid."""
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    param_dicts = [dict(zip(keys, combo)) for combo in combinations]
    print(f"Generated {len(param_dicts)} parameter combinations.")
    return param_dicts

def _run_backtest_worker(params: dict, data: pd.DataFrame, strategy_name: str, base_parameters: dict, metric: str) -> dict:
    """Helper function to run a single backtest instance for parallel processing."""
    # Combine the iteration-specific params with the base parameters
    current_run_params = {**base_parameters, **params}
    
    # Run the backtest. Use a copy of the data to avoid race conditions if it's modified in-place.
    results_df, _, _ = premade_backtest(data.copy(), strategy_name, current_run_params)
    
    # Safely get the score from the last row of the results DataFrame
    score = results_df[metric]
    
    if score is None:
        print(f"    Warning: Metric '{metric}' not found in results for params {params}. Skipping.")
        return {'params': params, 'score': None, 'results_df': None}

    return {'params': params, 'score': score, 'results_df': results_df}

def premade_optimizer(data: pd.DataFrame, strategy_name: str, parameters: dict, param_grid: dict):
    """
    Optimizes a trading strategy by searching through a grid of parameters, with optional parallel processing.

    Args:
        data (pd.DataFrame): The input data for backtesting.
        strategy_name (str): The name of the strategy to backtest.
        parameters (dict): A dictionary of base parameters, which can include:
                           'metric' (str): The metric to optimize (e.g., 'Sharpe Ratio').
                           'maximize' (bool): Whether to maximize or minimize the metric.
                           'parallel' (bool): Whether to run in parallel.
                           'n_jobs' (int): The number of parallel jobs to run (-1 for all cores).
        param_grid (dict): A dictionary where keys are parameter names and values are lists of values to test.

    Returns:
        A tuple containing the best results DataFrame, a dictionary with the best parameters,
        and a list of all results.
    """
    parameter_combinations = _generate_parameter_combinations(param_grid)
    num_combinations = len(parameter_combinations)

    # Extract settings from the parameters dict
    metric = parameters.get('metric', 'total_return_pct')
    maximize = parameters.get('maximize', True)
    parallel = parameters.get('parallel', False)
    n_jobs = parameters.get('n_jobs', -1)

    # Base parameters are those that are not part of the optimization grid settings
    base_parameters = {k: v for k, v in parameters.items() if k not in ['metric', 'maximize', 'parallel', 'n_jobs']}

    print(f"Starting optimization for {num_combinations} combinations...")
    print(f"Metric: {metric} ({'Maximize' if maximize else 'Minimize'}) | Parallel: {parallel}{f' (n_jobs={n_jobs})' if parallel else ''}")

    all_run_results = []
    if parallel:
        # Parallel execution
        results_list = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_run_backtest_worker)(params, data, strategy_name, base_parameters, metric)
            for params in parameter_combinations
        )
        # Filter out failed runs and structure results
        for res in results_list:
            if res and res['score'] is not None:
                all_run_results.append({'params': res['params'], 'results_summary': res['results_df'], 'score': res['score'], 'full_results': res['results_df']})
    else:
        # Sequential execution
        for i, params in enumerate(parameter_combinations):
            print(f"  Testing combination {i+1}/{num_combinations}: {params}")
            worker_result = _run_backtest_worker(params, data, strategy_name, base_parameters, metric)
            if worker_result and worker_result['score'] is not None:
                all_run_results.append({'params': worker_result['params'], 'results_summary': worker_result['results_df'], 'score': worker_result['score'], 'full_results': worker_result['results_df']})

    if not all_run_results:
        print("No valid results were generated. This might be due to the metric not being found in any backtest results.")
        return None, None, []

    # Find the best result from all runs
    best_run = max(all_run_results, key=lambda x: x['score']) if maximize else min(all_run_results, key=lambda x: x['score'])
    
    best_score = best_run['score']
    best_params = best_run['params']
    best_results = best_run['full_results']

    print("\nOptimization finished.")
    print(f"Best score ({metric}): {best_score:.4f}")
    print(f"Best parameters: {best_params}")

    return best_results, best_params, all_run_results