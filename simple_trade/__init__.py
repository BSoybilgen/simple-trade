# Import from data module
from .compute_indicators import download_data, compute_indicator, list_indicators
from .core import INDICATORS

# Import all indicators from core
from .core import (
    # Trend indicators
    sma, ema, wma, hma, adx, psa, tri, aro, str, vid,
    ich, ama, eit, fma, gma, htt, jma, kma, soa, tma, zma,
    ads, alm, dem, eac, lsm, mgd, swm, tem,

    # Momentum indicators
    rsi, mac, sto, cci, roc, wil, cmo, ult, dpo, eri, rmi, tsi, qst, crs, msi, fis, stc, ttm, kst, cog, vor, lsi,
    awo, bop, imi, pgo, ppo, psy, rvg, sri,

    # Volatility indicators
    bol, atr, kel, don, cha,
    acb, atp, bbw, cho, dvi, efr, fdi, grv, hav, hiv, mad, mai, nat, pav, pcw, pro, rsv, rvi, std, svi, uli, vhf, vqi, vsi, vra, tsv,

    # Volume indicators
    obv, vma, adl, cmf, vpt,
    vwa, mfi, foi, emv, pvo, vro, nvi, pvi, kvo, ado, vfi, bwm, fve, wad, voo
)

# Import configuration
from .config import BacktestConfig, get_default_config

# Import metrics functions
from .metrics import (
    compute_benchmark_return,
    calculate_performance_metrics,
    print_results,
    count_trades
)

# Import backtesting functions
from .run_band_trade_strategies import run_band_trade
from .run_cross_trade_strategies import run_cross_trade
from .run_combined_trade_strategies import run_combined_trade, plot_combined_results
from .optimize_custom_strategies import custom_optimizer, get_top_results, results_to_dataframe
from .optimize_premade_strategies import premade_optimizer

# Import premade backtest functions
from .run_premade_strategies import run_premade_trade, list_premade_strategies
from .compute_fibonacci_retracement import calculate_fibonacci_levels, plot_fibonacci_retracement
from .compute_resistance_support import find_pivot_points, find_resistance_support_lines, plot_resistance_support

# Import plotting functions
from .plot_ind import plot_indicator
from .plot_test import plot_backtest_results

__all__ = [
    # Configuration
    "BacktestConfig",
    "get_default_config",
    
    # Metrics functions
    "compute_benchmark_return",
    "calculate_performance_metrics",
    "print_results",
    "count_trades",
    
    # Backtesting functions
    "run_band_trade",
    "run_cross_trade",
    "run_combined_trade",
    "custom_optimizer",
    "premade_optimizer",
    "get_top_results",
    "results_to_dataframe",
    
    # Plotting functions
    "plot_indicator",
    "plot_backtest_results",
    "plot_combined_results",
    
    # Premade backtest
    "run_premade_trade",
    "list_premade_strategies",
    
    # Technical analysis tools
    "calculate_fibonacci_levels",
    "plot_fibonacci_retracement",
    "find_pivot_points",
    "find_resistance_support_lines",
    "plot_resistance_support",
    
    # Data functions
    "download_data", "compute_indicator", "list_indicators",
    
    # Indicators dictionary
    "INDICATORS",
    
    # Trend indicators
    "sma", "ema", "wma", "hma", "adx", "psa", "tri", "aro", "str", "vid",
    "ich", "ama", "eit", "fma", "gma", "htt", "jma", "kma", "soa", "tma", "zma",
    "ads", "alm", "dem", "eac", "lsm", "mgd", "swm", "tem",
    
    # Momentum indicators
    "rsi", "mac", "sto", "cci", "roc", "wil", "cmo", "ult", "dpo", "eri", "rmi", "tsi", "qst", "crs", "msi", "fis", "stc", "ttm", "kst", "cog", "vor", "lsi",
    "awo", "bop", "imi", "pgo", "ppo", "psy", "rvg", "sri",
    
    # Volatility indicators
    "bol", "atr", "kel", "don", "cha",
    "acb", "atp", "bbw", "cho", "dvi", "efr", "fdi", "grv", "hav", "hiv", "mad", "mai", "nat", "pav", "pcw", "pro", "rsv", "rvi", "std", "svi", "uli", "vhf", "vqi", "vsi", "vra", "tsv",
    
    # Volume indicators
    "obv", "vma", "adl", "cmf", "vpt",
    "vwa", "mfi", "foi", "emv", "pvo", "vro", "nvi", "pvi", "kvo", "ado", "vfi", "bwm", "fve", "wad", "voo"
]