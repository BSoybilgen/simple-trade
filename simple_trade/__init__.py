# Import from data module
from .compute_indicators import download_data, compute_indicator, list_indicators
from .core import INDICATORS

# Import all indicators from core
from .core import (
    # Moving Average indicators
    ads, alm, ama, dem, ema, fma, gma, hma, jma, kma, lsm, sma,
    soa, swm, tem, tma, vid, vma, wma, zma,

    # Trend indicators
    adx, aro, eac, eit, htt, ich, mgd, pro, psa, str, tri, vqi,

    # Momentum indicators
    awo, bop, cci, cmo, cog, crs, dpo, eri, fis, imi, kst, lsi,
    mac, msi, pgo, ppo, psy, qst, roc, rmi, rsi, rvg, sri, stc, sto,
    tsi, ttm, ult, vor, wad, wil,

    # Volatility indicators
    acb, atr, bbw, bol, cha, cho, don, dvi, efr, fdi, grv,
    hav, hiv, kel, mad, mai, nat, pav, pcw, rsv, rvi, svi, tsv,
    uli, vhf, vra, vsi,

    # Statistics indicators
    kur, mab, med, qua, skw, std, var, zsc,

    # Volume indicators
    ado, adl, bwm, cmf, emv, foi, fve, kvo, mfi, nvi, obv, pvi,
    pvo, vfi, voo, vpt, vro, vwa
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
    "calculate_performance_metrics",
    "compute_benchmark_return",
    "count_trades",
    "print_results",

    # Backtesting functions
    "custom_optimizer",
    "get_top_results",
    "premade_optimizer",
    "results_to_dataframe",
    "run_band_trade",
    "run_combined_trade",
    "run_cross_trade",

    # Plotting functions
    "plot_backtest_results",
    "plot_combined_results",
    "plot_indicator",

    # Premade backtest
    "list_premade_strategies",
    "run_premade_trade",

    # Technical analysis tools
    "calculate_fibonacci_levels",
    "find_pivot_points",
    "find_resistance_support_lines",
    "plot_fibonacci_retracement",
    "plot_resistance_support",

    # Data functions
    "compute_indicator", "download_data", "list_indicators",

    # Indicators dictionary
    "INDICATORS",

    # Moving Average indicators
    "ads", "alm", "ama", "dem", "ema", "fma", "gma", "hma", "jma", "kma", "lsm", "sma",
    "soa", "swm", "tem", "tma", "vid", "vma", "wma", "zma",

    # Trend indicators
    "adx", "aro", "eac", "eit", "htt", "ich", "mgd", "pro", "psa", "str", "tri", "vqi",

    # Momentum indicators
    "awo", "bop", "cci", "cmo", "cog", "crs", "dpo", "eri", "fis", "imi", "kst", "lsi",
    "mac", "msi", "pgo", "ppo", "psy", "qst", "roc", "rmi", "rsi", "rvg", "sri", "stc", "sto",
    "tsi", "ttm", "ult", "vor", "wad", "wil",

    # Volatility indicators
    "acb", "atp", "atr", "bbw", "bol", "cha", "cho", "don", "dvi", "efr", "fdi", "grv",
    "hav", "hiv", "kel", "mad", "mai", "nat", "pav", "pcw", "rsv", "rvi", "svi", "tsv",
    "uli", "vhf", "vra", "vsi",

    # Statistics indicators
    "kur", "mab", "med", "qua", "skw", "std", "var", "zsc",

    # Volume indicators
    "ado", "adl", "bwm", "cmf", "emv", "foi", "fve", "kvo", "mfi", "nvi", "obv", "pvi",
    "pvo", "vfi", "voo", "vpt", "vro", "vwa"
]