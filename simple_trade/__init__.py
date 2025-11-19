# Import from data module
from .indicator_handler import download_data, compute_indicator
from .core import INDICATORS

# Import all indicators from core
from .core import (
    # Trend indicators
    sma, ema, wma, hma, adx, psa, tri, aro, str, vid,
    ich, ama, eit, fma, gma, htt, jma, kma, soa, tma, zma,

    # Momentum indicators
    rsi, mac, sto, cci, roc, wil, cmo, ult, dpo, eri, rmi, tsi, qst, crs, msi, fis, stc, ttm, kst, cog, vor, lsi,

    # Volatility indicators
    bollinger_bands, atr, keltner_channels, donchian_channels, chaikin_volatility,

    # Volume indicators
    obv, vma, adline, cmf, vpt
)

# Import backtesting components
from .backtesting import Backtester
from .band_trade import BandTradeBacktester
from .cross_trade import CrossTradeBacktester
from .combine_trade import CombineTradeBacktester
from .premade_backtest import premade_backtest
from .fibonacci_retracement import calculate_fibonacci_levels, plot_fibonacci_retracement
from .resistance_support import find_pivot_points, find_resistance_support_lines, plot_resistance_support

# Import optimizer
from .optimizer import Optimizer

# Import plotting tools
from .plot_ind import IndicatorPlotter
from .plot_test import BacktestPlotter

__all__ = [
    # Main classes
    "Backtester",
    "BandTradeBacktester", 
    "CrossTradeBacktester",
    "Optimizer",
    "IndicatorPlotter",
    "BacktestPlotter",
    "CombineTradeBacktester",
    "premade_backtest",
    "calculate_fibonacci_levels",
    "plot_fibonacci_retracement",
    "find_pivot_points",
    "find_resistance_support_lines",
    "plot_resistance_support",
    
    # Data functions
    "download_data", "compute_indicator",
    
    # Indicators dictionary
    "INDICATORS",
    
    # Trend indicators
    "sma", "ema", "wma", "hma", "adx", "psa", "tri", "aro", "str", "vid",
    "ich", "ama", "eit", "fma", "gma", "htt", "jma", "kma", "soa", "tma", "zma",
    
    # Momentum indicators
    'rsi', 'mac', 'sto', 'cci', 'roc', 'wil', 'cmo', 'ult', 'dpo', 'eri', 'rmi', 'tsi', 'qst', 'crs', 'msi', 'fis', 'stc', 'ttm', 'kst', 'cog', 'vor', 'lsi',
    
    # Volatility indicators
    "bollinger_bands", "atr", "keltner_channels", "donchian_channels", "chaikin_volatility",
    
    # Volume indicators
    "obv", "vma", "adline", "cmf", "vpt"
]