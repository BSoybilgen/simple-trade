# Import from data module
from .data import download_data, compute_indicator
from .core import INDICATORS

# Import all indicators from core
from .core import (
    # Trend indicators
    sma, ema, wma, hma, adx, psar, trix, aroon, supertrend,
    ichimoku,
    
    # Momentum indicators
    rsi, macd, stoch, cci, roc,
    
    # Volatility indicators
    bollinger_bands, atr, keltner_channels, donchian_channels, chaikin_volatility,
    
    # Volume indicators
    obv, vma, adline, cmf, vpt
)

# Import backtesting components
from .backtesting import Backtester
from .band_trade import BandTradeBacktester
from .cross_trade import CrossTradeBacktester

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
    
    # Data functions
    "download_data", "compute_indicator", "download_and_compute_indicator",
    
    # Indicators dictionary
    "INDICATORS",
    
    # Trend indicators
    "sma", "ema", "wma", "hma", "adx", "psar", "trix", "aroon", "supertrend",
    "ichimoku",
    
    # Momentum indicators
    "rsi", "macd", "stoch", "cci", "roc",
    
    # Volatility indicators
    "bollinger_bands", "atr", "keltner_channels", "donchian_channels", "chaikin_volatility",
    
    # Volume indicators
    "obv", "vma", "adline", "cmf", "vpt"
]