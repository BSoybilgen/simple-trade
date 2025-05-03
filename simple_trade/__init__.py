from .data import download_data, compute_indicator, download_and_compute_indicator
from .core import INDICATORS
from .backtesting import Backtester # Import Backtester
from .plot_ind import IndicatorPlotter # Use new plotting module

# Import all indicators from core
from .core import sma, ema, wma, hma, rsi, macd, bollinger_bands, adx

__all__ = [
    # Main classes
    "Backtester", # Add Backtester to __all__
    "IndicatorPlotter", # Use new plotting module
    
    # Data functions
    "download_data", "compute_indicator", "download_and_compute_indicator",
    
    # Indicators dictionary
    "INDICATORS",
    
    # Individual indicators
    "sma", "ema", "wma", "hma", "rsi", "macd", "bollinger_bands", "adx"
]
