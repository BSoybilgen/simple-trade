"""
Trend indicators module
"""
from .sma import sma
from .ema import ema
from .wma import wma
from .hma import hma
from .adx import adx
from .psar import psar
from .ichi import ichimoku
from .trix import trix
from .aroon import aroon
from .strend import supertrend

__all__ = [
    'sma', 'ema', 'wma', 'hma', 'adx', 'psar', 
    'ichimoku',
    'trix', 'aroon', 'supertrend'
]