"""
Core module that imports and organizes all components of the simple_trading package.
"""
import pandas as pd
import numpy as np

# Import trend indicators
from simple_trading.trend.sma import sma
from simple_trading.trend.ema import ema
from simple_trading.trend.wma import wma
from simple_trading.trend.hma import hma
from simple_trading.trend.adx import adx
from simple_trading.trend.psar import psar
from simple_trading.trend.ichi import ichimoku, tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
from simple_trading.trend.strend import supertrend
from simple_trading.trend.trix import trix
from simple_trading.trend.aroon import aroon

# Import momentum indicators
from simple_trading.momentum.rsi import rsi
from simple_trading.momentum.macd import macd
from simple_trading.momentum.stoch import stoch
from simple_trading.momentum.cci import cci
from simple_trading.momentum.roc import roc

# Import volatility indicators
from simple_trading.volatility.bollin import bollinger_bands
from simple_trading.volatility.atr import atr
from simple_trading.volatility.kelt import keltner_channels
from simple_trading.volatility.donch import donchian_channels
from simple_trading.volatility.chaik import chaikin_volatility

# Import volume indicators
from simple_trading.volume.obv import obv
from simple_trading.volume.vma import vma
from simple_trading.volume.adline import adline
from simple_trading.volume.cmf import cmf
from simple_trading.volume.vpt import vpt

# Dictionary mapping indicator names to functions
INDICATORS = {
    'sma': sma,
    'ema': ema,
    'wma': wma,
    'hma': hma,
    'rsi': rsi,
    'macd': macd,
    'bollin': bollinger_bands,
    'adx': adx,
    'psar': psar,
    'ichimoku': ichimoku,
    'tenkan_sen': tenkan_sen,
    'kijun_sen': kijun_sen,
    'senkou_span_a': senkou_span_a,
    'senkou_span_b': senkou_span_b,
    'chikou_span': chikou_span,
    'supertrend': supertrend,
    'trix': trix,
    'aroon': aroon,
    'stoch': stoch,
    'cci': cci,
    'roc': roc,
    'atr': atr,
    'kelt': keltner_channels,
    'donch': donchian_channels,
    'chaik': chaikin_volatility,
    'obv': obv,
    'vma': vma,
    'adline': adline,
    'cmf': cmf,
    'vpt': vpt,
}

# Export all indicators
__all__ = [
    'sma', 'ema', 'wma', 'hma', 'adx', 'psar', 'supertrend', 'trix', 'aroon',   # Trend indicators
    'ichimoku', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span',  # Ichimoku indicators
    'rsi', 'macd', 'stoch', 'cci', 'roc',    # Momentum indicators
    'bollinger_bands', 'atr', 'keltner_channels', 'donchian_channels', 'chaikin_volatility',  # Volatility indicators
    'obv', 'vma', 'adline', 'cmf', 'vpt',  # Volume indicators
]
