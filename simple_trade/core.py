"""
Core module that imports and organizes all components of the simple_trade package.
"""
import pandas as pd
import numpy as np

# Import trend indicators
from simple_trade.trend.sma import sma
from simple_trade.trend.ema import ema
from simple_trade.trend.wma import wma
from simple_trade.trend.hma import hma
from simple_trade.trend.adx import adx
from simple_trade.trend.psar import psar
from simple_trade.trend.ichi import ichimoku, tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
from simple_trade.trend.trix import trix
from simple_trade.trend.aroon import aroon

# Import momentum indicators
from simple_trade.momentum.rsi import rsi
from simple_trade.momentum.macd import macd
from simple_trade.momentum.stoch import stoch
from simple_trade.momentum.cci import cci
from simple_trade.momentum.roc import roc

# Import volatility indicators
from simple_trade.volatility.bollin import bollinger_bands
from simple_trade.volatility.atr import atr
from simple_trade.volatility.kelt import keltner_channels
from simple_trade.volatility.donch import donchian_channels
from simple_trade.volatility.chaik import chaikin_volatility

# Import volume indicators
from simple_trade.volume.obv import obv
from simple_trade.volume.vma import vma
from simple_trade.volume.adline import adline
from simple_trade.volume.cmf import cmf
from simple_trade.volume.vpt import vpt

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
    'sma', 'ema', 'wma', 'hma', 'adx', 'psar', 'trix', 'aroon',   # Trend indicators
    'ichimoku', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span',  # Ichimoku indicators
    'rsi', 'macd', 'stoch', 'cci', 'roc',    # Momentum indicators
    'bollinger_bands', 'atr', 'keltner_channels', 'donchian_channels', 'chaikin_volatility',  # Volatility indicators
    'obv', 'vma', 'adline', 'cmf', 'vpt',  # Volume indicators
]
