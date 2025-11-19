"""
Core module that imports and organizes all components of the simple_trade package.
"""

# Import trend indicators
from simple_trade.trend.sma import sma
from simple_trade.trend.ema import ema
from simple_trade.trend.wma import wma
from simple_trade.trend.hma import hma
from simple_trade.trend.adx import adx
from simple_trade.trend.psa import psa
from simple_trade.trend.ich import ich
from simple_trade.trend.tri import tri
from simple_trade.trend.aro import aro
from simple_trade.trend.str import str
from simple_trade.trend.vid import vid
from simple_trade.trend.ama import ama
from simple_trade.trend.eit import eit
from simple_trade.trend.fma import fma
from simple_trade.trend.gma import gma
from simple_trade.trend.htt import htt
from simple_trade.trend.jma import jma
from simple_trade.trend.kma import kma
from simple_trade.trend.soa import soa
from simple_trade.trend.tma import tma
from simple_trade.trend.zma import zma

# Import momentum indicators
from simple_trade.momentum.rsi import rsi
from simple_trade.momentum.mac import mac
from simple_trade.momentum.sto import sto
from simple_trade.momentum.cci import cci
from simple_trade.momentum.roc import roc
from simple_trade.momentum.wil import wil
from simple_trade.momentum.cmo import cmo
from simple_trade.momentum.ult import ult
from simple_trade.momentum.dpo import dpo
from simple_trade.momentum.eri import eri
from simple_trade.momentum.rmi import rmi
from simple_trade.momentum.tsi import tsi
from simple_trade.momentum.qst import qst
from simple_trade.momentum.crs import crs
from simple_trade.momentum.msi import msi
from simple_trade.momentum.fis import fis
from simple_trade.momentum.stc import stc
from simple_trade.momentum.ttm import ttm
from simple_trade.momentum.kst import kst
from simple_trade.momentum.cog import cog
from simple_trade.momentum.vor import vor
from simple_trade.momentum.lsi import lsi

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
    'adx': adx,
    'psa': psa,
    'ich': ich,
    'tri': tri,
    'aro': aro,
    'str': str,
    'vid': vid,
    'ama': ama,
    'eit': eit,
    'fma': fma,
    'gma': gma,
    'htt': htt,
    'jma': jma,
    'kma': kma,
    'soa': soa,
    'tma': tma,
    'zma': zma,
    'rsi': rsi,
    'mac': mac,
    'bollin': bollinger_bands,
    'sto': sto,
    'cci': cci,
    'roc': roc,
    'wil': wil,
    'cmo': cmo,
    'ult': ult,
    'dpo': dpo,
    'eri': eri,
    'rmi': rmi,
    'tsi': tsi,
    'qst': qst,
    'crs': crs,
    'msi': msi,
    'atr': atr,
    'fis': fis,
    'stc': stc,
    'ttm': ttm,
    'kst': kst,
    'cog': cog,
    'vor': vor,
    'lsi': lsi,
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
    'sma', 'ema', 'wma', 'hma', 'adx', 'psa', 'tri', 'aro', 'str', 'vid', 'ama', 'eit', 'fma', 'gma', 'htt', 'jma', 'kma', 'soa', 'tma', 'zma', 'ich',  # Trend indicators
    'rsi', 'mac', 'sto', 'cci', 'roc', 'wil', 'cmo', 'ult', 'dpo', 'eri', 'rmi', 'tsi', 'qst', 'crs', 'msi', 'fis', 'stc', 'ttm', 'kst', 'cog', 'vor', 'lsi',    # Momentum indicators
    'bollinger_bands', 'atr', 'keltner_channels', 'donchian_channels', 'chaikin_volatility',  # Volatility indicators
    'obv', 'vma', 'adline', 'cmf', 'vpt',  # Volume indicators
]
