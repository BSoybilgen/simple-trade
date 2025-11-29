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
from simple_trade.trend.ads import ads
from simple_trade.trend.alm import alm
from simple_trade.trend.dem import dem
from simple_trade.trend.eac import eac
from simple_trade.trend.lsm import lsm
from simple_trade.trend.mgd import mgd
from simple_trade.trend.swm import swm
from simple_trade.trend.tem import tem

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
from simple_trade.momentum.awo import awo
from simple_trade.momentum.bop import bop
from simple_trade.momentum.imi import imi
from simple_trade.momentum.pgo import pgo
from simple_trade.momentum.ppo import ppo
from simple_trade.momentum.psy import psy
from simple_trade.momentum.rvg import rvg
from simple_trade.momentum.sri import sri

# Import volatility indicators
from simple_trade.volatility.acb import acb
from simple_trade.volatility.atr import atr
from simple_trade.volatility.atp import atp
from simple_trade.volatility.bbw import bbw
from simple_trade.volatility.bol import bol
from simple_trade.volatility.cha import cha
from simple_trade.volatility.cho import cho
from simple_trade.volatility.don import don
from simple_trade.volatility.dvi import dvi
from simple_trade.volatility.efr import efr
from simple_trade.volatility.fdi import fdi
from simple_trade.volatility.grv import grv
from simple_trade.volatility.hav import hav
from simple_trade.volatility.hiv import hiv
from simple_trade.volatility.kel import kel
from simple_trade.volatility.mad import mad
from simple_trade.volatility.mai import mai
from simple_trade.volatility.nat import nat
from simple_trade.volatility.pav import pav
from simple_trade.volatility.pcw import pcw
from simple_trade.volatility.pro import pro
from simple_trade.volatility.rsv import rsv
from simple_trade.volatility.rvi import rvi
from simple_trade.volatility.std import std
from simple_trade.volatility.svi import svi
from simple_trade.volatility.uli import uli
from simple_trade.volatility.vhf import vhf
from simple_trade.volatility.vqi import vqi
from simple_trade.volatility.vsi import vsi
from simple_trade.volatility.tsv import tsv
from simple_trade.volatility.vra import vra

# Import volume indicators
from simple_trade.volume.obv import obv
from simple_trade.volume.vma import vma
from simple_trade.volume.adl import adl
from simple_trade.volume.cmf import cmf
from simple_trade.volume.vpt import vpt
from simple_trade.volume.vwa import vwa
from simple_trade.volume.mfi import mfi
from simple_trade.volume.foi import foi
from simple_trade.volume.emv import emv
from simple_trade.volume.pvo import pvo
from simple_trade.volume.vro import vro
from simple_trade.volume.nvi import nvi
from simple_trade.volume.pvi import pvi
from simple_trade.volume.kvo import kvo
from simple_trade.volume.ado import ado
from simple_trade.volume.vfi import vfi
from simple_trade.volume.bwm import bwm
from simple_trade.volume.fve import fve
from simple_trade.volume.wad import wad
from simple_trade.volume.voo import voo

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
    'bol': bol,
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
    'awo': awo,
    'bop': bop,
    'imi': imi,
    'pgo': pgo,
    'ppo': ppo,
    'psy': psy,
    'rvg': rvg,
    'sri': sri,
    'kel': kel,
    'don': don,
    'cha': cha,
    'acb': acb,
    'atp': atp,
    'bbw': bbw,
    'cho': cho,
    'dvi': dvi,
    'efr': efr,
    'fdi': fdi,
    'grv': grv,
    'hav': hav,
    'hiv': hiv,
    'mad': mad,
    'mai': mai,
    'nat': nat,
    'pav': pav,
    'pcw': pcw,
    'pro': pro,
    'rsv': rsv,
    'rvi': rvi,
    'std': std,
    'svi': svi,
    'uli': uli,
    'vhf': vhf,
    'vqi': vqi,
    'vsi': vsi,
    'obv': obv,
    'vma': vma,
    'adl': adl,
    'cmf': cmf,
    'vpt': vpt,
    'vwa': vwa,
    'mfi': mfi,
    'foi': foi,
    'emv': emv,
    'pvo': pvo,
    'vro': vro,
    'nvi': nvi,
    'pvi': pvi,
    'kvo': kvo,
    'ado': ado,
    'vfi': vfi,
    'bwm': bwm,
    'fve': fve,
    'wad': wad,
    'voo': voo,
    'ads': ads,
    'alm': alm,
    'dem': dem,
    'eac': eac,
    'lsm': lsm,
    'mgd': mgd,
    'swm': swm,
    'tem': tem,
    'vra': vra,
    'tsv': tsv,
}

# Export all indicators
__all__ = [
    'sma', 'ema', 'wma', 'hma', 'adx', 'psa', 'tri', 'aro', 'str', 'vid', 'ama', 'eit', 'fma', 'gma', 'htt', 'jma', 'kma', 'soa', 'tma', 'zma', 'ich', 'ads', 'alm', 'dem', 'eac', 'lsm', 'mgd', 'swm', 'tem',  # Trend indicators
    'rsi', 'mac', 'sto', 'cci', 'roc', 'wil', 'cmo', 'ult', 'dpo', 'eri', 'rmi', 'tsi', 'qst', 'crs', 'msi', 'fis', 'stc', 'ttm', 'kst', 'cog', 'vor', 'lsi', 'awo', 'bop', 'imi', 'pgo', 'ppo', 'psy', 'rvg', 'sri',   # Momentum indicators
    'acb', 'atr', 'atp', 'bbw', 'bol', 'cha', 'cho', 'don', 'dvi', 'efr', 'fdi', 'grv', 'hav', 'hiv', 'kel', 'mad', 'mai', 'nat', 'pav', 'pcw', 'pro', 'rsv', 'rvi', 'std', 'svi', 'uli', 'vhf', 'vqi', 'vsi', 'vra', 'tsv',  # Volatility indicators
    'obv', 'vma', 'adl', 'cmf', 'vpt', 'vwa', 'mfi', 'foi', 'emv', 'pvo', 'vro', 'nvi', 'pvi', 'kvo', 'ado', 'vfi', 'bwm', 'fve', 'wad', 'voo',  # Volume indicators
]
