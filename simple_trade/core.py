"""
Core module that imports and organizes all components of the simple_trade package.
"""

# Import moving average indicators
from simple_trade.moving_average.ads import ads
from simple_trade.moving_average.alm import alm
from simple_trade.moving_average.ama import ama
from simple_trade.moving_average.dem import dem
from simple_trade.moving_average.ema import ema
from simple_trade.moving_average.fma import fma
from simple_trade.moving_average.gma import gma
from simple_trade.moving_average.hma import hma
from simple_trade.moving_average.jma import jma
from simple_trade.moving_average.kma import kma
from simple_trade.moving_average.lsm import lsm
from simple_trade.moving_average.sma import sma
from simple_trade.moving_average.soa import soa
from simple_trade.moving_average.swm import swm
from simple_trade.moving_average.tem import tem
from simple_trade.moving_average.tma import tma
from simple_trade.moving_average.vid import vid
from simple_trade.moving_average.vma import vma
from simple_trade.moving_average.wma import wma
from simple_trade.moving_average.zma import zma

# Import trend indicators
from simple_trade.trend.adx import adx
from simple_trade.trend.aro import aro
from simple_trade.trend.eac import eac
from simple_trade.trend.eit import eit
from simple_trade.trend.htt import htt
from simple_trade.trend.ich import ich
from simple_trade.trend.mgd import mgd
from simple_trade.trend.pro import pro
from simple_trade.trend.psa import psa
from simple_trade.trend.str import str
from simple_trade.trend.tri import tri
from simple_trade.trend.vqi import vqi

# Import momentum indicators
from simple_trade.momentum.awo import awo
from simple_trade.momentum.bop import bop
from simple_trade.momentum.cci import cci
from simple_trade.momentum.cmo import cmo
from simple_trade.momentum.cog import cog
from simple_trade.momentum.crs import crs
from simple_trade.momentum.dpo import dpo
from simple_trade.momentum.eri import eri
from simple_trade.momentum.fis import fis
from simple_trade.momentum.imi import imi
from simple_trade.momentum.kst import kst
from simple_trade.momentum.lsi import lsi
from simple_trade.momentum.mac import mac
from simple_trade.momentum.msi import msi
from simple_trade.momentum.pgo import pgo
from simple_trade.momentum.ppo import ppo
from simple_trade.momentum.psy import psy
from simple_trade.momentum.qst import qst
from simple_trade.momentum.roc import roc
from simple_trade.momentum.rmi import rmi
from simple_trade.momentum.rsi import rsi
from simple_trade.momentum.rvg import rvg
from simple_trade.momentum.sri import sri
from simple_trade.momentum.stc import stc
from simple_trade.momentum.sto import sto
from simple_trade.momentum.tsi import tsi
from simple_trade.momentum.ttm import ttm
from simple_trade.momentum.ult import ult
from simple_trade.momentum.vor import vor
from simple_trade.momentum.wad import wad
from simple_trade.momentum.wil import wil

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
from simple_trade.volatility.rsv import rsv
from simple_trade.volatility.rvi import rvi
from simple_trade.volatility.svi import svi
from simple_trade.volatility.tsv import tsv
from simple_trade.volatility.uli import uli
from simple_trade.volatility.vhf import vhf
from simple_trade.volatility.vra import vra
from simple_trade.volatility.vsi import vsi

# Import statistics indicators
from simple_trade.statistics.kur import kur
from simple_trade.statistics.mab import mab
from simple_trade.statistics.med import med
from simple_trade.statistics.qua import qua
from simple_trade.statistics.skw import skw
from simple_trade.statistics.std import std
from simple_trade.statistics.var import var
from simple_trade.statistics.zsc import zsc

# Import volume indicators
from simple_trade.volume.ado import ado
from simple_trade.volume.adl import adl
from simple_trade.volume.bwm import bwm
from simple_trade.volume.cmf import cmf
from simple_trade.volume.emv import emv
from simple_trade.volume.foi import foi
from simple_trade.volume.fve import fve
from simple_trade.volume.kvo import kvo
from simple_trade.volume.mfi import mfi
from simple_trade.volume.nvi import nvi
from simple_trade.volume.obv import obv
from simple_trade.volume.pvi import pvi
from simple_trade.volume.pvo import pvo
from simple_trade.volume.vfi import vfi
from simple_trade.volume.voo import voo
from simple_trade.volume.vpt import vpt
from simple_trade.volume.vro import vro
from simple_trade.volume.vwa import vwa

# Dictionary mapping indicator names to functions
INDICATORS = {
    'acb': acb,
    'adl': adl,
    'ado': ado,
    'ads': ads,
    'adx': adx,
    'alm': alm,
    'ama': ama,
    'aro': aro,
    'atp': atp,
    'atr': atr,
    'awo': awo,
    'bbw': bbw,
    'bol': bol,
    'bop': bop,
    'bwm': bwm,
    'cci': cci,
    'cha': cha,
    'cho': cho,
    'cmf': cmf,
    'cmo': cmo,
    'cog': cog,
    'crs': crs,
    'dem': dem,
    'don': don,
    'dpo': dpo,
    'dvi': dvi,
    'eac': eac,
    'efr': efr,
    'eit': eit,
    'ema': ema,
    'emv': emv,
    'eri': eri,
    'fdi': fdi,
    'fis': fis,
    'fma': fma,
    'foi': foi,
    'fve': fve,
    'gma': gma,
    'grv': grv,
    'hav': hav,
    'hiv': hiv,
    'hma': hma,
    'htt': htt,
    'ich': ich,
    'imi': imi,
    'jma': jma,
    'kel': kel,
    'kma': kma,
    'kst': kst,
    'kur': kur,
    'kvo': kvo,
    'lsm': lsm,
    'lsi': lsi,
    'mab': mab,
    'mac': mac,
    'mad': mad,
    'mai': mai,
    'mfi': mfi,
    'med': med,
    'mgd': mgd,
    'msi': msi,
    'nat': nat,
    'nvi': nvi,
    'obv': obv,
    'pav': pav,
    'pcw': pcw,
    'pgo': pgo,
    'ppo': ppo,
    'pro': pro,
    'psa': psa,
    'psy': psy,
    'pvi': pvi,
    'pvo': pvo,
    'qua': qua,
    'qst': qst,
    'roc': roc,
    'rmi': rmi,
    'rsi': rsi,
    'rsv': rsv,
    'rvg': rvg,
    'rvi': rvi,
    'skw': skw,
    'sma': sma,
    'soa': soa,
    'sri': sri,
    'std': std,
    'stc': stc,
    'sto': sto,
    'str': str,
    'svi': svi,
    'swm': swm,
    'tem': tem,
    'tma': tma,
    'tri': tri,
    'tsi': tsi,
    'ttm': ttm,
    'tsv': tsv,
    'ult': ult,
    'uli': uli,
    'var': var,
    'vid': vid,
    'vfi': vfi,
    'vhf': vhf,
    'vma': vma,
    'voo': voo,
    'vor': vor,
    'vpt': vpt,
    'vqi': vqi,
    'vra': vra,
    'vro': vro,
    'vsi': vsi,
    'vwa': vwa,
    'wad': wad,
    'wil': wil,
    'wma': wma,
    'zma': zma,
    'zsc': zsc,
}

# Export all indicators
__all__ = [
    'ads', 'alm', 'ama', 'dem', 'ema', 'fma', 'gma', 'hma', 'jma', 'kma', 'lsm', 'sma', 'soa', 'swm', 'tem', 'tma', 'vid', 'vma', 'wma', 'zma',  # Moving average indicators
    'adx', 'aro', 'eac', 'eit', 'htt', 'ich', 'mgd', 'pro', 'psa', 'str', 'tri', 'vqi',  # Trend indicators
    'awo', 'bop', 'cci', 'cmo', 'cog', 'crs', 'dpo', 'eri', 'fis', 'imi', 'kst', 'lsi', 'mac', 'msi', 'pgo', 'ppo', 'psy', 'qst', 'roc', 'rmi', 'rsi', 'rvg', 'sri', 'stc', 'sto', 'tsi', 'ttm', 'ult', 'vor', 'wad', 'wil',  # Momentum indicators
    'acb', 'atr', 'atp', 'bbw', 'bol', 'cha', 'cho', 'don', 'dvi', 'efr', 'fdi', 'grv', 'hav', 'hiv', 'kel', 'mad', 'mai', 'nat', 'pav', 'pcw', 'rsv', 'rvi', 'svi', 'tsv', 'uli', 'vhf', 'vra', 'vsi',  # Volatility indicators
    'adl', 'ado', 'bwm', 'cmf', 'emv', 'foi', 'fve', 'kvo', 'mfi', 'nvi', 'obv', 'pvi', 'pvo', 'vfi', 'voo', 'vpt', 'vro', 'vwa',  # Volume indicators
    'kur', 'mab', 'med', 'qua', 'skw', 'std', 'var', 'zsc',  # Statistics indicators
]
