"""
Trend indicators module
"""
from .sma import sma
from .ema import ema
from .wma import wma
from .hma import hma
from .adx import adx
from .psa import psa
from .soa import soa
from .ama import ama
from .kma import kma
from .tma import tma
from .fma import fma
from .gma import gma
from .jma import jma
from .zma import zma
from .htt import htt
from .eit import eit
from .ich import ich
from .tri import tri
from .aro import aro
from .str import str
from .dem import dem
from .tem import tem
from .alm import alm
from .mgd import mgd
from .lsm import lsm
from .swm import swm
from .ads import ads
from .eac import eac
from .vid import vid

__all__ = [
    'sma', 'ema', 'wma', 'hma', 'adx', 'psa', 'soa', 'ama', 'kma', 'tma', 'fma', 'gma', 'jma', 'zma', 'htt', 'eit',
    'ich',
    'tri', 'aro', 'str',
    'dem', 'tem', 'alm',
    'mgd', 'lsm', 'swm', 'ads', 'eac', 'vid'
]