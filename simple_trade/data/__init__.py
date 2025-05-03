"""
Data processing and indicators calculation module.
"""
from .indicator_handler import compute_indicator, download_data, download_and_compute_indicator

__all__ = ['compute_indicator', 'download_data', 'download_and_compute_indicator']
