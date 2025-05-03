"""Data module for downloading and processing financial data and computing technical indicators."""

# Re-export all functionality from the data package
from .data.indicator_handler import compute_indicator, download_data, download_and_compute_indicator

__all__ = ['compute_indicator', 'download_data', 'download_and_compute_indicator']
