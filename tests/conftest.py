"""
Shared fixtures for all test modules.
"""
import pytest
import pandas as pd
import numpy as np


# --- Premade Strategy Test Fixtures ---

@pytest.fixture
def sample_ohlcv_data():
    """Fixture to provide sample OHLCV data with DatetimeIndex"""
    index = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    close = pd.Series(np.linspace(100, 150, 100) + np.random.normal(0, 2, 100), index=index)
    high = close + np.random.uniform(0.5, 3, size=len(close))
    low = close - np.random.uniform(0.5, 3, size=len(close))
    low = pd.Series(np.minimum(low.values, close.values - 0.1), index=index)
    high = pd.Series(np.maximum(high.values, close.values + 0.1), index=index)
    # Open price between low and high, closer to previous close
    open_price = low + (high - low) * np.random.uniform(0.3, 0.7, size=len(close))
    volume = pd.Series(np.random.randint(1000, 10000, size=len(close)), index=index)
    
    df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    })
    return df


@pytest.fixture
def default_parameters():
    """Default parameters for run_premade_trade"""
    return {
        'initial_cash': 10000.0,
        'commission_long': 0.001,
        'commission_short': 0.001,
        'trading_type': 'long',
        'day1_position': 'none',
        'risk_free_rate': 0.02,
        'fig_control': 0
    }
