"""
Configuration dataclass for backtesting parameters.
"""
from dataclasses import dataclass


@dataclass
class BacktestConfig:
    """
    Configuration for backtesting parameters.
    
    This dataclass holds all the common configuration parameters used across
    different backtesting functions, reducing parameter repetition.
    
    Attributes:
        initial_cash: Starting cash balance for the backtest.
        commission_long: Commission rate for long trades (e.g., 0.001 for 0.1%).
        commission_short: Commission rate for short trades (e.g., 0.001 for 0.1%).
        short_borrow_fee_inc_rate: Time-based fee rate for holding short positions.
        long_borrow_fee_inc_rate: Time-based fee rate for holding long positions.
        long_entry_pct_cash: Percentage of cash to use for long entries (0.0 to 1.0).
        short_entry_pct_cash: Percentage of cash to use for short entries (0.0 to 1.0).
        risk_free_rate: Annual risk-free rate for Sharpe ratio calculation.
    
    Example:
        >>> config = BacktestConfig(initial_cash=50000, commission_long=0.002)
        >>> results, portfolio = run_band_trade(data, config, ...)
    """
    initial_cash: float = 10000.0
    commission_long: float = 0.001
    commission_short: float = 0.001
    short_borrow_fee_inc_rate: float = 0.0
    long_borrow_fee_inc_rate: float = 0.0
    long_entry_pct_cash: float = 0.9
    short_entry_pct_cash: float = 0.9
    risk_free_rate: float = 0.0
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if not (0.0 <= self.commission_long <= 1.0):
            raise ValueError("commission_long must be between 0.0 and 1.0")
        if not (0.0 <= self.commission_short <= 1.0):
            raise ValueError("commission_short must be between 0.0 and 1.0")
        if not (0.0 <= self.long_entry_pct_cash <= 1.0):
            raise ValueError("long_entry_pct_cash must be between 0.0 and 1.0")
        if not (0.0 <= self.short_entry_pct_cash <= 1.0):
            raise ValueError("short_entry_pct_cash must be between 0.0 and 1.0")
        if self.short_borrow_fee_inc_rate < 0:
            raise ValueError("short_borrow_fee_inc_rate must be non-negative")
        if self.long_borrow_fee_inc_rate < 0:
            raise ValueError("long_borrow_fee_inc_rate must be non-negative")


def get_default_config() -> BacktestConfig:
    """
    Returns a BacktestConfig with default values.
    
    Returns:
        BacktestConfig: Default configuration instance.
    """
    return BacktestConfig()
