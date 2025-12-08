import pandas as pd

from .plot_test import plot_backtest_results
from .config import BacktestConfig

# Import backtest functions from indicator modules
from .momentum.awo import strategy_awo
from .momentum.bop import strategy_bop
from .momentum.cci import strategy_cci
from .momentum.cmo import strategy_cmo
from .momentum.cog import strategy_cog
from .momentum.crs import strategy_crs
from .momentum.dpo import strategy_dpo
from .momentum.eri import strategy_eri
from .momentum.fis import strategy_fis
from .momentum.imi import strategy_imi
from .momentum.kst import strategy_kst
from .momentum.lsi import strategy_lsi
from .momentum.mac import strategy_mac
from .momentum.msi import strategy_msi
from .momentum.pgo import strategy_pgo
from .momentum.ppo import strategy_ppo
from .momentum.psy import strategy_psy
from .momentum.qst import strategy_qst
from .momentum.rmi import strategy_rmi
from .momentum.roc import strategy_roc
from .momentum.rsi import strategy_rsi
from .momentum.rvg import strategy_rvg
from .momentum.sri import strategy_sri
from .momentum.stc import strategy_stc
from .momentum.sto import strategy_sto
from .momentum.tsi import strategy_tsi
from .momentum.ttm import strategy_ttm
from .momentum.ult import strategy_ult
from .momentum.vor import strategy_vor
from .momentum.wad import strategy_wad
from .momentum.wil import strategy_wil

# Import moving average strategies
from .moving_average.ads import strategy_ads
from .moving_average.alm import strategy_alm
from .moving_average.ama import strategy_ama
from .moving_average.dem import strategy_dem
from .moving_average.ema import strategy_ema
from .moving_average.fma import strategy_fma
from .moving_average.gma import strategy_gma
from .moving_average.hma import strategy_hma
from .moving_average.jma import strategy_jma
from .moving_average.lsm import strategy_lsm
from .moving_average.sma import strategy_sma
from .moving_average.soa import strategy_soa
from .moving_average.swm import strategy_swm
from .moving_average.tem import strategy_tem
from .moving_average.tma import strategy_tma
from .moving_average.vid import strategy_vid
from .moving_average.vma import strategy_vma
from .moving_average.wma import strategy_wma
from .moving_average.zma import strategy_zma
from .moving_average.tt3 import strategy_tt3
from .moving_average.mam import strategy_mam
from .moving_average.evw import strategy_evw
from .moving_average.tsf import strategy_tsf

# Import trend strategies
from .trend.adx import strategy_adx
from .trend.aro import strategy_aro
from .trend.eac import strategy_eac
from .trend.eit import strategy_eit
from .trend.htt import strategy_htt
from .trend.ich import strategy_ich
from .trend.mgd import strategy_mgd
from .trend.pro import strategy_pro
from .trend.psa import strategy_psa
from .trend.str import strategy_str
from .trend.tri import strategy_tri
from .trend.vqi import strategy_vqi

from .volatility.acb import strategy_acb
from .volatility.atp import strategy_atp
from .volatility.atr import strategy_atr
from .volatility.bbw import strategy_bbw
from .volatility.bol import strategy_bol
from .volatility.cha import strategy_cha
from .volatility.cho import strategy_cho
from .volatility.don import strategy_don
from .volatility.dvi import strategy_dvi
from .volatility.efr import strategy_efr
from .volatility.fdi import strategy_fdi
from .volatility.grv import strategy_grv
from .volatility.hav import strategy_hav
from .volatility.hiv import strategy_hiv
from .volatility.kel import strategy_kel
from .volatility.mad import strategy_mad
from .volatility.mai import strategy_mai
from .volatility.nat import strategy_nat
from .volatility.pav import strategy_pav
from .volatility.pcw import strategy_pcw
from .volatility.rsv import strategy_rsv
from .volatility.rvi import strategy_rvi
from .volatility.svi import strategy_svi
from .volatility.tsv import strategy_tsv
from .volatility.uli import strategy_uli
from .volatility.vhf import strategy_vhf
from .volatility.vra import strategy_vra
from .volatility.vsi import strategy_vsi

from .volume.ado import strategy_ado
from .volume.adl import strategy_adl
from .volume.bwm import strategy_bwm
from .volume.cmf import strategy_cmf
from .volume.emv import strategy_emv
from .volume.foi import strategy_foi
from .volume.fve import strategy_fve
from .volume.kvo import strategy_kvo
from .volume.mfi import strategy_mfi
from .volume.nvi import strategy_nvi
from .volume.obv import strategy_obv
from .volume.pvi import strategy_pvi
from .volume.pvo import strategy_pvo
from .volume.vfi import strategy_vfi
from .volume.voo import strategy_voo
from .volume.vpt import strategy_vpt
from .volume.vro import strategy_vro
from .volume.vwa import strategy_vwa

# Import statistics strategies
from .statistics.kur import strategy_kur
from .statistics.mab import strategy_mab
from .statistics.med import strategy_med
from .statistics.qua import strategy_qua
from .statistics.skw import strategy_skw
from .statistics.std import strategy_std
from .statistics.var import strategy_var
from .statistics.zsc import strategy_zsc


# Strategy registry mapping strategy names to their backtest functions
_STRATEGY_REGISTRY = {
    # Momentum strategies
    'awo': strategy_awo,
    'bop': strategy_bop,
    'cci': strategy_cci,
    'cmo': strategy_cmo,
    'cog': strategy_cog,
    'crs': strategy_crs,
    'dpo': strategy_dpo,
    'eri': strategy_eri,
    'fis': strategy_fis,
    'imi': strategy_imi,
    'kst': strategy_kst,
    'lsi': strategy_lsi,
    'mac': strategy_mac,
    'msi': strategy_msi,
    'pgo': strategy_pgo,
    'ppo': strategy_ppo,
    'psy': strategy_psy,
    'qst': strategy_qst,
    'rmi': strategy_rmi,
    'roc': strategy_roc,
    'rsi': strategy_rsi,
    'rvg': strategy_rvg,
    'sri': strategy_sri,
    'stc': strategy_stc,
    'sto': strategy_sto,
    'tsi': strategy_tsi,
    'ttm': strategy_ttm,
    'ult': strategy_ult,
    'vor': strategy_vor,
    'wad': strategy_wad,
    'wil': strategy_wil,
    # Trend strategies
    'ads': strategy_ads,
    'adx': strategy_adx,
    'alm': strategy_alm,
    'ama': strategy_ama,
    'aro': strategy_aro,
    'dem': strategy_dem,
    'eac': strategy_eac,
    'eit': strategy_eit,
    'ema': strategy_ema,
    'fma': strategy_fma,
    'gma': strategy_gma,
    'hma': strategy_hma,
    'htt': strategy_htt,
    'ich': strategy_ich,
    'jma': strategy_jma,
    'lsm': strategy_lsm,
    'mgd': strategy_mgd,
    'pro': strategy_pro,
    'psa': strategy_psa,
    'sma': strategy_sma,
    'soa': strategy_soa,
    'str': strategy_str,
    'swm': strategy_swm,
    'tem': strategy_tem,
    'tma': strategy_tma,
    'tri': strategy_tri,
    'vid': strategy_vid,
    'vqi': strategy_vqi,
    'wma': strategy_wma,
    'zma': strategy_zma,
    'tt3': strategy_tt3,
    'mam': strategy_mam,
    'evw': strategy_evw,
    'tsf': strategy_tsf,
    # Volatility strategies
    'acb': strategy_acb,
    'atp': strategy_atp,
    'atr': strategy_atr,
    'bbw': strategy_bbw,
    'bol': strategy_bol,
    'cha': strategy_cha,
    'cho': strategy_cho,
    'don': strategy_don,
    'dvi': strategy_dvi,
    'efr': strategy_efr,
    'fdi': strategy_fdi,
    'grv': strategy_grv,
    'hav': strategy_hav,
    'hiv': strategy_hiv,
    'kel': strategy_kel,
    'mad': strategy_mad,
    'mai': strategy_mai,
    'nat': strategy_nat,
    'pav': strategy_pav,
    'pcw': strategy_pcw,
    'rsv': strategy_rsv,
    'rvi': strategy_rvi,
    'svi': strategy_svi,
    'tsv': strategy_tsv,
    'uli': strategy_uli,
    'vhf': strategy_vhf,
    'vra': strategy_vra,
    'vsi': strategy_vsi,
    # Volume strategies
    'ado': strategy_ado,
    'adl': strategy_adl,
    'bwm': strategy_bwm,
    'cmf': strategy_cmf,
    'emv': strategy_emv,
    'foi': strategy_foi,
    'fve': strategy_fve,
    'kvo': strategy_kvo,
    'mfi': strategy_mfi,
    'nvi': strategy_nvi,
    'obv': strategy_obv,
    'pvi': strategy_pvi,
    'pvo': strategy_pvo,
    'vfi': strategy_vfi,
    'vma': strategy_vma,
    'voo': strategy_voo,
    'vpt': strategy_vpt,
    'vro': strategy_vro,
    'vwa': strategy_vwa,
    # Statistics strategies
    'kur': strategy_kur,
    'mab': strategy_mab,
    'med': strategy_med,
    'qua': strategy_qua,
    'skw': strategy_skw,
    'std': strategy_std,
    'var': strategy_var,
    'zsc': strategy_zsc,
}


# Strategy categories mapping - groups strategies by their category
_STRATEGY_CATEGORIES = {
    'momentum': [
        'awo', 'bop', 'cci', 'cmo', 'cog', 'crs', 'dpo', 'eri', 'fis', 'imi',
        'kst', 'lsi', 'mac', 'msi', 'pgo', 'ppo', 'psy', 'qst', 'rmi', 'roc',
        'rsi', 'rvg', 'sri', 'stc', 'sto', 'tsi', 'ttm', 'ult', 'vor', 'wad', 'wil',
    ],
    'trend': [
        'ads', 'adx', 'alm', 'ama', 'aro', 'dem', 'eac', 'eit', 'ema', 'fma',
        'gma', 'hma', 'htt', 'ich', 'jma', 'lsm', 'mgd', 'pro', 'psa', 'sma',
        'soa', 'str', 'swm', 'tem', 'tma', 'tri', 'vid', 'vqi', 'wma', 'zma',
        'tt3', 'mam', 'evw', 'tsf',
    ],
    'volatility': [
        'acb', 'atp', 'atr', 'bbw', 'bol', 'cha', 'cho', 'don', 'dvi', 'efr',
        'fdi', 'grv', 'hav', 'hiv', 'kel', 'mad', 'mai', 'nat', 'pav', 'pcw',
        'rsv', 'rvi', 'svi', 'tsv', 'uli', 'vhf', 'vra', 'vsi',
    ],
    'volume': [
        'ado', 'adl', 'bwm', 'cmf', 'emv', 'foi', 'fve', 'kvo', 'mfi', 'nvi',
        'obv', 'pvi', 'pvo', 'vfi', 'vma', 'voo', 'vpt', 'vro', 'vwa',
    ],
    'statistics': [
        'kur', 'mab', 'med', 'qua', 'skw', 'std', 'var', 'zsc',
    ],
}


def _get_strategy_description(strategy_func) -> str:
    """Extract the first line of a strategy function's docstring as description."""
    import inspect
    doc = inspect.getdoc(strategy_func)
    if doc:
        # Get the first meaningful line (skip empty lines)
        lines = [line.strip() for line in doc.split('\n') if line.strip()]
        return lines[0] if lines else "No description available"
    return "No description available"


def _wrap_text(text: str, width: int = 70, indent: int = 0) -> list:
    """Wrap text to a specified width with optional indentation."""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        word_length = len(word)
        if current_length + word_length + len(current_line) > width:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
        else:
            current_line.append(word)
            current_length += word_length
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines


def list_premade_strategies(category: str = None, return_dict: bool = False) -> dict | None:
    """List all available premade backtest strategies with their descriptions.
    
    This function provides a comprehensive catalog of all backtest strategies available
    in the premade_backtest function, organized by category (momentum, trend, volatility, volume).
    
    Args:
        category: Optional filter by category. Options: 'momentum', 'trend', 'volatility', 'volume', 'statistics'.
                 If None, returns all strategies.
        return_dict: If True, returns a dictionary instead of printing. Default is False.
    
    Returns:
        dict or None: If return_dict=True, returns a nested dictionary with structure:
                     {category: {strategy_name: description}}
                     Otherwise, prints the strategies and returns None.
    
    Example:
        >>> list_strategies()  # Print all strategies
        >>> list_strategies(category='momentum')  # Print only momentum strategies
        >>> strategies = list_strategies(return_dict=True)  # Get dictionary of all strategies
    """
    # Filter by category if specified
    if category:
        if category.lower() not in _STRATEGY_CATEGORIES:
            valid_categories = ', '.join(_STRATEGY_CATEGORIES.keys())
            raise ValueError(f"Invalid category '{category}'. Valid options: {valid_categories}")
        categories_to_process = {category.lower(): _STRATEGY_CATEGORIES[category.lower()]}
    else:
        categories_to_process = _STRATEGY_CATEGORIES
    
    # Build the catalog dynamically from strategy function docstrings
    all_strategies = {}
    for cat_name, strategy_names in categories_to_process.items():
        all_strategies[cat_name] = {}
        for strategy_name in strategy_names:
            strategy_func = _STRATEGY_REGISTRY.get(strategy_name)
            if strategy_func and callable(strategy_func):
                description = _get_strategy_description(strategy_func)
                all_strategies[cat_name][strategy_name] = description
    
    # Return dictionary if requested
    if return_dict:
        return all_strategies
    
    # Otherwise, print formatted output
    print("\n" + "="*80)
    print("AVAILABLE PREMADE BACKTEST STRATEGIES")
    print("="*80 + "\n")
    
    for cat_name, strategies in all_strategies.items():
        print(f"\n{'─'*80}")
        print(f"{cat_name.upper()} STRATEGIES ({len(strategies)} total)")
        print(f"{'─'*80}\n")
        
        for strategy_name, description in sorted(strategies.items()):
            print(f"  • {strategy_name.upper()}")
            # Wrap long descriptions
            desc_lines = _wrap_text(description, width=72, indent=4)
            for line in desc_lines:
                print(f"    {line}")
            print()
    
    print("="*80)
    print(f"Total: {sum(len(strategies) for strategies in all_strategies.values())} strategies")
    print("="*80 + "\n")
    
    return None


def run_premade_trade(data: pd.DataFrame, strategy_name: str, parameters: dict = None):
    """
    Run a premade strategy on the provided data.
    
    This function provides a simple interface to run any of the 110 built-in
    trading strategies. Each strategy uses a specific technical indicator with
    predefined trading logic (crossover, mean reversion, band breakout, etc.).
    
    Args:
        data: DataFrame with OHLCV data (must have Open, High, Low, Close, Volume columns)
        strategy_name: Name of the strategy to run (e.g., 'rsi', 'macd', 'bol')
                      Use list_strategies() to see all available strategies.
        parameters: Optional dict with strategy-specific and backtest parameters:
            
            Backtest parameters (apply to all strategies):
                - initial_cash (float): Starting capital. Default 10000.0
                - commission_long (float): Commission rate for long trades. Default 0.001
                - commission_short (float): Commission rate for short trades. Default 0.001
                - short_borrow_fee_inc_rate (float): Short borrow fee rate. Default 0.0
                - long_borrow_fee_inc_rate (float): Long borrow fee rate. Default 0.0
                - long_entry_pct_cash (float): Fraction of cash for long entries. Default 1.0
                - short_entry_pct_cash (float): Fraction of cash for short entries. Default 1.0
                - trading_type (str): 'long', 'short', or 'both'. Default 'long'
                - day1_position (str): Initial position 'none', 'long', 'short'. Default 'none'
                - risk_free_rate (float): Risk-free rate for Sharpe ratio. Default 0.0
                - fig_control (int): 1 to generate plot, 0 for no plot. Default 0
            
            Strategy-specific parameters vary by indicator. See individual indicator
            documentation for available parameters.
    
    Returns:
        tuple: (results_dict, portfolio_df, fig)
            - results_dict: Dictionary with performance metrics
            - portfolio_df: DataFrame with portfolio history
            - fig: Plotly figure if fig_control=1, else None
    """
    if parameters is None:
        parameters = {}

    # Extract backtest configuration parameters
    initial_cash = float(parameters.get('initial_cash', 10000.0))
    commission_long = float(parameters.get('commission_long', 0.001))
    commission_short = float(parameters.get('commission_short', 0.001))
    short_borrow_fee_inc_rate = float(parameters.get('short_borrow_fee_inc_rate', 0.0))
    long_borrow_fee_inc_rate = float(parameters.get('long_borrow_fee_inc_rate', 0.0))
    long_entry_pct_cash = float(parameters.get('long_entry_pct_cash', 1.0))
    short_entry_pct_cash = float(parameters.get('short_entry_pct_cash', 1.0))
    trading_type = str(parameters.get('trading_type', 'long'))
    day1_position = str(parameters.get('day1_position', 'none'))
    risk_free_rate = float(parameters.get('risk_free_rate', 0.0))
    fig_control = int(parameters.get('fig_control', 0))

    # Create config for backtesting
    config = BacktestConfig(
        initial_cash=initial_cash,
        commission_long=commission_long,
        commission_short=commission_short,
        short_borrow_fee_inc_rate=short_borrow_fee_inc_rate,
        long_borrow_fee_inc_rate=long_borrow_fee_inc_rate,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        risk_free_rate=risk_free_rate,
    )

    # Validate strategy name
    strategy_name_lower = strategy_name.lower()
    if strategy_name_lower not in _STRATEGY_REGISTRY:
        available = ', '.join(sorted(_STRATEGY_REGISTRY.keys()))
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. "
            f"Use list_strategies() to see available strategies. "
            f"Available: {available}"
        )

    # Get the backtest function for this strategy
    backtest_func = _STRATEGY_REGISTRY[strategy_name_lower]

    # Make a copy of data to avoid modifying the original
    data_copy = data.copy()

    # Run the strategy's backtest function
    results, portfolio, indicator_cols_to_plot, data_with_indicators = backtest_func(
        data=data_copy,
        parameters=parameters,
        config=config,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash
    )

    # Generate plot if requested
    fig = None
    if fig_control == 1 and indicator_cols_to_plot:
        fig = plot_backtest_results(
            data_df=data_with_indicators,
            history_df=portfolio,
            price_col='Close',
            indicator_cols=indicator_cols_to_plot,
            title=f"{strategy_name.upper()} Strategy"
        )

    return results, portfolio, fig
