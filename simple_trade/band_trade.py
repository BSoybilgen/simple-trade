import pandas as pd
import numpy as np
from simple_trade.backtesting import Backtester

class BandTradeBacktester(Backtester):
    """Backtester extension for Band Trade strategies."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log = None

    def run_band_trade(self, data: pd.DataFrame, indicator_col: str, upper_band_col: str, lower_band_col: str, 
                            price_col: str = 'Close', long_entry_pct_cash: float = 0.9, short_entry_pct_cash: float = 0.1, 
                            trading_type: str = 'long', strategy_type: int = 1, day1_position: str = 'none',
                            risk_free_rate: int = 0.0) -> tuple:
        """
        Runs a backtest for a band trade strategy.

        Two strategy types are available via the `strategy_type` parameter:
        
        Strategy Type 1 (Mean Reversion - Default):
        - 'long' (default):
            Buys when the indicator crosses below the lower band.
            Sells (closes position) when the indicator crosses above the upper band.
            Only long positions are allowed.
        - 'short':
            Goes Short when the indicator crosses above the upper band.
            Covers Short when the indicator crosses below the lower band.
            No long positions are ever entered.
        - 'mixed':
            Allows both long and short positions with seamless transitions ('Cover and Buy', 'Sell and Short')
            without requiring an intermediate flat position.
            
        Strategy Type 2 (Breakout):
        - 'long' (default):
            Buys when the indicator crosses above the upper band.
            Sells (closes position) when the indicator crosses below the lower band.
            Only long positions are allowed.
        - 'short':
            Goes Short when the indicator crosses below the lower band.
            Covers Short when the indicator crosses above the upper band.
            No long positions are ever entered.
        - 'mixed':
            Allows both long and short positions with seamless transitions ('Cover and Buy', 'Sell and Short')
            without requiring an intermediate flat position.

        Assumes trading at the specified price_col value on the signal day (based on previous day's crossover).

        Args:
            data (pd.DataFrame): DataFrame containing price data and indicator/band columns. Must have a DatetimeIndex.
            indicator_col (str): Column name of the indicator (e.g., 'RSI', 'Close').
            upper_band_col (str): Column name of the upper band (e.g., 'BB_Upper', 'RSI_Upper').
            lower_band_col (str): Column name of the lower band (e.g., 'BB_Lower', 'RSI_Lower').
            price_col (str): Column name to use for trade execution prices (default: 'Close').
            long_entry_pct_cash (float): Pct of available cash to use for long entries (0.0 to 1.0, default 0.9).
            short_entry_pct_cash (float): Pct of available cash defining the value of short entries (0.0 to 1.0, default 0.1).
            trading_type (str): Defines the trading behavior ('long', 'short', 'mixed'). Default is 'long'.
            strategy_type (int): Defines the band trade logic (1: mean_reversion, 2: breakout). Default is 1.
            day1_position (str): Specifies whether to take a position on day 1. Options: 'none', 'long', 'short' (default: 'none').
            risk_free_rate (float): Risk-free rate for Sharpe and Sortino ratios (default: 0.0).

        Returns:
            tuple: A tuple containing:
                - dict: Dictionary with backtest summary results (final value, return, trades).
                - pd.DataFrame: DataFrame tracking daily portfolio evolution (cash, position, value, signals, actions).
        """
        # --- Input Validation ---
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")
        if data.empty:
             # Allow empty dataframe, return default results later
             pass # Or maybe raise ValueError("Input data cannot be empty.") depending on desired behavior

        # Check for required columns BEFORE trying to use them
        required_cols = [price_col, indicator_col, upper_band_col, lower_band_col]
        for col in required_cols:
            if col not in data.columns:
                if col == indicator_col:
                    raise ValueError(f"Indicator column '{col}' not found in DataFrame.")
                elif col == upper_band_col:
                    raise ValueError(f"Upper band column '{col}' not found in DataFrame.")
                elif col == lower_band_col:
                    raise ValueError(f"Lower band column '{col}' not found in DataFrame.")
                else:
                    raise ValueError(f"Price column '{col}' not found in DataFrame.")

        # Validate percentages
        if not (0.0 <= long_entry_pct_cash <= 1.0):
            raise ValueError("long_entry_pct_cash must be between 0.0 and 1.0")
        if not (0.0 <= short_entry_pct_cash <= 1.0):
            raise ValueError("short_entry_pct_cash must be between 0.0 and 1.0")

        valid_trading_types = ['long', 'short', 'mixed']
        if trading_type not in valid_trading_types:
            raise ValueError(f"Invalid trading_type '{trading_type}'. Must be one of {valid_trading_types}")
            
        # Validate strategy_type
        if strategy_type not in [1, 2]:
            raise ValueError(f"Invalid strategy_type: {strategy_type}. Must be 1 (mean reversion) or 2 (breakout).")
            
        # Validate day1_position
        valid_day1_positions = ['none', 'long', 'short']
        if day1_position not in valid_day1_positions:
            raise ValueError(f"Invalid day1_position '{day1_position}'. Must be one of {valid_day1_positions}")
        
        # Check compatibility between day1_position and trading_type
        if day1_position == 'long' and trading_type == 'short':
            raise ValueError("Cannot use day1_position='long' with trading_type='short'")
        if day1_position == 'short' and trading_type == 'long':
            raise ValueError("Cannot use day1_position='short' with trading_type='long'")

        # --- Signal Generation (delegated) ---
        df = self._generate_signals(data.copy(), indicator_col, upper_band_col, lower_band_col, strategy_type)

        # Drop NaNs created by shifts (moved back here from _generate_signals)
        df.dropna(inplace=True)

        # Check if DataFrame is empty *after* generating signals and dropping NaNs
        if df.empty:
            # Return default structure if no data remains for backtesting
            return {
                "error": "DataFrame became empty after signal generation/dropna, cannot run backtest.",
                "strategy": f"Band Trade ({indicator_col} vs {lower_band_col}/{upper_band_col} - {'Mean Reversion' if strategy_type == 1 else 'Breakout'}){' [Shorts Allowed]' if trading_type in ['short', 'mixed'] else ''}{' [Day1 ' + day1_position.capitalize() + ']' if day1_position != 'none' else ''}",
                "indicator_col": indicator_col,
                "upper_band_col": upper_band_col,
                "lower_band_col": lower_band_col,
                "strategy_type": strategy_type,
                "start_date": None,
                "end_date": None,
                "duration_days": 0,
                "initial_cash": self.initial_cash,
                "final_value": self.initial_cash,
                "total_return_pct": 0.0,
                "num_trades": 0,
                # Add other default metrics as needed
            }, pd.DataFrame() # Return empty DataFrame for portfolio details

        # --- Run Backtest --- 
        # Pass the signal df directly to the generalized backtest runner
        portfolio_log, end_state = self._run_backtest(
            signal_df=df, 
            price_col=price_col,
            trading_type=trading_type,
            long_entry_pct_cash=long_entry_pct_cash,
            short_entry_pct_cash=short_entry_pct_cash,
            day1_position=day1_position,
            strategy_type=strategy_type
        )

        # --- Prepare and Return Results ---
        # Use the end_state dataframe directly if the log is not empty
        # If the log is empty, _prepare_results should handle it (likely using initial state)
        final_df = end_state if portfolio_log else pd.DataFrame(index=df.index[[-1]]) # Provide minimal final_df if log is empty

        # Add necessary columns if final_df is minimal/empty and log is empty
        if not portfolio_log:
             # If no trades, create a minimal final_df structure based on initial state for _prepare_results
             final_df = pd.DataFrame({ 
                    'PositionSize': [0],
                    'PositionValue': [0.0],
                    'Cash': [self.initial_cash],
                    'PortfolioValue': [self.initial_cash],
                    'Close': [df[price_col].iloc[-1]]  # Add Close column for compatibility with compute_benchmark_return
             }, index=df.index[[-1]]) # Use last index from signal df
             # Add other expected cols minimally if needed by prepare_results

        results, portfolio_df = self._prepare_results(
            portfolio_log=portfolio_log, 
            final_df=final_df, 
            indicator_col=indicator_col, 
            upper_band_col=upper_band_col, 
            lower_band_col=lower_band_col, 
            strategy_type=strategy_type,
            trading_type=trading_type,
            day1_position=day1_position,
            risk_free_rate=risk_free_rate
        )

        return results, portfolio_df

    def _generate_signals(self, df: pd.DataFrame, indicator_col: str, upper_band_col: str, lower_band_col: str, strategy_type: int) -> pd.DataFrame:
        """Generates buy and sell signals based on indicator crossing bands."""
        # Use shift(1) to base signals on the *previous* day's state relative to bands
        prev_indicator = df[indicator_col].shift(1)
        prev_upper = df[upper_band_col].shift(1)
        prev_lower = df[lower_band_col].shift(1)
        prev_prev_indicator = df[indicator_col].shift(2)
        prev_prev_upper = df[upper_band_col].shift(2)
        prev_prev_lower = df[lower_band_col].shift(2)

        if strategy_type == 1:  # Mean Reversion Strategy
            # Buy/Cover Signal: Indicator crossed *below* lower band on the *previous* day
            df['buy_signal'] = (prev_indicator < prev_lower) & (prev_prev_indicator >= prev_prev_lower)

            # Sell/Short Signal: Indicator crossed *above* upper band on the *previous* day
            df['sell_signal'] = (prev_indicator > prev_upper) & (prev_prev_indicator <= prev_prev_upper)
        else:  # strategy_type == 2: Breakout Strategy
            # Buy/Cover Signal: Indicator crossed *above* upper band on the *previous* day
            df['buy_signal'] = (prev_indicator > prev_upper) & (prev_prev_indicator <= prev_prev_upper)

            # Sell/Short Signal: Indicator crossed *below* lower band on the *previous* day
            df['sell_signal'] = (prev_indicator < prev_lower) & (prev_prev_indicator >= prev_prev_lower)

        # Make sure boolean columns are boolean even if all False after dropna
        df['buy_signal'] = df['buy_signal'].astype(bool)
        df['sell_signal'] = df['sell_signal'].astype(bool)

        return df

    def _run_backtest(self, signal_df: pd.DataFrame, price_col: str, trading_type: str,
                      long_entry_pct_cash: float, short_entry_pct_cash: float, day1_position: str,
                      strategy_type: int = 1) -> tuple:
        """
        Runs the backtest simulation based on the generated signals.
        
        Args:
            signal_df (pd.DataFrame): DataFrame with buy/sell signals and price data.
            price_col (str): Column name for price data.
            trading_type (str): Trading type ('long', 'short', or 'mixed').
            long_entry_pct_cash (float): Percentage of cash to use for long entries.
            short_entry_pct_cash (float): Percentage of cash to use for short entries.
            day1_position (str): Initial position on day 1 ('none', 'long', or 'short').
            
        Returns:
            tuple: (portfolio_log, end_state)
                - portfolio_log: List of portfolio state snapshots.
                - end_state: DataFrame with final portfolio state.
        """
        # Initialize portfolio tracking
        portfolio_log = []
        cash = self.initial_cash
        position_size = 0  # Number of shares/contracts
        position_value = 0.0  # Market value of position
        position_type = 'none'  # 'none', 'long', or 'short'
        commission_paid = 0.0
        
        # Handle day1_position if not 'none'
        if day1_position != 'none' and not signal_df.empty:
            first_price = signal_df[price_col].iloc[0]
            
            if day1_position == 'long':
                # Calculate shares to buy
                shares_to_buy = int((cash * long_entry_pct_cash) / first_price)
                if shares_to_buy > 0:
                    # Calculate commission for long position
                    commission = shares_to_buy * first_price * self.commission_long
                    # Update portfolio
                    cash -= (shares_to_buy * first_price + commission)
                    position_size = shares_to_buy
                    position_value = shares_to_buy * first_price
                    position_type = 'long'
                    commission_paid += commission
            
            elif day1_position == 'short':
                # Calculate shares to short
                shares_to_short = int((cash * short_entry_pct_cash) / first_price)
                if shares_to_short > 0:
                    # Calculate commission for short position
                    commission = shares_to_short * first_price * self.commission_short
                    # When shorting, we receive cash from the sale, minus commission
                    cash += (shares_to_short * first_price - commission)
                    position_size = -shares_to_short  # Negative for short positions
                    position_value = abs(position_size) * first_price  # Use absolute size to calculate positive liability value
                    position_type = 'short'
                    commission_paid += commission
        
        # Process each day's signals and update portfolio
        for i, (date, row) in enumerate(signal_df.iterrows()):
            current_price = row[price_col]
            buy_signal = row.get('buy_signal', False)
            sell_signal = row.get('sell_signal', False)
            
            # Store start-of-day state for fee calculation
            start_of_day_position_type = position_type
            start_of_day_position_value = position_value # Value based on *previous* day's close
            
            # Apply borrow fees based on START of day position
            # This ensures the fee for holding overnight is applied even if position is closed today
            short_fee = 0.0
            long_fee = 0.0
            
            if start_of_day_position_type == 'short':
                # Apply short borrow fee rate
                # Use the positive position value from the start of the day (liability)
                short_fee = start_of_day_position_value * self.short_borrow_fee_inc_rate
                cash -= short_fee # Deduct fee from cash immediately
            
            elif start_of_day_position_type == 'long':
                # Apply incremental long borrow fee rate (typically for ETFs/leveraged positions)
                long_fee = start_of_day_position_value * self.long_borrow_fee_inc_rate
                cash -= long_fee # Deduct fee from cash immediately
            
            # Update position value based on the current day's closing price
            if position_type == 'long':
                position_value = position_size * current_price
            elif position_type == 'short':
                # Use absolute size to calculate positive liability value
                position_value = abs(position_size) * current_price
            else: # 'none'
                position_value = 0.0
            
            # Process signals based on trading_type
            if trading_type == 'long':
                # Long-only trading
                if buy_signal and position_type != 'long':
                    # Buy signal and not already long
                    shares_to_buy = int((cash * long_entry_pct_cash) / current_price)
                    if shares_to_buy > 0:
                        commission = shares_to_buy * current_price * self.commission_long
                        cash -= (shares_to_buy * current_price + commission)
                        position_size = shares_to_buy
                        position_value = shares_to_buy * current_price
                        position_type = 'long'
                        commission_paid += commission
                        action = 'BUY'  # Set action when buy is executed
                    else:
                        action = 'HOLD'
                
                elif sell_signal and position_type == 'long':
                    # Check for conflicting signals
                    if buy_signal:  # Both buy AND sell signals - conflicting
                        action = 'HOLD_CONFLICTING_SIGNAL'
                    else:
                        # Sell signal and currently long
                        commission = position_value * self.commission_long
                        cash += (position_value - commission)
                        position_size = 0
                        position_value = 0.0
                        position_type = 'none'
                        commission_paid += commission
                        action = 'SELL'  # Set action when sell is executed
                else:
                    action = 'HOLD'
            
            elif trading_type == 'short':
                # Short-only trading
                if sell_signal and position_type != 'short':
                    # Sell signal and not already short
                    shares_to_short = int((cash * short_entry_pct_cash) / current_price)
                    if shares_to_short > 0:
                        commission = shares_to_short * current_price * self.commission_short
                        # When shorting, we receive cash from the sale
                        cash += (shares_to_short * current_price - commission)
                        position_size = -shares_to_short
                        position_value = abs(position_size) * current_price
                        position_type = 'short'
                        commission_paid += commission
                        action = 'SHORT'
                    else:
                        action = 'HOLD'
                
                elif buy_signal and position_type == 'short':
                    # Check for conflicting signals
                    if sell_signal:  # Both buy AND sell signals - conflicting
                        action = 'HOLD_CONFLICTING_SIGNAL'
                    else:
                        # Buy signal and currently short (cover)
                        commission = position_value * self.commission_short
                        # When covering, we pay to buy back the shares
                        cash -= (position_value + commission)
                        position_size = 0
                        position_value = 0.0
                        position_type = 'none'
                        commission_paid += commission
                        action = 'COVER'
                else:
                    action = 'HOLD'
            
            elif trading_type == 'mixed':
                # Mixed long/short trading
                if buy_signal:
                    # Track previous position for combined actions
                    prev_position_type = position_type
                    
                    if position_type == 'short':
                        # Cover short position
                        # When covering, we need to pay to buy back the shares we borrowed
                        commission = position_value * self.commission_short
                        cash -= (position_value + commission)
                        commission_paid += commission
                        position_size = 0
                        position_value = 0.0
                        position_type = 'none'
                    
                    # Then go long (if not already long)
                    if position_type != 'long':
                        shares_to_buy = int((cash * long_entry_pct_cash) / current_price)
                        if shares_to_buy > 0:
                            commission = shares_to_buy * current_price * self.commission_long
                            cash -= (shares_to_buy * current_price + commission)
                            position_size = shares_to_buy
                            position_value = shares_to_buy * current_price
                            position_type = 'long'
                            commission_paid += commission
                            
                            # Set appropriate action based on previous position
                            if prev_position_type == 'short':
                                action = 'COVER AND BUY'
                            else:  # prev_position_type was 'none'
                                action = 'BUY'
                        else:
                            # Just cover if no shares bought
                            if prev_position_type == 'short':
                                action = 'COVER'
                            else:
                                action = 'HOLD'
                    else:
                        action = 'HOLD'  # Already long
                
                elif sell_signal:
                    # Track previous position for combined actions
                    prev_position_type = position_type
                    
                    if position_type == 'long':
                        # Sell long position
                        commission = position_value * self.commission_long
                        cash += (position_value - commission)
                        commission_paid += commission
                        position_size = 0
                        position_value = 0.0
                        position_type = 'none'
                    
                    # Then go short (if not already short)
                    if position_type != 'short':
                        shares_to_short = int((cash * short_entry_pct_cash) / current_price)
                        if shares_to_short > 0:
                            commission = shares_to_short * current_price * self.commission_short
                            # When shorting, we receive cash from the sale of borrowed shares
                            cash += (shares_to_short * current_price - commission)
                            position_size = -shares_to_short
                            position_value = abs(position_size) * current_price
                            position_type = 'short'
                            commission_paid += commission
                            
                            # Set appropriate action based on previous position
                            if prev_position_type == 'long':
                                action = 'SELL AND SHORT'
                            else:  # prev_position_type was 'none'
                                action = 'SHORT'
                        else:
                            # Just sell if no shares shorted
                            if prev_position_type == 'long':
                                action = 'SELL'
                            else:
                                action = 'HOLD'
                    else:
                        action = 'HOLD'  # Already short
                else:
                    action = 'HOLD'  # No buy or sell signal
            
            # Make sure action is set to HOLD if it hasn't been set by any of the trading logic
            if 'action' not in locals():
                action = 'HOLD'
            
            # Calculate portfolio value
            portfolio_value = cash
            if position_type == 'long':
                portfolio_value += position_value
            elif position_type == 'short':
                # For short positions, subtract the positive position value (liability)
                portfolio_value -= position_value
            
            # Create a snapshot of the current state
            snapshot = {
                'Date': date,
                'Price': current_price,
                'Close': current_price,  # Add Close column for compatibility with compute_benchmark_return
                'Cash': cash,
                'PositionSize': position_size,
                'PositionValue': position_value,
                'PositionType': position_type,
                'PortfolioValue': portfolio_value,
                'CommissionPaid': commission_paid,
                'ShortFee': short_fee,
                'LongFee': long_fee,
                'BuySignal': buy_signal,
                'SellSignal': sell_signal,
                'Action': action
            }
            
            # Add snapshot to portfolio log
            portfolio_log.append(snapshot)
        
        # Create end state DataFrame
        if portfolio_log:
            end_state = pd.DataFrame(portfolio_log)
            end_state.set_index('Date', inplace=True)
        else:
            # Create empty DataFrame with expected columns if no log entries
            end_state = pd.DataFrame(columns=['Price', 'Close', 'Cash', 'PositionSize', 'PositionValue', 
                                             'PositionType', 'PortfolioValue', 'CommissionPaid', 'ShortFee', 'LongFee',
                                             'BuySignal', 'SellSignal', 'Action'])
        
        return portfolio_log, end_state
        
    def _prepare_results(self, portfolio_log: list, final_df: pd.DataFrame, indicator_col: str, upper_band_col: str, lower_band_col: str,
                         strategy_type: int, trading_type: str, day1_position: str, risk_free_rate: float) -> tuple:
        portfolio_df = pd.DataFrame(portfolio_log).set_index('Date')
        portfolio_df = portfolio_df.drop(columns=['Cash']) # Drop the cash column
        
        # Calculate benchmark and improved results
        benchmark_results = self.compute_benchmark_return(final_df, price_col='Close')
        improved_results = self.calculate_performance_metrics(portfolio_df, risk_free_rate)

        # Calculate total fees
        total_short_fees = portfolio_df['ShortFee'].sum() if 'ShortFee' in portfolio_df.columns else 0
        total_long_fees = portfolio_df['LongFee'].sum() if 'LongFee' in portfolio_df.columns else 0
        total_fees = total_short_fees + total_long_fees

        # Merge all benchmark results
        results = {
            "strategy": f"Band Trade ({indicator_col} vs {lower_band_col}/{upper_band_col} - {'Mean Reversion' if strategy_type == 1 else 'Breakout'}){' [Shorts Allowed]' if trading_type in ['short', 'mixed'] else ''}{' [Day1 ' + day1_position.capitalize() + ']' if day1_position != 'none' else ''}",
            "indicator_col": indicator_col,
            "upper_band_col": upper_band_col,
            "lower_band_col": lower_band_col,
            "strategy_type": strategy_type,
            "initial_cash": self.initial_cash,
            "final_value": round(portfolio_df['PortfolioValue'].iloc[-1], 2),
            "total_return_pct": round(((portfolio_df['PortfolioValue'].iloc[-1] - self.initial_cash) / self.initial_cash) * 100, 2),
            "num_trades": portfolio_df['Action'].value_counts().get('BUY', 0) + portfolio_df['Action'].value_counts().get('SELL', 0) + portfolio_df['Action'].value_counts().get('SHORT', 0) + portfolio_df['Action'].value_counts().get('COVER', 0),
            "total_short_fees": round(total_short_fees, 2),
            "total_long_fees": round(total_long_fees, 2),
            "total_borrow_fees": round(total_fees, 2),
            }
        results.update(benchmark_results)
        results.update(improved_results)

        return results, portfolio_df

    def print_results(self, results: dict, detailed: bool = True):
        """
        Prints the backtest results in a nicely formatted way.
        
        Args:
            results (dict): The dictionary of backtest results.
            detailed (bool): Whether to print detailed metrics or just basic results.
        """
        print("\n" + "="*60)
        print(f"✨ {results['strategy']} ✨".center(60))
        print("="*60)
        
        # Time period information
        if 'start_date' in results and 'end_date' in results:
            print("\n🗓️ BACKTEST PERIOD:")
            start_date = results['start_date'].strftime('%Y-%m-%d') if hasattr(results['start_date'], 'strftime') else results['start_date']
            end_date = results['end_date'].strftime('%Y-%m-%d') if hasattr(results['end_date'], 'strftime') else results['end_date']
            print(f"  • Period: {start_date} to {end_date}")
            
            if 'duration_days' in results:
                print(f"  • Duration: {results['duration_days']} days")
            if 'days_in_backtest' in results:
                print(f"  • Trading Days: {results['days_in_backtest']}")
        
        # Basic metrics section
        print("\n📊 BASIC METRICS:")
        print(f"  • Initial Investment: ${results['initial_cash']:,.2f}")
        print(f"  • Final Portfolio Value: ${results['final_value']:,.2f}")
        print(f"  • Total Return: {results['total_return_pct']:,.2f}%")
        if 'annualized_return_pct' in results:
            print(f"  • Annualized Return: {results['annualized_return_pct']:,.2f}%")
        print(f"  • Number of Trades: {results['num_trades']}")
        if 'total_commissions' in results and results['total_commissions'] is not None:
            print(f"  • Total Commissions: ${results['total_commissions']:,.2f}")
        
        # Benchmark comparison
        if 'benchmark_return_pct' in results:
            print("\n📈 BENCHMARK COMPARISON:")
            print(f"  • Benchmark Return: {results['benchmark_return_pct']:,.2f}%")
            print(f"  • Benchmark Final Value: ${results['benchmark_final_value']:,.2f}")
            outperf = results['total_return_pct'] - results['benchmark_return_pct']
            outperf_sign = "+" if outperf >= 0 else ""
            print(f"  • Strategy vs Benchmark: {outperf_sign}{outperf:,.2f}%")
        
        # Only print detailed metrics if requested
        if detailed:
            # Risk metrics
            has_risk_metrics = any(metric in results for metric in 
                               ['sharpe_ratio', 'sortino_ratio', 'max_drawdown_pct', 'annualized_volatility_pct'])
            
            if has_risk_metrics:
                print("\n📉 RISK METRICS:")
                if 'sharpe_ratio' in results:
                    print(f"  • Sharpe Ratio: {results['sharpe_ratio']:,.3f}")
                if 'sortino_ratio' in results:
                    print(f"  • Sortino Ratio: {results['sortino_ratio']:,.3f}")
                if 'max_drawdown_pct' in results:
                    print(f"  • Maximum Drawdown: {results['max_drawdown_pct']:,.2f}%")
                if 'avg_drawdown_pct' in results:
                    print(f"  • Average Drawdown: {results['avg_drawdown_pct']:,.2f}%")
                if 'max_drawdown_duration_days' in results:
                    print(f"  • Max Drawdown Duration: {results['max_drawdown_duration_days']} days")
                if 'avg_drawdown_duration_days' in results:
                    print(f"  • Avg Drawdown Duration: {results['avg_drawdown_duration_days']} days")
                if 'annualized_volatility_pct' in results:
                    print(f"  • Annualized Volatility: {results['annualized_volatility_pct']:,.2f}%")
            
            # Fee information
            if 'total_borrow_fees' in results and results['total_borrow_fees'] > 0:
                print("\n💰 BORROW FEES:")
                if 'total_short_fees' in results and results['total_short_fees'] > 0:
                    print(f"  • Total Short Borrow Fees: ${results['total_short_fees']:,.2f}")
                if 'total_long_fees' in results and results['total_long_fees'] > 0:
                    print(f"  • Total Long Borrow Fees: ${results['total_long_fees']:,.2f}")
                print(f"  • Total Borrow Fees: ${results['total_borrow_fees']:,.2f}")
        
        print("\n" + "="*60)