import pandas as pd
import numpy as np
from .backtesting import Backtester

class CombineTradeBacktester(Backtester):
    """
    Backtester that combines trading signals from multiple strategy DataFrames.
    The final trading decision is based on an 'AND' logic applied to the 'PositionType'
    from each input DataFrame.
    """

    def __init__(self, **kwargs):
        """Initializes the CombineTradeBacktester."""
        super().__init__(**kwargs)

    def run_combined_trade(self, portfolio_dfs: list[pd.DataFrame], price_data: pd.DataFrame,
                             price_col: str = 'Close', long_entry_pct_cash: float = 0.9,
                             short_entry_pct_cash: float = 0.9, trading_type: str = 'long',
                             risk_free_rate: float = 0.0, combination_logic: str = 'unanimous') -> tuple:
        """
        Runs a backtest based on combined signals from multiple portfolio DataFrames.

        Args:
            portfolio_dfs (list[pd.DataFrame]): A list of portfolio DataFrames from other backtests.
                                                 Each DataFrame must contain a 'PositionType' column
                                                 and have a DatetimeIndex.
            price_data (pd.DataFrame): DataFrame with historical price data, including the `price_col`.
                                       Must have a DatetimeIndex.
            price_col (str): Column name for trade execution prices (default: 'Close').
            long_entry_pct_cash (float): Pct of cash for long entries (default: 0.9).
            short_entry_pct_cash (float): Pct of cash for short entries (default: 0.1).
            trading_type (str): Defines trading behavior ('long', 'short', 'mixed'). Default: 'mixed'.
            risk_free_rate (float): Annual risk-free rate for Sharpe ratio (default: 0.0).

        Returns:
            tuple: A tuple containing:
                - dict: Dictionary with backtest summary results.
                - pd.DataFrame: DataFrame tracking daily portfolio evolution.
        """
        # --- Input Validation ---
        if combination_logic not in ['unanimous', 'majority']:
            raise ValueError("combination_logic must be either 'unanimous' or 'majority'.")
        if not isinstance(portfolio_dfs, list) or not portfolio_dfs:
            raise ValueError("portfolio_dfs must be a non-empty list of DataFrames.")
        if not isinstance(price_data, pd.DataFrame) or not isinstance(price_data.index, pd.DatetimeIndex):
            raise TypeError("price_data must be a DataFrame with a DatetimeIndex.")
        if price_col not in price_data.columns:
            raise ValueError(f"Price column '{price_col}' not found in price_data.")

        # --- Signal Combination ---
        df = self._combine_signals(portfolio_dfs, price_data, price_col, combination_logic)

        if df.empty:
            return self._get_empty_results(), pd.DataFrame()

        # --- Run Backtest ---
        portfolio_log, end_state = self._run_backtest_loop(
            signal_df=df,
            price_col=price_col,
            trading_type=trading_type,
            long_entry_pct_cash=long_entry_pct_cash,
            short_entry_pct_cash=short_entry_pct_cash
        )

        # --- Prepare and Return Results ---
        results, portfolio_df = self._prepare_results(
            portfolio_log=portfolio_log,
            final_df=end_state,
            original_data=price_data,
            price_col=price_col,
            risk_free_rate=risk_free_rate,
            trading_type=trading_type
        )

        return results, portfolio_df

    def _combine_signals(self, portfolio_dfs: list[pd.DataFrame], price_data: pd.DataFrame, price_col: str, combination_logic: str) -> pd.DataFrame:
        """
        Merges PositionType from multiple DataFrames and generates final buy/sell signals
        based on the specified combination logic.
        """
        combined_df = price_data[[price_col]].copy()

        for i, portfolio_df in enumerate(portfolio_dfs):
            if 'PositionType' not in portfolio_df.columns:
                raise ValueError(f"DataFrame at index {i} is missing 'PositionType' column.")
            if not isinstance(portfolio_df.index, pd.DatetimeIndex):
                 raise TypeError(f"Index of DataFrame at index {i} must be a DatetimeIndex.")

            position_col = portfolio_df[['PositionType']].rename(columns={'PositionType': f'PositionType_{i}'})
            combined_df = combined_df.join(position_col, how='left')

        position_cols = [f'PositionType_{i}' for i in range(len(portfolio_dfs))]
        combined_df[position_cols] = combined_df[position_cols].ffill()
        combined_df.dropna(inplace=True)

        if combined_df.empty:
            return pd.DataFrame()

        def get_final_position(row):
            positions = [row[col] for col in position_cols]

            if combination_logic == 'unanimous':
                is_all_long = all(p == 'long' for p in positions)
                is_all_short = all(p == 'short' for p in positions)
                if is_all_long:
                    return 'long'
                elif is_all_short:
                    return 'short'
                else:
                    return 'none'

            elif combination_logic == 'majority':
                long_votes = positions.count('long')
                short_votes = positions.count('short')
                if long_votes > short_votes:
                    return 'long'
                elif short_votes > long_votes:
                    return 'short'
                else:  # Tie or only 'none' votes
                    return 'none'

            return 'none'  # Should not be reached if logic is validated in run_combined_trade

        combined_df['PositionType'] = combined_df.apply(get_final_position, axis=1)
        combined_df['prev_PositionType'] = combined_df['PositionType'].shift(1).fillna('none')

        buy_entry = (combined_df['PositionType'] == 'long') & (combined_df['prev_PositionType'] != 'long')
        short_exit = (combined_df['PositionType'] == 'long') & (combined_df['prev_PositionType'] == 'short')
        combined_df['buy_signal'] = buy_entry | short_exit

        short_entry = (combined_df['PositionType'] == 'short') & (combined_df['prev_PositionType'] != 'short')
        long_exit = (combined_df['PositionType'] != 'long') & (combined_df['prev_PositionType'] == 'long')
        combined_df['sell_signal'] = short_entry | long_exit

        return combined_df

    def _run_backtest_loop(self, signal_df: pd.DataFrame, price_col: str, trading_type: str,
                           long_entry_pct_cash: float, short_entry_pct_cash: float) -> tuple:
        """Runs the backtest simulation loop (Adapted from BandTradeBacktester)."""
        portfolio_log = []
        cash = self.initial_cash
        position_size = 0
        position_value = 0.0
        position_type = 'none'

        for date, row in signal_df.iterrows():
            current_price = row[price_col]
            buy_signal = row['buy_signal']
            sell_signal = row['sell_signal']
            action = 'HOLD'
            commission_paid = 0.0

            start_of_day_position_type = position_type
            start_of_day_position_value = abs(position_size) * current_price

            short_fee = 0.0
            long_fee = 0.0
            if start_of_day_position_type == 'short':
                short_fee = start_of_day_position_value * self.short_borrow_fee_inc_rate
                cash -= short_fee
            elif start_of_day_position_type == 'long':
                long_fee = start_of_day_position_value * self.long_borrow_fee_inc_rate
                cash -= long_fee

            if position_type == 'long':
                position_value = position_size * current_price
            elif position_type == 'short':
                position_value = abs(position_size) * current_price
            else:
                position_value = 0.0

            if trading_type == 'long':
                if buy_signal and position_type != 'long':
                    shares_to_buy = int((cash * long_entry_pct_cash) / current_price)
                    if shares_to_buy > 0:
                        commission = shares_to_buy * current_price * self.commission_long
                        cash -= (shares_to_buy * current_price + commission)
                        position_size = shares_to_buy
                        position_value = shares_to_buy * current_price
                        position_type = 'long'
                        commission_paid += commission
                        action = 'BUY'
                elif sell_signal and position_type == 'long':
                    commission = position_value * self.commission_long
                    cash += (position_value - commission)
                    position_size = 0
                    position_value = 0.0
                    position_type = 'none'
                    commission_paid += commission
                    action = 'SELL'

            elif trading_type == 'short':
                if sell_signal and position_type != 'short':
                    shares_to_short = int((cash * short_entry_pct_cash) / current_price)
                    if shares_to_short > 0:
                        commission = shares_to_short * current_price * self.commission_short
                        cash += (shares_to_short * current_price - commission)
                        position_size = -shares_to_short
                        position_value = abs(position_size) * current_price
                        position_type = 'short'
                        commission_paid += commission
                        action = 'SHORT'
                elif buy_signal and position_type == 'short':
                    commission = position_value * self.commission_short
                    cash -= (position_value + commission)
                    position_size = 0
                    position_value = 0.0
                    position_type = 'none'
                    commission_paid += commission
                    action = 'COVER'

            elif trading_type == 'mixed':
                if buy_signal:
                    prev_position_type = position_type
                    if position_type == 'short':
                        commission = position_value * self.commission_short
                        cash -= (position_value + commission)
                        commission_paid += commission
                        position_size = 0
                        position_value = 0.0
                        position_type = 'none'

                    if position_type != 'long':
                        shares_to_buy = int((cash * long_entry_pct_cash) / current_price)
                        if shares_to_buy > 0:
                            commission = shares_to_buy * current_price * self.commission_long
                            cash -= (shares_to_buy * current_price + commission)
                            position_size = shares_to_buy
                            position_value = shares_to_buy * current_price
                            position_type = 'long'
                            commission_paid += commission
                            action = 'COVER AND BUY' if prev_position_type == 'short' else 'BUY'

                elif sell_signal:
                    prev_position_type = position_type
                    if position_type == 'long':
                        commission = position_value * self.commission_long
                        cash += (position_value - commission)
                        commission_paid += commission
                        position_size = 0
                        position_value = 0.0
                        position_type = 'none'

                    if position_type != 'short':
                        shares_to_short = int((cash * short_entry_pct_cash) / current_price)
                        if shares_to_short > 0:
                            commission = shares_to_short * current_price * self.commission_short
                            cash += (shares_to_short * current_price - commission)
                            position_size = -shares_to_short
                            position_value = abs(position_size) * current_price
                            position_type = 'short'
                            commission_paid += commission
                            action = 'SELL AND SHORT' if prev_position_type == 'long' else 'SHORT'

            portfolio_value = cash
            if position_type == 'long':
                portfolio_value += position_value
            elif position_type == 'short':
                portfolio_value -= position_value

            portfolio_log.append({
                'Date': date,
                'Close': current_price,
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
            })

        end_state = pd.DataFrame(portfolio_log).set_index('Date') if portfolio_log else pd.DataFrame()
        return portfolio_log, end_state

    def _prepare_results(self, portfolio_log: list, final_df: pd.DataFrame, original_data: pd.DataFrame,
                         price_col: str, risk_free_rate: float, trading_type: str) -> tuple:
        """Prepares the final results dictionary and portfolio DataFrame (Adapted from BandTradeBacktester)."""
        if not portfolio_log:
            return self._get_empty_results(), pd.DataFrame()

        portfolio_df = pd.DataFrame(portfolio_log).set_index('Date')
        num_trades = len(portfolio_df[portfolio_df['Action'].isin(['BUY', 'SELL', 'SHORT', 'COVER', 'COVER AND BUY', 'SELL AND SHORT'])])

        performance_metrics = self.calculate_performance_metrics(portfolio_df, risk_free_rate)
        benchmark_metrics = self.compute_benchmark_return(original_data, price_col)

        results = {
            "strategy": f"Combined Strategy ({trading_type})",
            "initial_cash": self.initial_cash,
            "final_value": portfolio_df['PortfolioValue'].iloc[-1],
            "num_trades": num_trades,
        }
        results.update(performance_metrics)
        results.update(benchmark_metrics)

        return results, portfolio_df

    def _get_empty_results(self) -> dict:
        """Returns a dictionary for an empty/failed backtest."""
        return {
            "error": "Could not run combined backtest. Check input DataFrames.",
            "strategy": "Combined Strategy",
            "initial_cash": self.initial_cash,
            "final_value": self.initial_cash,
            "total_return_pct": 0.0,
            "num_trades": 0,
        }