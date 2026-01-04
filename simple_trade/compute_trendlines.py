import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


def find_trendline_points(data: pd.Series, window: int = 10) -> Tuple[pd.Series, pd.Series]:
    """Find pivot points suitable for trendline detection.
    
    Args:
        data (pd.Series): Time series data (e.g., close prices).
        window (int): The number of periods to look on each side to determine a pivot.
    
    Returns:
        tuple: A tuple containing two Series (pivot_highs, pivot_lows).
    """
    if not isinstance(data, pd.Series):
        raise TypeError("Input 'data' must be a pandas Series.")
    
    rolling_window_size = 2 * window + 1
    
    # Find local maxima
    local_max = data.rolling(window=rolling_window_size, center=True, min_periods=1).max()
    is_pivot_high = (data == local_max)
    
    # Find local minima
    local_min = data.rolling(window=rolling_window_size, center=True, min_periods=1).min()
    is_pivot_low = (data == local_min)
    
    pivot_highs = data[is_pivot_high]
    pivot_lows = data[is_pivot_low]
    
    return pivot_highs, pivot_lows


def calculate_trendline(x_points: np.ndarray, y_points: np.ndarray) -> Tuple[float, float]:
    """Calculate the slope and intercept of a trendline using linear regression.
    
    Args:
        x_points (np.ndarray): X coordinates (typically time indices).
        y_points (np.ndarray): Y coordinates (prices).
    
    Returns:
        tuple: (slope, intercept) of the trendline.
    """
    if len(x_points) < 2:
        return 0.0, 0.0
    
    # Convert to numeric indices for calculation
    x_numeric = np.arange(len(x_points))
    
    # Linear regression
    coefficients = np.polyfit(x_numeric, y_points, 1)
    slope, intercept = coefficients[0], coefficients[1]
    
    return slope, intercept


def find_best_trendlines(data: pd.Series, window: int = 10, 
                         min_touches: int = 3, 
                         tolerance: float = 0.02) -> Tuple[pd.DataFrame, List[dict], List[dict]]:
    """Find the best uptrend and downtrend lines based on pivot points.
    
    Args:
        data (pd.Series): The price data.
        window (int): The number of periods to look on each side to determine a pivot.
        min_touches (int): Minimum number of pivot points required to form a trendline.
        tolerance (float): The percentage tolerance for points to be considered on the line.
    
    Returns:
        tuple: A DataFrame with Close and Trendline columns, and two lists containing 
               uptrend and downtrend line dictionaries.
    """
    pivot_highs, pivot_lows = find_trendline_points(data, window)
    
    uptrend_lines = []
    downtrend_lines = []
    
    # Find uptrend lines (connecting pivot lows)
    if len(pivot_lows) >= min_touches:
        uptrend_lines = _find_trendlines_from_pivots(
            pivot_lows, data, min_touches, tolerance, trend_type='up'
        )
    
    # Find downtrend lines (connecting pivot highs)
    if len(pivot_highs) >= min_touches:
        downtrend_lines = _find_trendlines_from_pivots(
            pivot_highs, data, min_touches, tolerance, trend_type='down'
        )
    
    # Create DataFrame with Close and Trendline columns
    result_df = pd.DataFrame({'Close': data})
    
    # Combine all pivot points and sort by index for zigzag trendline
    all_pivots = pd.concat([pivot_highs, pivot_lows]).sort_index()
    all_pivots = all_pivots[~all_pivots.index.duplicated(keep='first')]
    
    # Create Pivot column with only the edge points (NaN elsewhere)
    result_df['Pivot'] = np.nan
    result_df.loc[all_pivots.index, 'Pivot'] = all_pivots.values
    
    # Create Trendline column with interpolated values between pivots
    result_df['Trendline'] = result_df['Pivot'].interpolate(method='index')
    
    return result_df, uptrend_lines, downtrend_lines


def _find_trendlines_from_pivots(pivots: pd.Series, data: pd.Series, 
                                  min_touches: int, tolerance: float,
                                  trend_type: str) -> List[dict]:
    """Helper function to find trendlines from pivot points.
    
    Args:
        pivots (pd.Series): Pivot points to analyze.
        data (pd.Series): Original price data.
        min_touches (int): Minimum number of touches required.
        tolerance (float): Percentage tolerance for line fitting.
        trend_type (str): Either 'up' or 'down'.
    
    Returns:
        list: List of trendline dictionaries.
    """
    trendlines = []
    pivot_indices = pivots.index.tolist()
    pivot_values = pivots.values
    
    # Try to find trendlines by iterating through combinations
    for i in range(len(pivot_indices) - min_touches + 1):
        for j in range(i + min_touches - 1, len(pivot_indices)):
            # Get the subset of pivots
            subset_indices = pivot_indices[i:j+1]
            subset_values = pivot_values[i:j+1]
            
            # Calculate trendline
            x_numeric = np.arange(len(subset_indices))
            slope, intercept = np.polyfit(x_numeric, subset_values, 1)
            
            # Check if slope direction matches trend type
            if trend_type == 'up' and slope < 0:
                continue
            if trend_type == 'down' and slope > 0:
                continue
            
            # Count touches (points close to the line)
            touches = 0
            for k, (idx, val) in enumerate(zip(subset_indices, subset_values)):
                predicted = slope * k + intercept
                if abs(val - predicted) <= abs(predicted * tolerance):
                    touches += 1
            
            # If enough touches, add to trendlines
            if touches >= min_touches:
                # Calculate line endpoints
                start_idx = subset_indices[0]
                end_idx = subset_indices[-1]
                start_price = slope * 0 + intercept
                end_price = slope * (len(subset_indices) - 1) + intercept
                
                trendlines.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_price': start_price,
                    'end_price': end_price,
                    'slope': slope,
                    'intercept': intercept,
                    'touches': touches,
                    'pivot_indices': subset_indices
                })
    
    # Sort by number of touches and return best ones
    trendlines.sort(key=lambda x: x['touches'], reverse=True)
    
    # Filter to avoid overlapping lines (keep the best ones)
    filtered_lines = []
    for line in trendlines:
        is_unique = True
        for existing in filtered_lines:
            # Check if lines are too similar
            if (abs(line['slope'] - existing['slope']) < 0.01 and
                abs(line['intercept'] - existing['intercept']) < data.mean() * 0.01):
                is_unique = False
                break
        if is_unique:
            filtered_lines.append(line)
            if len(filtered_lines) >= 3:  # Limit to top 3 lines
                break
    
    return filtered_lines


def plot_trendlines(data: pd.Series, window: int = 10, 
                   min_touches: int = 3, tolerance: float = 0.02):
    """Plot the price data with zigzag trendlines connecting pivot points.
    
    Args:
        data (pd.Series): The price data.
        window (int): The number of periods to look on each side to determine a pivot.
        min_touches (int): Minimum number of pivot points required (not used for zigzag).
        tolerance (float): The percentage tolerance (not used for zigzag).
    """
    pivot_highs, pivot_lows = find_trendline_points(data, window)
    
    # Combine all pivot points and sort by index
    all_pivots = pd.concat([pivot_highs, pivot_lows]).sort_index()
    
    # Remove duplicate indices (keep first occurrence)
    all_pivots = all_pivots[~all_pivots.index.duplicated(keep='first')]
    
    plt.figure(figsize=(15, 8))
    plt.plot(data, label='Price', linewidth=1.5, color='black')
    
    # Plot pivot points
    plt.scatter(pivot_highs.index, pivot_highs, color='red', marker='^', 
                s=50, alpha=0.8, label='Pivot Highs', zorder=5)
    plt.scatter(pivot_lows.index, pivot_lows, color='green', marker='v', 
                s=50, alpha=0.8, label='Pivot Lows', zorder=5)
    
    # Plot zigzag line connecting all pivots in succession
    if len(all_pivots) >= 2:
        plt.plot(all_pivots.index, all_pivots.values, color='red', linestyle='-', 
                linewidth=2, alpha=0.8, label='Trendline', zorder=4)
    
    plt.title('Automatic Trendlines')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
