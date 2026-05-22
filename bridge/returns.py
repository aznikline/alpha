"""Forward-return alignment — computes future returns and aligns with signal dates.

Critical concept: A factor signal on date D predicts returns AFTER date D.
Same-day returns (D to D) would be leakage — the signal was already known
when the return happened. Forward returns use D+1 (or D+N) as the start date.

Convention:
    signal.loc["2024-01-15", "000001.SZ"] = 0.34  (factor signal on entry date)
    forward_return.loc["2024-01-15", "000001.SZ"] = close["2024-01-16"] / close["2024-01-15"] - 1
    (return realized AFTER the signal date)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_forward_returns(
    price: pd.DataFrame,
    periods: int = 1,
    price_column: str | None = None,
) -> pd.DataFrame:
    """Compute forward (future) returns from a price DataFrame.
    
    Args:
        price: (Date, Stock) DataFrame with price data.
            If MultiIndex columns (stock, field), use price_column to select.
            If plain columns (stock codes), values are assumed to be the price.
        periods: Number of periods to look forward. Default 1 (next-day return).
        price_column: If price has MultiIndex columns, the field name to use
            (e.g. "close" or "vwap"). If None, assumes plain stock-code columns.
    
    Returns:
        pd.DataFrame: (Date, Stock) forward returns, aligned with signal dates.
            - Index: same dates as input price (the SIGNAL/ENTRY dates)
            - Values: return from D+periods relative to D
            - For 1-period: value at date D = price[D+1] / price[D] - 1
            - NaN for the last `periods` dates (no future price available)
            - dtype: float32
    """
    if price_column is not None and isinstance(price.columns, pd.MultiIndex):
        prices = price.xs(price_column, level=1, axis=1)
    else:
        prices = price
    
    # Compute forward return: price[D+periods] / price[D] - 1
    # shift(-periods) brings future prices to current row
    future_prices = prices.shift(-periods)
    forward_returns = (future_prices / prices - 1.0).astype(np.float32)
    
    return forward_returns


def align_signals_with_returns(
    signal_data: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = "inner",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align signal and forward-return DataFrames on shared dates and stocks.
    
    Both DataFrames must be (Date, Stock) orientation. Alignment ensures:
    - Only dates present in BOTH signal and returns are kept
    - Only stocks present in BOTH signal and returns are kept
    - This prevents look-ahead bias (no future returns without signals)
    - And prevents missing returns (no signals without returns to validate)
    
    Args:
        signal_data: (Date, Stock) normalized factor values
        forward_returns: (Date, Stock) forward returns from compute_forward_returns()
        method: Alignment method:
            - "inner": Keep only dates/stocks in BOTH (strict, recommended)
            - "outer": Keep all dates/stocks, fill missing with NaN
    
    Returns:
        Tuple of (aligned_signal, aligned_returns) with matching shape.
        Both have the same DatetimeIndex and same column order.
    
    Raises:
        ValueError: If either DataFrame is empty.
    """
    if signal_data.empty or forward_returns.empty:
        raise ValueError("Cannot align empty DataFrames")
    
    # Find shared dates
    shared_dates = signal_data.index.intersection(forward_returns.index)
    if method == "inner" and len(shared_dates) == 0:
        raise ValueError("No overlapping dates between signal and returns — check date ranges")
    
    # Find shared stocks
    shared_stocks = signal_data.columns.intersection(forward_returns.columns)
    if method == "inner" and len(shared_stocks) == 0:
        raise ValueError("No overlapping stocks between signal and returns — check stock code formats")
    
    # Reindex both to shared dates and shared stocks
    aligned_signal = signal_data.reindex(index=shared_dates, columns=shared_stocks)
    aligned_returns = forward_returns.reindex(index=shared_dates, columns=shared_stocks)
    
    return aligned_signal, aligned_returns


def check_no_leakage(
    signal_data: pd.DataFrame,
    same_day_returns: pd.DataFrame,
) -> bool:
    """Check that signal is NOT correlated with same-day returns (leakage detection).
    
    If a factor signal on date D is correlated with the return on date D,
    that means the signal contains information about the same day's price move
    — which is look-ahead bias (leakage).
    
    A properly constructed signal should have LOW correlation with same-day returns
    and HIGH correlation with forward returns.
    
    Args:
        signal_data: (Date, Stock) factor values
        same_day_returns: (Date, Stock) same-day returns (price[D]/price[D-1]-1)
    
    Returns:
        True if no significant leakage detected (same-day IC < 0.03).
        False if potential leakage (same-day IC > 0.03).
    """
    aligned_signal, aligned_returns = align_signals_with_returns(
        signal_data, same_day_returns, method="inner"
    )
    
    # Compute per-date IC (correlation between signal and same-day return)
    daily_ic = aligned_signal.corrwith(aligned_returns, axis=1, method="pearson")
    mean_ic = daily_ic.mean()
    
    # If same-day IC is significant (> 0.03), that's suspicious
    return abs(mean_ic) < 0.03