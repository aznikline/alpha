"""IC/IR evaluator — Information Coefficient and Information Ratio for factor validation.

These metrics assess factor quality AFTER the bridge pipeline has produced signals.
They are research validation tools, NOT dispatch gates — a factor can be sent to qmt
regardless of its IC/IR. These metrics help researchers decide which factors are worth
keeping and which should be discarded.

IC (Information Coefficient):
    Per-date Pearson correlation between factor signal and forward return.
    IC > 0.03 is typically considered meaningful for A-share factors.

IR (Information Ratio):
    Mean IC / Std IC — measures consistency of predictive power.
    IR > 0.5 over 20+ dates suggests a stable factor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ICResult:
    """Result of IC/IR analysis for a factor.
    
    Attributes:
        daily_ic: Series of per-date IC values (DatetimeIndex)
        mean_ic: Mean of daily IC across all dates
        std_ic: Standard deviation of daily IC
        ir: Information Ratio = mean_ic / std_ic (0 if std_ic == 0)
        hit_rate: Fraction of dates with IC > 0 (positive predictive direction)
        n_dates: Number of dates in the analysis
        date_range: (start_date, end_date) tuple
    """
    daily_ic: pd.Series
    mean_ic: float
    std_ic: float
    ir: float
    hit_rate: float
    n_dates: int
    date_range: tuple[str, str]


def compute_ic(
    signal_data: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = "pearson",
) -> ICResult:
    """Compute per-date IC between factor signal and forward returns.
    
    IC = correlation(signal, forward_return) computed per row (each date).
    Positive IC means the factor correctly predicts the direction of future returns.
    
    Args:
        signal_data: (Date, Stock) normalized factor values
        forward_returns: (Date, Stock) forward returns (aligned with signals)
        method: Correlation method ("pearson" or "spearman")
    
    Returns:
        ICResult with daily IC series, mean IC, IR, and statistics.
    
    Note: Both DataFrames should already be aligned (same dates and stocks).
    Use bridge.returns.align_signals_with_returns() before calling this.
    """
    # Per-date correlation: corr between signal row and return row
    daily_ic = signal_data.corrwith(forward_returns, axis=1, method=method)
    daily_ic = daily_ic.dropna()
    
    if daily_ic.empty:
        return ICResult(
            daily_ic=pd.Series(dtype=np.float32),
            mean_ic=0.0,
            std_ic=0.0,
            ir=0.0,
            hit_rate=0.0,
            n_dates=0,
            date_range=("", ""),
        )
    
    mean_ic = float(daily_ic.mean())
    std_ic = float(daily_ic.std())
    
    ir = mean_ic / std_ic if std_ic != 0 else 0.0
    hit_rate = float((daily_ic > 0).sum() / len(daily_ic))
    
    date_range = (
        str(daily_ic.index.min().strftime("%Y-%m-%d")),
        str(daily_ic.index.max().strftime("%Y-%m-%d")),
    )
    
    return ICResult(
        daily_ic=daily_ic,
        mean_ic=mean_ic,
        std_ic=std_ic,
        ir=ir,
        hit_rate=hit_rate,
        n_dates=len(daily_ic),
        date_range=date_range,
    )


def filter_by_ic(
    signal_data: pd.DataFrame,
    forward_returns: pd.DataFrame,
    min_ic: float = 0.02,
    min_ir: float = 0.5,
    window: int = 20,
) -> tuple[pd.DataFrame, ICResult]:
    """Filter signal dates by rolling IC/IR thresholds.
    
    This is a RESEARCH tool — it removes dates where the factor had low
    predictive power. It does NOT prevent a factor from being dispatched
    to qmt (that decision is left to the researcher).
    
    Args:
        signal_data: (Date, Stock) normalized factor values
        forward_returns: (Date, Stock) aligned forward returns
        min_ic: Minimum rolling mean IC threshold (default 0.02)
        min_ir: Minimum rolling IR threshold (default 0.5)
        window: Rolling window size for IC/IR computation (default 20 dates)
    
    Returns:
        Tuple of (filtered_signal, ic_result):
            - filtered_signal: (Date, Stock) signal data with low-IC dates removed
            - ic_result: ICResult with full IC statistics
    """
    ic_result = compute_ic(signal_data, forward_returns)
    
    if ic_result.n_dates == 0:
        return signal_data, ic_result
    
    # Rolling mean IC and IR
    rolling_mean_ic = ic_result.daily_ic.rolling(window=window, min_periods=window // 2).mean()
    rolling_std_ic = ic_result.daily_ic.rolling(window=window, min_periods=window // 2).std()
    rolling_ir = rolling_mean_ic / rolling_std_ic.where(rolling_std_ic != 0, 0)
    
    # Keep dates where rolling IC > min_ic AND rolling IR > min_ir
    valid_dates_mask = (rolling_mean_ic.abs() >= min_ic) & (rolling_ir.abs() >= min_ir)
    valid_dates_mask = valid_dates_mask.fillna(False)
    
    # Also keep dates that are in the original signal but not in the rolling window
    # (early dates before window fills) — include them by default
    early_dates = ic_result.daily_ic.index[:window // 2]
    
    filtered_dates = signal_data.index[
        signal_data.index.isin(valid_dates_mask[valid_dates_mask].index) |
        signal_data.index.isin(early_dates)
    ]
    
    filtered_signal = signal_data.reindex(index=filtered_dates)
    
    return filtered_signal, ic_result


def rolling_ic_summary(
    signal_data: pd.DataFrame,
    forward_returns: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """Compute rolling IC statistics as a DataFrame for analysis.
    
    Args:
        signal_data: (Date, Stock) normalized factor values
        forward_returns: (Date, Stock) aligned forward returns
        window: Rolling window size
    
    Returns:
        pd.DataFrame with columns:
            - ic: daily IC value
            - rolling_mean_ic: rolling mean IC
            - rolling_std_ic: rolling std IC
            - rolling_ir: rolling IR = mean/std
            - hit_rate: rolling fraction of positive IC dates
    """
    ic_result = compute_ic(signal_data, forward_returns)
    
    if ic_result.n_dates == 0:
        return pd.DataFrame()
    
    df = pd.DataFrame({
        "ic": ic_result.daily_ic,
    })
    
    df["rolling_mean_ic"] = df["ic"].rolling(window=window, min_periods=window // 2).mean()
    df["rolling_std_ic"] = df["ic"].rolling(window=window, min_periods=window // 2).std()
    df["rolling_ir"] = df["rolling_mean_ic"] / df["rolling_std_ic"].where(df["rolling_std_ic"] != 0, 0)
    df["hit_rate"] = (df["ic"] > 0).rolling(window=window, min_periods=window // 2).mean()
    
    return df