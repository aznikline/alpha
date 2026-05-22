"""SignalAlphaFactor — adapts pre-computed OpenAlpha factor signals to qmt's AlphaFactor interface."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Import qmt's AlphaFactor and FactorResult — the bridge inherits from qmt's ABC.
# This makes SignalAlphaFactor directly usable in MultiFactorStrategy.
# If qmt is not importable (e.g. running standalone tests without qmt installed),
# fall back to a local Protocol-based implementation.
try:
    from qmt_local.strategies.factor import AlphaFactor, FactorResult
    _HAS_QMT = True
except ImportError:
    from bridge._qmt_types import AlphaFactor, FactorResult
    _HAS_QMT = False


class SignalAlphaFactor(AlphaFactor):
    """Adapts pre-computed OpenAlpha factor signals to qmt's AlphaFactor interface.
    
    This is the core adapter: it stores a (Date, Stock) DataFrame of normalized
    factor values and implements qmt's compute(code, df) → float interface.
    
    The compute() method looks up the pre-computed value for the given
    stock on the latest date in df. No runtime factor computation occurs.
    
    Usage in qmt MultiFactorStrategy:
        signal_factor = SignalAlphaFactor(
            name="cs_rank(ts_delta(close, 5))",
            signal_data=normalized_df,  # (Date, Stock) normalized values
        )
        strategy = MultiFactorStrategy(
            factors=[(signal_factor, 1.0)],  # (factor, weight) tuple
            top_n=10,
            rebalance_period=1,
        )
    """

    def __init__(
        self,
        name: str,
        signal_data: pd.DataFrame,      # (Date, Stock) normalized values
                                        # Index: DatetimeIndex, Columns: qmt-format codes
        default_value: float = 0.0,     # Value returned for missing stock/date
    ):
        """Initialize SignalAlphaFactor with pre-computed signal data.
        
        Args:
            name: Factor name (typically the alpha expression string)
            signal_data: (Date, Stock) DataFrame with normalized factor values.
                Index must be DatetimeIndex, columns must be qmt-format stock codes.
            default_value: Value returned when stock/date lookup fails.
        """
        super().__init__(name=name)
        self._signal_data = signal_data
        self._default_value = default_value

    def compute(self, code: str, df: pd.DataFrame) -> float:
        """Look up pre-computed factor value from signal_data.
        
        Args:
            code: Stock code in qmt format (e.g. "000001.SZ")
            df: Market data DataFrame for this stock.
                The latest date df.index[-1] is used as the lookup date.
                Only used for date reference — factor value comes from signal_data.
        
        Returns:
            float: Normalized factor value for this stock on this date.
                   Returns default_value (0.0) if stock/date not found.
        """
        if df is None or df.empty:
            return self._default_value

        target_date = df.index[-1]

        # Convert target_date to Timestamp for lookup
        target_ts = pd.Timestamp(target_date)

        # Try exact date match first
        if target_ts in self._signal_data.index and code in self._signal_data.columns:
            return float(self._signal_data.loc[target_ts, code])

        # Try nearest date match (for cases where signal dates and market dates
        # don't align exactly, e.g. signal computed on trading days but df has all dates)
        try:
            nearest_idx = self._signal_data.index.get_indexer([target_ts], method="nearest")
            if nearest_idx[0] >= 0 and code in self._signal_data.columns:
                return float(self._signal_data.iloc[nearest_idx[0]][code])
        except (ValueError, IndexError):
            pass

        # Stock not in signal universe or date out of range
        return self._default_value

    @property
    def signal_data(self) -> pd.DataFrame:
        """Access the underlying signal DataFrame (Date, Stock)."""
        return self._signal_data