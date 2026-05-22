"""QMT-compatible type definitions — local copy of qmt AlphaFactor ABC and FactorResult.

These are copied from /Users/wizout/op/quant/qmt/src/qmt_local/strategies/factor.py
to eliminate the runtime dependency on qmt for the core bridge pipeline.

The original qmt file only depends on abc, dataclasses, numpy, pandas —
no loguru, no xtquant, no tushare. Safe to inline.

When qmt is available at runtime, the try/except pattern in signal_factor.py
will prefer qmt's original classes (to ensure full compatibility with qmt's
MultiFactorStrategy). When qmt is not available, these local copies serve
as a standalone fallback.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorResult:
    """Result of computing a factor on a universe.

    Attributes:
        values: Series indexed by stock code with factor values
        name: Factor name for identification
        date: The date this factor was computed for
    """

    values: pd.Series
    name: str
    date: Any


class AlphaFactor(ABC):
    """Abstract base class for an alpha factor.

    To implement your own alpha factor, subclass this and override
    ``compute``::

        class MyAlpha(AlphaFactor):
            def compute(self, code: str, df: pd.DataFrame) -> float:
                return df["close"].pct_change(20).iloc[-1]

    Then use it in a strategy::

        strategy = MultiFactorStrategy(
            factors=[(MyAlpha(), 1.0)],
            top_n=10,
        )

    Args:
        name: Human-readable factor name (auto-set from class if empty)
    """

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def compute(self, code: str, df: pd.DataFrame) -> float:
        """Compute the factor value for a single stock.

        Args:
            code: Stock code (e.g., "000001.SZ")
            df: DataFrame with historical OHLCV data for this stock.
                Guaranteed columns: open, high, low, close, volume.
                Index is typically integer or datetime.

        Returns:
            A single numeric factor value. NaN if the factor cannot be
            computed for this stock on this date.
        """
        raise NotImplementedError

    def compute_universe(
        self,
        data: dict[str, pd.DataFrame],
        date: Any = None,
    ) -> FactorResult:
        """Compute factor values for an entire universe.

        Args:
            data: Mapping from stock code → historical DataFrame
            date: The date these values correspond to (for tracking)

        Returns:
            FactorResult with values indexed by stock code
        """
        values: dict[str, float] = {}
        for code, df in data.items():
            if df is None or df.empty:
                continue
            try:
                val = self.compute(code, df)
                if pd.notna(val) and np.isfinite(val):
                    values[code] = float(val)
            except Exception:
                continue
        series = pd.Series(values, name=self.name)
        return FactorResult(values=series, name=self.name, date=date)