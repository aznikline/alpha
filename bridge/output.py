"""FactorOutput wrapper — converts AlphaExecutor ndarray output to labeled DataFrame."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorMetadata:
    """Metadata for a factor output."""
    expression: str
    normalization: list[str] = field(default_factory=lambda: ["at_nan2zero", "cs_booksize", "cs_rank"])
    universe: str = "csi_500"
    date_range: tuple[str, str] = ("", "")  # ("2020-01-01", "2025-12-31")
    operator_chain: list[str] = field(default_factory=list)


@dataclass
class FactorOutput:
    """Canonical OpenAlpha factor output before bridge conversion.
    
    Wraps a raw (Stock, Date) ndarray from AlphaExecutor.evaluate() into
    a labeled DataFrame with stock codes as index and dates as columns.
    
    Orientation: index=stock codes, columns=dates.
    This is OpenAlpha's natural orientation — transpose happens in bridge layer.
    """
    values: pd.DataFrame       # Index: stock codes, Columns: dates, dtype float32
    expression: str
    stocks: list[str]          # Stock codes from executor.context['stock_list']
    dates: pd.DatetimeIndex    # Dates parsed from executor.context['datestr']
    metadata: FactorMetadata
    
    def __post_init__(self):
        """Validate that values shape matches stocks/dates length."""
        if len(self.stocks) != self.values.shape[0]:
            raise ValueError(
                f"Stocks count mismatch: values has {self.values.shape[0]} rows, "
                f"stocks has {len(self.stocks)} entries"
            )
        if len(self.dates) != self.values.shape[1]:
            raise ValueError(
                f"Dates count mismatch: values has {self.values.shape[1]} columns, "
                f"dates has {len(self.dates)} entries"
            )


def wrap_factor_output(
    alpha: np.ndarray,
    stock_list: np.ndarray | list[str],
    datestr: np.ndarray | list[str],
    expression: str,
    normalization: list[str] | None = None,
    universe: str = "csi_500",
) -> FactorOutput:
    """Create FactorOutput from AlphaExecutor evaluate() result.
    
    Args:
        alpha: Raw ndarray from evaluate(), shape (N_stocks, N_dates)
        stock_list: Stock codes from executor.context['stock_list']
                   Format: '000001.SZ' / '600000.SH' (suffix format from data)
        datestr: Date strings from executor.context['datestr']
                 Format: 'YYYYMMDD' strings (e.g. '20200101')
        expression: The alpha expression that produced this factor
        normalization: Applied normalization steps (default from OpenAlpha pipeline)
        universe: Universe name (default 'csi_500')
    
    Returns:
        FactorOutput with labeled DataFrame and metadata.
    
    Raises:
        ValueError: If alpha is None or shapes don't match.
    """
    if alpha is None:
        raise ValueError("alpha is None — evaluate() returned None, expression may have errors")
    
    stocks = [str(s) for s in stock_list]
    dates = pd.to_datetime([str(d) for d in datestr], format="%Y%m%d")
    values = pd.DataFrame(
        alpha.astype(np.float32),
        index=stocks,
        columns=dates,
    )
    
    if normalization is None:
        normalization = ["at_nan2zero", "cs_booksize", "cs_rank"]
    
    date_range = (
        dates.min().strftime("%Y-%m-%d"),
        dates.max().strftime("%Y-%m-%d"),
    )
    
    metadata = FactorMetadata(
        expression=expression,
        normalization=normalization,
        universe=universe,
        date_range=date_range,
        operator_chain=_parse_operator_chain(expression),
    )
    
    return FactorOutput(
        values=values,
        expression=expression,
        stocks=stocks,
        dates=dates,
        metadata=metadata,
    )


def _parse_operator_chain(expression: str) -> list[str]:
    """Extract top-level operators from an expression string.
    
    Simple parser: extracts function calls at the start of each nesting level.
    E.g. 'cs_rank(ts_delta(close, 5))' → ['cs_rank', 'ts_delta', 'close']
    """
    import re
    # Match function_name( pattern
    tokens = re.findall(r'([a-z_]+)\s*\(', expression)
    # Also match bare field names (not in function calls)
    remaining = expression
    for token in tokens:
        remaining = remaining.replace(token, '', 1)
    # Extract remaining field names (alphanumeric, not in function calls)
    fields = re.findall(r'[a-z_][a-z0-9_]*', remaining)
    return tokens + fields