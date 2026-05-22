"""AlphaBridge.transpose — convert (Stock, Date) factor output to (Date, Stock) for qmt."""
from __future__ import annotations

import numpy as np
import pandas as pd

from bridge.output import FactorOutput
from bridge.code_mapper import StockCodeMapper


class AlphaBridge:
    """Bridge layer converting OpenAlpha factor output to qmt-compatible format.
    
    All methods are static — the bridge is a pure transformation pipeline
    with no state or configuration beyond method arguments.
    """

    @staticmethod
    def transpose(factor_output: FactorOutput) -> pd.DataFrame:
        """Transpose OpenAlpha output from (Stock, Date) to (Date, Stock).
        
        The qmt AlphaFactor.compute(code, df) interface needs factor values
        indexed by date with stock codes as columns, so it can look up
        the value for a specific stock on a specific date.
        
        Implementation:
            1. factor_output.values.T  (pandas transpose)
            2. Reindex to DatetimeIndex with daily frequency
            3. Sort index (ascending dates) — qmt convention
            4. Sort columns (ascending stock codes)
            5. Fill remaining NaN with 0.0
            6. Cast to float32
        
        Args:
            factor_output: FactorOutput with (Stock, Date) orientation.
            
        Returns:
            pd.DataFrame with:
              - index: pd.DatetimeIndex (daily, ascending dates)
              - columns: qmt-format stock codes (e.g. "000001.SZ")
              - dtype: float32
              - values: factor signal values
        
        Key invariant:
            After transpose, df.loc["2024-01-15", "000001.SZ"]
            gives the factor value for stock 000001 on date 2024-01-15.
        """
        df = factor_output.values.T.copy()
        
        # Ensure DatetimeIndex on rows
        df.index = pd.DatetimeIndex(df.index)
        
        # Sort both axes (qmt convention: ascending dates, ascending codes)
        df = df.sort_index().sort_index(axis=1)
        
        # Fill NaN with 0.0 (non-universe stocks or missing dates)
        df = df.fillna(0.0)
        
        # Preserve float32
        df = df.astype(np.float32)
        
        return df