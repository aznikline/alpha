"""AlphaBridge.normalize — cross-sectional normalization for factor signals."""
from __future__ import annotations

import numpy as np
import pandas as pd


class AlphaBridge:
    """Bridge layer converting OpenAlpha factor output to qmt-compatible format.
    
    NOTE: This is a SEPARATE class from bridge/transpose.py's AlphaBridge.
    In Phase 1, each file defines its own AlphaBridge namespace class with
    the relevant static methods. They will be unified into a single class
    when bridge/__init__.py is updated at the end of Phase 1.
    """

    @staticmethod
    def normalize(
        df: pd.DataFrame,
        method: str = "cs_rank_booksize",
        universe_mask: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Apply cross-sectional normalization to (Date, Stock) DataFrame.
        
        Each row (date) is normalized independently — this is "cross-sectional"
        normalization, meaning stocks are compared against each other on the same date.
        
        For "cs_rank_booksize" (default, matching OpenAlpha pipeline):
            1. Apply universe_mask (zero out non-universe stocks) if provided
            2. Per row (each date): rank stocks, subtract 0.5, scale by book size
            3. Result range ≈ [-1, 1], mean ≈ 0
            This matches OpenAlpha's internal normalization:
            cs_rank → subtract 0.5 → cs_booksize
        
        For "cs_zscore":
            1. Per row: z-score standardization
            2. Result range ≈ [-3, 3], mean = 0, std = 1
        
        For "raw":
            Pass through with NaN→0 only.
        
        Args:
            df: (Date, Stock) DataFrame, typically output of AlphaBridge.transpose()
            method: Normalization method:
                - "cs_rank_booksize": Rank → subtract 0.5 → book-size scaling (default, OpenAlpha convention)
                - "cs_zscore": Z-score standardization per date
                - "raw": No normalization, just NaN→0
            universe_mask: Optional (Date, Stock) bool DataFrame.
                True = in universe. If None, all non-zero stocks are considered in universe.
        
        Returns:
            pd.DataFrame: Same shape, normalized values. dtype float32.
        """
        result = df.copy().astype(np.float32)
        
        if universe_mask is not None:
            result = result.where(universe_mask, other=0.0)
        
        if method == "raw":
            return result.fillna(0.0).astype(np.float32)
        
        elif method == "cs_rank_booksize":
            return AlphaBridge._cs_rank_booksize(result)
        
        elif method == "cs_zscore":
            return AlphaBridge._cs_zscore(result)
        
        else:
            raise ValueError(
                f"Unknown normalization method: {method}. "
                f"Supported: 'cs_rank_booksize', 'cs_zscore', 'raw'"
            )

    @staticmethod
    def _cs_rank_booksize(df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional rank + book-size normalization.
        
        Matches OpenAlpha's internal pipeline: cs_rank → subtract 0.5 → cs_booksize.
        
        Per row (each date):
            1. Rank all non-zero values (rank 1 = smallest, rank N = largest)
            2. Subtract 0.5 → centered around 0
            3. Divide by N (number of non-zero stocks) → book-size scaling
            4. Result ≈ [-0.5, 0.5] after divide-by-N, then rescale to ≈ [-1, 1]
        
        The output range is approximately [-1, 1] with mean ≈ 0.
        This creates a long/short balanced signal suitable for VWAP backtesting.
        """
        result = df.copy()
        
        for idx in result.index:
            row = result.loc[idx]
            # Only rank non-zero values (zero = not in universe)
            nonzero_mask = row != 0.0
            n_nonzero = nonzero_mask.sum()
            
            if n_nonzero == 0:
                continue
            
            # Rank non-zero values
            ranked = row[nonzero_mask].rank(method="average")
            # Subtract 0.5 to center
            ranked = ranked - 0.5
            # Book-size: divide by number of positions → scale to [-0.5, 0.5] range
            ranked = ranked / n_nonzero
            # Scale to approximately [-1, 1] range (multiply by 2)
            ranked = ranked * 2
            
            # Assign back
            result.loc[idx, nonzero_mask] = ranked.astype(np.float32)
            result.loc[idx, ~nonzero_mask] = 0.0
        
        return result.astype(np.float32)

    @staticmethod
    def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional z-score normalization per date.
        
        Per row: subtract mean, divide by std.
        Result range ≈ [-3, 3], mean = 0, std = 1.
        """
        result = df.copy()
        
        for idx in result.index:
            row = result.loc[idx]
            nonzero_mask = row != 0.0
            
            if nonzero_mask.sum() == 0:
                continue
            
            nonzero_vals = row[nonzero_mask]
            mean = nonzero_vals.mean()
            std = nonzero_vals.std()
            
            if std == 0 or pd.isna(std):
                result.loc[idx, nonzero_mask] = 0.0
            else:
                result.loc[idx, nonzero_mask] = ((nonzero_vals - mean) / std).astype(np.float32)
            result.loc[idx, ~nonzero_mask] = 0.0
        
        return result.astype(np.float32)