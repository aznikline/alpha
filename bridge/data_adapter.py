"""Data adapter — thin wrapper over qmt DataManager with graceful fallback.

When qmt is importable, uses its DataManager for data fetching (with PIT support,
full A-share coverage, and Parquet+SQLite caching). When qmt is not available,
falls back to OpenAlpha's Parquet file loading.

This is NOT a DataProvider ABC or shared data platform. It's a practical adapter
that picks the best available source and converts formats for the bridge pipeline.

ADR reference: docs/data-layer-adr.md (ADR-001)

convergence-spec §6 B3: the qmt source path is configurable (env ``QMT_SRC_PATH`` or
constructor ``qmt_src_path``), no longer hardcoded. Real-data fetch is verified when qmt
and its dependencies are importable; otherwise the adapter reports ``available=False``
and callers fall back to synthetic data.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

from bridge.code_mapper import StockCodeMapper

# Default qmt source path: env override, else the workspace-relative default.
_DEFAULT_QMT_SRC = os.environ.get(
    "QMT_SRC_PATH", "/Users/wizout/op/quant/qmt/src"
)


class QmtDataAdapter:
    """Thin adapter that uses qmt DataManager when available, falls back otherwise.

    Usage:
        adapter = QmtDataAdapter(qmt_data_dir="~/.qmt_local/data")
        if adapter.available:
            price_df = adapter.get_daily_signal_frame(stocks, start, end, ["close", "vwap"])
            forward_returns = compute_forward_returns(price_df, price_column="close")
        else:
            # Use synthetic data or OpenAlpha's data_generator
    """

    def __init__(self, qmt_data_dir: str = "", qmt_src_path: str = ""):
        self._qmt_data_dir = os.path.expanduser(qmt_data_dir or "~/.qmt_local/data")
        self._qmt_src_path = qmt_src_path or _DEFAULT_QMT_SRC
        self._manager = None
        self._mapper = StockCodeMapper()
        self._available = False
        self._try_init_qmt()

    def _try_init_qmt(self) -> None:
        # Add qmt src to path (idempotent). Configurable via constructor/env.
        if self._qmt_src_path and self._qmt_src_path not in sys.path:
            sys.path.insert(0, self._qmt_src_path)
        try:
            from qmt_local.data.manager import DataManager
            self._manager = DataManager(cache_dir=self._qmt_data_dir)
            self._available = True
        except ImportError as exc:
            # qmt (or one of its deps) not importable from this venv. Name the missing
            # module so the user knows what to install — previously this was silent.
            # Review fix C5: qmt's DataManager imports loguru/akshare at import time;
            # alpha's venv lacks these, so available=False there until installed.
            import warnings

            warnings.warn(
                f"QmtDataAdapter: qmt DataManager not importable from this venv "
                f"(missing module: {exc.name or exc!s}). Real-data fetch disabled; "
                f"falling back to synthetic. To enable: install qmt's deps in this venv "
                f"(e.g. `pip install loguru akshare`) and ensure QMT_SRC_PATH points at "
                f"qmt/src. Set QMT_SRC_PATH={self._qmt_src_path!r}.",
                stacklevel=2,
            )
            self._manager = None
            self._available = False

    @staticmethod
    def _normalize_date_for_qmt(date: str) -> str:
        """Normalize a date string to compact YYYYMMDD (qmt/akshare convention).

        Accepts "YYYY-MM-DD", "YYYY/MM/DD", or "YYYYMMDD". Passes through anything
        that already looks compact.
        """
        s = str(date).strip()
        if "-" in s or "/" in s:
            return s.replace("-", "").replace("/", "")
        return s

    @property
    def available(self) -> bool:
        return self._available

    def has_pit(self) -> bool:
        return self._available

    def get_daily(
        self,
        stocks: list[str],
        start: str,
        end: str,
        fields: list[str] = ["open", "high", "low", "close", "volume", "amount"],
    ) -> pd.DataFrame:
        """Get daily data as (Date, Stock) DataFrame with multiple fields.

        Args:
            stocks: Stock codes in qmt suffix format (e.g. "000001.SZ")
            start: Date string — accepts "YYYY-MM-DD" or "YYYYMMDD" (normalized to
                   YYYYMMDD, which is what qmt DataManager.get_history expects).
            end: Date string — same format tolerance as ``start``.
            fields: Data fields to fetch

        Returns:
            pd.DataFrame with DatetimeIndex and MultiIndex(stock, field) columns.
            dtype: float32 for prices, float64 for volume/amount.

        Raises:
            RuntimeError: If qmt DataManager is not available.
        """
        if not self._available:
            raise RuntimeError("qmt DataManager not available — cannot fetch market data")

        # qmt's akshare provider expects compact YYYYMMDD dates, not ISO YYYY-MM-DD.
        start_qmt = self._normalize_date_for_qmt(start)
        end_qmt = self._normalize_date_for_qmt(end)

        raw = self._manager.get_history(
            codes=stocks, fields=fields, period="1d",
            start_date=start_qmt, end_date=end_qmt, adjust="qfq",
        )

        if not raw:
            raise RuntimeError(f"DataManager returned no data for {stocks}")

        # Convert dict[str, DataFrame] to (Date, Stock) MultiIndex DataFrame
        return self._assemble_daily(raw, fields)

    def _assemble_daily(
        self,
        raw: dict[str, pd.DataFrame],
        fields: list[str],
    ) -> pd.DataFrame:
        """Assemble per-stock DataFrames into (Date, Stock, Field) MultiIndex format.

        qmt DataManager returns {stock_code: DataFrame} where each DataFrame
        has time-indexed rows and standard column names (open, close, etc.).

        We assemble these into a single DataFrame with:
            - Index: DatetimeIndex (union of all dates)
            - Columns: MultiIndex with levels (stock_code, field_name)
        """
        all_dates = None
        data_frames = {}

        for code, df in raw.items():
            if df.empty:
                continue

            # Ensure time column is the index
            if "time" in df.columns:
                df = df.set_index("time")
            elif df.index.name in ("time", "date", "datetime"):
                pass  # already indexed
            else:
                continue

            # Select requested fields
            available_fields = [f for f in fields if f in df.columns]
            if not available_fields:
                continue

            df_selected = df[available_fields].astype(np.float32)

            # Track union of dates
            if all_dates is None:
                all_dates = df_selected.index
            else:
                all_dates = all_dates.union(df_selected.index)

            data_frames[code] = df_selected

        if not data_frames or all_dates is None:
            raise RuntimeError("No valid data assembled from DataManager results")

        # Reindex all stocks to same date range
        assembled = {}
        for code, df in data_frames.items():
            assembled[code] = df.reindex(all_dates)

        # Create MultiIndex DataFrame
        result = pd.concat(assembled, axis=1)
        result.index = pd.DatetimeIndex(result.index)
        result.columns = pd.MultiIndex.from_tuples(
            [(code, field) for code in assembled for field in fields if field in assembled[code].columns],
            names=["stock", "field"],
        )

        return result.sort_index(axis=0).sort_index(axis=1)

    def get_daily_signal_frame(
        self,
        stocks: list[str],
        start: str,
        end: str,
        price_field: str = "close",
    ) -> pd.DataFrame:
        """Get price data as (Date, Stock) DataFrame — bridge convention.

        Returns a plain DataFrame with DatetimeIndex and stock-code columns.
        Values are the specified price field (default: close).

        Ready for bridge.returns.compute_forward_returns().
        """
        multi_df = self.get_daily(stocks, start, end, [price_field])

        # Extract single field from MultiIndex → plain (Date, Stock) DataFrame
        if isinstance(multi_df.columns, pd.MultiIndex):
            result = multi_df.xs(price_field, level=1, axis=1)
        else:
            result = multi_df

        return result.astype(np.float32)

    def get_daily_ndarray(
        self,
        stocks: list[str],
        start: str,
        end: str,
        fields: list[str],
    ) -> dict[str, np.ndarray]:
        """Get daily data as (Stock, Date) ndarray dict — OpenAlpha convention.

        Returns dict mapping field_name -> np.ndarray with shape (n_stocks, n_dates).
        Stock codes are bare integers (OpenAlpha convention), not suffix format.

        **Row-order contract (review fix C4):** ndarray row ``i`` corresponds to
        ``stocks[i]`` (filtered to stocks that actually had data), in the **input order**.
        Previously ``.values`` discarded the index and an earlier ``sort_index(axis=1)``
        reordered columns alphabetically, so callers could not reliably map rows back to
        codes. Now the transposed frame is reindexed to the input ``stocks`` order before
        ``.values``; use :meth:`get_daily_codes` to get the ordered bare-code list.

        Note: This converts from (Date, Stock) bridge format to (Stock, Date)
        OpenAlpha format — the reverse of AlphaBridge.transpose().
        """
        multi_df = self.get_daily(stocks, start, end, fields)

        result = {}
        for field in fields:
            if isinstance(multi_df.columns, pd.MultiIndex):
                field_df = multi_df.xs(field, level=1, axis=1)
            else:
                field_df = multi_df

            # Transpose: (Date, Stock) → (Stock, Date)
            transposed = field_df.T.astype(np.float32)

            # Reindex rows to the INPUT stocks order (filtered to those present),
            # so ndarray row i == stocks[i]. This is the contract callers rely on.
            present = [c for c in stocks if c in transposed.index]
            if not present:
                result[field] = np.empty((0, len(field_df.index)), dtype=np.float32)
                continue
            transposed = transposed.reindex(present)

            # Rename columns to bare integer codes (OpenAlpha convention)
            bare_codes = [self._mapper.to_int(col) for col in transposed.index]
            transposed.index = [f"{code:06d}" for code in bare_codes]

            result[field] = transposed.values

        return result

    def get_daily_codes(
        self,
        stocks: list[str],
        start: str,
        end: str,
    ) -> list[str]:
        """Return the ordered bare-code list matching ``get_daily_ndarray`` row order.

        Companion to :meth:`get_daily_ndarray` so callers can map ndarray row ``i`` back
        to a stock code. Returns 6-digit zero-padded bare codes (OpenAlpha convention).
        """
        # Fetch a single field to determine which stocks have data in-range.
        multi_df = self.get_daily(stocks, start, end, ["close"])
        if isinstance(multi_df.columns, pd.MultiIndex):
            field_df = multi_df.xs("close", level=1, axis=1)
        else:
            field_df = multi_df
        present = [c for c in stocks if c in field_df.columns]
        return [f"{self._mapper.to_int(c):06d}" for c in present]