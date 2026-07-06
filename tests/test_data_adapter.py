"""Tests for bridge.data_adapter — QmtDataAdapter.

Since qmt DataManager is NOT available in the test environment, all tests
that require DataManager functionality use mocking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from unittest.mock import MagicMock, patch

from bridge.data_adapter import QmtDataAdapter
from bridge.returns import compute_forward_returns
from bridge.ic_filter import compute_ic
from bridge.code_mapper import StockCodeMapper


def _make_mock_datamanager_history(
    codes: list[str] | None = None,
    fields: list[str] | None = None,
    period: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
    adjust: str = "qfq",
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Generate mock get_history response matching DataManager signature."""
    stocks = codes or []
    req_fields = fields or ["close"]
    start = start_date or "2024-01-01"
    end = end_date or "2024-01-31"
    
    dates = pd.date_range(start, end, freq="B")
    result = {}
    
    for code in stocks:
        rng = np.random.default_rng(hash(code) % 2**32)
        data = {}
        
        for field in req_fields:
            if field in ("volume", "amount"):
                data[field] = rng.integers(1000, 1_000_000, len(dates))
            else:
                base = 10.0 + (hash(code) % 100)
                data[field] = base + rng.normal(0, 0.5, len(dates))
        
        df = pd.DataFrame(data, index=dates)
        df.index.name = "time"
        result[code] = df
    
    return result


def _create_mock_manager() -> MagicMock:
    """Create a mock DataManager instance with get_history method."""
    manager = MagicMock()
    manager.get_history = MagicMock(side_effect=_make_mock_datamanager_history)
    return manager


class TestQmtDataAdapterInit:
    """Tests for QmtDataAdapter initialization and availability detection."""
    
    def test_adapter_available_false_by_default(self, monkeypatch):
        """When qmt import fails, adapter.available should be False.

        Block the qmt_local import so this holds regardless of whether the venv has
        qmt's deps installed (review fix C5: alpha's venv now has them).
        """
        import builtins

        real_import = builtins.__import__

        def _block_qmt(name, *args, **kwargs):
            if name == "qmt_local" or name.startswith("qmt_local."):
                raise ImportError("qmt_local blocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_qmt)
        for mod in list(sys.modules):
            if mod == "qmt_local" or mod.startswith("qmt_local."):
                monkeypatch.delitem(sys.modules, mod, raising=False)
        adapter = QmtDataAdapter(qmt_src_path="/nonexistent/qmt/src")
        assert adapter.available is False

    def test_adapter_has_pit_returns_false_when_unavailable(self, monkeypatch):
        """has_pit() should return False when qmt not available."""
        import builtins

        real_import = builtins.__import__

        def _block_qmt(name, *args, **kwargs):
            if name == "qmt_local" or name.startswith("qmt_local."):
                raise ImportError("qmt_local blocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_qmt)
        for mod in list(sys.modules):
            if mod == "qmt_local" or mod.startswith("qmt_local."):
                monkeypatch.delitem(sys.modules, mod, raising=False)
        adapter = QmtDataAdapter(qmt_src_path="/nonexistent/qmt/src")
        assert adapter.has_pit() is False
    
    def test_adapter_available_true_when_qmt_import_succeeds(self):
        """When qmt DataManager import succeeds, adapter.available should be True."""
        with patch.object(QmtDataAdapter, '_try_init_qmt'):
            adapter = QmtDataAdapter.__new__(QmtDataAdapter)
            adapter._qmt_data_dir = "~/.qmt_local/data"
            adapter._manager = _create_mock_manager()
            adapter._mapper = StockCodeMapper()
            adapter._available = True
            
            assert adapter.available is True
            assert adapter.has_pit() is True


class TestQmtDataAdapterUnavailable:
    """Tests for error handling when qmt DataManager is not available.

    Block the qmt_local import via monkeypatch so available=False deterministically,
    independent of which venv the tests run in or whether an earlier test added
    qmt/src to sys.path (review fix C5).
    """

    @pytest.fixture(autouse=True)
    def _unavailable_adapter(self, monkeypatch):
        # Force qmt_local to be unimportable regardless of sys.path state.
        import builtins

        real_import = builtins.__import__

        def _block_qmt(name, *args, **kwargs):
            if name == "qmt_local" or name.startswith("qmt_local."):
                raise ImportError("qmt_local blocked for unavailable-test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_qmt)
        # Also remove any cached qmt_local modules so the blocked import is exercised.
        for mod in list(sys.modules):
            if mod == "qmt_local" or mod.startswith("qmt_local."):
                monkeypatch.delitem(sys.modules, mod, raising=False)
        self.adapter = QmtDataAdapter(qmt_src_path="/nonexistent/qmt/src")

    def test_get_daily_raises_runtimeerror_when_unavailable(self):
        """get_daily() should raise RuntimeError when qmt not available."""
        with pytest.raises(RuntimeError) as exc_info:
            self.adapter.get_daily(
                stocks=["000001.SZ", "600000.SH"],
                start="2024-01-01",
                end="2024-01-31",
            )

        assert "qmt DataManager not available" in str(exc_info.value)

    def test_get_daily_signal_frame_raises_runtimeerror_when_unavailable(self):
        """get_daily_signal_frame() should raise RuntimeError when qmt not available."""
        with pytest.raises(RuntimeError) as exc_info:
            self.adapter.get_daily_signal_frame(
                stocks=["000001.SZ", "600000.SH"],
                start="2024-01-01",
                end="2024-01-31",
            )

        assert "qmt DataManager not available" in str(exc_info.value)

    def test_get_daily_ndarray_raises_runtimeerror_when_unavailable(self):
        """get_daily_ndarray() should raise RuntimeError when qmt not available."""
        with pytest.raises(RuntimeError) as exc_info:
            self.adapter.get_daily_ndarray(
                stocks=["000001.SZ", "600000.SH"],
                start="2024-01-01",
                end="2024-01-31",
                fields=["close", "volume"],
            )
        
        assert "qmt DataManager not available" in str(exc_info.value)
    
    def test_runtimeerror_message_mentions_qmt_not_available(self):
        """RuntimeError message should clearly state qmt DataManager not available."""
        with pytest.raises(RuntimeError, match="qmt DataManager not available"):
            self.adapter.get_daily(
                stocks=["000001.SZ"],
                start="2024-01-01",
                end="2024-01-10",
            )


class TestQmtDataAdapterFormatConversion:
    """Tests for format conversion methods using mocked DataManager."""
    
    @pytest.fixture
    def mock_adapter(self) -> QmtDataAdapter:
        """Create adapter with mocked DataManager for format conversion tests."""
        adapter = QmtDataAdapter.__new__(QmtDataAdapter)
        adapter._qmt_data_dir = "~/.qmt_local/data"
        adapter._manager = _create_mock_manager()
        adapter._mapper = StockCodeMapper()
        adapter._available = True
        return adapter
    
    def test_get_daily_returns_multiindex_dataframe(self, mock_adapter):
        """get_daily() should return DataFrame with MultiIndex (stock, field) columns."""
        stocks = ["000001.SZ", "600000.SH", "300001.SZ"]
        fields = ["open", "close", "volume"]
        
        result = mock_adapter.get_daily(
            stocks=stocks,
            start="2024-01-01",
            end="2024-01-31",
            fields=fields,
        )
        
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.columns, pd.MultiIndex)
        assert result.columns.names == ["stock", "field"]
        assert isinstance(result.index, pd.DatetimeIndex)
        
        result_stocks = result.columns.get_level_values("stock").unique()
        for stock in stocks:
            assert stock in result_stocks
        
        result_fields = result.columns.get_level_values("field").unique()
        for field in fields:
            assert field in result_fields
    
    def test_get_daily_signal_frame_returns_plain_dataframe(self, mock_adapter):
        """get_daily_signal_frame() should return plain (Date, Stock) DataFrame."""
        stocks = ["000001.SZ", "600000.SH"]
        
        result = mock_adapter.get_daily_signal_frame(
            stocks=stocks,
            start="2024-01-01",
            end="2024-01-31",
            price_field="close",
        )
        
        assert isinstance(result, pd.DataFrame)
        assert not isinstance(result.columns, pd.MultiIndex)
        assert isinstance(result.index, pd.DatetimeIndex)
        
        for stock in stocks:
            assert stock in result.columns
        
        assert result.shape[1] == len(stocks)
    
    def test_get_daily_ndarray_returns_dict_with_correct_orientation(self, mock_adapter):
        """get_daily_ndarray() should return dict[str, ndarray] with (Stock, Date) orientation."""
        stocks = ["000001.SZ", "600000.SH"]
        fields = ["close", "volume"]
        
        result = mock_adapter.get_daily_ndarray(
            stocks=stocks,
            start="2024-01-01",
            end="2024-01-31",
            fields=fields,
        )
        
        assert isinstance(result, dict)
        assert set(result.keys()) == set(fields)
        
        for field, arr in result.items():
            assert isinstance(arr, np.ndarray)
            assert arr.shape[0] == len(stocks)
            assert arr.shape[1] > 0
    
    def test_stock_codes_converted_to_bare_format(self, mock_adapter):
        """Stock codes in ndarray output should be bare integers (OpenAlpha convention)."""
        stocks = ["000001.SZ", "600000.SH"]
        
        result = mock_adapter.get_daily_ndarray(
            stocks=stocks,
            start="2024-01-01",
            end="2024-01-10",
            fields=["close"],
        )
        
        close_arr = result["close"]
        assert close_arr.shape[0] == len(stocks)
        
        assert StockCodeMapper.to_int("000001.SZ") == 1
        assert StockCodeMapper.to_int("600000.SH") == 600000
    
    def test_all_numeric_outputs_are_float32(self, mock_adapter):
        """All numeric outputs should be float32 for memory efficiency."""
        stocks = ["000001.SZ", "600000.SH"]
        
        signal_frame = mock_adapter.get_daily_signal_frame(
            stocks=stocks,
            start="2024-01-01",
            end="2024-01-10",
        )
        assert all(signal_frame.dtypes == np.float32)
        
        daily_df = mock_adapter.get_daily(
            stocks=stocks,
            start="2024-01-01",
            end="2024-01-10",
            fields=["close", "open"],
        )
        assert all(daily_df.dtypes == np.float32)
        
        ndarray_dict = mock_adapter.get_daily_ndarray(
            stocks=stocks,
            start="2024-01-01",
            end="2024-01-10",
            fields=["close"],
        )
        assert ndarray_dict["close"].dtype == np.float32
    
    def test_empty_data_raises_runtimeerror(self, mock_adapter):
        """Empty data from DataManager should raise RuntimeError."""
        mock_adapter._manager.get_history = MagicMock(return_value={})
        
        with pytest.raises(RuntimeError, match="DataManager returned no data"):
            mock_adapter.get_daily(
                stocks=["000001.SZ"],
                start="2024-01-01",
                end="2024-01-10",
            )


class TestQmtDataAdapterIntegration:
    """Tests for integration with bridge.returns and bridge.ic_filter."""
    
    @pytest.fixture
    def mock_adapter(self) -> QmtDataAdapter:
        """Create adapter with mocked DataManager for integration tests."""
        adapter = QmtDataAdapter.__new__(QmtDataAdapter)
        adapter._qmt_data_dir = "~/.qmt_local/data"
        adapter._manager = _create_mock_manager()
        adapter._mapper = StockCodeMapper()
        adapter._available = True
        return adapter
    
    def test_signal_frame_works_with_compute_forward_returns(self, mock_adapter):
        """get_daily_signal_frame() output should work with compute_forward_returns()."""
        stocks = ["000001.SZ", "600000.SH", "300001.SZ"]
        
        price_df = mock_adapter.get_daily_signal_frame(
            stocks=stocks,
            start="2024-01-01",
            end="2024-01-31",
            price_field="close",
        )
        
        forward_returns = compute_forward_returns(price_df, periods=1)
        
        assert isinstance(forward_returns, pd.DataFrame)
        assert forward_returns.index.equals(price_df.index)
        assert set(forward_returns.columns) == set(price_df.columns)
        assert all(forward_returns.dtypes == np.float32)
        assert forward_returns.iloc[-1].isna().all()
    
    def test_signal_frame_has_datetimeindex_and_stock_columns(self, mock_adapter):
        """get_daily_signal_frame() output should have DatetimeIndex and stock-code columns."""
        stocks = ["000001.SZ", "600000.SH"]
        
        result = mock_adapter.get_daily_signal_frame(
            stocks=stocks,
            start="2024-01-01",
            end="2024-01-31",
        )
        
        assert isinstance(result.index, pd.DatetimeIndex)
        assert all(col in stocks for col in result.columns)
    
    def test_mock_data_forward_returns_compute_ic_produces_valid_result(self, mock_adapter):
        """Full pipeline: mock data → forward returns → compute_ic() → valid ICResult."""
        stocks = ["000001.SZ", "600000.SH", "300001.SZ", "000002.SZ"]
        
        price_df = mock_adapter.get_daily_signal_frame(
            stocks=stocks,
            start="2024-01-01",
            end="2024-02-28",
            price_field="close",
        )
        
        forward_returns = compute_forward_returns(price_df, periods=1)
        
        rng = np.random.default_rng(42)
        signal_data = pd.DataFrame(
            rng.standard_normal((len(price_df.index), len(stocks))),
            index=price_df.index,
            columns=price_df.columns,
        ).astype(np.float32)
        
        valid_idx = forward_returns.iloc[:-1].index
        signal_aligned = signal_data.loc[valid_idx]
        returns_aligned = forward_returns.loc[valid_idx]
        
        ic_result = compute_ic(signal_aligned, returns_aligned)
        
        assert ic_result.n_dates > 0
        assert isinstance(ic_result.mean_ic, float)
        assert isinstance(ic_result.std_ic, float)
        assert isinstance(ic_result.ir, float)
        assert isinstance(ic_result.hit_rate, float)
        assert 0.0 <= ic_result.hit_rate <= 1.0
        assert len(ic_result.date_range) == 2
        
        assert isinstance(ic_result.daily_ic, pd.Series)
        assert len(ic_result.daily_ic) == ic_result.n_dates


class TestQmtDataAdapterEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    @pytest.fixture
    def mock_adapter(self) -> QmtDataAdapter:
        """Create adapter with mocked DataManager."""
        adapter = QmtDataAdapter.__new__(QmtDataAdapter)
        adapter._qmt_data_dir = "~/.qmt_local/data"
        adapter._manager = _create_mock_manager()
        adapter._mapper = StockCodeMapper()
        adapter._available = True
        return adapter
    
    def test_single_stock_single_field(self, mock_adapter):
        """Test with single stock and single field."""
        result = mock_adapter.get_daily(
            stocks=["000001.SZ"],
            start="2024-01-01",
            end="2024-01-10",
            fields=["close"],
        )
        
        assert isinstance(result, pd.DataFrame)
        assert result.columns.get_level_values("stock").unique()[0] == "000001.SZ"
        assert result.columns.get_level_values("field").unique()[0] == "close"
    
    def test_custom_price_field(self, mock_adapter):
        """Test get_daily_signal_frame with custom price_field."""
        result = mock_adapter.get_daily_signal_frame(
            stocks=["000001.SZ", "600000.SH"],
            start="2024-01-01",
            end="2024-01-10",
            price_field="open",
        )
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 2
    
    def test_multiple_periods_forward_returns(self, mock_adapter):
        """Test compute_forward_returns with multiple periods."""
        price_df = mock_adapter.get_daily_signal_frame(
            stocks=["000001.SZ"],
            start="2024-01-01",
            end="2024-01-31",
        )
        
        forward_5 = compute_forward_returns(price_df, periods=5)
        
        assert forward_5.iloc[-5:].isna().all().all()
        assert forward_5.iloc[:-5].notna().any().any()
    
    def test_data_manager_returns_partial_fields(self, mock_adapter):
        """Test when DataManager returns only some requested fields."""
        def partial_history(codes, fields, start_date, end_date, **kwargs):
            dates = pd.date_range(start_date, end_date, freq="B")
            result = {}
            for code in codes:
                df = pd.DataFrame(
                    {"close": np.random.default_rng(hash(code) % 2**32).normal(10, 1, len(dates)),
                     "volume": np.random.default_rng(hash(code) % 2**32).integers(1000, 100000, len(dates))},
                    index=dates,
                )
                df.index.name = "time"
                result[code] = df
            return result
        
        mock_adapter._manager.get_history = MagicMock(side_effect=partial_history)
        
        result = mock_adapter.get_daily(
            stocks=["000001.SZ"],
            start="2024-01-01",
            end="2024-01-10",
            fields=["close", "volume", "vwap"],
        )
        
        assert isinstance(result, pd.DataFrame)
        available_fields = result.columns.get_level_values("field").unique()
        assert "close" in available_fields
        assert "volume" in available_fields


class TestDateNormalization:
    """convergence-spec B3: adapter normalizes ISO dates to qmt's YYYYMMDD convention."""

    def test_iso_with_dashes_compacted(self):
        assert QmtDataAdapter._normalize_date_for_qmt("2024-12-01") == "20241201"

    def test_iso_with_slashes_compacted(self):
        assert QmtDataAdapter._normalize_date_for_qmt("2024/12/01") == "20241201"

    def test_compact_passthrough(self):
        assert QmtDataAdapter._normalize_date_for_qmt("20241201") == "20241201"

    def test_get_daily_compacts_dates_before_calling_manager(self):
        """The adapter must pass YYYYMMDD to DataManager.get_history, not ISO."""
        adapter = QmtDataAdapter()
        captured = {}

        def fake_get_history(codes, fields, period, start_date, end_date, adjust, **kw):
            captured["start"] = start_date
            captured["end"] = end_date
            return {}

        adapter._available = True
        adapter._manager = MagicMock()
        adapter._manager.get_history = MagicMock(side_effect=fake_get_history)

        with pytest.raises(RuntimeError, match="returned no data"):
            adapter.get_daily(["000001.SZ"], "2024-12-01", "2024-12-31", ["close"])

        assert captured["start"] == "20241201", captured
        assert captured["end"] == "20241231", captured


class TestRealDataFetch:
    """Live integration: fetch real data through qmt DataManager.

    Skips when qmt (or its deps) is not importable from this venv. Run from a venv
    that has qmt installed (e.g. qmt's venv) to exercise the real path.
    """

    @pytest.fixture(autouse=True)
    def _require_qmt(self):
        adapter = QmtDataAdapter()
        if not adapter.available:
            pytest.skip("qmt DataManager not importable in this venv")

    def test_real_signal_frame_is_datetimeindex_float32(self):
        adapter = QmtDataAdapter()
        frame = adapter.get_daily_signal_frame(
            ["600000.SH"], "2024-12-01", "2024-12-31", price_field="close"
        )
        assert isinstance(frame.index, pd.DatetimeIndex)
        assert (frame.dtypes == np.float32).all()
        assert "600000.SH" in frame.columns
        assert len(frame) > 0

    def test_real_ndarray_is_stock_date(self):
        adapter = QmtDataAdapter()
        nd = adapter.get_daily_ndarray(
            ["600000.SH"], "2024-12-01", "2024-12-31", ["close"]
        )
        assert "close" in nd
        # (Stock, Date): 1 stock × N dates
        assert nd["close"].ndim == 2
        assert nd["close"].shape[0] == 1
        assert nd["close"].dtype == np.float32

    def test_real_ndarray_row_order_matches_input_codes(self):
        """Review fix C4: ndarray row i must correspond to stocks[i] (input order), and
        get_daily_codes returns the matching bare-code list. Previously .values discarded
        the index and sort_index(axis=1) reordered alphabetically.
        """
        adapter = QmtDataAdapter()
        stocks = ["600000.SH", "000001.SZ"]  # intentionally NOT alphabetical
        nd = adapter.get_daily_ndarray(stocks, "2024-12-01", "2024-12-31", ["close"])
        codes = adapter.get_daily_codes(stocks, "2024-12-01", "2024-12-31")
        # Row count matches codes count
        assert nd["close"].shape[0] == len(codes)
        # Row order follows the input stocks order (600000 before 000001)
        assert codes[0] == "600000"
        assert codes[1] == "000001"