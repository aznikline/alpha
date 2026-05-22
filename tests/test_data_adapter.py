"""Tests for bridge.data_adapter — QmtDataAdapter.

Since qmt DataManager is NOT available in the test environment, all tests
that require DataManager functionality use mocking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
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
    
    def test_adapter_available_false_by_default(self):
        """When qmt import fails, adapter.available should be False."""
        adapter = QmtDataAdapter()
        assert adapter.available is False
    
    def test_adapter_has_pit_returns_false_when_unavailable(self):
        """has_pit() should return False when qmt not available."""
        adapter = QmtDataAdapter()
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
    """Tests for error handling when qmt DataManager is not available."""
    
    def test_get_daily_raises_runtimeerror_when_unavailable(self):
        """get_daily() should raise RuntimeError when qmt not available."""
        adapter = QmtDataAdapter()
        
        with pytest.raises(RuntimeError) as exc_info:
            adapter.get_daily(
                stocks=["000001.SZ", "600000.SH"],
                start="2024-01-01",
                end="2024-01-31",
            )
        
        assert "qmt DataManager not available" in str(exc_info.value)
    
    def test_get_daily_signal_frame_raises_runtimeerror_when_unavailable(self):
        """get_daily_signal_frame() should raise RuntimeError when qmt not available."""
        adapter = QmtDataAdapter()
        
        with pytest.raises(RuntimeError) as exc_info:
            adapter.get_daily_signal_frame(
                stocks=["000001.SZ", "600000.SH"],
                start="2024-01-01",
                end="2024-01-31",
            )
        
        assert "qmt DataManager not available" in str(exc_info.value)
    
    def test_get_daily_ndarray_raises_runtimeerror_when_unavailable(self):
        """get_daily_ndarray() should raise RuntimeError when qmt not available."""
        adapter = QmtDataAdapter()
        
        with pytest.raises(RuntimeError) as exc_info:
            adapter.get_daily_ndarray(
                stocks=["000001.SZ", "600000.SH"],
                start="2024-01-01",
                end="2024-01-31",
                fields=["close", "volume"],
            )
        
        assert "qmt DataManager not available" in str(exc_info.value)
    
    def test_runtimeerror_message_mentions_qmt_not_available(self):
        """RuntimeError message should clearly state qmt DataManager not available."""
        adapter = QmtDataAdapter()
        
        with pytest.raises(RuntimeError, match="qmt DataManager not available"):
            adapter.get_daily(
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