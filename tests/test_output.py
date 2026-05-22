"""Tests for bridge.output — FactorOutput and wrap_factor_output."""
import numpy as np
import pandas as pd
import pytest

from bridge.output import FactorOutput, FactorMetadata, wrap_factor_output


class TestFactorMetadata:
    def test_default_normalization(self):
        meta = FactorMetadata(expression="cs_rank(close)")
        assert meta.normalization == ["at_nan2zero", "cs_booksize", "cs_rank"]
    
    def test_custom_normalization(self):
        meta = FactorMetadata(expression="close", normalization=["raw"])
        assert meta.normalization == ["raw"]
    
    def test_frozen(self):
        meta = FactorMetadata(expression="close")
        with pytest.raises(AttributeError):
            meta.expression = "other"  # frozen dataclass


class TestFactorOutput:
    @pytest.fixture
    def sample_output(self):
        """Create a sample FactorOutput for testing."""
        n_stocks = 5
        n_dates = 10
        alpha = np.random.randn(n_stocks, n_dates).astype(np.float32)
        stocks = ["000001.SZ", "000002.SZ", "600000.SH", "600036.SH", "300001.SZ"]
        dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
        
        values = pd.DataFrame(alpha, index=stocks, columns=dates)
        meta = FactorMetadata(
            expression="cs_rank(close)",
            date_range=("2024-01-01", "2024-01-12"),
        )
        
        return FactorOutput(
            values=values,
            expression="cs_rank(close)",
            stocks=stocks,
            dates=dates,
            metadata=meta,
        )
    
    def test_shape_matches_stocks_dates(self, sample_output):
        assert sample_output.values.shape == (len(sample_output.stocks), len(sample_output.dates))
    
    def test_dtype_is_float32(self, sample_output):
        assert sample_output.values.dtypes.iloc[0] == np.float32
    
    def test_index_is_stock_codes(self, sample_output):
        assert list(sample_output.values.index) == sample_output.stocks
    
    def test_columns_are_dates(self, sample_output):
        pd.testing.assert_index_equal(sample_output.values.columns, sample_output.dates)
    
    def test_orientation_is_stock_date(self, sample_output):
        # First axis = stocks, second axis = dates
        assert sample_output.values.shape[0] == len(sample_output.stocks)
        assert sample_output.values.shape[1] == len(sample_output.dates)
    
    def test_shape_validation_raises(self):
        # Mismatched stocks count
        with pytest.raises(ValueError, match="Stocks count mismatch"):
            FactorOutput(
                values=pd.DataFrame(np.zeros((3, 5))),
                expression="test",
                stocks=["a", "b"],  # 2 stocks but 3 rows
                dates=pd.date_range("2024-01-01", periods=5),
                metadata=FactorMetadata(expression="test"),
            )
    
    def test_shape_validation_dates_raises(self):
        # Mismatched dates count
        with pytest.raises(ValueError, match="Dates count mismatch"):
            FactorOutput(
                values=pd.DataFrame(np.zeros((3, 5))),
                expression="test",
                stocks=["a", "b", "c"],
                dates=pd.date_range("2024-01-01", periods=3),  # 3 dates but 5 columns
                metadata=FactorMetadata(expression="test"),
            )


class TestWrapFactorOutput:
    @pytest.fixture
    def mock_executor_output(self):
        """Simulate AlphaExecutor.evaluate() output."""
        n_stocks = 5
        n_dates = 10
        alpha = np.random.randn(n_stocks, n_dates).astype(np.float32)
        stock_list = np.array(["000001.SZ", "000002.SZ", "600000.SH", "600036.SH", "300001.SZ"])
        datestr = np.array([d.strftime("%Y%m%d") for d in pd.date_range("2024-01-01", periods=n_dates, freq="B")])
        return alpha, stock_list, datestr
    
    def test_wrap_creates_factor_output(self, mock_executor_output):
        alpha, stock_list, datestr = mock_executor_output
        result = wrap_factor_output(alpha, stock_list, datestr, "cs_rank(close)")
        assert isinstance(result, FactorOutput)
    
    def test_wrap_preserves_shape(self, mock_executor_output):
        alpha, stock_list, datestr = mock_executor_output
        result = wrap_factor_output(alpha, stock_list, datestr, "cs_rank(close)")
        assert result.values.shape == alpha.shape
    
    def test_wrap_preserves_dtype(self, mock_executor_output):
        alpha, stock_list, datestr = mock_executor_output
        result = wrap_factor_output(alpha, stock_list, datestr, "cs_rank(close)")
        assert result.values.dtypes.iloc[0] == np.float32
    
    def test_wrap_parses_dates(self, mock_executor_output):
        alpha, stock_list, datestr = mock_executor_output
        result = wrap_factor_output(alpha, stock_list, datestr, "cs_rank(close)")
        assert isinstance(result.dates, pd.DatetimeIndex)
        assert result.dates[0] == pd.Timestamp("2024-01-01")
    
    def test_wrap_sets_stocks(self, mock_executor_output):
        alpha, stock_list, datestr = mock_executor_output
        result = wrap_factor_output(alpha, stock_list, datestr, "cs_rank(close)")
        assert result.stocks == [str(s) for s in stock_list]
    
    def test_wrap_none_alpha_raises(self):
        stock_list = np.array(["000001.SZ"])
        datestr = np.array(["20240101"])
        with pytest.raises(ValueError, match="alpha is None"):
            wrap_factor_output(None, stock_list, datestr, "close")
    
    def test_wrap_metadata(self, mock_executor_output):
        alpha, stock_list, datestr = mock_executor_output
        result = wrap_factor_output(alpha, stock_list, datestr, "cs_rank(ts_delta(close, 5))")
        assert result.metadata.expression == "cs_rank(ts_delta(close, 5))"
        assert "cs_rank" in result.metadata.operator_chain
    
    def test_wrap_custom_normalization(self, mock_executor_output):
        alpha, stock_list, datestr = mock_executor_output
        result = wrap_factor_output(alpha, stock_list, datestr, "close", normalization=["raw"])
        assert result.metadata.normalization == ["raw"]
    
    def test_wrap_date_range(self, mock_executor_output):
        alpha, stock_list, datestr = mock_executor_output
        result = wrap_factor_output(alpha, stock_list, datestr, "close")
        assert result.metadata.date_range[0] == result.dates.min().strftime("%Y-%m-%d")
        assert result.metadata.date_range[1] == result.dates.max().strftime("%Y-%m-%d")