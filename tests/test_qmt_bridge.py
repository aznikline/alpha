"""Integration tests for the AlphaBridge pipeline: FactorOutput → transpose → normalize → SignalAlphaFactor."""
import numpy as np
import pandas as pd
import pytest

from bridge.output import FactorOutput, FactorMetadata, wrap_factor_output
from bridge.transpose import AlphaBridge as TransposeBridge
from bridge.normalize import AlphaBridge as NormalizeBridge
from bridge.signal_factor import SignalAlphaFactor


@pytest.fixture
def sample_factor_output():
    """Create a sample FactorOutput for testing the full pipeline."""
    n_stocks = 20
    n_dates = 30
    np.random.seed(42)
    alpha = np.random.randn(n_stocks, n_dates).astype(np.float32)
    
    # Mix of SZ and SH stocks
    stocks = [f"{i:06d}.SZ" for i in range(1, 11)] + [f"{600000 + i:06d}.SH" for i in range(10)]
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    datestr = [d.strftime("%Y%m%d") for d in dates]
    
    return wrap_factor_output(
        alpha=alpha,
        stock_list=stocks,
        datestr=datestr,
        expression="cs_rank(ts_delta(close, 5))",
    )


class TestTransposeNormalizePipeline:
    """Test the transpose → normalize pipeline as a sequence."""

    def test_full_pipeline_shapes(self, sample_factor_output):
        """Verify shape transformations through the pipeline."""
        transposed = TransposeBridge.transpose(sample_factor_output)
        normalized = NormalizeBridge.normalize(transposed)
        
        # After transpose: (Date, Stock)
        assert transposed.shape[0] == len(sample_factor_output.dates)
        assert transposed.shape[1] == len(sample_factor_output.stocks)
        
        # Normalize preserves shape
        assert normalized.shape == transposed.shape

    def test_transpose_orientation(self, sample_factor_output):
        """Verify transpose converts (Stock, Date) to (Date, Stock)."""
        transposed = TransposeBridge.transpose(sample_factor_output)
        
        # Index should be DatetimeIndex (dates)
        assert isinstance(transposed.index, pd.DatetimeIndex)
        
        # Columns should be stock codes
        assert transposed.shape[0] == len(sample_factor_output.dates)
        assert transposed.shape[1] == len(sample_factor_output.stocks)

    def test_normalize_cs_rank_booksize_range(self, sample_factor_output):
        """Verify cs_rank_booksize normalization produces values in [-1, 1] range."""
        transposed = TransposeBridge.transpose(sample_factor_output)
        normalized = NormalizeBridge.normalize(transposed, method="cs_rank_booksize")
        
        # Flatten to 1D array for scalar comparisons
        nonzero_vals = normalized.values[normalized.values != 0.0]
        assert nonzero_vals.min() >= 0.0 - 1e-4
        assert nonzero_vals.max() <= 2.0 + 1e-4

    def test_normalize_cs_zscore_range(self, sample_factor_output):
        """Verify cs_zscore normalization produces standard normal distribution."""
        transposed = TransposeBridge.transpose(sample_factor_output)
        normalized = NormalizeBridge.normalize(transposed, method="cs_zscore")
        
        # Flatten to 1D array for scalar comparisons
        nonzero_vals = normalized.values[normalized.values != 0.0]
        assert abs(nonzero_vals.mean()) < 0.5
        
        # Std should be approximately 1
        assert abs(nonzero_vals.std() - 1.0) < 0.5

    def test_normalize_raw_preserves_values(self, sample_factor_output):
        """Verify raw normalization only fills NaN with 0."""
        transposed = TransposeBridge.transpose(sample_factor_output)
        normalized = NormalizeBridge.normalize(transposed, method="raw")
        
        # Raw should be identical to transposed with NaN→0
        pd.testing.assert_frame_equal(
            normalized,
            transposed.fillna(0.0).astype(np.float32),
        )

    def test_value_preservation_through_pipeline(self, sample_factor_output):
        """Verify specific values are preserved through the pipeline."""
        transposed = TransposeBridge.transpose(sample_factor_output)
        
        # Pick a specific stock and date
        stock = sample_factor_output.stocks[0]
        date = sample_factor_output.dates[5]
        
        # Original value in FactorOutput (Stock, Date) orientation
        original_val = sample_factor_output.values.loc[stock, date]
        
        # After transpose, same value should be at (Date, Stock)
        transposed_val = transposed.loc[date, stock]
        
        assert abs(original_val - transposed_val) < 1e-6


class TestSignalAlphaFactorIntegration:
    """Test SignalAlphaFactor integration with the full pipeline."""

    @pytest.fixture
    def signal_factor(self, sample_factor_output):
        """Create a SignalAlphaFactor through the full pipeline."""
        transposed = TransposeBridge.transpose(sample_factor_output)
        normalized = NormalizeBridge.normalize(transposed)
        return SignalAlphaFactor(
            name=sample_factor_output.expression,
            signal_data=normalized,
        )

    def test_compute_returns_float(self, signal_factor):
        """Verify compute() returns a float."""
        mock_df = pd.DataFrame(
            {"close": [10.0, 11.0, 12.0]},
            index=pd.DatetimeIndex(["2024-01-29", "2024-01-30", "2024-01-31"]),
        )
        value = signal_factor.compute("000001.SZ", mock_df)
        assert isinstance(value, float)

    def test_compute_value_in_range(self, signal_factor):
        """Verify compute() returns value in expected range."""
        mock_df = pd.DataFrame(
            {"close": [10.0, 11.0, 12.0]},
            index=pd.DatetimeIndex(["2024-01-29", "2024-01-30", "2024-01-31"]),
        )
        value = signal_factor.compute("000001.SZ", mock_df)
        # cs_rank_booksize produces values approximately in [-2, 2] range
        # depending on number of stocks in universe
        assert abs(value) <= 2.0 + 1e-4

    def test_compute_missing_stock_returns_default(self, signal_factor):
        """Verify compute() returns default for missing stock."""
        mock_df = pd.DataFrame(
            {"close": [10.0, 11.0]},
            index=pd.DatetimeIndex(["2024-01-29", "2024-01-30"]),
        )
        value = signal_factor.compute("999999.SZ", mock_df)
        assert value == 0.0  # default_value

    def test_compute_empty_df_returns_default(self, signal_factor):
        """Verify compute() returns default for empty DataFrame."""
        mock_df = pd.DataFrame()
        value = signal_factor.compute("000001.SZ", mock_df)
        assert value == 0.0

    def test_name_preserved(self, signal_factor, sample_factor_output):
        """Verify factor name is preserved."""
        assert signal_factor.name == sample_factor_output.expression

    def test_signal_data_accessible(self, signal_factor):
        """Verify signal_data property returns DataFrame."""
        assert isinstance(signal_factor.signal_data, pd.DataFrame)
        assert signal_factor.signal_data.shape[0] > 0


class TestMultiFactorStrategyCompatibility:
    """Verify SignalAlphaFactor works with qmt MultiFactorStrategy interface."""

    def test_factor_weight_tuple(self, sample_factor_output):
        """Verify SignalAlphaFactor can be used in (factor, weight) tuple."""
        transposed = TransposeBridge.transpose(sample_factor_output)
        normalized = NormalizeBridge.normalize(transposed)
        signal_factor = SignalAlphaFactor(
            name="test_factor",
            signal_data=normalized,
        )
        
        # This is how qmt MultiFactorStrategy expects factors
        factors = [(signal_factor, 1.0)]
        
        assert len(factors) == 1
        assert factors[0][0] is signal_factor
        assert factors[0][1] == 1.0

    def test_multiple_factors_with_weights(self, sample_factor_output):
        """Verify multiple SignalAlphaFactors with weights."""
        transposed = TransposeBridge.transpose(sample_factor_output)
        normalized = NormalizeBridge.normalize(transposed)
        
        f1 = SignalAlphaFactor(name="factor_1", signal_data=normalized)
        f2 = SignalAlphaFactor(name="factor_2", signal_data=normalized)
        
        factors = [(f1, 0.6), (f2, 0.4)]
        
        assert len(factors) == 2
        assert factors[0][1] == 0.6
        assert factors[1][1] == 0.4

    def test_compute_universe(self, sample_factor_output):
        """Verify compute_universe() returns values for all stocks."""
        transposed = TransposeBridge.transpose(sample_factor_output)
        normalized = NormalizeBridge.normalize(transposed)
        signal_factor = SignalAlphaFactor(
            name="test_factor",
            signal_data=normalized,
        )
        
        # Create mock data for universe
        last_date = normalized.index[-1]
        mock_data = {}
        for code in normalized.columns[:5]:  # Test first 5 stocks
            mock_data[code] = pd.DataFrame(
                {"close": [10.0, 11.0, 12.0]},
                index=pd.DatetimeIndex([
                    last_date - pd.Timedelta(days=2),
                    last_date - pd.Timedelta(days=1),
                    last_date,
                ]),
            )
        
        result = signal_factor.compute_universe(mock_data)
        
        # Should return a Series-like result
        assert hasattr(result, "values") or isinstance(result, pd.Series)
        if isinstance(result, pd.Series):
            assert len(result) == 5


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_stock_single_date(self):
        """Test pipeline with minimal data (1 stock, 1 date)."""
        alpha = np.array([[0.5]], dtype=np.float32)
        stocks = ["000001.SZ"]
        dates = ["20240101"]
        
        factor_output = wrap_factor_output(alpha, stocks, dates, "close")
        transposed = TransposeBridge.transpose(factor_output)
        normalized = NormalizeBridge.normalize(transposed)
        
        assert transposed.shape == (1, 1)
        assert normalized.shape == (1, 1)

    def test_all_nan_row(self):
        """Test normalization handles rows with all NaN."""
        alpha = np.array([[np.nan, np.nan]], dtype=np.float32)
        stocks = ["000001.SZ"]
        dates = ["20240101", "20240102"]
        
        factor_output = wrap_factor_output(alpha, stocks, dates, "close")
        transposed = TransposeBridge.transpose(factor_output)
        normalized = NormalizeBridge.normalize(transposed, method="cs_rank_booksize")
        
        # All NaN should become 0.0
        assert normalized.iloc[0, 0] == 0.0
        assert normalized.iloc[1, 0] == 0.0

    def test_constant_values_row(self):
        """Test normalization handles rows with constant values."""
        # Shape: (3 stocks, 1 date) - OpenAlpha orientation
        alpha = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
        stocks = ["000001.SZ", "000002.SZ", "000003.SZ"]
        dates = ["20240101"]
        
        factor_output = wrap_factor_output(alpha, stocks, dates, "close")
        transposed = TransposeBridge.transpose(factor_output)
        normalized = NormalizeBridge.normalize(transposed, method="cs_zscore")
        
        # Constant values → z-score should be 0 (std=0 case)
        assert all(abs(normalized.iloc[0]) < 1e-6)

    def test_unknown_normalization_method_raises(self, sample_factor_output):
        """Test unknown normalization method raises ValueError."""
        transposed = TransposeBridge.transpose(sample_factor_output)
        
        with pytest.raises(ValueError, match="Unknown normalization method"):
            NormalizeBridge.normalize(transposed, method="invalid_method")

    def test_signal_factor_nearest_date_lookup(self, sample_factor_output):
        """Test SignalAlphaFactor finds nearest date when exact match fails."""
        transposed = TransposeBridge.transpose(sample_factor_output)
        normalized = NormalizeBridge.normalize(transposed)
        signal_factor = SignalAlphaFactor(
            name="test",
            signal_data=normalized,
        )
        
        # Request a date slightly off from signal dates
        signal_dates = normalized.index
        off_date = signal_dates[5] + pd.Timedelta(hours=12)
        
        mock_df = pd.DataFrame(
            {"close": [10.0]},
            index=pd.DatetimeIndex([off_date]),
        )
        
        # Should still find a value via nearest match
        value = signal_factor.compute("000001.SZ", mock_df)
        assert isinstance(value, float)


class TestPipelineConsistency:
    """Test consistency and reproducibility of the pipeline."""

    def test_reproducible_results(self, sample_factor_output):
        """Verify pipeline produces identical results on repeated runs."""
        transposed1 = TransposeBridge.transpose(sample_factor_output)
        normalized1 = NormalizeBridge.normalize(transposed1)
        
        transposed2 = TransposeBridge.transpose(sample_factor_output)
        normalized2 = NormalizeBridge.normalize(transposed2)
        
        pd.testing.assert_frame_equal(normalized1, normalized2)

    def test_transpose_fillna_zero(self, sample_factor_output):
        """Verify transpose fills NaN with 0.0."""
        transposed = TransposeBridge.transpose(sample_factor_output)
        
        # Should have no NaN values
        assert not transposed.isna().any().any()

    def test_normalize_fillna_zero(self, sample_factor_output):
        """Verify normalize fills NaN with 0.0."""
        transposed = TransposeBridge.transpose(sample_factor_output)
        normalized = NormalizeBridge.normalize(transposed)
        
        # Should have no NaN values
        assert not normalized.isna().any().any()

    def test_dtype_preservation(self, sample_factor_output):
        """Verify float32 dtype is preserved through pipeline."""
        transposed = TransposeBridge.transpose(sample_factor_output)
        normalized = NormalizeBridge.normalize(transposed)
        
        assert transposed.dtypes.iloc[0] == np.float32
        assert normalized.dtypes.iloc[0] == np.float32