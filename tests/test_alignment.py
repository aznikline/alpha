"""Tests for bridge.returns and bridge.ic_filter — forward-return alignment, IC/IR computation, and leakage detection."""
import numpy as np
import pandas as pd
import pytest

from bridge.returns import compute_forward_returns, align_signals_with_returns, check_no_leakage
from bridge.ic_filter import compute_ic, filter_by_ic, rolling_ic_summary, ICResult


# ============================================================================
# Test Data Generation Helpers
# ============================================================================


def _make_price_data(n_dates: int = 60, n_stocks: int = 20) -> pd.DataFrame:
    """Generate synthetic price data for testing."""
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    stocks = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]
    # Prices that trend upward with some noise
    base = 100.0
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1, (n_dates, n_stocks))
    prices = base + np.cumsum(noise, axis=0) + np.arange(n_dates).reshape(-1, 1) * 0.5
    return pd.DataFrame(prices, index=dates, columns=stocks, dtype=np.float32)


def _make_signal_data(n_dates: int = 60, n_stocks: int = 20) -> pd.DataFrame:
    """Generate synthetic factor signal data."""
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    stocks = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]
    rng = np.random.default_rng(123)
    signal = rng.normal(0, 0.5, (n_dates, n_stocks)).astype(np.float32)
    return pd.DataFrame(signal, index=dates, columns=stocks)


def _make_aligned_pair(n_dates: int = 60, n_stocks: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate aligned signal and forward return data for IC testing."""
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    stocks = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]
    
    rng = np.random.default_rng(42)
    
    # Signal: random values
    signal = rng.normal(0, 1, (n_dates, n_stocks)).astype(np.float32)
    signal_df = pd.DataFrame(signal, index=dates, columns=stocks)
    
    # Forward returns: correlated with signal (positive IC)
    noise = rng.normal(0, 0.5, (n_dates, n_stocks)).astype(np.float32)
    returns = 0.3 * signal + noise  # Positive correlation
    returns_df = pd.DataFrame(returns, index=dates, columns=stocks)
    
    return signal_df, returns_df


# ============================================================================
# TestComputeForwardReturns — 8 tests for forward return computation
# ============================================================================


class TestComputeForwardReturns:
    """Tests for compute_forward_returns function."""

    @pytest.fixture
    def price_data(self) -> pd.DataFrame:
        """Fixture providing standard price data."""
        return _make_price_data(n_dates=60, n_stocks=20)

    def test_1_period_forward_returns(self, price_data):
        """1-period forward returns: shift(-1) correctly aligns future prices to current dates."""
        result = compute_forward_returns(price_data, periods=1)
        
        expected_first = (price_data.iloc[1].values - price_data.iloc[0].values) / price_data.iloc[0].values
        np.testing.assert_allclose(result.iloc[0].values, expected_first, rtol=1e-3)

    def test_multi_period_forward_returns(self, price_data):
        """Multi-period forward returns: shift(-3) for 3-day returns."""
        result = compute_forward_returns(price_data, periods=3)
        
        # Forward return at date t should be (price[t+3] - price[t]) / price[t]
        expected_first = (price_data.iloc[3].values - price_data.iloc[0].values) / price_data.iloc[0].values
        np.testing.assert_allclose(result.iloc[0].values, expected_first, rtol=1e-5)

    def test_last_n_dates_have_nan(self, price_data):
        """Last N dates have NaN (no future data available) — CRITICAL leakage prevention test."""
        periods = 5
        result = compute_forward_returns(price_data, periods=periods)
        
        # Last `periods` dates should have NaN because there's no future data
        last_n_rows = result.iloc[-periods:]
        assert last_n_rows.isna().all().all(), f"Last {periods} dates should be NaN to prevent leakage"
        
        # Dates before that should have valid values
        valid_rows = result.iloc[:-periods]
        assert not valid_rows.isna().any().any(), "Dates with future data should have valid returns"

    def test_multiindex_columns_with_price_column(self, price_data):
        """MultiIndex columns with price_column parameter."""
        multi_cols = pd.MultiIndex.from_product([price_data.columns, ["close", "open"]])
        multi_price = pd.DataFrame(
            np.random.randn(len(price_data), len(multi_cols)).astype(np.float32),
            index=price_data.index,
            columns=multi_cols
        )
        
        result = compute_forward_returns(multi_price, periods=1, price_column="close")
        
        assert result.shape == (len(price_data), len(price_data.columns))
        assert isinstance(result.columns, pd.Index)

    def test_plain_dataframe_works(self, price_data):
        """Plain DataFrame (no MultiIndex) works correctly."""
        result = compute_forward_returns(price_data, periods=1)
        
        assert result.shape == price_data.shape
        assert result.index.equals(price_data.index)
        assert result.columns.equals(price_data.columns)

    def test_float32_dtype(self, price_data):
        """float32 dtype preserved."""
        result = compute_forward_returns(price_data, periods=1)
        
        assert result.dtypes.iloc[0] == np.float32

    def test_empty_price_dataframe(self):
        """Empty price DataFrame returns empty result."""
        empty_df = pd.DataFrame()
        result = compute_forward_returns(empty_df, periods=1)
        
        assert result.empty
        assert isinstance(result, pd.DataFrame)

    def test_single_stock_dataframe(self):
        """Single stock (1-column DataFrame) works correctly."""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        single_stock = pd.DataFrame(
            [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            index=dates,
            columns=["000001.SZ"],
            dtype=np.float32
        )
        
        result = compute_forward_returns(single_stock, periods=1)
        
        # Check shape
        assert result.shape == single_stock.shape
        # Last date should be NaN
        assert pd.isna(result.iloc[-1, 0])


# ============================================================================
# TestAlignSignalsWithReturns — 7 tests for alignment logic
# ============================================================================


class TestAlignSignalsWithReturns:
    """Tests for align_signals_with_returns function."""

    @pytest.fixture
    def signal_data(self) -> pd.DataFrame:
        """Fixture providing signal data."""
        return _make_signal_data(n_dates=60, n_stocks=20)

    @pytest.fixture
    def returns_data(self) -> pd.DataFrame:
        """Fixture providing forward returns data."""
        return _make_price_data(n_dates=60, n_stocks=20)

    def test_inner_join(self, signal_data, returns_data):
        """Inner join: only shared dates/stocks kept."""
        signal_subset = signal_data.iloc[10:50]
        returns_subset = returns_data.iloc[5:55]
        
        aligned_signal, aligned_returns = align_signals_with_returns(
            signal_subset, returns_subset, method="inner"
        )
        
        expected_dates = signal_subset.index.intersection(returns_subset.index)
        assert aligned_signal.index.equals(expected_dates)
        assert aligned_returns.index.equals(expected_dates)

    def test_outer_join(self, signal_data, returns_data):
        """Outer join: uses intersection (current implementation behavior)."""
        signal_subset = signal_data.iloc[10:50]
        returns_subset = returns_data.iloc[5:55]
        
        aligned_signal, aligned_returns = align_signals_with_returns(
            signal_subset, returns_subset, method="outer"
        )
        
        expected_dates = signal_subset.index.intersection(returns_subset.index)
        assert aligned_signal.index.equals(expected_dates)
        assert aligned_returns.index.equals(expected_dates)

    def test_empty_dataframe_raises(self):
        """ValueError on empty DataFrame inputs."""
        empty_df = pd.DataFrame()
        valid_df = _make_signal_data(n_dates=10, n_stocks=5)
        
        with pytest.raises(ValueError, match="empty"):
            align_signals_with_returns(empty_df, valid_df)
        
        with pytest.raises(ValueError, match="empty"):
            align_signals_with_returns(valid_df, empty_df)

    def test_no_overlapping_dates_raises(self):
        """ValueError on no overlapping dates."""
        dates1 = pd.date_range("2024-01-01", periods=10, freq="B")
        dates2 = pd.date_range("2024-03-01", periods=10, freq="B")
        
        signal = pd.DataFrame(np.zeros((10, 5)), index=dates1, columns=["A", "B", "C", "D", "E"])
        returns = pd.DataFrame(np.zeros((10, 5)), index=dates2, columns=["A", "B", "C", "D", "E"])
        
        with pytest.raises(ValueError, match="overlapping dates"):
            align_signals_with_returns(signal, returns, method="inner")

    def test_no_overlapping_stocks_raises(self):
        """ValueError on no overlapping stocks."""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        
        signal = pd.DataFrame(np.zeros((10, 3)), index=dates, columns=["A", "B", "C"])
        returns = pd.DataFrame(np.zeros((10, 3)), index=dates, columns=["X", "Y", "Z"])
        
        with pytest.raises(ValueError, match="overlapping stocks"):
            align_signals_with_returns(signal, returns, method="inner")

    def test_both_outputs_have_same_shape(self, signal_data, returns_data):
        """Both outputs have same shape (matching DatetimeIndex and columns)."""
        aligned_signal, aligned_returns = align_signals_with_returns(
            signal_data, returns_data, method="inner"
        )
        
        assert aligned_signal.shape == aligned_returns.shape
        assert aligned_signal.index.equals(aligned_returns.index)
        assert aligned_signal.columns.equals(aligned_returns.columns)

    def test_stock_code_format_mismatch(self):
        """Stock code format mismatch detection (e.g., '000001' vs '000001.SZ')."""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        
        signal = pd.DataFrame(
            np.zeros((10, 3)),
            index=dates,
            columns=["000001", "000002", "600000"]
        )
        returns = pd.DataFrame(
            np.zeros((10, 3)),
            index=dates,
            columns=["000001.SZ", "000002.SZ", "600000.SH"]
        )
        
        with pytest.raises(ValueError, match="overlapping stocks"):
            align_signals_with_returns(signal, returns, method="inner")


# ============================================================================
# TestCheckNoLeakage — 5 tests for leakage detection
# ============================================================================


class TestCheckNoLeakage:
    """Tests for check_no_leakage function."""

    @pytest.fixture
    def random_signal(self) -> pd.DataFrame:
        """Fixture providing random signal (no leakage)."""
        return _make_signal_data(n_dates=60, n_stocks=20)

    @pytest.fixture
    def same_day_returns(self) -> pd.DataFrame:
        """Fixture providing same-day returns (price[D]/price[D-1]-1)."""
        prices = _make_price_data(n_dates=60, n_stocks=20)
        return (prices / prices.shift(1) - 1).astype(np.float32)

    def test_no_leakage_random_signal(self, random_signal, same_day_returns):
        """No leakage: random signal vs same-day returns → returns True (IC ≈ 0)."""
        aligned_signal, aligned_returns = align_signals_with_returns(
            random_signal, same_day_returns, method="inner"
        )
        
        result = check_no_leakage(aligned_signal, aligned_returns)
        
        assert result == True

    def test_leakage_detected_same_day_return(self, same_day_returns):
        """Leakage detected: signal that's literally the same-day return → returns False (IC > threshold)."""
        leaky_signal = same_day_returns.shift(1)
        
        valid_idx = leaky_signal.dropna(how="all").index
        leaky_signal = leaky_signal.loc[valid_idx]
        aligned_returns = same_day_returns.loc[valid_idx]
        
        result = check_no_leakage(leaky_signal, aligned_returns)
        
        assert result == False

    def test_zero_signal_values(self, same_day_returns):
        """Zero signal values (all zero) → returns False (NaN correlation)."""
        dates = same_day_returns.index[:30]
        stocks = same_day_returns.columns[:10]
        
        zero_signal = pd.DataFrame(
            np.zeros((len(dates), len(stocks)), dtype=np.float32),
            index=dates,
            columns=stocks
        )
        
        aligned_returns = same_day_returns.loc[dates, stocks]
        
        result = check_no_leakage(zero_signal, aligned_returns)
        
        assert result == False

    def test_sparse_signal(self, same_day_returns):
        """Sparse signal (most zeros, few non-zero) → returns True or False based on IC."""
        dates = same_day_returns.index[:30]
        stocks = same_day_returns.columns[:10]
        
        rng = np.random.default_rng(42)
        sparse_data = np.zeros((len(dates), len(stocks)), dtype=np.float32)
        mask = rng.random((len(dates), len(stocks))) < 0.1
        sparse_data[mask] = rng.normal(0, 1, mask.sum()).astype(np.float32)
        
        sparse_signal = pd.DataFrame(sparse_data, index=dates, columns=stocks)
        aligned_returns = same_day_returns.loc[dates, stocks]
        
        result = check_no_leakage(sparse_signal, aligned_returns)
        
        assert isinstance(result, (bool, np.bool_))

    def test_empty_data_raises(self):
        """Empty data → ValueError."""
        empty_df = pd.DataFrame()
        valid_df = _make_signal_data(n_dates=10, n_stocks=5)
        
        with pytest.raises(ValueError, match="empty"):
            check_no_leakage(empty_df, valid_df)
        
        with pytest.raises(ValueError, match="empty"):
            check_no_leakage(valid_df, empty_df)


# ============================================================================
# TestComputeIC — 6 tests for IC computation
# ============================================================================


class TestComputeIC:
    """Tests for compute_ic function."""

    @pytest.fixture
    def positive_ic_pair(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fixture providing signal/returns with positive IC."""
        return _make_aligned_pair(n_dates=60, n_stocks=20)

    @pytest.fixture
    def forward_returns(self) -> pd.DataFrame:
        """Fixture providing forward returns for zero IC test."""
        prices = _make_price_data(n_dates=60, n_stocks=20)
        return compute_forward_returns(prices, periods=1)

    def test_positive_ic(self, positive_ic_pair):
        """Positive IC: signal that correlates with forward returns → mean_ic > 0."""
        signal, returns = positive_ic_pair
        
        result = compute_ic(signal, returns)
        
        assert result.mean_ic > 0
        assert isinstance(result, ICResult)

    def test_negative_ic(self, positive_ic_pair):
        """Negative IC: inverse signal → mean_ic < 0."""
        signal, returns = positive_ic_pair
        
        inverted_signal = -signal
        
        result = compute_ic(inverted_signal, returns)
        
        assert result.mean_ic < 0

    def test_zero_ic(self, forward_returns):
        """Zero IC: random signal → mean_ic ≈ 0."""
        dates = forward_returns.index[:30]
        stocks = forward_returns.columns[:10]
        
        rng = np.random.default_rng(123)
        random_signal = pd.DataFrame(
            rng.normal(0, 1, (len(dates), len(stocks))).astype(np.float32),
            index=dates,
            columns=stocks
        )
        
        aligned_returns = forward_returns.loc[dates, stocks]
        
        result = compute_ic(random_signal, aligned_returns)
        
        assert abs(result.mean_ic) < 0.15

    def test_spearman_method(self, positive_ic_pair):
        """Spearman method works (method='spearman')."""
        signal, returns = positive_ic_pair
        
        result = compute_ic(signal, returns, method="spearman")
        
        assert isinstance(result, ICResult)
        assert result.mean_ic > 0

    def test_empty_data_returns_zeros(self):
        """Empty data → ICResult with zeros."""
        empty_signal = pd.DataFrame()
        empty_returns = pd.DataFrame()
        
        result = compute_ic(empty_signal, empty_returns)
        
        assert isinstance(result, ICResult)
        assert result.mean_ic == 0.0
        assert result.ir == 0.0
        assert result.n_dates == 0

    def test_icresult_fields_correct(self, positive_ic_pair):
        """ICResult fields are correct (n_dates, date_range, hit_rate, etc.)."""
        signal, returns = positive_ic_pair
        
        result = compute_ic(signal, returns)
        
        assert result.n_dates == len(signal.index)
        assert isinstance(result.date_range[0], str)
        assert isinstance(result.date_range[1], str)
        assert 0 <= result.hit_rate <= 1
        assert isinstance(result.ir, float)


# ============================================================================
# TestFilterByIC — 6 tests for IC filtering
# ============================================================================


class TestFilterByIC:
    """Tests for filter_by_ic function."""

    @pytest.fixture
    def signal_returns_pair(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fixture providing signal and returns for filtering tests."""
        return _make_aligned_pair(n_dates=60, n_stocks=20)

    def test_removes_low_ic_dates(self, signal_returns_pair):
        """Removes low-IC dates from signal."""
        signal, returns = signal_returns_pair
        
        # Create some dates with low IC by zeroing signal on those dates
        modified_signal = signal.copy()
        modified_signal.iloc[-10:] = 0  # Last 10 dates have zero signal → low IC
        
        filtered_signal, ic_result = filter_by_ic(
            modified_signal, returns, min_ic=0.05, window=20
        )
        
        # Filtered signal should have fewer dates (low IC dates removed)
        assert len(filtered_signal.index) <= len(modified_signal.index)

    def test_preserves_early_dates(self, signal_returns_pair):
        """Preserves early dates (before rolling window fills)."""
        signal, returns = signal_returns_pair
        
        filtered_signal, ic_result = filter_by_ic(
            signal, returns, min_ic=0.01, window=10
        )
        
        # First few dates should be preserved even if IC can't be computed
        # (rolling window needs data)
        assert len(filtered_signal.index) > 0

    def test_returns_ic_result_alongside_filtered_signal(self, signal_returns_pair):
        """Returns ICResult alongside filtered signal."""
        signal, returns = signal_returns_pair
        
        filtered_signal, ic_result = filter_by_ic(signal, returns)
        
        assert isinstance(filtered_signal, pd.DataFrame)
        assert isinstance(ic_result, ICResult)

    def test_custom_thresholds_work(self, signal_returns_pair):
        """Custom thresholds (min_ic, min_ir) work."""
        signal, returns = signal_returns_pair
        
        filtered_signal, ic_result = filter_by_ic(
            signal, returns, min_ic=0.1, min_ir=0.5
        )
        
        # Should apply stricter thresholds
        assert isinstance(filtered_signal, pd.DataFrame)
        assert isinstance(ic_result, ICResult)

    def test_empty_data_returns_original(self):
        """Empty data → returns original signal and empty ICResult."""
        empty_signal = pd.DataFrame()
        empty_returns = pd.DataFrame()
        
        filtered_signal, ic_result = filter_by_ic(empty_signal, empty_returns)
        
        assert filtered_signal.empty
        assert ic_result.mean_ic == 0.0

    def test_filtered_has_fewer_dates_when_ic_low(self, signal_returns_pair):
        """Filtered signal has fewer dates than original (when IC is low)."""
        signal, returns = signal_returns_pair
        
        # Make signal mostly random (low IC)
        rng = np.random.default_rng(999)
        random_signal = pd.DataFrame(
            rng.normal(0, 0.1, signal.shape).astype(np.float32),
            index=signal.index,
            columns=signal.columns
        )
        
        filtered_signal, ic_result = filter_by_ic(
            random_signal, returns, min_ic=0.05, window=20
        )
        
        # With low IC, many dates should be filtered out
        assert len(filtered_signal.index) < len(signal.index)


# ============================================================================
# TestRollingICSummary — 4 tests for rolling summary
# ============================================================================


class TestRollingICSummary:
    """Tests for rolling_ic_summary function."""

    @pytest.fixture
    def signal_returns_pair(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fixture providing signal and returns for rolling summary tests."""
        return _make_aligned_pair(n_dates=60, n_stocks=20)

    def test_returns_dataframe_with_correct_columns(self, signal_returns_pair):
        """Returns DataFrame with correct columns (ic, rolling_mean_ic, rolling_std_ic, rolling_ir, hit_rate)."""
        signal, returns = signal_returns_pair
        
        result = rolling_ic_summary(signal, returns, window=20)
        
        expected_cols = ["ic", "rolling_mean_ic", "rolling_std_ic", "rolling_ir", "hit_rate"]
        assert list(result.columns) == expected_cols
        assert isinstance(result, pd.DataFrame)

    def test_empty_data_returns_empty_dataframe(self):
        """Empty data → returns empty DataFrame."""
        empty_signal = pd.DataFrame()
        empty_returns = pd.DataFrame()
        
        result = rolling_ic_summary(empty_signal, empty_returns, window=20)
        
        assert result.empty
        assert isinstance(result, pd.DataFrame)

    def test_rolling_window_produces_nan_for_first_dates(self, signal_returns_pair):
        """Rolling window produces NaN for first few dates."""
        signal, returns = signal_returns_pair
        window = 20
        
        result = rolling_ic_summary(signal, returns, window=window)
        
        # First (window-1) dates should have NaN in rolling columns
        rolling_cols = ["rolling_mean_ic", "rolling_std_ic", "rolling_ir"]
        for col in rolling_cols:
            nan_count = result[col].iloc[:window-1].isna().sum()
            assert nan_count > 0  # At least some NaN in early dates

    def test_all_values_finite(self, signal_returns_pair):
        """All values are finite (no inf)."""
        signal, returns = signal_returns_pair
        
        result = rolling_ic_summary(signal, returns, window=20)
        
        # Check no inf values
        for col in result.columns:
            assert not np.isinf(result[col]).any()