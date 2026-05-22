# Unified Data Layer Specification

> Version: 0.2 | Date: 2026-05-22 | Status: Proposed — verified against current code
>
> Last verified against: OpenAlpha commit [unknown], qmt commit [unknown]
> Files checked: OpenAlpha/src/simres/expr.py, qmt/src/qmt_local/data/manager.py
>
> This document specifies the shared data infrastructure that will serve all three
> projects (OpenAlpha, qmt, ptrade) with consistent, point-in-time-safe data access.

---

## 1. Problem Statement

Currently each project fetches data independently:

| Project | Source | Format | PIT Safety | Coverage |
|---------|--------|--------|------------|----------|
| OpenAlpha | akshare (inline) | Raw pandas | No PIT | CSI 500 OHLCV |
| qmt | xtquant SDK | NautilusTrader bars | PIT via xtquant | Full A-share |
| ptrade | akshare (inline) | Ptrade API format | No PIT | OHLCV + basic financials |

**Issues**:
- Duplicate downloads across projects
- No PIT protection in OpenAlpha/ptrade leads to future leakage in factor analysis
- Different formats require per-project normalization
- No caching leads to repeated network calls

---

## 2. Architecture

```
+---------------------------------------------------------------+
|                   Unified Data Layer                           |
|                                                               |
|  +-------------+    +-------------+    +-------------+       |
|  |DataProvider |--> | ParquetCache|--> | PITManager  |       |
|  |   ABC       |    |  (local fs) |    |  (safety)   |       |
|  +-------------+    +-------------+    +-------------+       |
|       |                   |                   |               |
|  +----v----+         +---v---+          +---v---+           |
|  |Akshare  |         |Local  |          |PIT    |           |
|  |Provider |         |Parquet|          |Index   |           |
|  +---------+         |Files  |          |(audit  |           |
|  +---------+         +-------+          | trail) |           |
|  |xtquant  |                            +-------+           |
|  |Provider |                            +-------+           |
|  |(future) |                            |Qlib   |           |
|  +---------+                            |Cache  |           |
|                                         |(future|           |
|                                         |)      |           |
|                                         +-------+           |
|                                                               |
|  +---------------------------------------------------+       |
|  | DataManager (Facade)                              |       |
|  | Single entry point for all data operations        |       |
|  +---------------------------------------------------+       |
+---------------------------------------------------------------+

         ^           ^           ^
         |           |           |
    OpenAlpha      qmt        ptrade
    (eval expr)   (backtest)  (strategy)
```

---

## 3. DataProvider ABC

### 3.1 Interface Definition

```python
from abc import ABC, abstractmethod
import pandas as pd

class DataProvider(ABC):
    """Unified data access interface for all three projects.
    
    All methods return (Date, Stock) orientation. The bridge layer
    handles conversion to OpenAlpha's (Stock, Date) when needed.
    """

    @abstractmethod
    def get_daily(
        self,
        stocks: list[str],            # qmt-format codes e.g. ["000001.SZ", "600000.SH"]
        start: str,                   # ISO date "YYYY-MM-DD"
        end: str,                     # ISO date "YYYY-MM-DD"
        fields: list[str] = ["open", "high", "low", "close", "volume", "amount"],
    ) -> pd.DataFrame:
        """Return daily market data.
        
        Returns:
            pd.DataFrame with:
              - index: pd.DatetimeIndex (trading dates)
              - columns: pd.MultiIndex with levels (stock_code, field)
              - dtype: float32 for prices, int64 for volume/amount
        
        Convention: (Date, Stock) orientation via MultiIndex columns.
        All values are float32. Volume/amount are int64.
        No NaN. Missing data filled with 0.0 for prices, 0 for volume.
        
        Example shape for 500 stocks over 60 trading days with 6 fields:
            (60, 3000) - 60 dates x (500 stocks x 6 fields)
        
        Selection example:
            df.loc["2024-01-15", ("000001.SZ", "close")]  # Single value
            df.loc["2024-01-15", "000001.SZ"]             # All fields for one stock on one date
            df.xs("close", level=1, axis=1)               # Close prices for all stocks, all dates
        """
        ...

    @abstractmethod
    def get_pit(
        self,
        stock: str,                   # qmt-format code
        field: str,                   # Financial field e.g. "eps", "revenue", "roe"
        date: str,                    # Point-in-time read date (ISO format)
    ) -> float | None:
        """Return financial data as known on `date`.
        
        CRITICAL: This prevents future leakage. Only returns data that was
        publicly available on or before `date`. Financial reports published
        after `date` are excluded.
        
        Args:
            stock: Stock code in qmt format
            field: Financial field name
            date: The "as-of" date for point-in-time read
        
        Returns:
            float: The PIT value, or None if not available on that date.
        
        PIT Rules:
            - Annual reports: available from announcement_date onward
            - Quarterly reports: available from announcement_date onward
            - If no report announced by `date`, return the last known value
            - If no value ever known, return None
        """
        ...

    @abstractmethod
    def get_pit_batch(
        self,
        stocks: list[str],
        field: str,
        date: str,
    ) -> pd.Series:
        """Batch PIT read for multiple stocks.
        
        Returns:
            pd.Series: Index=stock codes, Values=PIT float values
            NaN where no data available on that date.
        """
        ...

    @abstractmethod
    def get_index_weights(
        self,
        index: str,                   # "csi_500" | "csi_300" | "sse_50"
        date: str,                    # ISO date
    ) -> pd.Series:
        """Return index constituent weights on date.
        
        Returns:
            pd.Series: Index=stock codes (qmt format), Values=weight (float32)
            Sum of weights = 1.0
            Weights reflect the latest rebalance before `date`.
        """
        ...

    @abstractmethod
    def get_universe(
        self,
        index: str,                   # "csi_500" | "csi_300" | "all_a"
        date: str,                    # ISO date
    ) -> list[str]:
        """Return universe constituent stock codes on date.
        
        Returns:
            list[str]: Stock codes in qmt format
            Ordered by code value.
        """
        ...

    @abstractmethod
    def get_trading_dates(
        self,
        start: str,
        end: str,
    ) -> list[str]:
        """Return trading dates (excluding holidays) in range.
        
        Returns:
            list[str]: ISO date strings, ascending order
        """
        ...
```

---

## 4. AkshareProvider Implementation

### 4.1 Class Definition

```python
class AkshareProvider(DataProvider):
    """Data provider using akshare as the upstream source.
    
    Features:
    - CSI 500/300 constituent lists and weights
    - Daily OHLCV for all A-shares
    - Basic financial data (eps, revenue, roe) via announcement dates
    - Trading calendar (excluding A-share holidays)
    """

    def __init__(self, cache_dir: str = "~/.alpha_data/parquet"):
        self._cache = ParquetCache(cache_dir)
        self._pit_manager = PITManager()
        self._akshare = ak  # akshare module
    
    def get_daily(self, stocks, start, end, fields) -> pd.DataFrame:
        """Fetch daily data, cache to Parquet, return DatetimeIndex x MultiIndex columns."""
        # 1. Check cache first
        cached = self._cache.get_daily(stocks, start, end, fields)
        if cached is not None:
            return cached
        
        # 2. Download from akshare
        # akshare API: ak.stock_zh_a_hist(symbol, period="daily", start_date, end_date)
        # Need to loop per stock (akshare has no batch API for A-share daily)
        raw = {}
        for code_qmt in stocks:
            symbol = StockCodeMapper.to_openalpha(code_qmt)  # "000001.SZ" -> 1
            symbol_str = f"{symbol:06d}"
            df = ak.stock_zh_a_hist(symbol=symbol_str, period="daily",
                                     start_date=start.replace("-",""),
                                     end_date=end.replace("-",""))
            raw[code_qmt] = df
        
        # 3. Assemble into DatetimeIndex x MultiIndex(stock, field) format
        result = self._assemble_daily(raw, fields)
        
        # 4. Cache result
        self._cache.put_daily(stocks, start, end, fields, result)
        
        return result
    
    def get_pit(self, stock, field, date) -> float | None:
        """PIT read via announcement dates from akshare financial data."""
        return self._pit_manager.get(stock, field, date)
    
    def get_index_weights(self, index, date) -> pd.Series:
        """Fetch index weights from akshare.
        akshare API: ak.index_stock_cons_weight_csindex(symbol=index_code, date=date)
        """
        index_map = {"csi_500": "000905", "csi_300": "000300", "sse_50": "000016"}
        symbol = index_map[index]
        df = ak.index_stock_cons_weight_csindex(symbol=symbol, date=date.replace("-",""))
        # Convert to Series with qmt-format codes
        weights = pd.Series(...)
        return weights
    
    def get_universe(self, index, date) -> list[str]:
        """Fetch universe constituents from akshare.
        akshare API: ak.index_stock_cons_csindex(symbol=index_code)
        """
        ...
    
    def get_trading_dates(self, start, end) -> list[str]:
        """Fetch trading calendar from akshare.
        akshare API: ak.tool_trade_date_hist_sina()
        """
        ...
```

### 4.2 akshare API Mapping

| Our Method | akshare API | Notes |
|-----------|------------|-------|
| `get_daily()` | `ak.stock_zh_a_hist(symbol, period="daily")` | Per-stock, no batch |
| `get_pit()` | `ak.stock_financial_analysis_indicator(symbol)` | Needs announcement date parsing |
| `get_index_weights()` | `ak.index_stock_cons_weight_csindex(symbol, date)` | Direct batch |
| `get_universe()` | `ak.index_stock_cons_csindex(symbol)` | Direct batch |
| `get_trading_dates()` | `ak.tool_trade_date_hist_sina()` | Full calendar |

---

## 5. ParquetCache

### 5.1 Cache Strategy

```python
class ParquetCache:
    """Local Parquet file cache for DataProvider results.
    
    Cache Structure:
        ~/.alpha_data/parquet/
        |-- daily/
        |   |-- 2024/
        |   |   |-- 01/
        |   |   |   |-- csi_500_ohlcv.parquet   # Full month, all stocks
        |   |   |   |-- csi_500_ohlcv.meta.json  # Metadata (fields, date range)
        |   |   |-- 02/
        |   |   |   |-- ...
        |-- pit/
        |   |-- eps.parquet                       # All stocks, all dates
        |   |-- revenue.parquet
        |   |-- roe.parquet
        |-- index/
        |   |-- csi_500_weights.parquet           # All rebalance dates
        |   |-- csi_300_weights.parquet
        |-- universe/
        |   |-- csi_500.parquet
        |   |-- csi_300.parquet
        |-- calendar/
            |-- trading_dates.parquet
    """

    def __init__(self, cache_dir: str):
        self._cache_dir = Path(cache_dir).expanduser()
    
    def get_daily(self, stocks, start, end, fields) -> pd.DataFrame | None:
        """Check if cached data covers the requested range.
        
        Logic:
            1. Find all monthly Parquet files covering start->end
            2. If all months present and fields match -> load and concatenate
            3. If any month missing -> return None (trigger download)
        """
        ...
    
    def put_daily(self, stocks, start, end, fields, df) -> None:
        """Cache daily data, partitioned by month.
        
        Logic:
            1. Split df by month (based on date index)
            2. Write each month as a separate Parquet file
            3. Write metadata JSON alongside
        """
        ...
    
    def invalidate(self, data_type: str, start: str, end: str) -> None:
        """Remove cached files in a date range.
        
        Use when upstream data changes (e.g. index rebalance).
        """
        ...
```

### 5.2 Cache Invariants

- **Daily data**: Partitioned by year/month. Each file contains all stocks for that month.
- **PIT data**: Single file per field, covering all stocks and all dates. Updated incrementally.
- **Index data**: Single file per index, all rebalance dates. Updated when new rebalance occurs.
- **Cache key**: `(data_type, year, month, fields, universe)` ensures correct data is returned.
- **Compression**: Parquet with `snappy` compression (good balance of speed/size).
- **Schema enforcement**: All Parquet files have explicit schemas with `float32` for prices, `int64` for volume.

---

## 6. PITManager

### 6.1 Point-in-Time Safety

```python
class PITManager:
    """Ensures financial data reads are point-in-time safe.
    
    Core principle: Only return data that was publicly available on or before
    the requested date. This prevents future leakage in factor analysis.
    
    Implementation:
    - Maintains an announcement_date index per stock per field
    - announcement_date is when the financial report was published
    - When get() is called with date=X, only returns values from reports
      with announcement_date <= X
    """

    def __init__(self, source: str = "akshare"):
        self._source = source
        self._announcement_index: dict[str, dict[str, pd.Series]] = {}
        # Structure: {field: {stock_code: Series(date->announcement_date)}}
    
    def get(self, stock: str, field: str, date: str) -> float | None:
        """PIT read for a single stock/field/date.
        
        Algorithm:
            1. Look up announcement dates for this stock/field
            2. Find the latest announcement_date <= date
            3. Return the value from that announcement
            4. If no announcement found, return None
        """
        announcements = self._announcement_index.get(field, {}).get(stock)
        if announcements is None:
            return None
        
        # Filter: only announcements before or on date
        valid = announcements[announcements <= pd.Timestamp(date)]
        if valid.empty:
            return None
        
        latest_announce = valid.max()
        return self._get_value(stock, field, latest_announce)
    
    def get_batch(self, stocks: list[str], field: str, date: str) -> pd.Series:
        """Batch PIT read. Returns Series(stock -> value)."""
        results = {}
        for stock in stocks:
            results[stock] = self.get(stock, field, date)
        return pd.Series(results, dtype=np.float32)
    
    def refresh(self, field: str) -> None:
        """Refresh announcement index from source.
        
        For akshare: download financial data with announcement dates,
        build index mapping (report_date -> announcement_date).
        """
        ...
```

### 6.2 PIT Audit Trail

```python
@dataclass
class PITAuditEntry:
    """Audit trail for PIT reads."""
    stock: str           # Stock code
    field: str           # Financial field
    request_date: str    # Date the user requested
    announce_date: str   # Date the report was announced
    report_date: str     # Period end date of the report
    value: float         # Value returned
    source: str          # "akshare" | "xtquant" | "qlib"

class PITAuditLog:
    """Maintains audit trail of all PIT reads for debugging and verification."""
    
    def log(self, entry: PITAuditEntry) -> None:
        """Log a PIT read."""
        ...
    
    def verify(self, stock: str, field: str, date: str) -> bool:
        """Verify that a PIT read was correct (no future leakage).
        
        Check: announce_date <= request_date for all logged reads.
        """
        ...
```

### 6.3 PIT Rules Reference

| Report Type | Frequency | Available From | Lag (Typical) |
|-------------|-----------|---------------|----------------|
| Annual report | Yearly | announcement_date | ~1-2 months after fiscal year end |
| Q1 quarterly | Quarterly | announcement_date | ~1 month after Q1 end (April 30 deadline) |
| Q2 semi-annual | Semi-annual | announcement_date | ~1-2 months after H1 end (August 31 deadline) |
| Q3 quarterly | Quarterly | announcement_date | ~1 month after Q3 end (October 31 deadline) |

**A-share disclosure deadlines**:
- Annual: April 30
- Q1: April 30
- H1: August 31
- Q3: October 31

---

## 7. DataManager Facade

### 7.1 Definition

```python
class DataManager:
    """Single entry point for all data operations across the three projects.
    
    Wraps DataProvider + ParquetCache + PITManager into a unified interface.
    """

    def __init__(
        self,
        providers: list[DataSource] | None = None,   # Multiple data sources
        storage: Storage | None = None,              # Storage backend
        cache_dir: str = "",                         # Cache directory path
    ):
        self._providers = providers or []
        self._storage = storage
        self._cache_dir = cache_dir
        self._pit = PITManager()
    
    def daily(self, stocks, start, end, fields) -> pd.DataFrame:
        """Get daily data. Returns DatetimeIndex x MultiIndex(stock, field)."""
        # Implementation uses configured providers
        ...
    
    def pit(self, stock, field, date) -> float | None:
        """Get PIT data. Point-in-time safe."""
        return self._pit.get(stock, field, date)
    
    def pit_batch(self, stocks, field, date) -> pd.Series:
        """Batch PIT read."""
        return self._pit.get_batch(stocks, field, date)
    
    def index_weights(self, index, date) -> pd.Series:
        """Get index weights."""
        ...
    
    def universe(self, index, date) -> list[str]:
        """Get universe stocks."""
        ...
    
    def trading_dates(self, start, end) -> list[str]:
        """Get trading calendar."""
        ...
    
    # Orientation helpers
    def daily_for_openalpha(self, stocks, start, end, fields=["close"]) -> pd.DataFrame:
        """Get single-field daily data in (Stock, Date) orientation for OpenAlpha.
        
        Only works for a single field at a time. OpenAlpha expects simple
        (Stock, Date) layout without multi-level columns.
        
        Note: OpenAlpha's evaluate() currently returns np.ndarray (Stock, Date),
        not DataFrame. The proposed FactorOutput would wrap this in a DataFrame.
        """
        df = self.daily(stocks, start, end, fields)
        # Reshape: select single field, pivot to (Stock, Date)
        if len(fields) == 1:
            field = fields[0]
            single = df.xs(field, level=1, axis=1)  # (Date, Stock) for single field
            return single.T  # (Stock, Date)
        else:
            raise ValueError("daily_for_openalpha() only supports single field. Use daily() for multi-field.")
    
    def daily_for_qmt(self, stocks, start, end, fields) -> pd.DataFrame:
        """Get daily data in DatetimeIndex x MultiIndex(stock, field) for qmt.
        
        Returns the same format as daily() - no transformation needed.
        qmt expects (Date, Stock) orientation with MultiIndex columns.
        """
        return self.daily(stocks, start, end, fields)
```

---

## 8. Wiring: OpenAlpha -> DataProvider

### 8.1 Current State

```python
# OpenAlpha/src/simres/expr.py (current)
class AlphaExecutor:
    def __init__(self, data_dir: str, alpha_dir: str | None = None):
        # Loads raw CSV/parquet files from data_dir
        # alpha_dir optional for alpha expression storage
        # No PIT, no caching, raw akshare download inline
```

### 8.2 Proposed State

```python
# OpenAlpha/src/simres/expr.py (modified)
class AlphaExecutor:
    def __init__(
        self,
        data_dir: str | None = None,   # Legacy: still accept raw data dir
        alpha_dir: str | None = None,  # Alpha expression storage
        data_manager: DataManager | None = None,  # NEW: accept DataManager
    ):
        if data_manager is not None:
            self._data = data_manager
        else:
            self._data = LegacyDataLoader(data_dir)  # Backward compat
    
    def evaluate(self, expr, universe, start, end) -> FactorOutput:
        """Evaluate alpha expression over universe and date range.
        
        Note: Current OpenAlpha returns np.ndarray (Stock, Date).
        Proposed FactorOutput would wrap this in a DataFrame with
        proper index/columns for downstream consumers.
        """
        # Use self._data.daily_for_openalpha() for market data
        # Use self._data.pit_batch() for financial data (PIT-safe)
        # Use self._data.universe() for universe definition
```

**Migration path**: Add `data_manager` parameter without removing `data_dir`. Existing scripts continue working. New bridge code uses DataManager.

---

## 9. Wiring: qmt -> DataProvider

### 9.1 Current State

```python
# qmt/src/qmt_local/data/manager.py (current)
class DataManager:
    def __init__(
        self,
        providers: list[DataSource] | None = None,
        storage: Storage | None = None,
        cache_dir: str = "",
    ):
        # providers: list of DataSource instances (plural)
        # storage: Storage backend for persistence
        # DataSource wraps xtquant SDK (Windows-only)
```

### 9.2 Proposed State

```python
# qmt/src/qmt_local/data/manager.py (modified)
class DataManager:
    def __init__(
        self,
        providers: list[DataSource] | None = None,   # Legacy: xtquant DataSource list
        storage: Storage | None = None,              # Storage backend
        cache_dir: str = "",                         # Cache directory
        shared_provider: DataProvider | None = None, # NEW: shared DataProvider
    ):
        if shared_provider is not None:
            self._provider = shared_provider
        elif providers is not None:
            self._provider = XtquantProvider(providers)  # Wrap existing
        else:
            raise ValueError("Must provide providers or shared_provider")
```

**Migration path**: qmt's DataManager accepts our shared DataProvider. When running on Windows with xtquant available, it uses XtquantProvider (wrapper). When on Mac/Linux, it uses AkshareProvider.

---

## 10. Future Data Providers

### 10.1 XtquantProvider (Windows-only, future)

```python
class XtquantProvider(DataProvider):
    """Data provider using xtquant SDK (Windows-only).
    
    Advantages over AkshareProvider:
    - Full A-share coverage including real-time data
    - Native PIT support (xtquant has announcement_date fields)
    - No network calls when xtquant is connected locally
    """
    
    def __init__(self, xt_session):  # xtquant.xtdata.XtQuantSession
        self._session = xt_session
    
    def get_daily(self, stocks, start, end, fields) -> pd.DataFrame:
        # xtquant.xtdata.get_market_data_ex(...)
        ...
    
    def get_pit(self, stock, field, date) -> float | None:
        # xtquant has native PIT support
        ...
```

### 10.2 QlibDataProvider (future)

```python
class QlibDataProvider(DataProvider):
    """Data provider using Microsoft Qlib's data infrastructure.
    
    Advantages:
    - Qlib's cached and normalized data (CN market focus)
    - Expressive data queries (Qlib D expression)
    - Pre-computed features for ML
    """
    
    def __init__(self, qlib_dir: str):
        import qlib
        qlib.init(provider_uri=qlib_dir)
        ...
```

---

## 11. Data Format Specifications

### 11.1 Daily Data Parquet Schema

```python
# Parquet schema for daily OHLCV
daily_schema = pa.schema([
    pa.field("date", pa.timestamp("ns")),      # Trading date
    pa.field("stock", pa.string()),             # qmt-format code "000001.SZ"
    pa.field("open", pa.float32()),
    pa.field("high", pa.float32()),
    pa.field("low", pa.float32()),
    pa.field("close", pa.float32()),            # Adjusted close
    pa.field("volume", pa.int64()),             # Shares traded
    pa.field("amount", pa.float32()),           # Turnover in CNY
])
```

### 11.2 PIT Data Parquet Schema

```python
# Parquet schema for PIT financial data
pit_schema = pa.schema([
    pa.field("stock", pa.string()),             # qmt-format code
    pa.field("field", pa.string()),             # Financial field name
    pa.field("value", pa.float32()),            # Reported value
    pa.field("report_date", pa.timestamp("ns")), # Period end date
    pa.field("announce_date", pa.timestamp("ns")),# When report was published
    pa.field("source", pa.string()),            # "akshare" | "xtquant"
])
```

### 11.3 Index Data Parquet Schema

```python
# Parquet schema for index weights
index_schema = pa.schema([
    pa.field("date", pa.timestamp("ns")),       # Rebalance date
    pa.field("stock", pa.string()),             # Constituent code
    pa.field("weight", pa.float32()),           # Weight (sum = 1.0)
    pa.field("index", pa.string()),             # "csi_500" | "csi_300"
])
```

---

## 12. Orientation Convention Summary

| Component | Orientation | Access Method |
|-----------|-------------|---------------|
| DataProvider.get_daily() | DatetimeIndex x MultiIndex(stock, field) | Default (qmt convention) |
| DataManager.daily_for_qmt() | DatetimeIndex x MultiIndex(stock, field) | Same as provider |
| DataManager.daily_for_openalpha() | (Stock, Date) single field | xs() + transpose |
| AlphaBridge.transpose() | (Stock,Date)->(Date,Stock) | For bridge output |
| AlphaBridge.normalize() | DatetimeIndex x plain stock-code columns | Factor signal input and output |
| SignalAlphaFactor.signal_data | DatetimeIndex x plain stock-code columns | One scalar signal per qmt code/date |
| Parquet storage | DatetimeIndex x MultiIndex(stock, field) | Default storage orientation |

**Rule**: Market data storage and provider transport use DatetimeIndex x MultiIndex(stock, field). Factor-signal bridge data uses a plain DatetimeIndex x stock-code-column DataFrame because qmt `AlphaFactor.compute(code, df)` needs one scalar per code/date. Only OpenAlpha's internal computation uses (Stock, Date) for single fields. The DataManager facade handles market-data conversion.

---

## 13. Testing Strategy

### 13.1 PIT Correctness Test

```python
def test_pit_no_future_leakage():
    """Verify that PIT reads never return data from future announcements."""
    manager = DataManager()
    
    # eps for 000001.SZ: annual report for 2023 announced on 2024-04-28
    # Request on 2024-04-27 -> should return 2022 annual eps (last known)
    value_before = manager.pit("000001.SZ", "eps", "2024-04-27")
    
    # Request on 2024-04-28 -> should return 2023 annual eps
    value_on = manager.pit("000001.SZ", "eps", "2024-04-28")
    
    # value_before should be 2022 value, value_on should be 2023 value
    # They should be different (unless coincidentally equal)
    # The announcement_date cutoff is the key check
    
    audit = manager._pit.audit_log
    assert audit.verify("000001.SZ", "eps", "2024-04-27")  # No future leakage
    assert audit.verify("000001.SZ", "eps", "2024-04-28")  # No future leakage
```

### 13.2 Cache Consistency Test

```python
def test_cache_roundtrip():
    """Verify data survives cache -> read -> compare cycle."""
    provider = AkshareProvider()
    original = provider.get_daily(["000001.SZ"], "2024-01-01", "2024-01-31", ["close"])
    
    # Second call should hit cache
    cached = provider.get_daily(["000001.SZ"], "2024-01-01", "2024-01-31", ["close"])
    
    pd.testing.assert_frame_equal(original, cached)
```

### 13.3 Orientation Consistency Test

```python
def test_orientation_consistency():
    """Verify DataManager produces correct orientation for each project."""
    manager = DataManager()
    
    # qmt format: DatetimeIndex x MultiIndex(stock, field)
    qmt_data = manager.daily_for_qmt(["000001.SZ"], "2024-01-01", "2024-01-31", ["close"])
    assert isinstance(qmt_data.index, pd.DatetimeIndex)
    assert isinstance(qmt_data.columns, pd.MultiIndex)
    
    # OpenAlpha format: (Stock, Date) single field
    alpha_data = manager.daily_for_openalpha(["000001.SZ"], "2024-01-01", "2024-01-31", ["close"])
    assert isinstance(alpha_data.columns, pd.DatetimeIndex)  # Date as columns
    assert isinstance(alpha_data.index, pd.Index)  # Stock codes as index
    
    # Values should match (single field case)
    # qmt_data.xs("close", level=1, axis=1) gives (Date, Stock) for close prices
    close_qmt = qmt_data.xs("close", level=1, axis=1)
    # alpha_data is (Stock, Date) - transpose to compare
    assert close_qmt.equals(alpha_data.T)
```

---

## 14. Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| get_daily (500 stocks, 60 days) | < 500ms cached, < 30s fresh | Network download is bottleneck |
| get_pit (single stock) | < 10ms cached | Single Series lookup |
| get_pit_batch (500 stocks) | < 100ms cached | Vectorized Series operation |
| get_index_weights | < 50ms cached | Single Parquet read |
| Cache write (monthly) | < 200ms | Single Parquet write |
| PIT verification | < 1ms per entry | Audit log scan |
