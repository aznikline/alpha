# ADR-001: Data Convergence Layer Decision

> Status: Accepted | Date: 2026-05-22 | Decision ID: ADR-001
>
> Decides: reuse qmt DataManager, thin adapter, or shared provider
> Based on: P3.1 evidence from data-layer-spec.md and code analysis

---

## Context

### Current State (P3.1 Evidence)

Three projects fetch market data independently with different contracts:

| Project | Data Source | Format | Orientation | PIT Safety | Abstraction Level | Stock Codes |
|---------|-------------|--------|-------------|------------|-------------------|-------------|
| **OpenAlpha** | Parquet files in `data/` dir | `np.ndarray` float32 | `(Stock, Date)` | No | None (raw file loading) | Integer bare codes (000001) |
| **qmt** | Tushare → AKShare → Local fallback | `dict[str, DataFrame]` per stock | `(Date, fields)` per stock | No | Full (DataSource + Storage ABC) | Suffix codes (000001.SZ) |
| **ptrade** | AKShare inline + Ptrade API | `DataFrame` or dict | Mixed | No | Minimal (DataSource class) | Suffix codes (000001.SS) |

### qmt DataManager Capabilities

The qmt DataManager (`/qmt/src/qmt_local/data/manager.py`) provides:

- **Provider abstraction**: `DataSource` base class with priority-based fallback chain
- **Storage abstraction**: `Storage` base class with Parquet + SQLite implementations
- **QMT-compatible API**: `get_market_data_ex()`, `download_history_data()`, `get_local_data()`
- **Data quality validation**: NaN checks, staleness detection
- **In-memory caching**: Per-code DataFrame cache
- **Subscription hooks**: Real-time quote subscription support

### OpenAlpha Data Needs

OpenAlpha's `AlphaExecutor` requires:

- Daily OHLCV for CSI 500 universe
- (Stock, Date) orientation for factor computation
- Integer stock codes for expression evaluation
- No PIT financial data (deferred to Phase 2+)

### Gap Analysis

| Gap | OpenAlpha | qmt DataManager | Impact |
|-----|-----------|-----------------|--------|
| Orientation mismatch | (Stock, Date) ndarray | dict[str, DataFrame] per stock | Requires transpose wrapper |
| Stock code format | Integer | Suffix | Requires code mapper |
| PIT financials | Not needed | Not provided | No gap |
| Multi-field batch | Single field per call | Multi-field per stock | No gap |
| Universe definition | CSI 500 hardcoded | Index constituents API | Minor gap |

**Duplication observed**:
- OpenAlpha and ptrade both call AKShare independently
- OpenAlpha downloads CSI 500 OHLCV that qmt already has in Parquet cache
- No PIT protection in OpenAlpha → potential future leakage in factor analysis
- Different stock code formats require per-project normalization

**Options considered**:

1. **Build shared DataProvider under `/alpha/data`** — Full unified data layer with ABC, ParquetCache, PITManager, DataManager facade (as described in data-layer-spec.md)
2. **Reuse qmt DataManager directly** — Import qmt's `DataManager` into OpenAlpha and call `get_history()` for market data
3. **Thin adapter wrapping qmt DataManager** — A `QMTDataAdapter` class that bridges qmt's `(stock, fields)` format to OpenAlpha's `(Stock, Date)` ndarray format, while also providing `(Date, Stock)` signal-aligned data for the bridge layer

---

## Decision

**Option 3: Thin adapter wrapping qmt DataManager.**

Create `bridge/data_adapter.py` with a `QMTDataAdapter` class that:
- Wraps qmt's `DataManager` for data fetching (no duplicate downloads)
- Converts qmt's per-stock DataFrame format to OpenAlpha's `(Stock, Date)` ndarray
- Converts qmt's per-stock DataFrame format to bridge's `(Date, Stock)` signal format
- Delegates PIT reads to qmt's `PITDatabase` (reuse, don't rebuild)
- Handles stock code format conversion via `StockCodeMapper`
- Does NOT create a new `/alpha/data` directory or shared data infrastructure

---

## Rationale

### Why NOT Option 1 (shared DataProvider)?

1. **qmt already has a mature DataManager** — Tushare/AKShare/Local fallback, Parquet cache, SQLite metadata, PIT database. Building another DataProvider ABC would create parallel infrastructure that drifts from qmt's reality.

2. **data-layer-spec.md describes Proposed interfaces** — They were written before implementation. The actual qmt implementation differs in important ways:
   - qmt uses `DataSource` ABC (not `DataProvider`)
   - qmt uses `DataRequest` dataclass for requests
   - qmt's storage is per-stock Parquet (not monthly partitioned)
   - qmt's PIT is per-stock asof join (not announcement-date indexed)

3. **OpenAlpha only needs market data for factor evaluation** — It doesn't need PIT, index weights, or universe lists. It needs OHLCV arrays. A thin adapter that calls qmt for this data is simpler and avoids building infrastructure OpenAlpha doesn't use.

4. **Risk of overbuild** — The roadmap explicitly warns: "Data-layer overbuild: Parallel data framework drifts from qmt | Require Phase 3 ADR before adding shared provider"

### Why NOT Option 2 (reuse qmt DataManager directly)?

1. **Format mismatch** — qmt returns `dict[str, pd.DataFrame]` (one DataFrame per stock), while OpenAlpha needs `np.ndarray` with shape `(Stock, Date)`. Direct import would require format conversion code scattered across OpenAlpha.

2. **Stock code mismatch** — qmt uses suffix codes (000001.SZ), OpenAlpha uses bare codes (000001). Conversion needs centralized handling.

3. **Dependency coupling** — OpenAlpha would directly import from qmt, creating a hard dependency. The bridge layer should mediate this.

### Why Option 3 (thin adapter)?

1. **Single conversion point** — All format/code/orientation conversions happen in one class, not scattered across call sites.

2. **Reuses qmt infrastructure** — No duplicate downloads, no duplicate PIT logic, no duplicate cache. The adapter delegates to qmt's DataManager.

3. **Bridge layer consistency** — The adapter outputs match the bridge's expected formats:
   - For factor evaluation: `(Stock, Date)` ndarray (OpenAlpha convention)
   - For signal alignment: `(Date, Stock)` DataFrame (bridge convention)
   - For forward returns: `(Date, Stock)` DataFrame (returns.py convention)

4. **Loose coupling** — OpenAlpha doesn't import qmt directly. The adapter mediates. If qmt changes its data format, only the adapter needs updating.

5. **PIT reuse** — For Phase 2's forward-return alignment, the adapter can provide PIT-safe financial data by delegating to qmt's PITDatabase. No need to build a new PITManager.

6. **Incremental** — If a future need arises for a full shared data layer (e.g. ptrade integration), the adapter can be expanded or replaced. But we don't build that now.

---

## Consequences

### Positive

- No duplicate data downloads (qmt cache serves both projects)
- PIT safety available for factor evaluation (via qmt PITDatabase)
- Single conversion point for format/code mismatches
- No new `/alpha/data` directory needed
- OpenAlpha stays independent of qmt import structure

### Negative

- Adapter depends on qmt's DataManager API — if qmt changes, adapter needs updating
- qmt DataManager requires xtquant/Tushare config — adapter needs same environment setup
- Running without qmt installed means no market data access (but bridge still works with synthetic data)
- No offline data for OpenAlpha without qmt cache populated first

### Mitigations

- Adapter uses try/except import pattern (same as SignalAlphaFactor) — works standalone with fallback
- Adapter interface is defined in bridge module, not in qmt — changes are localized
- Synthetic data generation already exists in OpenAlpha (data_generator.py) for testing without qmt

---

## Implementation Plan

### P3.3: QMTDataAdapter

```python
class QMTDataAdapter:
    """Thin adapter wrapping qmt DataManager for OpenAlpha data needs.
    
    Converts between qmt's per-stock DataFrame format and:
    - OpenAlpha's (Stock, Date) ndarray format (for factor evaluation)
    - Bridge's (Date, Stock) DataFrame format (for signal alignment)
    - Returns module's (Date, Stock) format (for forward-return computation)
    
    Delegates data fetching and PIT to qmt's DataManager.
    Handles stock code conversion via StockCodeMapper.
    """

    def __init__(self, qmt_data_dir: str = ""):
        # try/except import pattern — works without qmt installed
        try:
            from qmt_local.data.manager import DataManager
            self._manager = DataManager(cache_dir=qmt_data_dir)
            self._pit = None  # Will use manager's PIT when needed
            self._available = True
        except ImportError:
            self._manager = None
            self._available = False

    def get_daily_ndarray(
        self,
        stocks: list[str],
        start_date: str,
        end_date: str,
        fields: list[str],
    ) -> dict[str, np.ndarray]:
        """Fetch daily data as (Stock, Date) ndarray dict — OpenAlpha convention.
        
        Returns dict mapping field_name -> np.ndarray with shape (n_stocks, n_dates).
        Stock codes are bare integers (OpenAlpha convention).
        """
        ...

    def get_daily_signal_frame(
        self,
        stocks: list[str],
        start_date: str,
        end_date: str,
        fields: list[str],
    ) -> pd.DataFrame:
        """Fetch daily data as (Date, Stock) DataFrame — bridge convention.
        
        Returns DataFrame with DatetimeIndex and stock-code columns.
        Stock codes use suffix format (000001.SZ).
        MultiIndex columns for multiple fields.
        """
        ...

    def get_forward_return_data(
        self,
        stocks: list[str],
        start_date: str,
        end_date: str,
        price_field: str = "vwap",
    ) -> pd.DataFrame:
        """Fetch price data suitable for compute_forward_returns().
        
        Returns (Date, Stock) DataFrame with price values only.
        Ready for bridge.returns.compute_forward_returns().
        """
        ...

    @property
    def available(self) -> bool:
        return self._available
```

### No `/alpha/data` directory

The adapter uses qmt's existing data infrastructure. No new shared directory is created.

### Fallback for standalone mode

When qmt is not available (ImportError), the adapter marks `available = False`. Tests and examples can use synthetic data generators instead.

### Test Requirements

```python
def test_adapter_transpose():
    """Verify dict[str, DataFrame] -> (Stock, Date) preserves values."""
    mock_dict = {
        "000001.SZ": pd.DataFrame({"close": [10, 11, 12]}, index=pd.date_range("2024-01-01", periods=3)),
        "600000.SH": pd.DataFrame({"close": [20, 21, 22]}, index=pd.date_range("2024-01-01", periods=3)),
    }
    adapter = QMTDataAdapter()
    matrix = adapter._transpose_to_matrix(mock_dict, [1, 600000], "close")
    
    # matrix[stock_idx, date_idx] matches original DataFrame value
    assert matrix[0, 0] == 10  # Stock 1 (000001.SZ), first date
    assert matrix[1, 0] == 20  # Stock 600000, first date

def test_adapter_code_mapping():
    """Verify integer -> suffix conversion is correct."""
    adapter = QMTDataAdapter()
    qmt_codes = adapter._mapper.batch_to_qmt([1, 600000])
    assert qmt_codes == ["000001.SZ", "600000.SH"]

def test_adapter_signal_frame():
    """Verify (Date, Stock) output matches bridge convention."""
    adapter = QMTDataAdapter()
    frame = adapter.get_daily_signal_frame(["000001.SZ"], "2024-01-01", "2024-01-03", ["close"])
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert "000001.SZ" in frame.columns
```

### Migration Path

1. **Phase 0**: Add `QMTDataAdapter` in `bridge/data_adapter.py`
2. **Phase 1**: Wire adapter to bridge pipeline (signal creation)
3. **Phase 1**: Add fallback to legacy file loading if adapter unavailable
4. **Phase 2**: Add tests for transpose and code mapping correctness
5. **Future**: Replace adapter with shared provider if PIT/ptrade needs justify

---

## Alternatives Considered

| Alternative | Rejected Reason |
|-------------|-----------------|
| Reuse qmt DataManager directly | Orientation and code format mismatch requires OpenAlpha core changes; creates hard dependency |
| Build shared DataProvider ABC | Duplicates qmt's existing DataSource abstraction; overbuild for MVP scope |
| Build new ParquetCache | Duplicates qmt's ParquetStorage; no justification for parallel infrastructure |
| Build PITManager now | No current usage in OpenAlpha; deferred to Phase 2+ research validation |
| Import ptrade DataSource | ptrade integration deferred per roadmap decision; ptrade has simpler abstraction |

---

## References

- `/Users/wizout/op/quant/alpha/docs/data-layer-spec.md` — Full data layer specification (proposed interfaces)
- `/Users/wizout/op/quant/alpha/docs/integration-roadmap.md` — Phase 3 decision context and risk register
- `/Users/wizout/op/quant/alpha/docs/bridge-spec.md` — Bridge layer interface signatures
- `/Users/wizout/op/quant/qmt/src/qmt_local/data/manager.py` — qmt DataManager implementation (306 lines)
- `/Users/wizout/op/quant/qmt/src/qmt_local/data/provider/base.py` — qmt DataSource ABC
- `/Users/wizout/op/quant/ptrade/local_ptrade/data_manager.py` — ptrade DataManager implementation (678 lines)

---

## Open Questions

1. **Should OpenAlpha's `AlphaExecutor.load_all_data()` be refactored to use the adapter?**
   - Current: loads from local Parquet files directly
   - Future: could call `QMTDataAdapter.get_daily_ndarray()` instead
   - Recommendation: defer until bridge proves stable. OpenAlpha's current file-based loading works for research. Adapter is for bridge pipeline only.

2. **Should ptrade be included in the adapter scope?**
   - Recommendation: no. ptrade has its own `data_source.py` and Ptrade API. The adapter is for OpenAlpha → qmt bridge needs only.

3. **What about OpenAlpha's synthetic data generator (`data_generator.py`)?**
   - It stays as-is. Used for testing without qmt. Not replaced by the adapter.

4. **When should shared provider be reconsidered?**
   - Trigger: PIT financial data becomes blocking for factor research
   - Trigger: ptrade integration becomes active (requires third consumer)
   - Trigger: qmt DataManager API changes significantly (adapter maintenance burden)
   - Default: defer until evidence proves need