# OpenAlpha × qmt Integration Roadmap

> Version: 0.3 | Date: 2026-05-22 | Status: Proposed — MVP-first, verified against current code
> Last verified against: OpenAlpha commit [unknown], qmt commit [unknown]
> Files checked: OpenAlpha/src/simres/expr.py, qmt/src/qmt_local/strategies/factor.py, qmt/src/qmt_local/strategies/multi_factor.py

---

## 1. Goal

Build the smallest verifiable integration path from OpenAlpha factor expressions to qmt factor strategies:

```
OpenAlpha expression
  → AlphaExecutor.evaluate()
  → (Stock, Date) ndarray/DataFrame
  → AlphaBridge.transpose()
  → (Date, Stock) signal DataFrame
  → SignalAlphaFactor
  → qmt MultiFactorStrategy backtest
```

The first milestone is not a platform, data lake, dashboard, or live execution system. It is a tested factor-signal bridge that can be imported, run, and verified locally.

---

## 2. Project Roles

| Project | Current Role | Integration Use | Notes |
|---------|--------------|-----------------|-------|
| **OpenAlpha** | Factor discovery and expression evaluation | Produces factor matrices from expressions | Current `AlphaExecutor.evaluate()` returns `np.ndarray`, not DataFrame |
| **qmt** | Local event-driven quant framework | Primary consumer of OpenAlpha signals | Has `AlphaFactor`, `FeatureEngine`, and `MultiFactorStrategy` interfaces |
| **ptrade** | Lightweight PTrade-style local strategy sandbox | Optional future adapter/reference | Defer until OpenAlpha → qmt bridge is proven |

**Decision**: qmt is the primary integration target. ptrade is not part of the MVP path.

---

## 3. Current API Facts

### 3.1 OpenAlpha AlphaExecutor

**Location**: `OpenAlpha/src/simres/expr.py`

```python
class AlphaExecutor:
    def __init__(self, data_dir: str, alpha_dir: str | None = None):
        ...

    def load_all_data(self) -> None:
        """Loads parquet data into evaluation context."""

    def evaluate(self, expression: str) -> np.ndarray | None:
        """Returns a (Stock, Date) float matrix."""

    def backtest(self, alpha: np.ndarray, price: str = "vwap") -> dict:
        """Runs OpenAlpha's VWAP-to-VWAP research backtest."""
```

OpenAlpha orientation:

- `axis=0` = cross-section / stock dimension
- `axis=1` = time-series / date dimension
- `evaluate()` returns `np.ndarray`, so dates/stocks must be attached by a wrapper before crossing the bridge

### 3.2 qmt AlphaFactor Framework

**Location**: `qmt/src/qmt_local/strategies/factor.py` (external repo)

```python
class AlphaFactor(ABC):
    @abstractmethod
    def compute(self, code: str, df: pd.DataFrame) -> float:
        """Single stock, current strategy context -> float."""

class FeatureEngine:
    def process(self, s: pd.Series) -> pd.Series:
        """winsorize -> fillna -> zscore -> rank."""
```

Current `FeatureEngine` is only a preprocessing pipeline. It does not provide `register_factor()` or `compute_features()`.

### 3.3 qmt MultiFactorStrategy

**Location**: `qmt/src/qmt_local/strategies/multi_factor.py` (external repo)

```python
class MultiFactorStrategy:
    def __init__(self, factors: list[tuple[AlphaFactor, float]], top_n: int = 10, ...):
        ...
```

The bridge must output one or more `AlphaFactor` instances that can be passed as `(factor, weight)` tuples.

---

## 4. Non-Goals for MVP

These are intentionally deferred:

- Redis, ZeroMQ, HTTP, gRPC, or any cross-machine signal bus
- Unified data layer under `/alpha/data`
- ptrade adapter
- Dashboard or FastAPI service
- qmt live trading, MiniQMT, xtquant proxy, or Windows runtime work
- Replacing qmt `DataManager`
- PIT financial data infrastructure
- IC/IR as a hard gate before qmt can consume a factor

---

## 5. Target Contracts

### 5.1 FactorOutput

```python
@dataclass
class FactorOutput:
    """Canonical OpenAlpha factor output before bridge conversion."""

    values: pd.DataFrame        # index: stock code, columns: date, dtype float32
    expression: str
    stocks: list[str]
    dates: pd.DatetimeIndex
    metadata: dict[str, Any]
```

`FactorOutput.values` keeps OpenAlpha's natural `(Stock, Date)` orientation.

### 5.2 BridgeSignalFrame

The qmt-facing bridge output is a simple two-dimensional signal frame:

```python
signal_data: pd.DataFrame
# index: pd.DatetimeIndex
# columns: canonical stock codes, e.g. 000001.SZ
# values: float32 normalized factor scores
```

**Rule**: factor signals use `(Date, Stock)` with plain stock-code columns. MultiIndex columns are reserved for market data such as OHLCV fields.

### 5.3 SignalAlphaFactor

```python
class SignalAlphaFactor(AlphaFactor):
    def __init__(
        self,
        name: str,
        signal_data: pd.DataFrame,
        default_value: float = 0.0,
    ):
        ...

    def compute(self, code: str, df: pd.DataFrame) -> float:
        """Return signal_data.loc[current_date, code], or default_value if missing."""
```

The adapter should be intentionally small. It should not fetch market data, publish messages, run IC filters, or decide portfolio weights.

### 5.4 AlphaBridge

```python
class AlphaBridge:
    @staticmethod
    def from_executor_result(
        values: np.ndarray,
        stocks: list[str],
        dates: pd.DatetimeIndex,
        expression: str,
    ) -> FactorOutput:
        ...

    @staticmethod
    def transpose(factor_output: FactorOutput) -> pd.DataFrame:
        """Convert (Stock, Date) to (Date, Stock)."""

    @staticmethod
    def normalize(signal_data: pd.DataFrame, method: str = "cs_rank") -> pd.DataFrame:
        """Apply row-wise cross-sectional normalization."""

    @staticmethod
    def to_qmt_factor(name: str, signal_data: pd.DataFrame) -> SignalAlphaFactor:
        ...
```

---

## 6. Phase Roadmap

### Phase 0 — Package and Output Wrapper

**Objective**: Make OpenAlpha importable and able to produce a labeled factor output.

| ID | Task | Files | Success Criteria |
|----|------|-------|------------------|
| P0.1 | Add minimal package config | `pyproject.toml` or `OpenAlpha/pyproject.toml` | `pip install -e .` works in local dev |
| P0.2 | Add `FactorOutput` wrapper | `bridge/output.py` | ndarray + stocks + dates becomes labeled `(Stock, Date)` DataFrame |
| P0.3 | Add stock-code mapper | `bridge/code_mapper.py` | `000001` maps deterministically to qmt-style codes |
| P0.4 | Add unit tests for wrapper/mapper | `tests/` | Shape, dtype, index, columns, and code conversion are verified |

**Deliverable**: A labeled `FactorOutput` can be created from current `AlphaExecutor.evaluate()` output.

### Phase 1 — qmt Bridge MVP

**Objective**: Convert one OpenAlpha factor into one qmt-compatible `AlphaFactor`.

| ID | Task | Files | Success Criteria |
|----|------|-------|------------------|
| P1.1 | Implement `AlphaBridge.transpose()` | `bridge/transpose.py` | `(Stock, Date)` becomes `(Date, Stock)` with preserved labels |
| P1.2 | Implement `AlphaBridge.normalize()` | `bridge/normalize.py` | Cross-sectional rank/zscore can run row-wise by date |
| P1.3 | Implement `SignalAlphaFactor` | `bridge/signal_factor.py` | `compute(code, df)` returns the expected signal for qmt |
| P1.4 | Add qmt integration test | `tests/test_qmt_bridge.py` | `SignalAlphaFactor` works as `[(factor, weight)]` input to `MultiFactorStrategy` or a narrow qmt-compatible harness |
| P1.5 | Add one runnable example | `examples/openalpha_to_qmt_factor.py` | Example runs expression -> signal -> qmt factor without Redis/data layer |

**Deliverable**: The local OpenAlpha -> qmt factor bridge works without external services.

### Phase 2 — Research Validation

**Objective**: Add quality checks after the bridge works.

| ID | Task | Files | Success Criteria |
|----|------|-------|------------------|
| P2.1 | Add forward-return alignment helper | `bridge/returns.py` | Signal dates and return dates are explicitly shifted and tested |
| P2.2 | Add IC/IR evaluator | `bridge/ic_filter.py` | IC/IR metrics are computed, not required for dispatch |
| P2.3 | Add leakage tests | `tests/test_alignment.py` | Same-day and future-return mistakes fail tests |
| P2.4 | Document validation workflow | `docs/bridge-spec.md` | Bridge spec separates signal creation from factor evaluation |

**Deliverable**: A factor can be scored and rejected by research metrics, but qmt consumption does not depend on Redis or a new data platform.

### Phase 3 — Data Convergence Decision

**Objective**: Decide whether a shared `/alpha/data` layer is justified by real duplication.

| ID | Task | Files | Success Criteria |
|----|------|-------|------------------|
| P3.1 | Compare OpenAlpha, qmt, and ptrade data contracts | `docs/data-layer-spec.md` | Current formats, providers, cache locations, and PIT gaps are documented |
| P3.2 | Write data-layer ADR | `docs/data-layer-adr.md` | Decision: reuse qmt DataManager, create thin adapter, or build shared provider |
| P3.3 | Prototype only the chosen adapter | TBD by ADR | No parallel data framework unless ADR proves need |

**Deliverable**: A documented decision about data ownership before adding `/alpha/data`.

### Phase 4 — Optional Platform Extensions

**Objective**: Add operational features only after the local bridge has evidence.

| Extension | Trigger | Candidate Files |
|-----------|---------|-----------------|
| Redis or HTTP signal bus | Need Mac/Linux research process to feed Windows MiniQMT runtime | `bridge/signal_bus.py` |
| ptrade adapter | Need PTrade-style local strategy examples using OpenAlpha signals | `bridge/ptrade_adapter.py` |
| Dashboard | Need repeated visual inspection of IC, turnover, quantiles, and signal drift | `dashboard/` |
| qmt live execution adapter | Backtest bridge is stable and MiniQMT workflow is selected | qmt-side adapter or external proxy |

---

## 7. MVP Directory Structure

Only create files needed by the current phase.

```
alpha/
├── bridge/
│   ├── __init__.py
│   ├── output.py          # FactorOutput wrapper
│   ├── code_mapper.py     # stock code conversion
│   ├── transpose.py       # (Stock, Date) -> (Date, Stock)
│   ├── normalize.py       # row-wise signal normalization
│   └── signal_factor.py   # qmt AlphaFactor adapter
├── examples/
│   └── openalpha_to_qmt_factor.py
├── tests/
│   ├── test_output.py
│   ├── test_transpose.py
│   ├── test_normalize.py
│   └── test_signal_factor.py
└── pyproject.toml
```

Do not create `/alpha/data`, `/alpha/dashboard`, `signal_bus.py`, or ptrade adapters until the phase that needs them starts.

---

## 8. Data Format Rules

### 8.1 Orientation

| Data Type | Shape | Column Convention |
|-----------|-------|-------------------|
| OpenAlpha internal matrix | `(Stock, Date)` | no labels unless wrapped |
| `FactorOutput.values` | `(Stock, Date)` | dates as columns |
| qmt-facing `signal_data` | `(Date, Stock)` | plain stock-code columns |
| Market data with multiple fields | `DatetimeIndex × MultiIndex(stock, field)` | only for OHLCV/fundamental data |

### 8.2 Numeric Rules

- Preserve `float32` where practical.
- Normalize cross-sectionally by date.
- Convert missing signal values to `0.0` only after the missingness rule is explicit.
- Keep raw factor output and normalized signal output as separate artifacts.

### 8.3 Stock Codes

| Project | Example | Notes |
|---------|---------|-------|
| OpenAlpha | `000001` | Current data may use bare codes |
| qmt | `000001.SZ` | Use as bridge canonical code |
| ptrade | `000001` or platform-specific suffixes | Future adapter concern |

The bridge should make suffix mapping deterministic and testable.

---

## 9. Simplified Service Shape

The MVP service is a pure local adapter:

```python
class AlphaBridgeService:
    def process_factor(
        self,
        factor_output: FactorOutput,
        *,
        normalize_method: str = "cs_rank",
    ) -> SignalAlphaFactor:
        signal_data = AlphaBridge.transpose(factor_output)
        signal_data = AlphaBridge.normalize(signal_data, method=normalize_method)
        return SignalAlphaFactor(factor_output.expression, signal_data)
```

No data provider, Redis bus, ptrade branch, dashboard branch, or IC filter belongs in the MVP service.

---

## 10. Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Orientation mismatch | Silent factor inversion or date/stock lookup errors | Unit tests for every transpose and lookup path |
| Missing labels from OpenAlpha ndarray | qmt receives signals for wrong dates/stocks | `FactorOutput` must require stocks and dates |
| Code suffix mismatch | qmt lookup misses signals | Centralize conversion in `StockCodeMapper` |
| qmt strategy context date ambiguity | `SignalAlphaFactor.compute()` may not know current date | Define and test how date is read from qmt `df` or strategy context before broad integration |
| OpenAlpha packaging gap | Examples and tests are brittle | Phase 0 package/import cleanup before bridge MVP |
| Data-layer overbuild | Parallel data framework drifts from qmt | Require Phase 3 ADR before adding shared provider |

---

## 11. Success Metrics

| Metric | Phase 0 | Phase 1 | Phase 2 | Phase 3+ |
|--------|---------|---------|---------|----------|
| Importability | Package import works | Example imports both OpenAlpha bridge and qmt factor ABC | Same | Same |
| Shape correctness | Labeled `(Stock, Date)` wrapper tested | `(Date, Stock)` signal tested | Alignment tested | Provider/adapter tested only if built |
| qmt compatibility | — | `SignalAlphaFactor.compute()` verified | Multi-factor validation path verified | Optional execution/live adapters |
| External services | None | None | None | Added only by explicit trigger |
| Research quality | — | Manual inspection possible | IC/IR and leakage tests | Dashboard optional |

---

## 12. Decisions

| Topic | Decision | Rationale |
|-------|----------|-----------|
| Primary consumer | qmt first | It already has `AlphaFactor` and `MultiFactorStrategy` abstractions |
| ptrade | Defer | Useful as lightweight sandbox, but not needed for qmt bridge proof |
| Redis/signal bus | Defer | Local in-process bridge is enough for MVP |
| Unified data layer | Defer pending ADR | Current qmt already has a data manager; adding another one now increases drift |
| Signal format | Plain `(Date, Stock)` DataFrame | qmt factor lookup needs one scalar per code/date |
| Market data format | MultiIndex allowed | OHLCV/fundamental data needs stock + field dimensions |
| IC/IR | Phase 2 metric, not Phase 1 dispatch gate | Bridge correctness should be proven before research filtering |
| Dashboard | Optional extension | Visual analysis is useful after stable artifacts exist |
