# AlphaBridge Technical Specification

> Version: 0.2 | Date: 2026-05-22 | Status: Proposed — verified against current code
>
> Last verified against: OpenAlpha commit [unknown], qmt commit [unknown]
> Files checked: OpenAlpha/src/simres/expr.py, qmt/src/qmt_local/strategies/factor.py, qmt/src/qmt_local/strategies/multi_factor.py
>
> This document specifies the exact interface signatures, data transformations, and
> wiring points for the bridge layer connecting OpenAlpha factor discovery to qmt
> trading execution.

---

## 1. Purpose

The AlphaBridge transforms OpenAlpha's (Stock, Date) factor output into formats
compatible with qmt's (Date, Stock) AlphaFactor interface. The MVP path is a
local in-process adapter; IC/IR filtering and Redis-based real-time dispatch are
optional later extensions.

---

## 2. Source: OpenAlpha Factor Output

### 2.1 Current OpenAlpha API (As-Is)

Location: `/OpenAlpha/src/simres/expr.py`

```python
class AlphaExecutor:
    def __init__(self, data_dir: str, alpha_dir: str | None = None):
        """Initialize executor with data directory.
        
        Must call load_all_data() separately before evaluate().
        """
    
    def evaluate(self, expression: str) -> np.ndarray | None:
        """Execute expression via eval().
        
        Returns (Stock, Date) ndarray or None on error.
        Shape: (N_stocks, N_dates)
        """
    
    def backtest(self, alpha: np.ndarray, price: str = 'vwap') -> dict:
        """Backtest a factor array.
        
        Args:
            alpha: (Stock, Date) factor values from evaluate()
            price: Price column for returns ('vwap' or 'close')
        
        Returns dict with keys:
            - datestr: list of date strings
            - net_ret: net returns
            - long_ret: long side returns
            - short_ret: short side returns
            - tvr: turnover
            - long_num: number of long positions
            - short_num: number of short positions
        """
```

**Key observations:**
- `evaluate()` returns raw `np.ndarray`, NOT a `FactorOutput` dataclass
- No `universe`, `start`, `end` parameters — operates on pre-loaded data
- Must call `load_all_data()` after `__init__` before using `evaluate()`

### 2.2 Proposed Bridge API (To-Be)

The following interfaces are proposed additions to support the bridge layer:

```python
@dataclass
class FactorOutput:
    """Proposed: Canonical output format for bridge integration."""
    values: pd.DataFrame       # Index: stock codes (integer format), Columns: dates (DatetimeIndex)
                                # dtype: float32, NaN→0 after normalization
                                # Shape example: (500, 1200) for CSI500 over ~5 years
    metadata: FactorMetadata

@dataclass
class FactorMetadata:
    """Proposed: Metadata for factor output."""
    expression: str             # Original expression string
    normalization: list[str]    # Applied normalization steps
                                # Default: ["at_nan2zero", "cs_booksize", "cs_rank"]
    universe: str               # "csi_500" | "csi_300" | "all_a"
    date_range: tuple[str, str] # ("2020-01-01", "2025-12-31")
    operator_chain: list[str]   # Top-down parse of expression

class AlphaExecutor:
    """Proposed: Enhanced evaluate() with universe and date range parameters."""
    
    def evaluate(
        self,
        expr: str,                  # Factor expression e.g. "cs_rank(ts_delta(close, 5))"
        universe: str = "csi_500",   # Universe name
        start: str | None = None,    # ISO date start
        end: str | None = None,      # ISO date end
    ) -> FactorOutput:
        """Evaluate a factor expression over the specified universe and date range.
        
        Returns FactorOutput with (Stock, Date) orientation.
        """
```

### 2.3 Raw Output Before Normalization

The AlphaExecutor internally computes a raw `(Stock, Date)` matrix before applying
the normalization pipeline. The pipeline is:

```
raw_values → at_mask(expr, ts_fill(csi_500_weight) > 0) 
           → cs_rank → - 0.5 
           → cs_booksize 
           → at_nan2zero
```

Each step preserves the `(Stock, Date)` orientation. Final range: approximately `[-1, 1]`.

---

## 3. Target: qmt AlphaFactor Interface

### 3.1 Current qmt API (As-Is)

Location: `/qmt/src/qmt_local/strategies/factor.py`

```python
class AlphaFactor(ABC):
    """Base class for alpha factors in qmt.
    
    Interface contract: compute a single float value for a single stock on a single date.
    """
    
    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def compute(self, code: str, df: pd.DataFrame) -> float:
        """Compute factor value for a stock.
        
        Args:
            code: Stock code in qmt format (e.g. "000001.SZ")
            df: Market data DataFrame for this stock, indexed by date.
               Columns: open, high, low, close, volume, etc.
               Shape: (N_dates, M_fields)
               The latest row (df.index[-1]) is the target date.
        
        Returns:
            float: Normalized factor value for this stock on the target date.
                   Returns NaN if cannot compute.
        """
    
    def compute_universe(self, data: dict[str, pd.DataFrame], date: Any = None) -> FactorResult:
        """Compute for entire universe.
        
        Args:
            data: {stock_code: DataFrame} for all stocks
            date: Target date (optional, defaults to latest)
        
        Returns:
            FactorResult with values for all stocks
        """
```

**Key observations:**
- `name` is a regular attribute set in `__init__`, NOT a `@property`
- `compute()` returns `float` (NaN if cannot compute)
- `compute_universe()` is a concrete method that iterates over all stocks

### 3.2 Current FactorResult (As-Is)

```python
@dataclass(frozen=True)
class FactorResult:
    """Result of computing a factor for a universe."""
    values: pd.Series   # Series indexed by stock code with factor values
    name: str           # Factor name
    date: Any           # Date when computed
```

### 3.3 Current FeatureEngine (As-Is)

Location: `/qmt/src/qmt_local/strategies/multi_factor.py`

```python
class FeatureEngine:
    """Preprocessing pipeline for factor values.
    
    NOTE: This is NOT a factor registry. It only preprocesses factor values.
    """
    
    def __init__(
        self,
        winsorize: float | None = None,  # Winsorize threshold (e.g. 0.05)
        zscore: bool = False,             # Apply z-score normalization
        rank: bool = False,               # Apply rank normalization
        fillna: float | None = 0.0,       # Fill NaN values
        neutralize: str | None = None,    # Industry neutralization
    ):
        """Initialize preprocessing pipeline."""
    
    def process(self, s: pd.Series) -> pd.Series:
        """Apply preprocessing pipeline to factor values.
        
        Pipeline order: winsorize → fillna → zscore → rank
        
        Args:
            s: pd.Series indexed by stock code
        
        Returns:
            pd.Series: Preprocessed factor values
        """
```

**CRITICAL: There is NO `register_factor()` or `compute_features()` method.**
FeatureEngine is a preprocessing pipeline only. The correct wiring pattern is:

```python
# Compute factor for universe
result = factor.compute_universe(data, date)
# Preprocess factor values
processed = feature_engine.process(result.values)
```

### 3.4 Current MultiFactorStrategy (As-Is)

```python
class MultiFactorStrategy(QMTStrategy):
    """Strategy combining multiple AlphaFactors with weights."""
    
    def __init__(
        self,
        factors: list[tuple[AlphaFactor, float]],  # List of (factor, weight) tuples
        top_n: int = 10,                           # Number of top stocks to hold
        rebalance_period: int = 1,                 # Rebalance every N days
        feature_engine: FeatureEngine | None = None,  # Optional preprocessing
        position_value: float | None = None,      # Position value per stock
    ):
        """Initialize multi-factor strategy.
        
        Args:
            factors: List of (factor_instance, weight) tuples
            top_n: Number of stocks in portfolio
            rebalance_period: Days between rebalancing
            feature_engine: Optional preprocessing pipeline
            position_value: Fixed position value (optional)
        """
```

**Key observations:**
- `factors` is a `list[tuple[AlphaFactor, float]]`, NOT a dict
- `rebalance_period` (int), NOT `rebalance_freq` (str)
- No `generate_signals()` method in current implementation

### 3.5 Proposed Bridge API (To-Be)

The following interfaces are proposed for enhanced bridge integration:

```python
class FeatureEngine:
    """Proposed: Factor registry with preprocessing."""
    
    def register_factor(self, factor: AlphaFactor) -> None:
        """Register an AlphaFactor instance."""
    
    def compute_features(self, code: str, df: pd.DataFrame) -> dict[str, float]:
        """Compute all registered factors for a stock.
        
        Returns: {factor_name: factor_value}
        """

class MultiFactorStrategy:
    """Proposed: Enhanced strategy with signal generation."""
    
    def generate_signals(self, date: str, universe: list[str]) -> list[Signal]:
        """Generate trading signals for all stocks on a date.
        
        Returns list of Signal objects with stock, direction, weight.
        """
```

---

## 4. Bridge Transformations

### 4.1 Transformation Pipeline

```
FactorOutput (Stock, Date) float32
    │
    ▼ AlphaBridge.transpose()
pd.DataFrame (Date, Stock) float32
    │
    ▼ AlphaBridge.normalize()  [if re-normalization needed]
pd.DataFrame (Date, Stock) float32, range [-1, 1]
    │
    ▼ ICFilter.filter()        [optional Phase 2 research validation]
pd.DataFrame (Date, Stock) float32, filtered dates only
    │
    ├──→ SignalAlphaFactor      [for qmt]
    ├──→ PtradeStrategyAdapter  [optional future ptrade adapter]
    └──→ AnalysisReport         [optional future analysis UI]
```

### 4.2 AlphaBridge.transpose()

```python
class AlphaBridge:
    @staticmethod
    def transpose(factor_output: FactorOutput) -> pd.DataFrame:
        """Transpose OpenAlpha output from (Stock, Date) to (Date, Stock).
        
        Implementation:
            1. factor_output.values.T  (pandas transpose)
            2. Reindex to DatetimeIndex with daily frequency
            3. Sort index (ascending dates) — qmt convention
            4. Sort columns (ascending stock codes)
            5. Fill remaining NaN with 0.0
            6. Cast to float32
        
        Returns:
            pd.DataFrame: Index=DatetimeIndex(daily), Columns=stock codes (suffix format)
        """
        df = factor_output.values.T
        df.index = pd.DatetimeIndex(df.index, freq="D")
        df = df.sort_index().sort_index(axis=1)
        df = df.fillna(0.0).astype(np.float32)
        return df
```

**Key invariant**: After transpose, `df.loc["2024-01-15", "000001.SZ"]` gives the
factor value for stock 000001 on date 2024-01-15.

### 4.3 AlphaBridge.normalize()

```python
class AlphaBridge:
    @staticmethod
    def normalize(
        df: pd.DataFrame,                # (Date, Stock) after transpose
        method: str = "cs_rank_booksize", # "cs_rank_booksize" | "cs_zscore" | "raw"
        universe_mask: pd.DataFrame | None = None, # (Date, Stock) bool mask
    ) -> pd.DataFrame:
        """Apply cross-sectional normalization.
        
        For "cs_rank_booksize" (default, matching OpenAlpha pipeline):
            1. Apply universe_mask (zero out non-universe stocks)
            2. Per row (each date): rank stocks, subtract 0.5, scale by book size
            3. Result range ≈ [-1, 1], mean ≈ 0
        
        For "cs_zscore":
            1. Per row: z-score standardization
            2. Result range ≈ [-3, 3], mean = 0, std = 1
        
        For "raw":
            Pass through with NaN→0 only.
        
        Args:
            df: (Date, Stock) DataFrame
            method: Normalization method
            universe_mask: Boolean mask, True = in universe. If None, all stocks included.
        
        Returns:
            pd.DataFrame: Same shape, normalized values.
        """
```

### 4.4 StockCodeMapper

```python
class StockCodeMapper:
    """Bidirectional mapping between OpenAlpha (integer) and qmt (suffix) code formats."""
    
    # Known code ranges
    # Shanghai: 600xxx (main), 601xxx, 603xxx, 605xxx, 688xxx (科创板)
    SH_CODES = set(range(600000, 690000))  # Shanghai main + 科创板
    
    # Shenzhen: 000xxx, 001xxx, 002xxx, 003xxx (main), 300xxx (创业板)
    SZ_CODES = set(range(0, 4000)) | set(range(300000, 301000))  # SZ main + 创业板
    
    @staticmethod
    def to_qmt(code: int | str) -> str:
        """Convert OpenAlpha integer code to qmt suffix format.
        000001 → "000001.SZ"
        600000 → "600000.SH"
        300001 → "300001.SZ"  (创业板)
        """
        code_int = int(code)
        if code_int in StockCodeMapper.SH_CODES:
            return f"{code_int:06d}.SH"
        elif code_int in StockCodeMapper.SZ_CODES:
            return f"{code_int:06d}.SZ"
        else:
            raise ValueError(f"Unknown exchange for code {code_int}")
    
    @staticmethod
    def to_openalpha(code: str) -> int:
        """Convert qmt suffix format to OpenAlpha integer code.
        "000001.SZ" → 1
        "600000.SH" → 600000
        """
        return int(code.split(".")[0])
    
    @staticmethod
    def batch_to_qmt(codes: list[int | str]) -> list[str]:
        """Batch conversion. Returns list of qmt-format codes."""
        return [StockCodeMapper.to_qmt(c) for c in codes]
    
    @staticmethod
    def batch_to_openalpha(codes: list[str]) -> list[int]:
        """Batch conversion. Returns list of integer codes."""
        return [StockCodeMapper.to_openalpha(c) for c in codes]
```

**Column mapping during transpose**: `factor_output.values.index` (OpenAlpha integer codes)
→ `transposed.columns` (qmt suffix codes) via `StockCodeMapper.batch_to_qmt()`.

**Note**: For the proposed `FactorOutput`, `values` is defined as:
- Index: stock codes (integer format)
- Columns: dates (DatetimeIndex)

Therefore, after transpose, `transposed.columns` are stock codes (suffix format), and
`transposed.index` are dates.

---

## 5. SignalAlphaFactor Adapter

### 5.1 Class Definition

```python
from qmt_local.strategies.factor import AlphaFactor

class SignalAlphaFactor(AlphaFactor):
    """Adapts pre-computed OpenAlpha factor signals to qmt's AlphaFactor interface.
    
    This is the core adapter: it stores a (Date, Stock) DataFrame of normalized
    factor values and implements qmt's compute(code, df) → float interface.
    
    The compute() method simply looks up the pre-computed value for the given
    stock on the latest date in df. No runtime factor computation occurs.
    """

    def __init__(
        self,
        name: str,
        signal_data: pd.DataFrame,      # (Date, Stock) normalized values
                                        # Index: DatetimeIndex, Columns: qmt-format codes
        ic_threshold: float = 0.02,     # Minimum IC to include this factor
    ):
        self.name = name  # Set name attribute (not a property)
        self._signal_data = signal_data
        self._ic_threshold = ic_threshold
    
    def compute(self, code: str, df: pd.DataFrame) -> float:
        """Look up pre-computed factor value.
        
        Args:
            code: Stock code in qmt format (e.g. "000001.SZ")
            df: Market data DataFrame (used only for date reference)
               The latest date df.index[-1] is the lookup date.
        
        Returns:
            float: Normalized factor value for this stock on this date.
                   Returns 0.0 if stock/date not found in signal_data.
        
        Raises:
            KeyError: If code not in signal_data columns (should not happen
                     after proper column mapping).
        """
        target_date = df.index[-1]
        try:
            return float(self._signal_data.loc[target_date, code])
        except KeyError:
            return 0.0  # Stock not in universe or date not computed
```

### 5.2 Wiring with Current qmt API

```python
# In AlphaBridgeService.process_factor():
# Using PROPOSED FactorOutput (bridge enhancement)
bridge_output = AlphaBridge.transpose(factor_output)
normalized = AlphaBridge.normalize(bridge_output)

# Create adapter
signal_factor = SignalAlphaFactor(
    name=factor_output.metadata.expression,  # e.g. "cs_rank(ts_delta(close, 5))"
    signal_data=normalized,
    ic_threshold=0.02,
)

# Wire with CURRENT MultiFactorStrategy
# factors is list of (factor, weight) tuples
strategy = MultiFactorStrategy(
    factors=[(signal_factor, 1.0)],  # Single factor with weight 1.0
    top_n=10,
    rebalance_period=1,
    feature_engine=None,  # Optional: FeatureEngine for preprocessing
    position_value=None,
)

# If using FeatureEngine for preprocessing:
feature_engine = FeatureEngine(winsorize=0.05, zscore=True)
result = signal_factor.compute_universe(data, date)
processed = feature_engine.process(result.values)
```

---

## 6. Optional Redis Signal Bus

This section is not part of the MVP integration roadmap. Keep the local
OpenAlpha -> qmt bridge working before adding a cross-process signal bus.

### 6.1 Message Protocol

```json
// Channel: alpha:signal:{factor_name}
// Example: alpha:signal:cs_rank_ts_delta_close_5

{
  "date": "2024-01-15",
  "factor_name": "cs_rank(ts_delta(close, 5))",
  "ic": 0.035,           // Rolling IC for this date
  "ir": 0.8,             // Rolling IR for this date
  "signals": {
    "000001.SZ": 0.12,
    "000002.SZ": -0.05,
    "600000.SH": 0.34,
    // ... all stocks in universe
  },
  "universe": "csi_500",
  "normalization": ["cs_rank", "cs_booksize", "at_nan2zero"]
}
```

### 6.2 qmt Subscriber

```python
class QMTSignalSubscriber:
    """Receives factor signals from Redis and creates SignalAlphaFactor instances."""
    
    def __init__(self):
        self._signal_factors: dict[str, SignalAlphaFactor] = {}
    
    async def on_signal(self, channel: str, message: dict) -> None:
        """Callback when a new factor signal arrives.
        
        1. Parse message → DataFrame (Date=message.date, Stock=message.signals keys)
        2. Create SignalAlphaFactor from parsed data
        3. Store for later use in strategy construction
        """
        date = message["date"]
        signals = message["signals"]
        
        # Convert signals dict to single-row DataFrame
        df = pd.DataFrame(signals, index=[pd.Timestamp(date)])
        
        signal_factor = SignalAlphaFactor(
            name=message["factor_name"],
            signal_data=df,
            ic_threshold=message.get("ic", 0.02),
        )
        
        self._signal_factors[message["factor_name"]] = signal_factor
    
    def get_signal_factors(self) -> list[tuple[SignalAlphaFactor, float]]:
        """Get all signal factors as list of (factor, weight) tuples for MultiFactorStrategy."""
        # Default weight of 1.0 for each factor
        return [(f, 1.0) for f in self._signal_factors.values()]
```

---

## 7. Wiring Points (Minimal qmt Modifications)

The MVP bridge requires only local strategy construction changes in qmt. A
runtime subscriber is optional and belongs to a later cross-process deployment
phase.

### 7.1 Wiring Point 1: Strategy Construction

```python
# In qmt's strategy initialization (e.g. strategies/multi_factor.py)
# BEFORE (manual factor construction):
class MyStrategy(MultiFactorStrategy):
    def __init__(self):
        super().__init__(
            factors=[
                (MomentumFactor(), 0.5),
                (ValueFactor(), 0.5),
            ],
            top_n=10,
            rebalance_period=1,
        )

# AFTER (bridge-powered construction):
class MyStrategy(MultiFactorStrategy):
    def __init__(self, signal_factors: list[tuple[SignalAlphaFactor, float]]):
        # Combine existing factors with bridge-provided factors
        all_factors = [
            (MomentumFactor(), 0.3),
            (ValueFactor(), 0.3),
        ] + signal_factors  # Add bridge factors with their weights
        
        super().__init__(
            factors=all_factors,
            top_n=10,
            rebalance_period=1,
        )
```

### 7.2 Optional Extension: Signal Subscriber Startup

```python
# Optional future runtime integration.
# Start a Redis subscriber alongside existing qmt engine components only when
# cross-process signal delivery is required.
async def start_engine(config: dict):
    engine = ...  # existing qmt engine/runtime object
    
    # Bridge integration: subscribe to factor signals
    subscriber = QMTSignalSubscriber()
    bus = RedisSignalBus(redis_url=config["redis_url"])
    await bus.subscribe(
        factor_names=config["bridge_factors"],
        callback=subscriber.on_signal
    )
    
    # Later, when constructing strategy:
    signal_factors = subscriber.get_signal_factors()
    strategy = MyStrategy(signal_factors=signal_factors)
```

---

## 8. Phase 2: Factor Evaluation Tools

Phase 2 adds research-oriented evaluation tools that operate on pre-computed signals.
These are **NOT dispatch gates** — they validate factor quality before deployment.

### 8.1 Conceptual Separation

| Phase | Purpose | Pipeline |
|-------|---------|----------|
| **Phase 1: Signal Creation** | Build deployable signals | `FactorOutput → transpose → normalize → SignalAlphaFactor` |
| **Phase 2: Factor Evaluation** | Validate factor quality | `compute_forward_returns → compute_ic → filter_by_ic` |

**Key distinction**: Phase 1 produces signals for qmt execution. Phase 2 produces metrics for research decisions.

### 8.2 Forward-Return Alignment

```python
def compute_forward_returns(
    prices: pd.DataFrame,       # (Date, Stock) price values
    horizon: int = 1,           # Forward horizon in days (default: 1)
    price_col: str = "close",   # Price column to use
) -> pd.DataFrame:
    """Compute forward returns for IC calculation.
    
    Returns (Date, Stock) DataFrame where each value is the forward return
    from that date to date + horizon.
    
    Example: horizon=1 → returns[t, stock] = (price[t+1] - price[t]) / price[t]
    
    CRITICAL: Last `horizon` dates will have NaN returns (no future data).
    """
    
def align_signals_with_returns(
    signals: pd.DataFrame,      # (Date, Stock) normalized factor values
    returns: pd.DataFrame,      # (Date, Stock) forward returns
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align signal and return dates for IC computation.
    
    Drops dates where either signals or returns are NaN.
    Returns aligned (signals, returns) with matching date indices.
    """

def check_no_leakage(
    signals: pd.DataFrame,      # (Date, Stock) factor values
    returns: pd.DataFrame,      # (Date, Stock) forward returns
    threshold: float = 0.3,     # Correlation threshold for leakage detection
) -> dict[str, Any]:
    """Detect potential same-day leakage (look-ahead bias).
    
    Leakage occurs when factor values correlate with SAME-DAY returns instead of
    forward returns. This indicates the factor may be using future information.
    
    Returns dict with:
        - "same_day_ic": IC computed on same-day returns (should be near 0)
        - "forward_ic": IC computed on forward returns (expected signal)
        - "leakage_detected": bool, True if same_day_ic > threshold
        - "warning": str, explanation if leakage detected
    
    Example output:
        {"same_day_ic": 0.02, "forward_ic": 0.05, "leakage_detected": False}
        {"same_day_ic": 0.45, "forward_ic": 0.03, "leakage_detected": True, 
         "warning": "Factor correlates with same-day returns — possible look-ahead bias"}
    """
```

### 8.3 IC/IR Metrics

```python
@dataclass
class ICResult:
    """Result of IC computation for a single date."""
    date: pd.Timestamp
    ic: float              # Information Coefficient (rank correlation)
    ir: float              # Information Ratio (IC / std(IC) over window)
    n_stocks: int          # Number of stocks in cross-section
    significant: bool      # Whether IC is statistically significant

def compute_ic(
    signals: pd.DataFrame,      # (Date, Stock) factor values
    returns: pd.DataFrame,      # (Date, Stock) forward returns
    method: str = "spearman",   # "spearman" (rank) or "pearson" (linear)
) -> list[ICResult]:
    """Compute per-date IC between signals and forward returns.
    
    For each date:
        1. Extract cross-section of signals and returns
        2. Compute correlation (spearman rank or pearson linear)
        3. Record IC, date, and stock count
    
    Returns list of ICResult for each valid date.
    
    Typical good IC: 0.02-0.05 for daily factors
    Typical good IR: 0.5-1.0 (IC consistent over time)
    """

def filter_by_ic(
    signals: pd.DataFrame,      # (Date, Stock) factor values
    returns: pd.DataFrame,      # (Date, Stock) forward returns
    min_ic: float = 0.02,       # Minimum IC threshold
    min_ir: float = 0.5,        # Minimum IR threshold
    window: int = 20,           # Rolling window for IR computation
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter signal dates by IC/IR quality thresholds.
    
    For each date:
        1. Compute rolling IC over window
        2. Compute IR = mean(IC) / std(IC)
        3. Keep date if IC > min_ic AND IR > min_ir
    
    Returns (filtered_signals, filtered_returns) with only high-quality dates.
    
    USE CASE: Remove dates where factor predictive power is weak or unstable.
    """

def rolling_ic_summary(
    ic_results: list[ICResult],
    window: int = 20,
) -> pd.DataFrame:
    """Compute rolling IC/IR statistics over time.
    
    Returns DataFrame with columns:
        - date: Timestamp
        - ic_mean: Rolling mean IC
        - ic_std: Rolling std IC
        - ir: Rolling Information Ratio
        - ic_decay: IC trend (positive = improving, negative = decaying)
    
    USE CASE: Monitor factor quality over time, detect IC decay.
    """
```

### 8.4 Usage Pattern

```python
# Phase 1: Create deployable signal
factor_output = executor.evaluate("cs_rank(ts_delta(close, 5))")
transposed = AlphaBridge.transpose(factor_output)
normalized = AlphaBridge.normalize(transposed)
signal_factor = SignalAlphaFactor(name="momentum_5d", signal_data=normalized)

# Phase 2: Evaluate factor quality (research-only)
prices = load_prices()  # (Date, Stock) DataFrame
returns = compute_forward_returns(prices, horizon=1)
aligned_signals, aligned_returns = align_signals_with_returns(normalized, returns)

# Check for leakage (look-ahead bias)
leakage_check = check_no_leakage(normalized, returns)
if leakage_check["leakage_detected"]:
    print(f"WARNING: {leakage_check['warning']}")
    # Do NOT deploy this factor — it has look-ahead bias

# Compute IC metrics
ic_results = compute_ic(aligned_signals, aligned_returns)
summary = rolling_ic_summary(ic_results, window=20)

# Filter by quality (optional, for research analysis)
filtered_signals, filtered_returns = filter_by_ic(
    aligned_signals, aligned_returns, min_ic=0.02, min_ir=0.5
)

# Decision: Deploy only if IC/IR meet thresholds
mean_ic = sum(r.ic for r in ic_results) / len(ic_results)
if mean_ic > 0.03 and not leakage_check["leakage_detected"]:
    # Deploy to qmt
    strategy = MultiFactorStrategy(factors=[(signal_factor, 1.0)], top_n=10)
else:
    # Reject factor — insufficient predictive power or leakage detected
    print(f"Factor rejected: mean_ic={mean_ic:.3f}")
```

### 8.5 Leakage Detection Details

**Same-day returns are suspicious**: If a factor correlates strongly with returns on the SAME day (not forward), it likely contains look-ahead information.

| Scenario | Same-Day IC | Forward IC | Interpretation |
|----------|-------------|------------|----------------|
| Clean factor | ~0.00 | 0.03-0.05 | Factor predicts future, not present |
| Leaky factor | 0.30+ | ~0.00 | Factor uses same-day information |
| Mixed | 0.10 | 0.02 | Partial leakage — investigate |

**Common leakage sources**:
- Using `close` when `vwap` is the execution price
- Including same-day volume/turnover in signal
- Normalization that ranks on same-day returns

---

## 9. Integration Test Specification

### 9.1 End-to-End Bridge Test

```python
def test_bridge_e2e():
    """Full pipeline: expression → ndarray → DataFrame → transpose → normalize → SignalAlphaFactor → compute."""
    
    # 1. OpenAlpha: evaluate factor expression (CURRENT API)
    executor = AlphaExecutor(data_dir="tests/fixtures/data")
    executor.load_all_data()  # Must call before evaluate()
    
    alpha_array = executor.evaluate("cs_rank(ts_delta(close, 5))")
    
    # Verify orientation (Stock, Date)
    assert alpha_array is not None
    assert alpha_array.shape[0] < alpha_array.shape[1]  # (Stock, Date) - fewer stocks than dates
    
    # 2. Convert ndarray to DataFrame (PROPOSED FactorOutput simulation)
    # This step would be handled by proposed FactorOutput wrapper
    factor_output = FactorOutput(
        values=pd.DataFrame(alpha_array),  # Index=Stock, Columns=Date
        metadata=FactorMetadata(
            expression="cs_rank(ts_delta(close, 5))",
            normalization=["cs_rank", "cs_booksize", "at_nan2zero"],
            universe="csi_500",
            date_range=("2024-01-01", "2024-03-31"),
            operator_chain=["cs_rank", "ts_delta", "close"],
        )
    )
    
    # 3. Bridge: transpose
    transposed = AlphaBridge.transpose(factor_output)
    assert transposed.shape == factor_output.values.T.shape  # (Date, Stock)
    assert isinstance(transposed.index, pd.DatetimeIndex)
    
    # 4. Bridge: normalize
    normalized = AlphaBridge.normalize(transposed, method="cs_rank_booksize")
    assert normalized.max() <= 1.0 + 1e-6
    assert normalized.min() >= -1.0 - 1e-6
    
    # 5. Create SignalAlphaFactor
    signal_factor = SignalAlphaFactor(
        name="cs_rank(ts_delta(close, 5))",
        signal_data=normalized,
    )
    assert signal_factor.name == "cs_rank(ts_delta(close, 5))"
    
    # 6. Compute for a single stock (CURRENT AlphaFactor.compute)
    mock_df = pd.DataFrame(
        {"close": [10.0, 11.0, 12.0]},
        index=pd.DatetimeIndex(["2024-01-29", "2024-01-30", "2024-01-31"]),
    )
    value = signal_factor.compute("000001.SZ", mock_df)
    assert isinstance(value, float)
    assert abs(value) <= 1.0 + 1e-6
    
    # 7. Compute for universe (CURRENT AlphaFactor.compute_universe)
    mock_data = {
        "000001.SZ": mock_df,
        "000002.SZ": mock_df.copy(),
    }
    result = signal_factor.compute_universe(mock_data)
    assert isinstance(result, FactorResult)
    assert isinstance(result.values, pd.Series)
    assert result.name == "cs_rank(ts_delta(close, 5))"
    
    # 8. Preprocess with FeatureEngine (CURRENT FeatureEngine.process)
    feature_engine = FeatureEngine(winsorize=0.05, zscore=True)
    processed = feature_engine.process(result.values)
    assert isinstance(processed, pd.Series)
```

### 9.2 Orientation Test

```python
def test_orientation_invariant():
    """Verify (Stock,Date) → (Date,Stock) transpose preserves all values."""
    executor = AlphaExecutor(data_dir="tests/fixtures/data")
    executor.load_all_data()
    
    alpha_array = executor.evaluate("close")
    assert alpha_array is not None
    
    # Simulate FactorOutput
    stock_codes = [1, 2, 600000]  # OpenAlpha integer codes
    dates = pd.date_range("2024-01-01", "2024-01-31")
    
    factor_output = FactorOutput(
        values=pd.DataFrame(alpha_array[:len(stock_codes), :len(dates)], 
                           index=stock_codes, columns=dates),
        metadata=FactorMetadata(...)
    )
    
    transposed = AlphaBridge.transpose(factor_output)
    
    # Every value in transposed should match the corresponding value in output
    for stock_idx, stock_code in enumerate(factor_output.values.index[:5]):
        qmt_code = StockCodeMapper.to_qmt(stock_code)
        for date_idx, date in enumerate(factor_output.values.columns[:5]):
            pd_date = pd.Timestamp(date)
            original_val = factor_output.values.iloc[stock_idx, date_idx]
            transposed_val = transposed.loc[pd_date, qmt_code]
            assert abs(original_val - transposed_val) < 1e-6
```

### 9.3 IC Filter Test

```python
def test_ic_filter():
    """Verify IC filter removes low-quality dates."""
    factor_df = pd.DataFrame(...)  # (Date, Stock) normalized values
    returns_df = pd.DataFrame(...)  # (Date, Stock) forward returns
    
    filter = ICFilter(min_ic=0.02, min_ir=0.5)
    filtered, stats = filter.filter(factor_df, returns_df, window=20)
    
    # All remaining dates should have IC > 0.02
    assert all(stats["ic"][filtered.index] > 0.02)
    # Some dates should have been removed
    assert len(filtered) < len(factor_df)
```

### 9.4 MultiFactorStrategy Wiring Test

```python
def test_multi_factor_strategy_wiring():
    """Verify SignalAlphaFactor works with CURRENT MultiFactorStrategy."""
    
    # Create signal factor
    signal_data = pd.DataFrame(
        {"000001.SZ": [0.1], "000002.SZ": [-0.05]},
        index=[pd.Timestamp("2024-01-15")],
    )
    signal_factor = SignalAlphaFactor(name="test_factor", signal_data=signal_data)
    
    # Wire with CURRENT MultiFactorStrategy (list of tuples)
    strategy = MultiFactorStrategy(
        factors=[(signal_factor, 1.0)],  # (factor, weight) tuple
        top_n=10,
        rebalance_period=1,
    )
    
    # Verify strategy has the factor
    assert len(strategy.factors) == 1
    assert strategy.factors[0][0] == signal_factor
    assert strategy.factors[0][1] == 1.0
```

---

## 10. Performance Requirements

| Operation | Target Latency | Notes |
|-----------|---------------|-------|
| Transpose (500×1200) | < 50ms | Pandas `.T` is O(n*m), negligible for this size |
| Normalize (1200×500) | < 100ms | Per-row rank operation, vectorized |
| SignalAlphaFactor.compute() | < 1ms | Single DataFrame lookup |
| IC filter (1200 dates) | < 200ms | Per-date correlation, vectorized |
| Redis publish | < 10ms | Optional signal-bus extension only |
| Redis subscribe callback | < 50ms | Optional signal-bus extension only |

---

## 11. Dependency Matrix

| Bridge Component | Depends On | Version |
|-----------------|-----------|---------|
| AlphaBridge.transpose | pandas, numpy | pandas≥2.0, numpy≥1.24 |
| AlphaBridge.normalize | pandas, numpy | same |
| SignalAlphaFactor | pandas, qmt.strategies.factor | qmt (local) |
| StockCodeMapper | (none) | — |
| ICFilter | pandas, scipy | scipy≥1.10 (for stats) |
| RedisSignalBus | redis[hiredis], asyncio | Optional extension; redis≥5.0 |
| AlphaBridgeService | all above | — |
| Integration tests | pytest, pytest-asyncio | pytest≥8.0 |
