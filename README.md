# alpha — Quant Factor Discovery & Integration Workspace

> Current state: **OpenAlpha is runnable** (factor evaluation + backtest), bridge and data layer are **planned only** (see docs/).

---

## What Is This

This workspace contains three quant projects that will be integrated via a bridge layer:

| Project | Location | Role | Status |
|---------|----------|------|--------|
| **OpenAlpha** | `./OpenAlpha/` | Factor discovery — 30+ operators, eval engine, VWAP L/S backtest | Runnable |
| **qmt** | `../qmt/` | Local event-driven quant framework — data, strategy, backtest, trading, risk, gateway modules | Runnable (Windows needed for live MiniQMT paths) |
| **ptrade** | `../ptrade/` | Lightweight strategy — Ptrade API compat, local backtest | Runnable |

**Integration goal**: Build a bridge that connects OpenAlpha's factor discovery output → qmt's AlphaFactor execution interface.

---

## Directory Structure

```
alpha/
├── README.md                  # This file
├── docs/                      # Integration specs (all Proposed, not yet implemented)
│   ├── integration-roadmap.md # MVP-first phases, decisions, risks
│   ├── bridge-spec.md         # Bridge layer interface signatures
│   └── data-layer-spec.md     # Unified data infrastructure spec
└── OpenAlpha/                 # Factor discovery project
    ├── src/simres/
    │   ├── expr.py            # AlphaExecutor (evaluate + backtest)
    │   └── operators.py       # 30+ vectorized operators (ts_*, cs_*, at_*)
    ├── data/20251231/         # Market data directory (CSV/Parquet)
    ├── run_alpha.py           # MAIN ENTRY POINT
    ├── factor_factory.py      # Factor template generation
    ├── factor_combination.py  # Portfolio optimization (MVO/RP)
    ├── gp_enhanced.py         # Genetic programming factor mining
    ├── gp_mining.py           # GP mining runner
    ├── factor_analysis.py     # Factor IC analysis
    ├── build_strategy.py      # Strategy builder
    ├── data_generator.py      # Mock data generator
    ├── param_sensitivity.py   # Parameter sensitivity analysis
    ├── README.md              # Factor gallery (expressions + charts)
    └── [various .png, .pkl, .txt reports]
```

---

## How to Run OpenAlpha

### 1. Prerequisites

```bash
cd OpenAlpha
# No pyproject.toml yet — install dependencies manually:
pip install numpy pandas bottleneck akshare matplotlib scipy
```

### 2. Prepare Data

Place market data in `./data/20251231/` directory. Files can be CSV or Parquet format.
Each file should contain daily OHLCV data (open, high, low, close, vwap, volume, etc.)
with stock codes as rows and dates as columns.

Required fields in data:
- `vwap`, `close`, `open`, `high`, `low`, `volume`, `amount`
- `csi_500_weight` (index constituent weights)
- Industry classification data (for `cs_indneut`)

### 3. Run Factor Evaluation + Backtest

```bash
python run_alpha.py
```

This script:
1. Initializes `AlphaExecutor(data_dir='./data/20251231')`
2. Calls `executor.load_all_data()` — loads all CSV/Parquet files as (Stock, Date) float32 matrices
3. Reads factor expressions from `src/ruiqiwang_csi_500.txt`
4. Evaluates each expression via `executor.evaluate(full_expr)` — returns np.ndarray
5. Backtests via `executor.backtest(alpha)` — returns dict with returns/turnover
6. Prints annualized return, volatility, Sharpe ratio, max drawdown

### 4. Current API (Key Methods)

```python
from simres.expr import AlphaExecutor

executor = AlphaExecutor(data_dir='./data/20251231')
executor.load_all_data()  # Required before evaluate()

# Evaluate single factor expression
alpha = executor.evaluate("cs_rank(ts_delta(close, 5))")  # Returns np.ndarray (Stock, Date) or None

# Backtest
result = executor.backtest(alpha, price='vwap')  # Returns dict
# result keys: datestr, net_ret, long_ret, short_ret, tvr, long_num, short_num
```

**Data orientation**: (Stock, Date) with axis=0=CrossSectional, axis=1=TimeSeries.

---

## Documentation Map

| Document | What It Covers | Implementation Status |
|----------|---------------|----------------------|
| `docs/integration-roadmap.md` | MVP-first OpenAlpha → qmt roadmap, task IDs, risks, decisions | **Proposed** — not yet implemented |
| `docs/bridge-spec.md` | Bridge layer interfaces: transpose, normalize, SignalAlphaFactor, optional signal bus | **Proposed** — not yet implemented |
| `docs/data-layer-spec.md` | DataProvider ABC, ParquetCache, PITManager, DataManager facade | **Proposed** — not yet implemented |
| `OpenAlpha/README.md` | Factor gallery (expressions + performance charts) | Gallery only — no API docs |

**Important**: All `docs/` content describes **planned** interfaces. They contain "Current API" sections documenting existing code and "Proposed API" sections for planned additions. Do not treat Proposed sections as implemented.

---

## Key Facts for Integration

| Aspect | OpenAlpha Current | qmt Current | Bridge Need |
|--------|------------------|-------------|-------------|
| Data orientation | `(Stock, Date)` np.ndarray | `(Date, Stock)` pd.DataFrame | Transpose adapter |
| Stock codes | Integer (000001) | Suffix (000001.SZ) | StockCodeMapper |
| Factor interface | `evaluate(expr) → ndarray` | `AlphaFactor.compute(code, df) → float` | SignalAlphaFactor adapter |
| Backtest output | dict with 7 keys | FactorAnalysisReport dataclass | Format conversion |
| Package structure | None (no pyproject.toml) | Full package | Phase 0: add packaging |

---

## What's Missing (Not Yet Built)

- `/alpha/bridge/` — Bridge layer directory (planned in Phase 1)
- `/alpha/tests/` — Test suite (planned in Phase 0)
- `/alpha/pyproject.toml` — Package config (planned in Phase 0)
- `/alpha/data/` — Unified data layer or adapter (deferred until Phase 3 ADR)
- `/alpha/dashboard/` — Web dashboard (optional Phase 4 extension)

---

## Related Projects

- **qmt**: `/Users/wizout/op/quant/qmt/` — Trading execution layer
- **ptrade**: `/Users/wizout/op/quant/ptrade/` — Lightweight strategy layer
- **Reference**: See `docs/integration-roadmap.md` for the current MVP-first integration plan
