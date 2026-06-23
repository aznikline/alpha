# alpha — Quant Factor Discovery & Integration Workspace

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-116%20passed%20%7C%2095%25%20cov-brightgreen.svg)](./tests)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

> OpenAlpha factor discovery → qmt execution bridge. Factor mining (expression / genetic programming), an integration **bridge** layer (transpose → normalize → `SignalAlphaFactor`), and 116 tests at 95% coverage.

## What Is This

This workspace connects factor discovery with execution: it mines alpha factors with OpenAlpha, then adapts them into qmt's `AlphaFactor` interface via a bridge layer.

| Component | Location | Role | Status |
|----------|----------|------|--------|
| **OpenAlpha** | `./OpenAlpha/` | Factor discovery — 30+ operators, eval engine, VWAP L/S backtest | Runnable |
| **bridge/** | `./bridge/` | OpenAlpha → qmt adapter: transpose, normalize, `SignalAlphaFactor`, code mapping | **Implemented** + tested |
| **tests/** | `./tests/` | Integration tests for the bridge pipeline | **116 tests, 95% cov** |
| qmt (private) | separate repo | Author's local event-driven quant framework — the execution target | Private (trading logic) |
| ptrade (private) | separate repo | Author's lightweight strategy layer | Private |

> `qmt` and `ptrade` are the author's private execution frameworks — the bridge targets qmt's `AlphaFactor` interface, but ships with a Protocol fallback so `bridge/` and its tests run standalone without qmt installed. They're intentionally not published (to isolate trading logic), but the bridge is fully usable and tested independently.

---

## Directory Structure

```
alpha/
├── README.md
├── LICENSE
├── pyproject.toml             # openalpha-bridge package config
├── bridge/                    # OpenAlpha → qmt adapter (implemented)
│   ├── transpose.py           # (Stock,Date) ↔ (Date,Stock)
│   ├── normalize.py           # factor normalization
│   ├── signal_factor.py       # SignalAlphaFactor — adapts to qmt AlphaFactor
│   ├── code_mapper.py         # integer code ↔ 000001.SZ
│   ├── data_adapter.py        # data orientation adapter
│   ├── ic_filter.py           # IC-based filtering
│   ├── output.py              # FactorOutput / wrap_factor_output
│   ├── returns.py             # return calc helpers
│   └── _qmt_types.py          # Protocol fallback when qmt not installed
├── tests/                      # 116 pytest integration tests (95% bridge cov)
│   ├── test_qmt_bridge.py     # full pipeline: transpose → normalize → SignalAlphaFactor
│   ├── test_alignment.py      # IC computation (Pearson/Spearman)
│   ├── test_code_mapper.py
│   ├── test_data_adapter.py
│   └── test_output.py
├── docs/                       # Integration specs (planned vs implemented)
│   ├── integration-roadmap.md
│   ├── bridge-spec.md
│   ├── data-layer-spec.md
│   └── data-layer-adr.md
├── examples/
└── OpenAlpha/                  # Factor discovery project
    ├── src/simres/
    │   ├── expr.py            # AlphaExecutor (evaluate + backtest)
    │   └── operators.py       # 30+ vectorized operators (ts_*, cs_*, at_*)
    ├── run_alpha.py           # MAIN ENTRY POINT
    ├── factor_factory.py      # Factor template generation
    ├── factor_combination.py  # Portfolio optimization (MVO/RP)
    ├── gp_enhanced.py         # Genetic programming factor mining
    ├── gp_mining.py           # GP mining runner
    ├── factor_analysis.py     # Factor IC analysis
    ├── build_strategy.py      # Strategy builder
    └── data_generator.py      # Mock data generator
```

---

## How to Run

### Bridge tests

```bash
pip install -e ".[dev]"        # numpy pandas scipy pytest pytest-cov etc.
pytest                          # 116 tests, 95% bridge coverage
```

### OpenAlpha factor evaluation + backtest

```bash
cd OpenAlpha
pip install numpy pandas bottleneck akshare matplotlib scipy
python run_alpha.py
```

Requires market data in `./data/20251231/` (CSV/Parquet, OHLCV with `vwap`/`close`/`open`/`high`/`low`/`volume`/`amount` + `csi_500_weight` + industry classification for `cs_indneut`). Use `data_generator.py` for mock data.

### Factor evaluation API

```python
from simres.expr import AlphaExecutor

executor = AlphaExecutor(data_dir='./data/20251231')
executor.load_all_data()  # required before evaluate()

alpha = executor.evaluate("cs_rank(ts_delta(close, 5))")  # → np.ndarray (Stock, Date)
result = executor.backtest(alpha, price='vwap')           # → dict (returns/turnover)
```

**Data orientation**: (Stock, Date) with axis=0=CrossSectional, axis=1=TimeSeries.

---

## Documentation Map

| Document | Covers | Status |
|----------|--------|--------|
| `docs/integration-roadmap.md` | MVP-first OpenAlpha → qmt roadmap, task IDs, risks | Proposed roadmap |
| `docs/bridge-spec.md` | Bridge layer interfaces: transpose, normalize, SignalAlphaFactor | **Implemented** in `bridge/` |
| `docs/data-layer-spec.md` | DataProvider ABC, ParquetCache, PITManager, DataManager | Proposed — not yet built |
| `docs/data-layer-adr.md` | Data layer architecture decision record | Proposed |
| `OpenAlpha/README.md` | Factor gallery (expressions + performance) | Gallery |

> `docs/` specs describe interfaces — `bridge-spec.md` is now implemented; `data-layer-*` remain proposed.

---

## Integration Facts

| Aspect | OpenAlpha | qmt (external) | Bridge provides |
|--------|-----------|-----------------|-----------------|
| Data orientation | `(Stock, Date)` np.ndarray | `(Date, Stock)` pd.DataFrame | `transpose` adapter |
| Stock codes | Integer (000001) | Suffix (000001.SZ) | `code_mapper` |
| Factor interface | `evaluate(expr) → ndarray` | `AlphaFactor.compute(code, df) → float` | `SignalAlphaFactor` |
| Backtest output | dict with 7 keys | `FactorAnalysisReport` | `output` module |
| Package structure | `openalpha-bridge` (this repo) | qmt package | bridge imports qmt with Protocol fallback |

---

## What's Not Yet Built

- `data-layer` — Unified data infrastructure (DataProvider ABC, ParquetCache, PITManager). See `docs/data-layer-spec.md`. Deferred until data-layer ADR is finalized.
- `dashboard/` — Web dashboard (optional, Phase 4).

The bridge layer, test suite, and package config are **all implemented**.

---

## Related

- **alpha-mining-system** — sibling project, a broader Alpha factor mining platform with genetic programming + DeepAlpha: https://github.com/aznikline/alpha-mining-system
- Integration target (`qmt`, `ptrade`) referenced in `docs/integration-roadmap.md`.

## License

[MIT License](./LICENSE) © 2026 aznikline
