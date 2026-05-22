"""Bridge package — converts OpenAlpha factor output to qmt-compatible format.

Provides the full pipeline: FactorOutput → transpose → normalize → SignalAlphaFactor.
Phase 2 adds research validation: forward-return alignment, IC/IR evaluation.
Phase 3 adds data adapter: QmtDataAdapter (thin wrapper over qmt DataManager).

qmt runtime dependency is optional:
- AlphaFactor/FactorResult are copied locally in bridge/_qmt_types.py
- signal_factor.py prefers qmt's originals when available, falls back to local copies
- data_adapter.py requires qmt DataManager for data fetching (graceful fallback if absent)

Typical usage:
    from bridge import (
        AlphaBridge,
        FactorOutput,
        SignalAlphaFactor,
        wrap_factor_output,
        compute_forward_returns,
        compute_ic,
    )
"""
from bridge._qmt_types import AlphaFactor, FactorResult
from bridge.code_mapper import StockCodeMapper
from bridge.data_adapter import QmtDataAdapter
from bridge.ic_filter import ICResult, compute_ic, filter_by_ic, rolling_ic_summary
from bridge.output import FactorMetadata, FactorOutput, wrap_factor_output
from bridge.returns import align_signals_with_returns, check_no_leakage, compute_forward_returns
from bridge.signal_factor import SignalAlphaFactor


class AlphaBridge:
    """Unified bridge class combining pipeline and validation static methods.

    Pipeline methods are defined in separate modules but assembled here
    as a single namespace for convenience.

    Phase 2 validation methods are also available as namespace attributes.
    """

    transpose = __import__("bridge.transpose", fromlist=["AlphaBridge"]).AlphaBridge.transpose
    normalize = __import__("bridge.normalize", fromlist=["AlphaBridge"]).AlphaBridge.normalize

    compute_forward_returns = compute_forward_returns
    align_signals_with_returns = align_signals_with_returns
    check_no_leakage = check_no_leakage
    compute_ic = compute_ic
    filter_by_ic = filter_by_ic
    rolling_ic_summary = rolling_ic_summary


__all__ = [
    "AlphaBridge",
    "AlphaFactor",
    "FactorOutput",
    "FactorMetadata",
    "FactorResult",
    "wrap_factor_output",
    "StockCodeMapper",
    "SignalAlphaFactor",
    "compute_forward_returns",
    "align_signals_with_returns",
    "check_no_leakage",
    "compute_ic",
    "filter_by_ic",
    "rolling_ic_summary",
    "ICResult",
    "QmtDataAdapter",
]