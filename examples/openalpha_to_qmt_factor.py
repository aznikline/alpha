"""Example: OpenAlpha factor expression → qmt-compatible SignalAlphaFactor.

Demonstrates the full Phase 1 bridge pipeline using mock data:
1. Create mock factor output (simulating AlphaExecutor.evaluate() result)
2. Wrap into FactorOutput with labeled DataFrame
3. Transpose from (Stock, Date) to (Date, Stock)
4. Normalize cross-sectionally (cs_rank_booksize)
5. Create SignalAlphaFactor for qmt consumption
6. Compute factor values for individual stocks
7. Show (factor, weight) tuple wiring pattern for MultiFactorStrategy

Run: python examples/openalpha_to_qmt_factor.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bridge.output import FactorOutput, FactorMetadata, wrap_factor_output
from bridge.code_mapper import StockCodeMapper
from bridge.transpose import AlphaBridge as TransposeBridge
from bridge.normalize import AlphaBridge as NormalizeBridge
from bridge.signal_factor import SignalAlphaFactor


def create_mock_factor_output() -> FactorOutput:
    """Create mock factor output simulating AlphaExecutor.evaluate() result.
    
    In production, this would come from:
        executor = AlphaExecutor(data_dir='./data/20251231')
        executor.load_all_data()
        alpha_array = executor.evaluate("cs_rank(ts_delta(close, 5))")
        factor_output = wrap_factor_output(
            alpha=alpha_array,
            stock_list=executor.context['stock_list'],
            datestr=executor.context['datestr'],
            expression="cs_rank(ts_delta(close, 5))",
        )
    """
    n_stocks = 50
    n_dates = 120
    np.random.seed(42)
    
    alpha = np.random.randn(n_stocks, n_dates).astype(np.float32)
    
    # Generate realistic stock codes (mix of SZ and SH)
    sz_codes = [f"{i:06d}.SZ" for i in range(1, 26)]
    sh_codes = [f"{600000 + i:06d}.SH" for i in range(25)]
    stocks = sz_codes + sh_codes
    
    # Generate trading dates
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    datestr = [d.strftime("%Y%m%d") for d in dates]
    
    return wrap_factor_output(
        alpha=alpha,
        stock_list=np.array(stocks),
        datestr=np.array(datestr),
        expression="cs_rank(ts_delta(close, 5))",
    )


def main() -> None:
    print("=" * 60)
    print("OpenAlpha → qmt Bridge Pipeline Demo")
    print("=" * 60)
    
    # Step 1: Create mock factor output
    print("\n[Step 1] Creating mock FactorOutput...")
    factor_output = create_mock_factor_output()
    print(f"  Expression: {factor_output.expression}")
    print(f"  Shape: {factor_output.values.shape} (Stock, Date)")
    print(f"  Stocks: {len(factor_output.stocks)} ({factor_output.stocks[0]}, ..., {factor_output.stocks[-1]})")
    print(f"  Dates: {len(factor_output.dates)} ({factor_output.dates[0]}, ..., {factor_output.dates[-1]})")
    print(f"  Dtype: {factor_output.values.dtypes.iloc[0]}")
    
    # Step 2: Transpose to (Date, Stock) for qmt
    print("\n[Step 2] Transposing to (Date, Stock)...")
    transposed = TransposeBridge.transpose(factor_output)
    print(f"  Shape: {transposed.shape} (Date, Stock)")
    print(f"  Index type: {type(transposed.index).__name__}")
    print(f"  Columns: {transposed.columns[0]}, ..., {transposed.columns[-1]}")
    
    # Step 3: Normalize cross-sectionally
    print("\n[Step 3] Normalizing (cs_rank_booksize)...")
    normalized = NormalizeBridge.normalize(transposed, method="cs_rank_booksize")
    nonzero_values = normalized.values[normalized.values != 0.0]
    print(f"  Shape: {normalized.shape}")
    print(f"  Non-zero value range: [{nonzero_values.min():.4f}, {nonzero_values.max():.4f}]")
    print(f"  Mean of non-zero: {nonzero_values.mean():.6f} (≈0)")
    print(f"  Dtype: {normalized.dtypes.iloc[0]}")
    
    # Step 4: Create SignalAlphaFactor
    print("\n[Step 4] Creating SignalAlphaFactor...")
    signal_factor = SignalAlphaFactor(
        name=factor_output.expression,
        signal_data=normalized,
    )
    print(f"  Name: {signal_factor.name}")
    print(f"  Signal data shape: {signal_factor.signal_data.shape}")
    
    # Step 5: Compute for individual stocks
    print("\n[Step 5] Computing factor values for individual stocks...")
    last_date = normalized.index[-1]
    mock_df = pd.DataFrame(
        {"close": [10.0, 11.0, 12.0]},
        index=pd.DatetimeIndex([last_date - pd.Timedelta(days=2), last_date - pd.Timedelta(days=1), last_date]),
    )
    
    test_stocks = ["000001.SZ", "600000.SH", "000025.SZ"]
    for code in test_stocks:
        value = signal_factor.compute(code, mock_df)
        print(f"  {code} on {last_date.strftime('%Y-%m-%d')}: {value:.6f}")
    
    # Step 6: Show MultiFactorStrategy wiring pattern
    print("\n[Step 6] MultiFactorStrategy wiring pattern...")
    factors = [(signal_factor, 1.0)]
    print(f"  factors = [(SignalAlphaFactor(name='{signal_factor.name}'), 1.0)]")
    print(f"  Number of factors: {len(factors)}")
    print(f"  Factor 0: name='{factors[0][0].name}', weight={factors[0][1]}")
    
    # StockCodeMapper demo
    print("\n[Bonus] StockCodeMapper examples...")
    test_codes = [1, 600000, 300001, 688001]
    for code in test_codes:
        qmt_code = StockCodeMapper.to_qmt(code)
        exchange = StockCodeMapper.get_exchange(code)
        print(f"  {code} → {qmt_code} (exchange: {exchange})")
    
    print("\n" + "=" * 60)
    print("Pipeline demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()