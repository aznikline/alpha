"""
FactorDiagnostics - Alphalens 风格因子分层诊断

输出 IC 序列、分位数组合收益、单调性评分、Top 分组换手等核心体检指标。
"""
import warnings
warnings.filterwarnings("ignore")

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .factor_lab import FactorLab, FactorResult


class FactorDiagnostics:
    """因子分层诊断器"""

    def __init__(self, lab: Optional[FactorLab] = None):
        self.lab = lab or FactorLab()

    def analyze(self, expr: str, quantiles: int = 5) -> Dict:
        result = self.lab.test_alpha(expr)
        if 'error' in result.metrics:
            return {
                'expr': expr,
                'error': result.metrics['error'],
                'summary': {},
                'ic': pd.DataFrame(),
                'quantiles': pd.DataFrame(),
                'turnover': pd.DataFrame(),
            }

        return self.from_result(result, quantiles=quantiles)

    def from_result(self, result: FactorResult, quantiles: int = 5) -> Dict:
        alpha = result.alpha_matrix
        ret1 = self.lab.executor.context['ret1']
        dates = pd.to_datetime(self.lab.executor.context['datestr'])

        ic = self._ic_series(alpha, ret1, dates)
        quantile_returns = self._quantile_returns(alpha, ret1, dates, quantiles)
        turnover = self._turnover(alpha, dates, quantiles)
        summary = self._summary(result, ic, quantile_returns, turnover)

        return {
            'expr': result.expr,
            'summary': summary,
            'ic': ic,
            'quantiles': quantile_returns,
            'turnover': turnover,
        }

    def print_report(self, report: Dict, max_rows: int = 8) -> None:
        if 'error' in report:
            print(f"诊断失败: {report['error']}")
            return

        summary = report['summary']
        print("\n" + "=" * 70)
        print("🧪 因子分层诊断报告")
        print("=" * 70)
        print(f"表达式: {report['expr']}")
        print(f"IC 均值: {summary['ic_mean']:+.4f} | ICIR: {summary['ic_ir']:+.3f}")
        print(f"多空分位收益: {summary['long_short_return']:+.2%}")
        print(f"单调性评分: {summary['monotonic_score']:.1f}/100")
        print(f"Top 分组换手: {summary['top_quantile_turnover']:.2%}")

        print("\n分位组年化收益:")
        cols = ['quantile', 'ann_ret', 'ann_vol', 'sr', 'avg_count']
        print(report['quantiles'][cols].to_string(index=False))

        print("\n最近 IC:")
        print(report['ic'].tail(max_rows).to_string(index=False))
        print("=" * 70 + "\n")

    def _ic_series(self, alpha: np.ndarray, ret1: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
        rows = []
        for t in range(min(alpha.shape[1], ret1.shape[1])):
            a = alpha[:, t]
            r = ret1[:, t]
            mask = ~np.isnan(a) & ~np.isnan(r)
            if mask.sum() < 10:
                continue
            ic, _ = stats.spearmanr(a[mask], r[mask])
            rows.append({'date': dates[t], 'ic': ic})
        return pd.DataFrame(rows)

    def _quantile_returns(
        self,
        alpha: np.ndarray,
        ret1: np.ndarray,
        dates: pd.DatetimeIndex,
        quantiles: int,
    ) -> pd.DataFrame:
        returns_by_q = {q: [] for q in range(1, quantiles + 1)}
        counts_by_q = {q: [] for q in range(1, quantiles + 1)}

        for t in range(min(alpha.shape[1], ret1.shape[1])):
            a = alpha[:, t]
            r = ret1[:, t]
            mask = ~np.isnan(a) & ~np.isnan(r)
            if mask.sum() < quantiles * 5:
                continue

            ranks = pd.Series(a[mask]).rank(method='first')
            labels = pd.qcut(ranks, quantiles, labels=False) + 1
            period_ret = pd.Series(r[mask])

            for q in range(1, quantiles + 1):
                selected = period_ret[labels == q]
                returns_by_q[q].append(selected.mean() if len(selected) else np.nan)
                counts_by_q[q].append(len(selected))

        rows = []
        for q in range(1, quantiles + 1):
            values = np.array(returns_by_q[q], dtype=float)
            valid = values[~np.isnan(values)]
            ann_ret = np.nanmean(valid) * 252 if len(valid) else 0.0
            ann_vol = np.nanstd(valid) * np.sqrt(252) if len(valid) else 0.0
            rows.append({
                'quantile': q,
                'ann_ret': ann_ret,
                'ann_vol': ann_vol,
                'sr': ann_ret / (ann_vol + 1e-12),
                'avg_count': np.nanmean(counts_by_q[q]) if counts_by_q[q] else 0,
            })

        return pd.DataFrame(rows)

    def _turnover(self, alpha: np.ndarray, dates: pd.DatetimeIndex, quantiles: int) -> pd.DataFrame:
        top_sets = []
        bottom_sets = []
        rows = []

        for t in range(alpha.shape[1]):
            a = alpha[:, t]
            mask = ~np.isnan(a)
            if mask.sum() < quantiles * 5:
                continue

            stock_idx = np.where(mask)[0]
            ranks = pd.Series(a[mask]).rank(method='first')
            labels = pd.qcut(ranks, quantiles, labels=False) + 1
            top = set(stock_idx[labels == quantiles])
            bottom = set(stock_idx[labels == 1])

            if top_sets:
                prev_top = top_sets[-1]
                prev_bottom = bottom_sets[-1]
                top_turnover = 1 - len(top & prev_top) / max(len(top), 1)
                bottom_turnover = 1 - len(bottom & prev_bottom) / max(len(bottom), 1)
                rows.append({
                    'date': dates[t],
                    'top_turnover': top_turnover,
                    'bottom_turnover': bottom_turnover,
                })

            top_sets.append(top)
            bottom_sets.append(bottom)

        return pd.DataFrame(rows)

    def _summary(
        self,
        result: FactorResult,
        ic: pd.DataFrame,
        quantile_returns: pd.DataFrame,
        turnover: pd.DataFrame,
    ) -> Dict:
        ic_values = ic['ic'].dropna() if len(ic) > 0 else pd.Series(dtype=float)
        ic_mean = ic_values.mean() if len(ic_values) else 0.0
        ic_ir = ic_mean / (ic_values.std() + 1e-12) if len(ic_values) else 0.0

        quantile_returns = quantile_returns.sort_values('quantile')
        q_ret = quantile_returns['ann_ret'].values if len(quantile_returns) else np.array([])
        long_short_return = q_ret[-1] - q_ret[0] if len(q_ret) >= 2 else 0.0
        monotonic_score = self._monotonic_score(q_ret)
        top_turnover = turnover['top_turnover'].mean() if len(turnover) > 0 else 0.0

        return {
            'val_sr': result.metrics.get('val_sr', 0.0),
            'ic_mean': float(ic_mean),
            'ic_ir': float(ic_ir),
            'long_short_return': float(long_short_return),
            'monotonic_score': float(monotonic_score),
            'top_quantile_turnover': float(top_turnover),
        }

    def _monotonic_score(self, values: np.ndarray) -> float:
        if len(values) < 2:
            return 0.0
        diffs = np.diff(values)
        direction = np.sign(values[-1] - values[0]) or 1
        aligned = (np.sign(diffs) == direction).mean()
        spread = abs(values[-1] - values[0])
        return float(np.clip(aligned * 70 + min(spread * 100, 30), 0, 100))


if __name__ == '__main__':
    lab = FactorLab()
    diagnostics = FactorDiagnostics(lab=lab)
    expr = lab.get_template('momentum_price', w=20)
    diagnostics.print_report(diagnostics.analyze(expr))
