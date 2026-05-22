"""
FactorDecayMonitor - 因子衰减监控与 Walk-forward 验证

核心能力:
- 对单个因子做滚动窗口样本外健康度评估
- 批量监控 FactorTracker 中的 Top 因子
- 输出 active / watch / retired 状态与退役建议
"""
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .factor_lab import FactorLab, FactorResult
from .factor_tracker import FactorTracker, tracker as get_tracker


class FactorDecayMonitor:
    """因子滚动健康度监控器"""

    def __init__(self, lab: Optional[FactorLab] = None, tracker: Optional[FactorTracker] = None):
        self.lab = lab or FactorLab()
        self.tracker = tracker or get_tracker()

    def analyze(
        self,
        expr: str,
        window: int = 60,
        step: int = 20,
        min_obs: int = 20,
    ) -> Dict:
        result = self.lab.test_alpha(expr)
        if 'error' in result.metrics:
            return {
                'expr': expr,
                'status': 'error',
                'health_score': 0.0,
                'reason': result.metrics['error'],
                'windows': pd.DataFrame(),
                'summary': {},
            }

        windows = self._rolling_windows(result, window=window, step=step, min_obs=min_obs)
        summary = self._summarize_windows(windows, result.metrics)
        status = self._classify(summary)

        return {
            'expr': expr,
            'status': status,
            'health_score': summary['health_score'],
            'reason': self._status_reason(status, summary),
            'windows': windows,
            'summary': summary,
            'evaluated_at': datetime.now().isoformat(),
        }

    def monitor_top(
        self,
        metric: str = 'val_sr',
        k: int = 20,
        window: int = 60,
        step: int = 20,
        min_obs: int = 20,
        source: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> pd.DataFrame:
        candidates = self._candidate_factors(metric=metric, k=k, source=source, tag=tag)
        reports = []

        for _, row in candidates.iterrows():
            report = self.analyze(row['expr'], window=window, step=step, min_obs=min_obs)
            summary = report['summary']
            reports.append({
                'status': report['status'],
                'health_score': report['health_score'],
                'reason': report['reason'],
                'expr': report['expr'],
                'base_metric': row.get(metric, np.nan),
                'recent_sr': summary.get('recent_sr', np.nan),
                'early_sr': summary.get('early_sr', np.nan),
                'sr_decay': summary.get('sr_decay', np.nan),
                'recent_ic_mean': summary.get('recent_ic_mean', np.nan),
                'positive_window_ratio': summary.get('positive_window_ratio', np.nan),
                'window_count': summary.get('window_count', 0),
            })

        if not reports:
            return pd.DataFrame()

        df = pd.DataFrame(reports)
        return df.sort_values(['health_score', 'base_metric'], ascending=False).reset_index(drop=True)

    def print_report(self, report: Dict, max_windows: int = 8) -> None:
        print("\n" + "=" * 70)
        print("📉 因子衰减监控报告")
        print("=" * 70)
        print(f"表达式: {report['expr']}")
        print(f"状态: {report['status']} | 健康度: {report['health_score']:.1f}/100")
        print(f"原因: {report['reason']}")

        summary = report.get('summary', {})
        if summary:
            print("\n核心指标:")
            print(f"  早期 SR: {summary.get('early_sr', np.nan):+.3f}")
            print(f"  近期 SR: {summary.get('recent_sr', np.nan):+.3f}")
            print(f"  SR 衰减: {summary.get('sr_decay', np.nan):+.3f}")
            print(f"  近期 IC: {summary.get('recent_ic_mean', np.nan):+.4f}")
            print(f"  正收益窗口比例: {summary.get('positive_window_ratio', np.nan):.1%}")

        windows = report.get('windows', pd.DataFrame())
        if len(windows) > 0:
            print("\n最近窗口:")
            cols = ['start', 'end', 'sr', 'ann_ret', 'ann_vol', 'ic_mean', 'tvr']
            print(windows[cols].tail(max_windows).to_string(index=False))
        print("=" * 70 + "\n")

    def _rolling_windows(self, result: FactorResult, window: int, step: int, min_obs: int) -> pd.DataFrame:
        bt = result.bt_result
        alpha = result.alpha_matrix
        ret1 = self.lab.executor.context['ret1']
        net_ret = bt['net_ret']
        dates = pd.to_datetime(bt['datestr'])
        rows = []

        for start in range(0, len(net_ret) - min_obs + 1, step):
            end = min(start + window, len(net_ret))
            if end - start < min_obs:
                continue

            period_ret = net_ret[start:end]
            valid_ret = period_ret[~np.isnan(period_ret)]
            if len(valid_ret) < min_obs:
                continue

            ann_ret = np.nanmean(valid_ret) * 252
            ann_vol = np.nanstd(valid_ret) * np.sqrt(252)
            sr = ann_ret / (ann_vol + 1e-12)
            period_tvr = np.nanmean(bt['tvr'][start:end])
            ic_values = self._window_ic(alpha[:, start:end], ret1[:, start:end])

            rows.append({
                'start': dates[start].strftime('%Y-%m-%d'),
                'end': dates[end - 1].strftime('%Y-%m-%d'),
                'start_idx': start,
                'end_idx': end,
                'sr': sr,
                'ann_ret': ann_ret,
                'ann_vol': ann_vol,
                'ic_mean': np.nanmean(ic_values) if ic_values else 0.0,
                'ic_ir': np.nanmean(ic_values) / (np.nanstd(ic_values) + 1e-12) if ic_values else 0.0,
                'tvr': period_tvr,
                'obs': len(valid_ret),
            })

        return pd.DataFrame(rows)

    def _window_ic(self, alpha: np.ndarray, ret1: np.ndarray) -> List[float]:
        values = []
        for t in range(alpha.shape[1]):
            a = alpha[:, t]
            r = ret1[:, t]
            mask = ~np.isnan(a) & ~np.isnan(r)
            if mask.sum() >= 10:
                ic, _ = stats.spearmanr(a[mask], r[mask])
                if not np.isnan(ic):
                    values.append(ic)
        return values

    def _summarize_windows(self, windows: pd.DataFrame, base_metrics: Dict) -> Dict:
        if len(windows) == 0:
            return {
                'health_score': 0.0,
                'window_count': 0,
                'early_sr': np.nan,
                'recent_sr': np.nan,
                'sr_decay': np.nan,
                'recent_ic_mean': np.nan,
                'positive_window_ratio': 0.0,
            }

        split = max(1, len(windows) // 3)
        early = windows.head(split)
        recent = windows.tail(split)
        early_sr = early['sr'].mean()
        recent_sr = recent['sr'].mean()
        sr_decay = recent_sr - early_sr
        recent_ic_mean = recent['ic_mean'].mean()
        positive_window_ratio = (windows['sr'] > 0).mean()
        turnover_penalty = max(0.0, min(1.0, (base_metrics.get('tvr', 0.0) - 1.0) / 2.0))

        score = 50.0
        score += np.clip(recent_sr, -2, 2) * 15
        score += np.clip(recent_ic_mean * 100, -2, 2) * 8
        score += (positive_window_ratio - 0.5) * 30
        score += np.clip(sr_decay, -2, 2) * 8
        score -= turnover_penalty * 15
        score = float(np.clip(score, 0, 100))

        return {
            'health_score': score,
            'window_count': len(windows),
            'early_sr': float(early_sr),
            'recent_sr': float(recent_sr),
            'sr_decay': float(sr_decay),
            'recent_ic_mean': float(recent_ic_mean),
            'positive_window_ratio': float(positive_window_ratio),
            'base_val_sr': float(base_metrics.get('val_sr', np.nan)),
            'base_ic_ir': float(base_metrics.get('ic_ir', np.nan)),
            'base_tvr': float(base_metrics.get('tvr', np.nan)),
        }

    def _classify(self, summary: Dict) -> str:
        if summary.get('window_count', 0) == 0:
            return 'retired'
        if summary['health_score'] >= 65 and summary['recent_sr'] > 0 and summary['positive_window_ratio'] >= 0.5:
            return 'active'
        if summary['health_score'] >= 40 and summary['recent_sr'] > -0.5:
            return 'watch'
        return 'retired'

    def _status_reason(self, status: str, summary: Dict) -> str:
        if summary.get('window_count', 0) == 0:
            return '有效滚动窗口不足，无法继续使用'
        if status == 'active':
            return '近期收益、IC 与窗口稳定性仍满足使用条件'
        if status == 'watch':
            return '因子仍有部分信号，但近期表现或稳定性已出现衰减'
        return '近期表现显著衰减，建议从生产因子库退役'

    def _candidate_factors(
        self,
        metric: str,
        k: int,
        source: Optional[str],
        tag: Optional[str],
    ) -> pd.DataFrame:
        if source is None and tag is None:
            return self.tracker.top_k(metric=metric, k=k)

        df = self.tracker.query(tag=tag, source=source)
        if len(df) == 0 or metric not in df.columns:
            return pd.DataFrame()
        return df.sort_values(metric, ascending=False).head(k).reset_index(drop=True)


if __name__ == '__main__':
    monitor = FactorDecayMonitor()
    top = monitor.tracker.top_k(metric='val_sr', k=1)
    if len(top) == 0:
        print("暂无已记录因子，请先运行 demo_complete_workflow.py 或记录实验。")
    else:
        decay_report = monitor.analyze(top.iloc[0]['expr'])
        monitor.print_report(decay_report)
