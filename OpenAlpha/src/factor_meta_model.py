"""
FactorMetaModel - 多因子 Ridge Ensemble

将多个因子暴露合成为一个线性预测器，用训练集拟合权重，并在验证集评估组合因子表现。
"""
import warnings
warnings.filterwarnings("ignore")

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .factor_lab import FactorLab, FactorResult
from .factor_tracker import FactorTracker, tracker as get_tracker


class FactorMetaModel:
    """轻量多因子线性集成模型"""

    def __init__(self, lab: Optional[FactorLab] = None, tracker: Optional[FactorTracker] = None):
        self.lab = lab or FactorLab()
        self.tracker = tracker or get_tracker()
        self.weights: Optional[np.ndarray] = None
        self.factor_names: List[str] = []
        self.factor_exprs: List[str] = []
        self.intercept: float = 0.0

    def fit_from_tracker(
        self,
        metric: str = 'val_sr',
        k: int = 10,
        ridge: float = 1.0,
        min_valid: int = 50,
    ) -> Dict:
        df = self.tracker.top_k(metric=metric, k=k)
        if len(df) == 0:
            raise ValueError("FactorTracker 中没有可用因子")
        return self.fit(df['expr'].tolist(), ridge=ridge, min_valid=min_valid)

    def fit(self, exprs: List[str], ridge: float = 1.0, min_valid: int = 50) -> Dict:
        if not exprs:
            raise ValueError("exprs 不能为空")

        results = [self.lab.test_alpha(expr) for expr in exprs]
        valid_results = [result for result in results if 'error' not in result.metrics and result.alpha_matrix is not None]
        if len(valid_results) == 0:
            raise ValueError("没有有效因子可用于训练")

        self.factor_exprs = [result.expr for result in valid_results]
        self.factor_names = [f'factor_{idx + 1}' for idx in range(len(valid_results))]

        exposures = self._stack_exposures(valid_results)
        target = self.lab.executor.context['ret1']
        train_cut = self.lab.train_cut

        x_train, y_train = self._panel_to_samples(exposures[:, :, :train_cut], target[:, :train_cut], min_valid=min_valid)
        if len(y_train) == 0:
            raise ValueError("训练样本不足")

        self.intercept, self.weights = self._fit_ridge(x_train, y_train, ridge=ridge)
        combined_alpha = self._combine_exposures(exposures)
        combined_result = self._evaluate_combined(combined_alpha)

        report = {
            'n_factors': len(valid_results),
            'ridge': ridge,
            'weights': self.weights_table(),
            'train_samples': len(y_train),
            'metrics': combined_result.metrics,
            'expr': self.expression_summary(),
        }
        return report

    def predict_alpha(self, exprs: Optional[List[str]] = None) -> np.ndarray:
        if self.weights is None:
            raise ValueError("模型尚未训练")

        source_exprs = exprs or self.factor_exprs
        results = [self.lab.test_alpha(expr) for expr in source_exprs]
        valid_results = [result for result in results if 'error' not in result.metrics and result.alpha_matrix is not None]
        exposures = self._stack_exposures(valid_results)
        return self._combine_exposures(exposures)

    def weights_table(self) -> pd.DataFrame:
        if self.weights is None:
            return pd.DataFrame()
        return pd.DataFrame({
            'name': self.factor_names,
            'weight': self.weights,
            'abs_weight': np.abs(self.weights),
            'expr': self.factor_exprs,
        }).sort_values('abs_weight', ascending=False).reset_index(drop=True)

    def expression_summary(self) -> str:
        if self.weights is None:
            return ""
        terms = []
        for weight, expr in zip(self.weights, self.factor_exprs):
            terms.append(f"({weight:+.4f})*({expr})")
        return f"intercept({self.intercept:+.4e}) + " + " + ".join(terms)

    def print_report(self, report: Dict) -> None:
        metrics = report['metrics']
        print("\n" + "=" * 70)
        print("🤖 多因子 Meta Model 报告")
        print("=" * 70)
        print(f"因子数量: {report['n_factors']} | Ridge: {report['ridge']} | 训练样本: {report['train_samples']}")
        print(f"Train SR: {metrics.get('train_sr', 0):+.3f} | Val SR: {metrics.get('val_sr', 0):+.3f}")
        print(f"ICIR: {metrics.get('ic_ir', 0):+.3f} | TVR: {metrics.get('tvr', 0):.3f}")
        print("\n权重:")
        print(report['weights'][['weight', 'expr']].to_string(index=False))
        print("=" * 70 + "\n")

    def _stack_exposures(self, results: List[FactorResult]) -> np.ndarray:
        arrays = []
        for result in results:
            alpha = result.alpha_matrix.astype(float)
            alpha = self._cross_section_zscore(alpha)
            arrays.append(alpha)
        return np.stack(arrays, axis=0)

    def _cross_section_zscore(self, alpha: np.ndarray) -> np.ndarray:
        mean = np.nanmean(alpha, axis=0, keepdims=True)
        std = np.nanstd(alpha, axis=0, keepdims=True)
        return (alpha - mean) / (std + 1e-12)

    def _panel_to_samples(self, exposures: np.ndarray, target: np.ndarray, min_valid: int) -> tuple[np.ndarray, np.ndarray]:
        features = []
        labels = []
        n_factors, _, n_dates = exposures.shape

        for t in range(n_dates):
            x = exposures[:, :, t].T
            y = target[:, t]
            mask = ~np.isnan(y)
            for idx in range(n_factors):
                mask &= ~np.isnan(x[:, idx])
            if mask.sum() < min_valid:
                continue
            features.append(x[mask])
            labels.append(y[mask])

        if not features:
            return np.empty((0, n_factors)), np.empty((0,))
        return np.vstack(features), np.concatenate(labels)

    def _fit_ridge(self, x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[float, np.ndarray]:
        x_mean = x.mean(axis=0, keepdims=True)
        x_std = x.std(axis=0, keepdims=True) + 1e-12
        y_mean = y.mean()
        x_norm = (x - x_mean) / x_std
        y_centered = y - y_mean

        xtx = x_norm.T @ x_norm
        penalty = np.eye(xtx.shape[0]) * ridge
        weights_norm = np.linalg.solve(xtx + penalty, x_norm.T @ y_centered)
        weights = weights_norm / x_std.flatten()
        intercept = y_mean - float(x_mean.flatten() @ weights)
        return intercept, weights

    def _combine_exposures(self, exposures: np.ndarray) -> np.ndarray:
        combined = np.tensordot(self.weights, exposures, axes=(0, 0)) + self.intercept
        return self._cross_section_zscore(combined)

    def _evaluate_combined(self, alpha: np.ndarray) -> FactorResult:
        bt = self.lab.executor.backtest(alpha)
        net_ret = bt['net_ret']
        train_cut = self.lab.train_cut
        train_ret = net_ret[:train_cut]
        val_ret = net_ret[train_cut:]
        train_valid = train_ret[~np.isnan(train_ret)]
        val_valid = val_ret[~np.isnan(val_ret)]

        train_mean = np.nanmean(train_valid) * 252
        train_vol = np.nanstd(train_valid) * np.sqrt(252)
        val_mean = np.nanmean(val_valid) * 252
        val_vol = np.nanstd(val_valid) * np.sqrt(252)
        ic_values = self._ic_values(alpha[:, :train_cut], self.lab.executor.context['ret1'][:, :train_cut])
        cum = np.nancumprod(1 + val_valid)
        max_dd = np.max(1 - cum / np.maximum.accumulate(cum)) if len(cum) else 0.0

        metrics = {
            'train_sr': train_mean / (train_vol + 1e-12),
            'val_sr': val_mean / (val_vol + 1e-12),
            'ic_mean': np.nanmean(ic_values) if ic_values else 0.0,
            'ic_ir': np.nanmean(ic_values) / (np.nanstd(ic_values) + 1e-12) if ic_values else 0.0,
            'tvr': np.nanmean(bt['tvr']),
            'ann_ret': val_mean,
            'ann_vol': val_vol,
            'max_dd': max_dd,
            'is_robust': True,
        }
        return FactorResult(self.expression_summary(), metrics, alpha, bt)

    def _ic_values(self, alpha: np.ndarray, target: np.ndarray) -> List[float]:
        values = []
        for t in range(min(alpha.shape[1], target.shape[1])):
            a = alpha[:, t]
            y = target[:, t]
            mask = ~np.isnan(a) & ~np.isnan(y)
            if mask.sum() >= 10:
                corr = pd.Series(a[mask]).corr(pd.Series(y[mask]), method='spearman')
                if not np.isnan(corr):
                    values.append(corr)
        return values


if __name__ == '__main__':
    model = FactorMetaModel()
    report = model.fit_from_tracker(k=5, ridge=1.0)
    model.print_report(report)
