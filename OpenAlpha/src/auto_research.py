"""
AutoResearchPipeline - 一键因子研究流水线

自动串联: 生成候选因子 -> 表达式体检 -> 并行评估 -> 实验记录 -> 分层诊断 -> 衰减监控 -> Meta Model -> Dashboard。
"""
import warnings
warnings.filterwarnings("ignore")

import json
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .factor_dashboard import FactorDashboard
from .factor_decay_monitor import FactorDecayMonitor
from .factor_diagnostics import FactorDiagnostics
from .factor_expression_inspector import FactorExpressionInspector
from .factor_lab import FactorLab
from .factor_meta_model import FactorMetaModel
from .factor_tracker import FactorTracker, tracker as get_tracker
from .fast_eval import FastEval


class AutoResearchPipeline:
    """一键因子研究流水线"""

    def __init__(
        self,
        lab: Optional[FactorLab] = None,
        tracker: Optional[FactorTracker] = None,
        engine: Optional[FastEval] = None,
    ):
        self.lab = lab or FactorLab()
        self.tracker = tracker or get_tracker()
        self.engine = engine
        self.inspector = FactorExpressionInspector()

    def run(
        self,
        templates: Optional[List[str]] = None,
        windows: Optional[List[int]] = None,
        top_k: int = 20,
        workers: int = 4,
        enable_ic: bool = True,
        dashboard_file: str = './openalpha_auto_dashboard.html',
        artifact_dir: Optional[str] = None,
    ) -> Dict:
        templates = templates or ['momentum_price', 'volume_price_corr', 'volatility_std', 'reg_beta_market']
        windows = windows or [5, 10, 20, 40, 60]
        engine = self.engine or FastEval(n_workers=workers, use_cache=True, enable_ic=enable_ic)
        artifact_path = Path(artifact_dir) if artifact_dir else None
        if artifact_path:
            artifact_path.mkdir(parents=True, exist_ok=True)
            dashboard_file = str(artifact_path / 'dashboard.html')

        print("\n" + "=" * 70)
        print("🤖 OpenAlpha AutoResearch Pipeline")
        print("=" * 70)

        candidates = self._generate_candidates(templates=templates, windows=windows)
        print(f"生成候选因子: {len(candidates)}")

        inspected = self._inspect_candidates(candidates)
        valid = inspected[inspected['status'] != 'fail'].copy()
        print(f"静态体检通过/警告: {len(valid)} | 失败: {len(inspected) - len(valid)}")
        if len(valid) == 0:
            raise ValueError("没有通过静态体检的候选因子")

        evaluated = engine.evaluate(valid['expr'].tolist(), show_progress=True)
        if evaluated.empty:
            raise ValueError("候选因子评估结果为空")

        results = valid.merge(evaluated, on='expr', how='inner')
        if 'score' not in results.columns:
            results['score'] = results['val_sr'] * 0.5 + results.get('ic_ir', 0) * 0.3 - results['tvr'] * 0.2
        results = results.sort_values('score', ascending=False).reset_index(drop=True)
        print(f"评估成功: {len(results)}")

        self._log_results(results)
        top = results.head(top_k).copy()
        diagnostics = self._diagnose_top(top.head(min(5, top_k)))
        decay = self._monitor_decay(top_k=min(10, top_k))
        meta_report = self._fit_meta(top['expr'].head(min(5, len(top))).tolist())
        dashboard = FactorDashboard(lab=self.lab, tracker=self.tracker).build(output_file=dashboard_file, top_k=top_k)

        summary = {
            'generated_at': datetime.now().isoformat(),
            'config': {
                'templates': templates,
                'windows': windows,
                'top_k': top_k,
                'workers': workers,
                'enable_ic': enable_ic,
            },
            'n_candidates': len(candidates),
            'n_valid': len(valid),
            'n_evaluated': len(results),
            'top': top,
            'diagnostics': diagnostics,
            'decay': decay,
            'meta': meta_report,
            'dashboard': dashboard,
            'artifact_dir': str(artifact_path) if artifact_path else None,
        }
        if artifact_path:
            self._save_artifacts(
                artifact_path=artifact_path,
                summary=summary,
                candidates=candidates,
                inspected=inspected,
                results=results,
                top=top,
                diagnostics=diagnostics,
                decay=decay,
                meta_report=meta_report,
            )
        self._print_summary(summary)
        return summary

    def _generate_candidates(self, templates: List[str], windows: List[int]) -> pd.DataFrame:
        rows = []
        for template in templates:
            if template not in self.lab.templates:
                print(f"跳过未知模板: {template}")
                continue

            raw_template = self.lab.templates[template]
            if '{w}' in raw_template:
                for window in windows:
                    rows.append({'template': template, 'params': {'w': window}, 'expr': self.lab.get_template(template, w=window)})
            elif '{d}' in raw_template:
                for window in windows:
                    rows.append({'template': template, 'params': {'d': window}, 'expr': self.lab.get_template(template, d=window)})
            elif '{w1}' in raw_template and '{w2}' in raw_template:
                for w1, w2 in product(windows, windows):
                    if w1 < w2:
                        rows.append({'template': template, 'params': {'w1': w1, 'w2': w2}, 'expr': self.lab.get_template(template, w1=w1, w2=w2)})
            else:
                rows.append({'template': template, 'params': {}, 'expr': self.lab.get_template(template)})

        return pd.DataFrame(rows).drop_duplicates('expr').reset_index(drop=True)

    def _inspect_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for row in candidates.itertuples(index=False):
            report = self.inspector.inspect(row.expr)
            rows.append({
                'template': row.template,
                'params': row.params,
                'expr': row.expr,
                'status': report['status'],
                'complexity': report['complexity'],
                'issue_count': len(report['issues']),
            })
        return pd.DataFrame(rows)

    def _log_results(self, results: pd.DataFrame) -> None:
        for _, row in results.iterrows():
            self.tracker.log(
                expr=row['expr'],
                metrics={
                    'val_sr': row['val_sr'],
                    'train_sr': row['train_sr'],
                    'ic_ir': row.get('ic_ir', 0),
                    'tvr': row['tvr'],
                    'score': row['score'],
                    'complexity': row['complexity'],
                },
                tags=['auto_research', row['template']],
                source='auto_research',
                skip_duplicate=True,
            )

    def _diagnose_top(self, top: pd.DataFrame) -> pd.DataFrame:
        diagnostics = FactorDiagnostics(lab=self.lab)
        rows = []
        for _, row in top.iterrows():
            report = diagnostics.analyze(row['expr'])
            if 'error' in report:
                continue
            summary = report['summary']
            rows.append({
                'expr': row['expr'],
                'ic_mean': summary['ic_mean'],
                'ic_ir': summary['ic_ir'],
                'long_short_return': summary['long_short_return'],
                'monotonic_score': summary['monotonic_score'],
                'top_quantile_turnover': summary['top_quantile_turnover'],
            })
        return pd.DataFrame(rows)

    def _monitor_decay(self, top_k: int) -> pd.DataFrame:
        monitor = FactorDecayMonitor(lab=self.lab, tracker=self.tracker)
        return monitor.monitor_top(metric='val_sr', k=top_k, source='auto_research')

    def _fit_meta(self, exprs: List[str]) -> Dict:
        if len(exprs) < 2:
            return {'error': '候选因子不足，跳过 Meta Model'}
        model = FactorMetaModel(lab=self.lab, tracker=self.tracker)
        return model.fit(exprs, ridge=1.0)

    def _save_artifacts(
        self,
        artifact_path: Path,
        summary: Dict,
        candidates: pd.DataFrame,
        inspected: pd.DataFrame,
        results: pd.DataFrame,
        top: pd.DataFrame,
        diagnostics: pd.DataFrame,
        decay: pd.DataFrame,
        meta_report: Dict,
    ) -> None:
        candidates.to_csv(artifact_path / 'candidates.csv', index=False)
        inspected.to_csv(artifact_path / 'inspection.csv', index=False)
        results.to_csv(artifact_path / 'evaluation.csv', index=False)
        top.to_csv(artifact_path / 'top_factors.csv', index=False)
        diagnostics.to_csv(artifact_path / 'diagnostics.csv', index=False)
        decay.to_csv(artifact_path / 'decay.csv', index=False)

        if isinstance(meta_report, dict) and 'weights' in meta_report:
            meta_report['weights'].to_csv(artifact_path / 'meta_weights.csv', index=False)

        config = summary['config']
        (artifact_path / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')

        summary_payload = {
            'generated_at': summary['generated_at'],
            'n_candidates': summary['n_candidates'],
            'n_valid': summary['n_valid'],
            'n_evaluated': summary['n_evaluated'],
            'dashboard': summary['dashboard'],
            'artifact_dir': summary['artifact_dir'],
            'best_factor': self._best_factor_payload(top),
            'meta_metrics': self._meta_metrics_payload(meta_report),
        }
        (artifact_path / 'summary.json').write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"📦 Research Artifact 已保存: {artifact_path}")

    def _best_factor_payload(self, top: pd.DataFrame) -> Dict:
        if len(top) == 0:
            return {}
        row = top.iloc[0]
        return {
            'expr': row.get('expr', ''),
            'score': float(row.get('score', 0)),
            'val_sr': float(row.get('val_sr', 0)),
            'train_sr': float(row.get('train_sr', 0)),
            'ic_ir': float(row.get('ic_ir', 0)) if 'ic_ir' in top.columns else 0.0,
            'tvr': float(row.get('tvr', 0)),
        }

    def _meta_metrics_payload(self, meta_report: Dict) -> Dict:
        if not isinstance(meta_report, dict) or 'metrics' not in meta_report:
            return {}
        metrics = meta_report['metrics']
        return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}

    def _print_summary(self, summary: Dict) -> None:
        print("\n" + "=" * 70)
        print("AutoResearch Summary")
        print("=" * 70)
        print(f"候选: {summary['n_candidates']} | 体检通过: {summary['n_valid']} | 评估成功: {summary['n_evaluated']}")
        print(f"Dashboard: {summary['dashboard']}")
        if summary.get('artifact_dir'):
            print(f"Artifact: {summary['artifact_dir']}")

        print("\nTop 因子:")
        cols = ['score', 'val_sr', 'train_sr', 'ic_ir', 'tvr', 'expr']
        print(summary['top'][[col for col in cols if col in summary['top'].columns]].head(10).to_string(index=False))

        if isinstance(summary['decay'], pd.DataFrame) and len(summary['decay']) > 0:
            print("\n衰减监控:")
            print(summary['decay'][['status', 'health_score', 'recent_sr', 'sr_decay', 'expr']].head(10).to_string(index=False))

        meta = summary.get('meta', {})
        if 'metrics' in meta:
            metrics = meta['metrics']
            print("\nMeta Model:")
            print(f"  Train SR={metrics.get('train_sr', 0):+.3f} | Val SR={metrics.get('val_sr', 0):+.3f} | ICIR={metrics.get('ic_ir', 0):+.3f}")
        print("=" * 70)


if __name__ == '__main__':
    AutoResearchPipeline().run()
