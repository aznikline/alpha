"""
RunArtifacts - AutoResearch 研究产物对比

扫描 runs/ 目录中的 Research Artifact，生成跨实验排行榜。
"""
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


class RunArtifacts:
    """AutoResearch artifact 管理与对比"""

    REQUIRED_FILES = {'summary.json', 'config.json', 'top_factors.csv'}

    def __init__(self, runs_dir: str = './runs'):
        self.runs_dir = Path(runs_dir)

    def leaderboard(self) -> pd.DataFrame:
        rows = []
        for run_dir in self._run_dirs():
            rows.append(self._load_run(run_dir))

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        sort_cols = [col for col in ['best_val_sr', 'meta_val_sr', 'n_evaluated'] if col in df.columns]
        return df.sort_values(sort_cols, ascending=False).reset_index(drop=True)

    def print_leaderboard(self, top: int = 20) -> None:
        df = self.leaderboard()
        print("\n" + "=" * 90)
        print("🏁 AutoResearch Run Leaderboard")
        print("=" * 90)
        if df.empty:
            print(f"未发现 artifact，请先运行: python3 openalpha_cli.py auto --artifact {self.runs_dir}/<run_id>")
            print("=" * 90 + "\n")
            return

        cols = [
            'run_id', 'generated_at', 'n_candidates', 'n_evaluated',
            'best_val_sr', 'best_score', 'meta_val_sr', 'dashboard',
        ]
        print(df[[col for col in cols if col in df.columns]].head(top).to_string(index=False))
        print("=" * 90 + "\n")

    def compare_top_factors(self, top: int = 50) -> pd.DataFrame:
        rows = []
        for run_dir in self._run_dirs():
            top_file = run_dir / 'top_factors.csv'
            if not top_file.exists():
                continue
            df = pd.read_csv(top_file).head(top).copy()
            df['run_id'] = run_dir.name
            rows.append(df)

        if not rows:
            return pd.DataFrame()
        combined = pd.concat(rows, ignore_index=True)
        return combined.sort_values('val_sr', ascending=False).reset_index(drop=True)

    def export_leaderboard(self, output_file: str = './runs_leaderboard.csv') -> str:
        df = self.leaderboard()
        df.to_csv(output_file, index=False)
        print(f"📦 Run leaderboard 已导出: {output_file}")
        return output_file

    def _run_dirs(self) -> List[Path]:
        if not self.runs_dir.exists():
            return []
        return [path for path in self.runs_dir.iterdir() if path.is_dir() and self._is_artifact(path)]

    def _is_artifact(self, run_dir: Path) -> bool:
        existing = {path.name for path in run_dir.iterdir() if path.is_file()}
        return self.REQUIRED_FILES.issubset(existing)

    def _load_run(self, run_dir: Path) -> Dict:
        summary = self._load_json(run_dir / 'summary.json')
        config = self._load_json(run_dir / 'config.json')
        top = self._load_top(run_dir / 'top_factors.csv')
        meta_weights = self._load_meta_weights(run_dir / 'meta_weights.csv')
        meta_metrics = summary.get('meta_metrics', {})
        best = summary.get('best_factor', {})

        return {
            'run_id': run_dir.name,
            'path': str(run_dir),
            'generated_at': summary.get('generated_at', ''),
            'n_candidates': summary.get('n_candidates', 0),
            'n_valid': summary.get('n_valid', 0),
            'n_evaluated': summary.get('n_evaluated', 0),
            'templates': ','.join(config.get('templates', [])),
            'windows': ','.join(str(value) for value in config.get('windows', [])),
            'best_expr': best.get('expr', top.get('expr', '')),
            'best_score': best.get('score', top.get('score', 0.0)),
            'best_val_sr': best.get('val_sr', top.get('val_sr', 0.0)),
            'best_ic_ir': best.get('ic_ir', top.get('ic_ir', 0.0)),
            'best_tvr': best.get('tvr', top.get('tvr', 0.0)),
            'meta_val_sr': meta_metrics.get('val_sr', 0.0),
            'meta_ic_ir': meta_metrics.get('ic_ir', 0.0),
            'meta_factor_count': len(meta_weights),
            'dashboard': summary.get('dashboard', str(run_dir / 'dashboard.html')),
        }

    def _load_json(self, path: Path) -> Dict:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding='utf-8'))

    def _load_top(self, path: Path) -> Dict:
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        if df.empty:
            return {}
        row = df.iloc[0]
        return {
            'expr': row.get('expr', ''),
            'score': float(row.get('score', 0)),
            'val_sr': float(row.get('val_sr', 0)),
            'ic_ir': float(row.get('ic_ir', 0)) if 'ic_ir' in df.columns else 0.0,
            'tvr': float(row.get('tvr', 0)),
        }

    def _load_meta_weights(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)


if __name__ == '__main__':
    RunArtifacts().print_leaderboard()
