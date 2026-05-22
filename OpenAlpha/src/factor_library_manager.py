"""
FactorLibraryManager - 版本化因子库管理

从实验记录和衰减监控中筛选可用因子，生成可复用、可追溯的因子库版本。
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .factor_decay_monitor import FactorDecayMonitor
from .factor_lab import FactorLab
from .factor_tracker import FactorTracker, tracker as get_tracker


class FactorLibraryManager:
    """版本化因子库管理器"""

    def __init__(self, library_dir: str = './factor_libraries', lab: Optional[FactorLab] = None, tracker: Optional[FactorTracker] = None):
        self.library_dir = Path(library_dir)
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.lab = lab or FactorLab()
        self.tracker = tracker or get_tracker()

    def promote(
        self,
        name: str,
        metric: str = 'val_sr',
        k: int = 50,
        min_metric: float = 0.0,
        allowed_status: Optional[List[str]] = None,
        include_decay: bool = True,
    ) -> str:
        allowed_status = allowed_status or ['active', 'watch']
        top = self.tracker.top_k(metric=metric, k=k)
        if len(top) == 0:
            raise ValueError("没有可提升的实验因子")

        top = top[top[metric] >= min_metric].copy() if metric in top.columns else top
        if include_decay and len(top) > 0:
            decay = FactorDecayMonitor(lab=self.lab, tracker=self.tracker).monitor_top(metric=metric, k=k)
            top = top.merge(decay[['expr', 'status', 'health_score', 'recent_sr', 'sr_decay']], on='expr', how='left')
            top = top[top['status'].isin(allowed_status)].copy()
        else:
            top['status'] = 'unverified'
            top['health_score'] = None

        if len(top) == 0:
            raise ValueError("没有满足提升条件的因子")

        version = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = self.library_dir / name / version
        output_dir.mkdir(parents=True, exist_ok=True)
        factors = self._library_records(top)
        metadata = {
            'name': name,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'metric': metric,
            'k': k,
            'min_metric': min_metric,
            'allowed_status': allowed_status,
            'include_decay': include_decay,
            'count': len(factors),
        }
        payload = {'metadata': metadata, 'factors': factors}

        (output_dir / 'library.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
        pd.DataFrame(factors).to_csv(output_dir / 'factors.csv', index=False)
        (self.library_dir / name / 'latest.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"📚 因子库已提升: {output_dir / 'library.json'}")
        return str(output_dir)

    def list_libraries(self) -> pd.DataFrame:
        rows = []
        for library in sorted(self.library_dir.iterdir() if self.library_dir.exists() else []):
            if not library.is_dir():
                continue
            latest = library / 'latest.json'
            if not latest.exists():
                continue
            payload = json.loads(latest.read_text(encoding='utf-8'))
            metadata = payload.get('metadata', {})
            rows.append({
                'name': metadata.get('name', library.name),
                'latest_version': metadata.get('version', ''),
                'count': metadata.get('count', 0),
                'metric': metadata.get('metric', ''),
                'created_at': metadata.get('created_at', ''),
                'path': str(latest),
            })
        return pd.DataFrame(rows)

    def show(self, name: str, version: str = 'latest') -> Dict:
        path = self.library_dir / name / ('latest.json' if version == 'latest' else f'{version}/library.json')
        if not path.exists():
            raise FileNotFoundError(f"因子库不存在: {path}")
        return json.loads(path.read_text(encoding='utf-8'))

    def print_library(self, name: str, version: str = 'latest', top: int = 20) -> None:
        payload = self.show(name=name, version=version)
        metadata = payload['metadata']
        factors = pd.DataFrame(payload['factors'])
        print("\n" + "=" * 80)
        print(f"📚 Factor Library: {metadata['name']} @ {metadata['version']}")
        print("=" * 80)
        print(f"创建时间: {metadata['created_at']} | 因子数: {metadata['count']} | 指标: {metadata['metric']}")
        cols = [col for col in ['val_sr', 'ic_ir', 'tvr', 'status', 'health_score', 'expr'] if col in factors.columns]
        print(factors[cols].head(top).to_string(index=False))
        print("=" * 80 + "\n")

    def _library_records(self, df: pd.DataFrame) -> List[Dict]:
        cols = [
            'expr', 'val_sr', 'train_sr', 'ic_ir', 'tvr', 'score', 'source',
            'tags', 'status', 'health_score', 'recent_sr', 'sr_decay', 'note',
        ]
        available = [col for col in cols if col in df.columns]
        records = df[available].to_dict('records')
        for idx, record in enumerate(records, start=1):
            record['rank'] = idx
        return records


if __name__ == '__main__':
    manager = FactorLibraryManager()
    print(manager.list_libraries().to_string(index=False))
