"""
FactorTracker - 因子实验追踪与版本管理

核心特性:
- 📝 自动记录所有因子实验，无需手动维护
- 🔍 强大的查询与筛选功能，快速找到历史优质因子
- 📊 版本对比，分析因子演进过程
- 🎯 重复因子自动去重
- 📈 实验趋势分析

使用方式:
    from src.factor_tracker import tracker

    # 记录实验
    tracker.log(
        expr="ts_correlation(close, volume, 10)",
        metrics={'val_sr': 0.85, 'ic_ir': 2.3},
        tags=['volume', 'correlation'],
        note="基础量价相关性因子"
    )

    # 查询 Top 因子
    top = tracker.top_k(metric='val_sr', k=20)

    # 按标签筛选
    vol_factors = tracker.query(tag='volume')
"""
import warnings
warnings.filterwarnings("ignore")

import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

import pandas as pd
import numpy as np

TRACKER_DIR = './.factor_experiments'
os.makedirs(TRACKER_DIR, exist_ok=True)


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if pd.isna(value) if isinstance(value, (float, np.floating)) else False:
        return None
    return value


class FactorExperiment:
    """单次因子实验记录"""

    def __init__(self, expr: str, metrics: Dict[str, float],
                 tags: List[str] = None, note: str = None,
                 source: str = 'manual',
                 expr_hash: str = None):
        self.expr = expr
        self.metrics = _json_safe(metrics)
        self.tags = _json_safe(tags or [])
        self.note = note or ''
        self.source = source
        self.expr_hash = expr_hash or hashlib.md5(expr.encode()).hexdigest()
        self.created_at = datetime.now().isoformat()
        self.id = f"{self.expr_hash}_{int(datetime.now().timestamp())}"

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'expr': self.expr,
            'expr_hash': self.expr_hash,
            'metrics': self.metrics,
            'tags': self.tags,
            'note': self.note,
            'source': self.source,
            'created_at': self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'FactorExperiment':
        exp = cls(
            expr=data['expr'],
            metrics=data['metrics'],
            tags=data.get('tags', []),
            note=data.get('note', ''),
            source=data.get('source', 'manual'),
            expr_hash=data.get('expr_hash'),
        )
        exp.created_at = data['created_at']
        exp.id = data['id']
        return exp


class FactorTracker:
    """
    因子实验追踪器

    设计目标:
    1. 零配置启动，自动持久化
    2. 自动去重，避免重复记录
    3. 丰富的查询筛选能力
    4. 趋势分析与洞察
    """

    def __init__(self, data_dir: str = TRACKER_DIR, auto_save: bool = True):
        self.data_dir = data_dir
        self.auto_save = auto_save
        self.experiments: List[FactorExperiment] = []
        self.expr_hashes: Set[str] = set()
        os.makedirs(self.data_dir, exist_ok=True)

        # 加载历史数据
        self._load()

        print(f"📒 FactorTracker 已加载 {len(self.experiments)} 条历史记录")

    def _get_experiment_file(self) -> str:
        return os.path.join(self.data_dir, 'experiments.jsonl')

    def _load(self):
        """加载历史实验"""
        exp_file = self._get_experiment_file()
        if not os.path.exists(exp_file):
            return

        try:
            with open(exp_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        exp = FactorExperiment.from_dict(data)
                        self.experiments.append(exp)
                        self.expr_hashes.add(exp.expr_hash)
        except Exception as e:
            print(f"⚠️ 加载历史实验失败: {e}")

    def _save(self):
        """保存所有实验"""
        if not self.auto_save:
            return

        exp_file = self._get_experiment_file()
        try:
            with open(exp_file, 'w', encoding='utf-8') as f:
                for exp in self.experiments:
                    f.write(json.dumps(exp.to_dict(), ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"⚠️ 保存实验失败: {e}")

    def log(self, expr: str, metrics: Dict[str, float],
            tags: List[str] = None, note: str = None,
            source: str = 'manual',
            skip_duplicate: bool = True) -> Optional[FactorExperiment]:
        """
        记录一次因子实验

        Args:
            expr: 因子表达式
            metrics: 指标字典
            tags: 标签列表
            note: 备注
            source: 来源 (manual/gp_factory/random_search)
            skip_duplicate: 是否跳过重复因子

        Returns:
            实验记录对象，如果是重复则返回 None
        """
        expr_hash = hashlib.md5(expr.encode()).hexdigest()

        # 重复检查
        if skip_duplicate and expr_hash in self.expr_hashes:
            return None

        exp = FactorExperiment(
            expr=expr,
            metrics=metrics,
            tags=tags or [],
            note=note,
            source=source,
            expr_hash=expr_hash,
        )

        self.experiments.append(exp)
        self.expr_hashes.add(expr_hash)
        self._save()

        return exp

    def log_batch(self, results_df: pd.DataFrame, source: str = 'batch',
                  expr_col: str = 'expr', tags: List[str] = None):
        """
        批量记录实验结果

        Args:
            results_df: 因子结果 DataFrame
            source: 来源标识
            expr_col: 表达式列名
            tags: 统一添加的标签
        """
        n_added = 0
        for _, row in results_df.iterrows():
            expr = row[expr_col]
            metrics = row.drop(expr_col).to_dict()
            # 清洗非数值指标
            metrics = {k: float(v) for k, v in metrics.items()
                      if isinstance(v, (int, float, np.number)) and not pd.isna(v)}

            exp = self.log(expr, metrics, tags=tags, source=source, skip_duplicate=True)
            if exp:
                n_added += 1

        print(f"✅ 批量记录完成，新增 {n_added} 个因子")

    def exists(self, expr: str) -> bool:
        """检查因子是否已存在"""
        expr_hash = hashlib.md5(expr.encode()).hexdigest()
        return expr_hash in self.expr_hashes

    def top_k(self, metric: str = 'val_sr', k: int = 10,
              min_tvr: float = None, max_tvr: float = None) -> pd.DataFrame:
        """
        获取 Top K 因子

        Args:
            metric: 排序指标
            k: 返回数量
            min_tvr: 最小换手率过滤
            max_tvr: 最大换手率过滤

        Returns:
            因子列表 DataFrame
        """
        if not self.experiments:
            return pd.DataFrame()

        df = self.to_df()

        # 换手率过滤
        if min_tvr is not None and 'tvr' in df.columns:
            df = df[df['tvr'] >= min_tvr]
        if max_tvr is not None and 'tvr' in df.columns:
            df = df[df['tvr'] <= max_tvr]

        if metric not in df.columns:
            print(f"⚠️ 指标 {metric} 不存在，可用指标: {list(df.columns)}")
            return pd.DataFrame()

        df = df.sort_values(metric, ascending=False).head(k).reset_index(drop=True)
        return df

    def query(self, tag: str = None, source: str = None,
              expr_contains: str = None,
              metric_filter: Dict[str, tuple] = None) -> pd.DataFrame:
        """
        按条件查询因子

        Args:
            tag: 按标签筛选
            source: 按来源筛选
            expr_contains: 表达式包含字符串
            metric_filter: 指标筛选，如 {'val_sr': (0.5, None), 'tvr': (None, 2.0)}

        Returns:
            筛选后的 DataFrame
        """
        df = self.to_df()

        if tag is not None and 'tags' in df.columns:
            df = df[df['tags'].apply(lambda x: tag in x if isinstance(x, list) else False)]

        if source is not None and 'source' in df.columns:
            df = df[df['source'] == source]

        if expr_contains is not None:
            df = df[df['expr'].str.contains(expr_contains)]

        if metric_filter:
            for metric, (min_val, max_val) in metric_filter.items():
                if metric in df.columns:
                    if min_val is not None:
                        df = df[df[metric] >= min_val]
                    if max_val is not None:
                        df = df[df[metric] <= max_val]

        return df.reset_index(drop=True)

    def to_df(self) -> pd.DataFrame:
        """导出为 DataFrame"""
        if not self.experiments:
            return pd.DataFrame()

        rows = []
        for exp in self.experiments:
            row = {
                'id': exp.id,
                'expr': exp.expr,
                'tags': exp.tags,
                'note': exp.note,
                'source': exp.source,
                'created_at': exp.created_at,
                **exp.metrics
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def stats(self) -> Dict[str, Any]:
        """统计摘要"""
        df = self.to_df()

        if len(df) == 0:
            return {'total': 0}

        stats = {
            'total': len(df),
            'by_source': df['source'].value_counts().to_dict() if 'source' in df.columns else {},
            'date_range': [df['created_at'].min(), df['created_at'].max()] if 'created_at' in df.columns else [],
        }

        for col in ['val_sr', 'train_sr', 'ic_ir', 'tvr']:
            if col in df.columns:
                stats[f'{col}_mean'] = df[col].mean()
                stats[f'{col}_max'] = df[col].max()
                stats[f'{col}_min'] = df[col].min()

        return stats

    def print_summary(self):
        """打印摘要"""
        s = self.stats()

        print("\n" + "=" * 60)
        print("📊 因子实验追踪摘要")
        print("=" * 60)
        print(f"  总记录数: {s.get('total', 0)}")

        if 'by_source' in s:
            print(f"  来源分布:")
            for src, cnt in s['by_source'].items():
                print(f"    - {src}: {cnt}")

        for k, v in s.items():
            if k.endswith('_mean') and isinstance(v, (int, float)):
                metric = k.replace('_mean', '')
                print(f"  {metric}: mean={s.get(k, 0):.3f}, max={s.get(f'{metric}_max', 0):.3f}")

        print("=" * 60 + "\n")

    def export_library(self, metric: str = 'val_sr', threshold: float = 0.5,
                      output_file: str = './factor_library.json'):
        """
        导出因子库

        Args:
            metric: 排序指标
            threshold: 入选阈值
            output_file: 输出文件路径
        """
        df = self.top_k(metric=metric, k=9999)

        if metric in df.columns:
            df = df[df[metric] >= threshold]

        export_cols = [col for col in ['expr', metric, 'tvr', 'ic_ir', 'tags', 'note'] if col in df.columns]
        library = {
            'exported_at': datetime.now().isoformat(),
            'metric': metric,
            'threshold': threshold,
            'count': len(df),
            'factors': df[export_cols].to_dict('records')
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(library, f, indent=2, ensure_ascii=False)

        print(f"📦 已导出 {len(df)} 个因子到 {output_file}")

    def load_library(self, input_file: str) -> pd.DataFrame:
        """加载因子库"""
        with open(input_file, 'r', encoding='utf-8') as f:
            library = json.load(f)

        df = pd.DataFrame(library['factors'])
        print(f"📦 已加载 {len(df)} 个因子 (阈值: {library.get('threshold', 'N/A')})")
        return df

    def tag_stats(self) -> pd.DataFrame:
        """标签统计"""
        df = self.to_df()
        if len(df) == 0 or 'tags' not in df.columns:
            return pd.DataFrame()

        all_tags = []
        for tags in df['tags']:
            if isinstance(tags, list):
                all_tags.extend(tags)

        if not all_tags:
            return pd.DataFrame()

        tag_counts = pd.Series(all_tags).value_counts()
        return pd.DataFrame({'tag': tag_counts.index, 'count': tag_counts.values})

    def trend_analysis(self, days: int = 30, metric: str = 'val_sr') -> Dict:
        """
        实验趋势分析

        Args:
            days: 最近 N 天
            metric: 分析指标

        Returns:
            趋势数据字典
        """
        df = self.to_df()
        if len(df) == 0 or 'created_at' not in df.columns:
            return {}

        df['date'] = pd.to_datetime(df['created_at']).dt.date
        daily = df.groupby('date').agg({
            metric: ['count', 'mean', 'max'],
        }).reset_index()

        daily.columns = ['date', 'count', 'mean', 'max']
        daily = daily.sort_values('date').tail(days)

        return {
            'daily_stats': daily,
            'trend_days': days,
            'improving': daily['mean'].iloc[-1] > daily['mean'].iloc[0] if len(daily) >= 2 else None,
        }


# 全局单例
_tracker_instance = None


def tracker() -> FactorTracker:
    """获取全局追踪器实例"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = FactorTracker()
    return _tracker_instance


# 便捷装饰器，自动记录函数产生的因子
def track_results(source: str = None, tags: List[str] = None):
    """
    装饰器：自动追踪函数返回的因子结果

    使用方式:
        @track_results(source='my_strategy', tags=['custom', 'test'])
        def my_factor_generator():
            return factor_results_df
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, pd.DataFrame) and 'expr' in result.columns:
                tracker().log_batch(result, source=source or func.__name__, tags=tags)
            return result
        return wrapper
    return decorator


if __name__ == '__main__':
    # 演示使用方式
    t = tracker()

    print("\n📊 当前实验统计:")
    t.print_summary()

    # 演示记录
    t.log(
        expr="ts_correlation(close, volume, 10)",
        metrics={'val_sr': 0.85, 'train_sr': 0.92, 'ic_ir': 2.3, 'tvr': 1.2},
        tags=['volume', 'correlation', 'demo'],
        note="演示因子：量价相关性"
    )

    print("\n✅ 演示完成")
