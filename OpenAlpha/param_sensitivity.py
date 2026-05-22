#!/usr/bin/env python3
"""
OpenAlpha 参数敏感性分析
========================

复用 FactorLab / FastEval / FactorTracker，对模板参数进行系统化扫描，输出最优参数与稳定性评分。
"""
import warnings
warnings.filterwarnings("ignore")

from itertools import product
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from src.factor_lab import FactorLab
from src.fast_eval import FastEval
from src.factor_tracker import tracker as get_tracker


SWEEP_CONFIGS = [
    {
        'name': 'momentum_price',
        'params': {'w': [2, 3, 5, 10, 20, 40, 60]},
    },
    {
        'name': 'volume_price_corr',
        'params': {'w': [2, 3, 5, 10, 20, 40, 60]},
    },
    {
        'name': 'volatility_std',
        'params': {'w': [2, 3, 5, 10, 20, 40, 60]},
    },
    {
        'name': 'volatility_skew',
        'params': {'w': [3, 5, 10, 20, 40, 60]},
    },
    {
        'name': 'reg_beta_market',
        'params': {'w': [5, 10, 20, 40, 60]},
    },
    {
        'name': 'dual_mean',
        'params': {'w1': [2, 5, 10], 'w2': [20, 40, 60]},
    },
]


def expand_grid(params: Dict[str, List[int]]) -> List[Dict[str, int]]:
    keys = list(params.keys())
    return [dict(zip(keys, values)) for values in product(*(params[key] for key in keys))]


def build_sweep_factors(lab: FactorLab) -> pd.DataFrame:
    rows = []
    for config in SWEEP_CONFIGS:
        for params in expand_grid(config['params']):
            if 'w1' in params and 'w2' in params and params['w1'] >= params['w2']:
                continue
            expr = lab.get_template(config['name'], **params)
            rows.append({
                'template': config['name'],
                'params': params,
                'expr': expr,
            })
    return pd.DataFrame(rows)


def score_stability(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values('score', ascending=False).copy()
    best_score = group['score'].max()
    group['rank_in_template'] = range(1, len(group) + 1)
    group['score_gap_to_best'] = best_score - group['score']
    group['stable_zone'] = group['score_gap_to_best'] <= max(0.1, abs(best_score) * 0.2)
    return group


def plot_sensitivity(results: pd.DataFrame, output_file: str) -> None:
    templates = results['template'].drop_duplicates().tolist()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for idx, template in enumerate(templates[:6]):
        ax = axes[idx]
        subset = results[results['template'] == template].copy()
        if subset.empty:
            ax.set_visible(False)
            continue

        if 'w' in subset.columns:
            subset = subset.sort_values('w')
            x = subset['w']
            xlabel = 'w'
        else:
            subset = subset.sort_values(['w1', 'w2'])
            x = [f"{row.w1}/{row.w2}" for row in subset.itertuples()]
            xlabel = 'w1/w2'

        ax.plot(x, subset['val_sr'], marker='o', label='Val SR', color='#2E86AB')
        ax.plot(x, subset['train_sr'], marker='s', label='Train SR', color='#A23B72', alpha=0.75)
        ax2 = ax.twinx()
        ax2.plot(x, subset['tvr'], marker='^', label='TVR', color='#F18F01', alpha=0.7)

        ax.set_title(template)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Sharpe')
        ax2.set_ylabel('TVR')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=35)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize=8, loc='best')

    for idx in range(len(templates), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('OpenAlpha Parameter Sensitivity', fontsize=15)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def main() -> None:
    print("=" * 70)
    print("OpenAlpha - 参数敏感性分析")
    print("=" * 70)

    lab = FactorLab()
    engine = FastEval(n_workers=4, use_cache=True, enable_ic=True)
    experiment_tracker = get_tracker()

    sweep_df = build_sweep_factors(lab)
    print(f"\n生成 {len(sweep_df)} 个参数组合")

    eval_df = engine.evaluate(sweep_df['expr'].tolist(), show_progress=True)
    if eval_df.empty:
        print("没有可用评估结果")
        return

    results = sweep_df.merge(eval_df, on='expr', how='inner')
    params_df = pd.json_normalize(results['params'])
    results = pd.concat([results.drop(columns=['params']), params_df], axis=1)
    results = results.groupby('template', group_keys=False).apply(score_stability)
    results = results.sort_values(['template', 'rank_in_template']).reset_index(drop=True)

    for _, row in results.iterrows():
        experiment_tracker.log(
            expr=row['expr'],
            metrics={
                'val_sr': row['val_sr'],
                'train_sr': row['train_sr'],
                'ic_ir': row.get('ic_ir', 0),
                'tvr': row['tvr'],
                'score': row['score'],
                'param_rank': row['rank_in_template'],
            },
            tags=['param_sensitivity', row['template']],
            source='param_sensitivity',
            skip_duplicate=True,
        )

    output_csv = './param_sensitivity.csv'
    output_png = './param_sensitivity.png'
    results.to_csv(output_csv, index=False)
    plot_sensitivity(results, output_png)

    print("\n" + "=" * 70)
    print("最优参数汇总")
    print("=" * 70)
    summary_cols = ['template', 'rank_in_template', 'val_sr', 'train_sr', 'ic_ir', 'tvr', 'score', 'stable_zone', 'expr']
    best = results[results['rank_in_template'] == 1][summary_cols]
    print(best.to_string(index=False))

    stable = results[results['stable_zone']]
    print(f"\n稳定参数区间候选: {len(stable)} 个")
    print(stable[['template', 'val_sr', 'score', 'stable_zone', 'expr']].head(20).to_string(index=False))

    print(f"\n详细结果已保存: {output_csv}")
    print(f"参数敏感性图已保存: {output_png}")
    print("=" * 70)
    print("参数敏感性分析完成")


if __name__ == '__main__':
    main()
