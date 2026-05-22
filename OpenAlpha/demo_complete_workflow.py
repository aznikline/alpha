#!/usr/bin/env python3
"""
OpenAlpha 完整因子研究工作流演示
==================================

展示从因子生成 -> 并行回测 -> 实验追踪 -> 结果分析的完整流程

使用方式:
    python3 demo_complete_workflow.py
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.factor_lab import FactorLab
from src.fast_eval import FastEval
from src.factor_tracker import tracker as get_tracker


def build_candidate_factors(lab: FactorLab) -> list[str]:
    factors = []

    for w in [5, 10, 20, 40, 60]:
        factors.append(lab.get_template('momentum_price', w=w))
        factors.append(lab.get_template('momentum_ret', w=w))

    for w in [5, 10, 20, 40]:
        factors.append(lab.get_template('volatility_std', w=w))
        factors.append(lab.get_template('volatility_range', w=w))

    for w in [10, 20, 40]:
        factors.append(lab.get_template('volume_price_corr', w=w))
        factors.append(lab.get_template('volume_breakout', w=w))

    for w in [20, 40, 60]:
        factors.append(lab.get_template('reg_beta_market', w=w))
        factors.append(lab.get_template('reg_alpha_volume', w=w))

    return factors


def infer_tags(expr: str) -> list[str]:
    tags = ['demo_batch']
    if 'ts_mean' in expr or 'ret1' in expr:
        tags.append('momentum')
    if 'std' in expr or 'high-low' in expr:
        tags.append('volatility')
    if 'volume' in expr or 'amount' in expr:
        tags.append('volume')
    if 'regression' in expr:
        tags.append('regression')
    return tags


def main() -> None:
    print("=" * 70)
    print("🚀 OpenAlpha 完整因子研究工作流演示")
    print("=" * 70)

    print("\n📦 步骤 1: 初始化核心模块")
    print("-" * 70)
    lab = FactorLab()
    engine = FastEval(n_workers=4, use_cache=True, enable_ic=True)
    experiment_tracker = get_tracker()

    print(f"   ✓ FactorLab 已加载 {len(lab.templates)} 个因子模板")
    print(f"   ✓ FastEval 已启动 ({engine.n_workers} 进程, 缓存={'已启用' if engine.use_cache else '已禁用'})")
    print(f"   ✓ FactorTracker 已加载 {len(experiment_tracker.experiments)} 条历史记录")

    print("\n🔬 步骤 2: 快速测试单个因子")
    print("-" * 70)
    result = lab.quick_test('momentum_price', w=20)
    lab.report(result)

    print("\n🧬 步骤 3: 批量生成候选因子")
    print("-" * 70)
    test_factors = build_candidate_factors(lab)
    print(f"   生成了 {len(test_factors)} 个候选因子")

    print("\n⚡ 步骤 4: 并行回测评估")
    print("-" * 70)
    results_df = engine.evaluate(test_factors, show_progress=True)

    print(f"\n   ✓ 成功评估 {len(results_df)} 个因子")
    if len(results_df) > 0:
        print(f"   最佳 val_sr: {results_df['val_sr'].max():.3f}")
        print(f"   最佳 ic_ir: {results_df.get('ic_ir', 0).max():.3f}")

    print("\n📝 步骤 5: 记录到实验追踪器")
    print("-" * 70)
    for _, row in results_df.iterrows():
        expr = row['expr']
        metrics = {
            'val_sr': row['val_sr'],
            'train_sr': row['train_sr'],
            'ic_ir': row.get('ic_ir', 0),
            'tvr': row['tvr'],
            'ann_ret': row.get('val_ret', 0),
            'ann_vol': row.get('val_vol', 0),
            'max_dd': row.get('max_dd', 0),
        }
        experiment_tracker.log(
            expr=expr,
            metrics=metrics,
            tags=infer_tags(expr),
            source='demo_workflow',
            skip_duplicate=True,
        )
    print("   ✓ 已记录到 FactorTracker")

    print("\n🏆 步骤 6: 查询与分析 Top 因子")
    print("-" * 70)
    top_10 = experiment_tracker.top_k(metric='val_sr', k=10)
    print("\nTop 10 因子 (按 val_sr):")
    if len(top_10) > 0:
        print(top_10[['val_sr', 'ic_ir', 'tvr', 'expr']].to_string(index=False))

    print("\n📊 动量类因子:")
    momentum_factors = experiment_tracker.query(tag='momentum')
    if len(momentum_factors) > 0:
        print(momentum_factors[['val_sr', 'ic_ir', 'tvr', 'expr']].head().to_string(index=False))

    print("\n📈 步骤 7: 生成详细分析报告")
    print("-" * 70)
    if len(top_10) > 0:
        best_expr = top_10.iloc[0]['expr']
        best_sr = top_10.iloc[0]['val_sr']
        print(f"\n   为最佳因子生成完整报告: val_sr = {best_sr:.3f}")
        print(f"   因子表达式: {best_expr[:80]}...")
        lab.report(lab.test_alpha(best_expr))

    print("\n📦 步骤 8: 导出因子库")
    print("-" * 70)
    experiment_tracker.export_library(
        metric='val_sr',
        threshold=0.5,
        output_file='./demo_factor_library.json',
    )

    print("\n" + "=" * 70)
    print("📊 实验统计摘要")
    print("=" * 70)
    experiment_tracker.print_summary()

    print("\n" + "=" * 70)
    print("✅ 工作流演示完成！")
    print("=" * 70)
    print("\n📌 生成的文件:")
    print("   - ./.factor_cache/           因子计算缓存")
    print("   - ./.factor_experiments/     实验记录数据库")
    print("   - ./demo_factor_library.json 导出的因子库")
    print("\n🎯 下一步操作建议:")
    print("   1. 编辑 src/factor_lab.py 添加自定义因子模板")
    print("   2. 运行 gp_enhanced.py 进行遗传编程大规模挖掘")
    print("   3. 使用 factor_combination.py 构建因子组合")
    print("   4. 在 interactive_research.ipynb 中进行交互式研究")


if __name__ == '__main__':
    main()
