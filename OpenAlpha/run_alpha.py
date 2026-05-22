import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'simres'))

import numpy as np
import pandas as pd
from simres.expr import AlphaExecutor
import simres.operators as op

# 初始化执行器
data_dir = './data/20251231'
executor = AlphaExecutor(data_dir=data_dir)
executor.load_all_data()

# 读取因子表达式列表
with open('src/ruiqiwang_csi_500.txt', 'r') as f:
    alpha_list = [line.strip() for line in f.read().split('\n') if line.strip()]

print(f"共加载 {len(alpha_list)} 个因子表达式")
print("=" * 60)

# 选择前5个因子进行测试
test_alphas = alpha_list[:5]

results = []
for i, expr in enumerate(test_alphas):
    alpha_id = 5000001 + i
    print(f"\n[{alpha_id}] 表达式: {expr}")

    try:
        # 计算因子值（带CSI 500掩码和booksize归一化）
        full_expr = f'at_nan2zero(cs_booksize(cs_rank(at_mask({expr},ts_fill(csi_500_weight)>0))-0.5))'
        alpha = executor.evaluate(full_expr)

        if alpha is None:
            print("  计算失败")
            continue

        print(f"  因子矩阵形状: {alpha.shape}")
        print(f"  因子均值: {np.nanmean(alpha):.6f}")
        print(f"  因子标准差: {np.nanstd(alpha):.6f}")

        # 回测
        btresult = executor.backtest(alpha)

        # 计算回测指标
        net_daily = btresult['net_ret']
        ann_ret = np.nanmean(net_daily) * 252
        ann_vol = np.nanstd(net_daily) * np.sqrt(252)
        sr = ann_ret / ann_vol if ann_vol != 0 else 0

        # 计算最大回撤
        cumulative = np.nancumsum(net_daily)
        peak = np.maximum.accumulate(cumulative)
        dd = np.max(peak - cumulative)

        tvr_avg = np.nanmean(btresult['tvr'])

        print(f"  年化收益: {ann_ret*100:.2f}%")
        print(f"  年化波动: {ann_vol*100:.2f}%")
        print(f"  夏普比率: {sr:.3f}")
        print(f"  最大回撤: {dd*100:.2f}%")
        print(f"  平均换手率: {tvr_avg:.3f}")

        results.append({
            'alpha_id': alpha_id,
            'expression': expr,
            'ann_ret': ann_ret,
            'ann_vol': ann_vol,
            'sr': sr,
            'dd': dd,
            'tvr': tvr_avg,
            'shape': alpha.shape
        })

    except Exception as e:
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()

# 汇总结果
print("\n" + "=" * 60)
print("回测结果汇总")
print("=" * 60)

if results:
    summary_df = pd.DataFrame(results)
    print(summary_df[['alpha_id', 'ann_ret', 'ann_vol', 'sr', 'dd', 'tvr']].to_string(index=False))
