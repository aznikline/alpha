import warnings
warnings.filterwarnings("ignore")

import sys
import os
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'simres'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from simres.expr import AlphaExecutor
import simres.operators as op

# ============== 配置 ==============
DATA_DIR = './data/20251231'
ALPHA_DIR = './alphas/20251231'
RESULT_DIR = f'{ALPHA_DIR}/simres'
MATRIX_DIR = f'{ALPHA_DIR}/matrix'

# 创建输出目录
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MATRIX_DIR, exist_ok=True)

# ============== 初始化 ==============
print("=" * 60)
print("OpenAlpha - 多因子组合策略构建")
print("=" * 60)

executor = AlphaExecutor(data_dir=DATA_DIR)
executor.load_all_data()

# 读取因子列表
with open('src/ruiqiwang_csi_500.txt', 'r') as f:
    alpha_list = [line.strip() for line in f.read().split('\n') if line.strip()]

print(f"\n共加载 {len(alpha_list)} 个因子表达式")

# ============== 阶段 1: 批量计算所有因子 ==============
print("\n" + "=" * 60)
print("阶段 1: 批量计算所有因子")
print("=" * 60)

all_results = []
all_matrices = []

for i, expr in enumerate(alpha_list):
    alpha_id = 5000001 + i
    full_expr = f'at_nan2zero(cs_booksize(cs_rank(at_mask({expr},ts_fill(csi_500_weight)>0))-0.5))'

    try:
        alpha = executor.evaluate(full_expr)
        if alpha is None:
            continue

        # 保存因子矩阵
        df = pd.DataFrame(alpha,
                          index=executor.context['stock_list'],
                          columns=executor.context['datestr'])
        df.to_parquet(f"{MATRIX_DIR}/{alpha_id}")

        # 回测
        bt = executor.backtest(alpha)
        bt['alpha_id'] = str(alpha_id)
        bt['expression'] = expr

        # 保存回测结果
        with open(f"{RESULT_DIR}/{alpha_id}.pkl", "wb") as f:
            pickle.dump(bt, f)

        # 计算指标
        net_daily = bt['net_ret']
        ann_ret = np.nanmean(net_daily) * 252
        ann_vol = np.nanstd(net_daily) * np.sqrt(252)
        sr = ann_ret / ann_vol if ann_vol != 0 else 0
        cumulative = np.nancumsum(net_daily)
        peak = np.maximum.accumulate(cumulative)
        dd = np.max(peak - cumulative)
        tvr = np.nanmean(bt['tvr'])

        all_results.append({
            'alpha_id': alpha_id,
            'expression': expr,
            'ann_ret': ann_ret,
            'ann_vol': ann_vol,
            'sr': sr,
            'dd': dd,
            'tvr': tvr,
        })
        all_matrices.append(alpha)

        print(f"[{alpha_id}] SR={sr:.3f} Ret={ann_ret*100:.1f}% DD={dd*100:.1f}% Tvr={tvr:.2f} | {expr[:50]}...")

    except Exception as e:
        print(f"[{alpha_id}] 错误: {e}")

# ============== 阶段 2: 因子评估与筛选 ==============
print("\n" + "=" * 60)
print("阶段 2: 因子评估与筛选")
print("=" * 60)

results_df = pd.DataFrame(all_results)
print(f"\n成功计算 {len(results_df)} / {len(alpha_list)} 个因子")

# 筛选标准：夏普 > 0.2, 回撤 < 0.3, 换手 < 2.0
filtered = results_df[
    (results_df['sr'] > 0.2) &
    (results_df['dd'] < 0.30) &
    (results_df['tvr'] < 2.0)
].sort_values('sr', ascending=False)

print(f"\n通过筛选的因子: {len(filtered)} 个")
print(filtered[['alpha_id', 'sr', 'ann_ret', 'dd', 'tvr']].to_string(index=False))

# 计算因子相关性矩阵
if len(all_matrices) > 1:
    print("\n计算因子收益相关性...")
    # 使用因子值的截面相关性作为替代
    n = min(len(all_matrices), len(all_results))
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            # 计算每个因子的日度收益的截面均值序列的相关性
            ret_i = np.nanmean(all_matrices[i][:, 2:] * executor.context['ret1'][:, 2:], axis=0)
            ret_j = np.nanmean(all_matrices[j][:, 2:] * executor.context['ret1'][:, 2:], axis=0)
            mask = ~np.isnan(ret_i) & ~np.isnan(ret_j)
            if mask.sum() > 10:
                c = np.corrcoef(ret_i[mask], ret_j[mask])[0, 1]
                corr_matrix[i, j] = corr_matrix[j, i] = c

    # 剔除高相关性因子 (保留夏普最高的)
    selected_ids = []
    selected_idx = []
    for _, row in filtered.iterrows():
        idx = row.name
        too_correlated = False
        for sel_idx in selected_idx:
            if abs(corr_matrix[idx, sel_idx]) > 0.7:
                too_correlated = True
                break
        if not too_correlated:
            selected_ids.append(row['alpha_id'])
            selected_idx.append(idx)

    print(f"\n剔除高相关后保留: {len(selected_ids)} 个因子")
    for aid in selected_ids:
        row = results_df[results_df['alpha_id'] == aid].iloc[0]
        print(f"  {aid}: SR={row['sr']:.3f} | {row['expression'][:60]}")
else:
    selected_ids = filtered['alpha_id'].tolist()[:5]
    print(f"\n因子数量较少，直接选取前 {len(selected_ids)} 个")

# ============== 阶段 3: 构建组合策略 ==============
print("\n" + "=" * 60)
print("阶段 3: 构建组合策略")
print("=" * 60)

# 加载选中因子的矩阵并等权组合
combo_alpha = None
weights = []

for aid in selected_ids:
    matrix_path = f"{MATRIX_DIR}/{aid}"
    if os.path.exists(matrix_path):
        df = pd.read_parquet(matrix_path)
        mat = df.values.astype(np.float32)
        if combo_alpha is None:
            combo_alpha = np.zeros_like(mat)
        combo_alpha += mat
        weights.append(1.0)

if combo_alpha is not None and len(weights) > 0:
    combo_alpha /= len(weights)
    print(f"组合因子形状: {combo_alpha.shape}")
    print(f"等权组合 {len(weights)} 个因子")

    # 再次booksize归一化
    combo_alpha = op.cs_booksize(combo_alpha)

    # 保存组合因子
    combo_df = pd.DataFrame(combo_alpha,
                            index=executor.context['stock_list'],
                            columns=executor.context['datestr'])
    combo_df.to_parquet(f"{MATRIX_DIR}/COMBO")

    # 组合回测
    combo_bt = executor.backtest(combo_alpha)
    combo_bt['alpha_id'] = 'COMBO'

    with open(f"{RESULT_DIR}/COMBO.pkl", "wb") as f:
        pickle.dump(combo_bt, f)

    # 计算组合指标
    net_daily = combo_bt['net_ret']
    ann_ret = np.nanmean(net_daily) * 252
    ann_vol = np.nanstd(net_daily) * np.sqrt(252)
    sr = ann_ret / ann_vol if ann_vol != 0 else 0
    cumulative = np.nancumsum(net_daily)
    peak = np.maximum.accumulate(cumulative)
    dd = np.max(peak - cumulative)
    tvr = np.nanmean(combo_bt['tvr'])

    # 计算Calmar比率
    calmar = ann_ret / dd if dd > 0 else 0

    print("\n组合策略表现:")
    print(f"  年化收益: {ann_ret*100:.2f}%")
    print(f"  年化波动: {ann_vol*100:.2f}%")
    print(f"  夏普比率: {sr:.3f}")
    print(f"  最大回撤: {dd*100:.2f}%")
    print(f"  Calmar比率: {calmar:.3f}")
    print(f"  平均换手: {tvr:.3f}")

    # ============== 阶段 4: 可视化 ==============
    print("\n" + "=" * 60)
    print("阶段 4: 生成回测图表")
    print("=" * 60)

    dates = pd.to_datetime(executor.context['datestr'])
    scale = 10000

    long_pnl = np.nancumsum(combo_bt['long_ret']) * scale
    short_pnl = np.nancumsum(combo_bt['short_ret']) * scale
    net_pnl = np.nancumsum(net_daily) * scale

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_linewidth(2)
    fig.patch.set_edgecolor('black')
    ax.set_facecolor('white')
    ax.grid(True, which='both', color='lightgray', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.plot(dates, long_pnl, color='black', label='Long', linewidth=0.8, alpha=0.8)
    ax.plot(dates, short_pnl, color='green', label='Short', linewidth=0.8, alpha=0.8)
    ax.plot(dates, net_pnl, color='red', label='COMBO', linewidth=1.5, zorder=5)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m%d'))
    plt.xticks(rotation=30, ha='right', fontsize=8)

    header = (f"COMBO | sr:{sr:.3f} ret:{ann_ret:.3f} tvr:{tvr:.3f} "
              f"dd:{dd*100:.1f}% ({dates[0].strftime('%Y%m%d')}-{dates[-1].strftime('%Y%m%d')})")
    plt.title(header, loc='left', fontsize=10, family='monospace', pad=15)

    ax.set_ylabel('Thousand Currencies', fontsize=8)
    ax.legend(loc='upper left', fontsize=8, frameon=True, edgecolor='lightgray')
    ax.set_xlim(dates[0], dates[-1])

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
    plt.savefig('./combo_backtest.png', dpi=150, bbox_inches='tight')
    print("图表已保存: ./combo_backtest.png")
    plt.close()

    # 保存策略汇总报告
    report = {
        'selected_factors': selected_ids,
        'factor_count': len(selected_ids),
        'performance': {
            'ann_ret': ann_ret,
            'ann_vol': ann_vol,
            'sharpe': sr,
            'max_dd': dd,
            'calmar': calmar,
            'avg_tvr': tvr,
        },
        'all_factor_results': all_results,
    }

    with open('./strategy_report.pkl', 'wb') as f:
        pickle.dump(report, f)

    print("\n策略报告已保存: ./strategy_report.pkl")

print("\n" + "=" * 60)
print("策略构建完成!")
print("=" * 60)
