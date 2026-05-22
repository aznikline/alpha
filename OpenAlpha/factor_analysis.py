import warnings
warnings.filterwarnings("ignore")

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'simres'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simres.expr import AlphaExecutor
import simres.operators as op

# ============== 配置 ==============
DATA_DIR = './data/20251231'
ALPHA_DIR = './alphas/20251231'
MATRIX_DIR = f'{ALPHA_DIR}/matrix'

executor = AlphaExecutor(data_dir=DATA_DIR)
executor.load_all_data()

# 读取因子列表
with open('src/ruiqiwang_csi_500.txt', 'r') as f:
    alpha_list = [line.strip() for line in f.read().split('\n') if line.strip()]

print("=" * 70)
print("OpenAlpha - 因子 IC 分析与衰减分析")
print("=" * 70)

# 获取未来收益率 (t+1 到 t+N)
ret1 = executor.context['ret1']  # (Stock, Date)
dates = executor.context['datestr']
n_dates = len(dates)

# 计算未来多期收益
future_rets = {}
for horizon in [1, 2, 3, 5, 10, 20]:
    if horizon == 1:
        future_rets[horizon] = ret1
    else:
        # 累积收益
        cum = np.ones_like(ret1)
        for h in range(1, horizon + 1):
            cum *= (1 + op.ts_delay(ret1, -h))
        future_rets[horizon] = cum - 1


def compute_ic(alpha, future_ret):
    """计算每日 IC (截面秩相关系数)"""
    n_stocks, n_dates = alpha.shape
    ics = []
    for t in range(n_dates):
        a = alpha[:, t]
        r = future_ret[:, t]
        mask = ~np.isnan(a) & ~np.isnan(r)
        if mask.sum() < 10:
            ics.append(np.nan)
            continue
        # Spearman rank correlation
        from scipy import stats
        ic, _ = stats.spearmanr(a[mask], r[mask])
        ics.append(ic)
    return np.array(ics)


def compute_pearson_ic(alpha, future_ret):
    """计算每日 Pearson IC"""
    n_stocks, n_dates = alpha.shape
    ics = []
    for t in range(n_dates):
        a = alpha[:, t]
        r = future_ret[:, t]
        mask = ~np.isnan(a) & ~np.isnan(r)
        if mask.sum() < 10:
            ics.append(np.nan)
            continue
        ic = np.corrcoef(a[mask], r[mask])[0, 1]
        ics.append(ic)
    return np.array(ics)


# ============== 阶段 1: 批量计算 IC ==============
print("\n" + "=" * 70)
print("阶段 1: 批量计算因子 IC")
print("=" * 70)

all_ic_results = []

for i, expr in enumerate(alpha_list):
    alpha_id = 5000001 + i
    full_expr = f'at_nan2zero(cs_booksize(cs_rank(at_mask({expr},ts_fill(csi_500_weight)>0))-0.5))'

    try:
        alpha = executor.evaluate(full_expr)
        if alpha is None:
            continue

        # 计算不同预测周期的 IC
        for horizon, future_ret in future_rets.items():
            ics = compute_ic(alpha, future_ret)
            valid_ics = ics[~np.isnan(ics)]
            if len(valid_ics) == 0:
                continue

            ic_mean = np.mean(valid_ics)
            ic_std = np.std(valid_ics)
            icir = ic_mean / ic_std if ic_std > 0 else 0
            ic_pos_ratio = np.mean(valid_ics > 0)

            all_ic_results.append({
                'alpha_id': alpha_id,
                'expression': expr,
                'horizon': horizon,
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'icir': icir,
                'ic_pos_ratio': ic_pos_ratio,
                'ic_series': ics,
            })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  已处理 {i+1}/{len(alpha_list)} 个因子")

    except Exception as e:
        print(f"  [{alpha_id}] 错误: {e}")

ic_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'ic_series'}
                       for r in all_ic_results])

# ============== 阶段 2: IC 汇总报告 ==============
print("\n" + "=" * 70)
print("阶段 2: IC 汇总报告 (预测周期=1天)")
print("=" * 70)

ic_1d = ic_df[ic_df['horizon'] == 1].sort_values('icir', ascending=False)
print(f"\n共 {len(ic_1d)} 个因子")
print(ic_1d[['alpha_id', 'ic_mean', 'ic_std', 'icir', 'ic_pos_ratio']].to_string(index=False))

# 保存完整IC结果
ic_df.to_csv('./ic_analysis.csv', index=False)
print("\n完整IC分析已保存: ./ic_analysis.csv")

# ============== 阶段 3: 因子衰减分析 ==============
print("\n" + "=" * 70)
print("阶段 3: 因子衰减分析")
print("=" * 70)

# 选取 ICIR 最高的 5 个因子做衰减曲线
top5 = ic_1d.head(5)['alpha_id'].tolist()

decay_results = []
for aid in top5:
    rows = ic_df[ic_df['alpha_id'] == aid].sort_values('horizon')
    print(f"\n  Alpha {aid}:")
    for _, row in rows.iterrows():
        print(f"    周期={row['horizon']:2d}天 | IC={row['ic_mean']:+.4f} | ICIR={row['icir']:+.3f} | 正IC占比={row['ic_pos_ratio']:.1%}")
        decay_results.append({
            'alpha_id': aid,
            'horizon': row['horizon'],
            'ic_mean': row['ic_mean'],
            'icir': row['icir'],
        })

# ============== 阶段 4: 可视化 ==============
print("\n" + "=" * 70)
print("阶段 4: 生成可视化图表")
print("=" * 70)

# 图1: IC分布直方图 (top因子)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, aid in enumerate(top5[:5]):
    result = [r for r in all_ic_results if r['alpha_id'] == aid and r['horizon'] == 1][0]
    ics = result['ic_series']
    valid_ics = ics[~np.isnan(ics)]

    axes[idx].hist(valid_ics, bins=30, edgecolor='black', alpha=0.7)
    axes[idx].axvline(np.mean(valid_ics), color='red', linestyle='--', linewidth=2, label=f"Mean={np.mean(valid_ics):.3f}")
    axes[idx].set_title(f"Alpha {aid}\nICIR={result['icir']:.3f}", fontsize=10)
    axes[idx].set_xlabel("IC")
    axes[idx].set_ylabel("Freq")
    axes[idx].legend(fontsize=8)

axes[5].axis('off')
plt.suptitle("IC Distribution (Top 5 Factors)", fontsize=14)
plt.tight_layout()
plt.savefig('./ic_distribution.png', dpi=150, bbox_inches='tight')
print("IC分布图已保存: ./ic_distribution.png")
plt.close()

# 图2: 因子衰减曲线
fig, ax = plt.subplots(figsize=(10, 6))
decay_df = pd.DataFrame(decay_results)
for aid in top5:
    subset = decay_df[decay_df['alpha_id'] == aid]
    ax.plot(subset['horizon'], subset['ic_mean'], marker='o', label=f"Alpha {aid}")

ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel("Prediction Horizon (Days)")
ax.set_ylabel("Mean IC")
ax.set_title("Factor Decay Curve (Mean IC vs Horizon)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./factor_decay.png', dpi=150, bbox_inches='tight')
print("因子衰减曲线已保存: ./factor_decay.png")
plt.close()

# 图3: ICIR热力图 (因子 × 预测周期)
pivot_icir = ic_df.pivot_table(values='icir', index='alpha_id', columns='horizon')
fig, ax = plt.subplots(figsize=(10, 12))
im = ax.imshow(pivot_icir.values, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
ax.set_xticks(range(len(pivot_icir.columns)))
ax.set_xticklabels(pivot_icir.columns)
ax.set_yticks(range(len(pivot_icir.index)))
ax.set_yticklabels(pivot_icir.index)
ax.set_xlabel("Prediction Horizon (Days)")
ax.set_ylabel("Alpha ID")
ax.set_title("ICIR Heatmap")
plt.colorbar(im, ax=ax, label='ICIR')
# 标注数值
for i in range(len(pivot_icir.index)):
    for j in range(len(pivot_icir.columns)):
        val = pivot_icir.iloc[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                   color="white" if abs(val) > 0.5 else "black")
plt.tight_layout()
plt.savefig('./icir_heatmap.png', dpi=150, bbox_inches='tight')
print("ICIR热力图已保存: ./icir_heatmap.png")
plt.close()

# ============== 阶段 5: 优质因子筛选建议 ==============
print("\n" + "=" * 70)
print("阶段 5: 优质因子筛选建议")
print("=" * 70)

# 筛选标准: |IC| > 0.02, ICIR > 0.3, 正IC占比 > 55%
good_factors = ic_1d[
    (abs(ic_1d['ic_mean']) > 0.02) &
    (abs(ic_1d['icir']) > 0.3) &
    (ic_1d['ic_pos_ratio'] > 0.55)
].sort_values('icir', ascending=False)

print(f"\n满足优质标准的因子: {len(good_factors)} 个")
print(good_factors[['alpha_id', 'ic_mean', 'icir', 'ic_pos_ratio']].to_string(index=False))

# 保存报告
with open('./factor_ic_report.txt', 'w') as f:
    f.write("OpenAlpha Factor IC Analysis Report\n")
    f.write("=" * 70 + "\n\n")
    f.write("Top 10 Factors by ICIR (1-day horizon):\n")
    f.write(ic_1d.head(10)[['alpha_id', 'ic_mean', 'icir', 'ic_pos_ratio']].to_string(index=False))
    f.write("\n\n")
    f.write("Good Quality Factors (|IC|>0.02, |ICIR|>0.3, PosRatio>55%):\n")
    f.write(good_factors[['alpha_id', 'ic_mean', 'icir', 'ic_pos_ratio']].to_string(index=False))
    f.write("\n")

print("\n报告已保存: ./factor_ic_report.txt")
print("\n" + "=" * 70)
print("IC 分析完成!")
print("=" * 70)
