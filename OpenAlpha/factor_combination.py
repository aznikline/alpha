import warnings
warnings.filterwarnings("ignore")

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'simres'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from simres.expr import AlphaExecutor
import simres.operators as op

# =============================================================================
# 因子组合优化器
# =============================================================================

print("=" * 80)
print("OpenAlpha - 因子组合优化")
print("=" * 80)

executor = AlphaExecutor(data_dir='./data/20251231')
executor.load_all_data()
ret1 = executor.context['ret1']
datestr = executor.context['datestr']
n_dates = len(datestr)

# 训练/验证分割
train_cut = int(n_dates * 0.7)
train_slice = slice(None, train_cut)
val_slice = slice(train_cut, None)

# =============================================================================
# 一、Top因子列表（来自因子工厂报告）
# =============================================================================

TOP_FACTORS = [
    # (名称, 表达式, 分类)
    ("value_ratio_ret1_volume", "cs_rank(ret1/volume)", "value"),
    ("value_ratio_amount_ret1", "cs_rank(amount/ret1)", "value"),
    ("reg_residual_window40_high_ret1", "ts_regression(high,ret1,40,0)", "quality"),
    ("reg_residual_window10_close_ret1", "ts_regression(close,ret1,10,0)", "quality"),
    ("ols_residual_window2_volume_close", "ts_ols(volume,close,2)[2]", "quality"),
    ("reg_residual_window3_vwap_high", "ts_regression(vwap,high,3,0)", "quality"),
    ("reversal_delay_shift1_amount", "(amount-ts_delay(amount,1))", "reversal"),
    ("volume_price_window2_amount_vwap", "ts_correlation(amount,vwap,2)", "volume"),
    ("reg_residual_window5_amount_vwap", "ts_regression(amount,vwap,5,0)", "quality"),
    ("ols_residual_window2_high_open", "ts_ols(high,open,2)[2]", "quality"),
    ("volume_price_window2_high_volume", "ts_correlation(high,volume,2)", "volume"),
    ("value_ratio_volume_vwap", "cs_rank(volume/vwap)", "value"),
    ("reg_residual_window10_volume_low", "ts_regression(volume,low,10,0)", "quality"),
    ("reg_residual_window3_high_close", "ts_regression(high,close,3,0)", "quality"),
    ("reg_residual_window3_high_low", "ts_regression(high,low,3,0)", "quality"),
    ("reg_residual_window10_ret1_amount", "ts_regression(ret1,amount,10,0)", "quality"),
    ("reg_residual_window40_amount_high", "ts_regression(amount,high,40,0)", "quality"),
    ("reg_residual_window5_high_volume", "ts_regression(high,volume,5,0)", "quality"),
    ("reg_residual_window40_amount_vwap", "ts_regression(amount,vwap,40,0)", "quality"),
    ("reg_residual_window2_low_low", "ts_regression(low,low,2,0)", "quality"),
]

N_FACTORS = len(TOP_FACTORS)
print(f"\n选取 Top {N_FACTORS} 个因子进行组合优化")

# =============================================================================
# 二、计算所有因子的日度因子值矩阵
# =============================================================================

print("\n" + "=" * 80)
print("阶段 1: 计算因子值矩阵")
print("=" * 80)

factor_matrices = []  # 存储每个因子的 (股票 x 日期) 矩阵
factor_names = []
factor_returns = []  # 存储每个因子的日度收益率 (日期,)

for idx, (name, expr, category) in enumerate(TOP_FACTORS):
    full_expr = f'at_nan2zero(cs_booksize(cs_rank(at_mask({expr},ts_fill(csi_500_weight)>0))-0.5))'
    try:
        alpha = executor.evaluate(full_expr)
        if alpha is not None and not np.all(np.isnan(alpha)):
            factor_matrices.append(alpha)
            factor_names.append(name)

            # 计算因子日度收益率
            bt = executor.backtest(alpha)
            factor_returns.append(bt['net_ret'])

            if (idx + 1) % 5 == 0:
                print(f"  已计算 {idx+1}/{N_FACTORS} 个因子")
    except Exception as e:
        print(f"  因子 {name} 计算失败: {e}")

print(f"\n成功计算 {len(factor_matrices)} 个因子")

# 转换为 numpy 数组 (n_factors, n_dates)
factor_returns_np = np.array(factor_returns)
n_valid_factors = len(factor_returns_np)

# =============================================================================
# 三、因子相关性分析
# =============================================================================

print("\n" + "=" * 80)
print("阶段 2: 因子相关性分析")
print("=" * 80)

# 计算相关系数矩阵（基于日度收益率）
corr_matrix = np.corrcoef(factor_returns_np)

# 转换为 DataFrame 便于查看
corr_df = pd.DataFrame(
    corr_matrix,
    index=[f"F{i+1}" for i in range(n_valid_factors)],
    columns=[f"F{i+1}" for i in range(n_valid_factors)]
)

print("\n因子相关系数矩阵 (前10个因子):")
print(corr_df.iloc[:10, :10].round(3).to_string())

# 找出高相关性因子对 (|corr| > 0.7)
high_corr_pairs = []
for i in range(n_valid_factors):
    for j in range(i + 1, n_valid_factors):
        if abs(corr_matrix[i, j]) > 0.7:
            high_corr_pairs.append({
                'factor1': factor_names[i],
                'factor2': factor_names[j],
                'correlation': corr_matrix[i, j]
            })

if high_corr_pairs:
    print(f"\n发现 {len(high_corr_pairs)} 对高相关性因子 (|corr| > 0.7):")
    for pair in high_corr_pairs:
        print(f"  {pair['factor1'][:25]:25s} <-> {pair['factor2'][:25]:25s} : {pair['correlation']:.3f}")
else:
    print("\n未发现高相关性因子对 (|corr| > 0.7)")

# =============================================================================
# 四、均值方差优化 (Mean-Variance Optimization)
# =============================================================================

print("\n" + "=" * 80)
print("阶段 3: 均值方差优化")
print("=" * 80)

# 使用训练集计算预期收益率和协方差矩阵
train_factor_returns = factor_returns_np[:, train_slice]
expected_returns = np.mean(train_factor_returns, axis=1) * 252  # 年化
cov_matrix = np.cov(train_factor_returns) * 252  # 年化协方差

def portfolio_variance(weights):
    """计算组合方差"""
    return weights.T @ cov_matrix @ weights

def portfolio_return(weights):
    """计算组合预期收益率"""
    return weights.T @ expected_returns

def mvo_objective(weights):
    """MVO 目标函数：最大化夏普 = 收益/风险"""
    return -portfolio_return(weights) / (np.sqrt(portfolio_variance(weights)) + 1e-12)

# 约束条件
constraints = [
    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # 权重和为1
]

# 边界：每个因子权重在 [0, 0.3] 之间（避免单因子权重过大）
bounds = tuple((0, 0.3) for _ in range(n_valid_factors))

# 初始权重：等权
initial_weights = np.ones(n_valid_factors) / n_valid_factors

# 优化
mvo_result = minimize(
    mvo_objective,
    initial_weights,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 1000}
)

mvo_weights = mvo_result.x

print("\nMVO 优化结果:")
print(f"  优化成功: {mvo_result.success}")
print(f"  预期年化收益率: {portfolio_return(mvo_weights):.4f}")
print(f"  预期年化波动率: {np.sqrt(portfolio_variance(mvo_weights)):.4f}")
print(f"  预期夏普比率: {portfolio_return(mvo_weights) / np.sqrt(portfolio_variance(mvo_weights)):.4f}")

print("\nTop 10 因子权重 (MVO):")
mvo_weight_df = pd.DataFrame({
    'Factor': factor_names,
    'Weight': mvo_weights,
    'Category': [cat for _, _, cat in TOP_FACTORS]
}).sort_values('Weight', ascending=False)

for _, row in mvo_weight_df.head(10).iterrows():
    print(f"  {row['Factor'][:30]:30s} : {row['Weight']:>6.1%} [{row['Category']}]")

# =============================================================================
# 五、风险平价优化 (Risk Parity)
# =============================================================================

print("\n" + "=" * 80)
print("阶段 4: 风险平价优化")
print("=" * 80)

def risk_contribution(weights, cov_matrix):
    """计算每个因子的风险贡献"""
    portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights + 1e-12)
    marginal_risk = cov_matrix @ weights
    component_risk = weights * marginal_risk / portfolio_vol
    return component_risk

def risk_parity_objective(weights):
    """风险平价目标函数：最小化风险贡献差异"""
    rc = risk_contribution(weights, cov_matrix)
    return np.sum((rc - np.mean(rc)) ** 2)

# 优化
rp_result = minimize(
    risk_parity_objective,
    initial_weights,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 1000}
)

rp_weights = rp_result.x

print("\n风险平价优化结果:")
print(f"  优化成功: {rp_result.success}")
print(f"  预期年化收益率: {portfolio_return(rp_weights):.4f}")
print(f"  预期年化波动率: {np.sqrt(portfolio_variance(rp_weights)):.4f}")
print(f"  预期夏普比率: {portfolio_return(rp_weights) / np.sqrt(portfolio_variance(rp_weights)):.4f}")

print("\nTop 10 因子权重 (Risk Parity):")
rp_weight_df = pd.DataFrame({
    'Factor': factor_names,
    'Weight': rp_weights,
    'Category': [cat for _, _, cat in TOP_FACTORS]
}).sort_values('Weight', ascending=False)

for _, row in rp_weight_df.head(10).iterrows():
    print(f"  {row['Factor'][:30]:30s} : {row['Weight']:>6.1%} [{row['Category']}]")

# 验证风险贡献是否均等
print("\n风险贡献验证 (前10个因子):")
rc = risk_contribution(rp_weights, cov_matrix)
rc_df = pd.DataFrame({
    'Factor': factor_names,
    'RiskContribution': rc
}).sort_values('RiskContribution', ascending=False)
for _, row in rc_df.head(10).iterrows():
    print(f"  {row['Factor'][:30]:30s} : {row['RiskContribution']:>6.1%}")

# =============================================================================
# 六、等权组合 (Equal Weight)
# =============================================================================

print("\n" + "=" * 80)
print("阶段 5: 等权组合")
print("=" * 80)

equal_weights = np.ones(n_valid_factors) / n_valid_factors

print(f"  预期年化收益率: {portfolio_return(equal_weights):.4f}")
print(f"  预期年化波动率: {np.sqrt(portfolio_variance(equal_weights)):.4f}")
print(f"  预期夏普比率: {portfolio_return(equal_weights) / np.sqrt(portfolio_variance(equal_weights)):.4f}")

# =============================================================================
# 七、验证集回测
# =============================================================================

print("\n" + "=" * 80)
print("阶段 6: 验证集回测")
print("=" * 80)

val_factor_returns = factor_returns_np[:, val_slice]

# 计算各组合的验证集表现
mvo_val_returns = mvo_weights @ val_factor_returns
rp_val_returns = rp_weights @ val_factor_returns
equal_val_returns = equal_weights @ val_factor_returns

def calculate_metrics(daily_returns, name):
    """计算组合绩效指标"""
    mean_ret = np.mean(daily_returns) * 252
    vol = np.std(daily_returns) * np.sqrt(252)
    sharpe = mean_ret / (vol + 1e-12)
    cum_ret = np.cumprod(1 + daily_returns)
    max_dd = np.max(1 - cum_ret / np.maximum.accumulate(cum_ret))

    return {
        'Name': name,
        'AnnualReturn': mean_ret,
        'Volatility': vol,
        'Sharpe': sharpe,
        'MaxDrawdown': max_dd,
        'TotalReturn': cum_ret[-1] - 1
    }

mvo_metrics = calculate_metrics(mvo_val_returns, 'MVO')
rp_metrics = calculate_metrics(rp_val_returns, 'RiskParity')
equal_metrics = calculate_metrics(equal_val_returns, 'EqualWeight')

# 最佳单因子作为基准
best_single_idx = np.argmax([np.mean(val_factor_returns[i]) / np.std(val_factor_returns[i]) for i in range(n_valid_factors)])
best_single_metrics = calculate_metrics(val_factor_returns[best_single_idx], f'BestSingle_{factor_names[best_single_idx][:20]}')

metrics_df = pd.DataFrame([mvo_metrics, rp_metrics, equal_metrics, best_single_metrics])

print("\n验证集表现对比:")
print("-" * 80)
print(f"{'组合名称':25s} {'年化收益':>10s} {'年化波动':>10s} {'夏普比率':>10s} {'最大回撤':>10s} {'累计收益':>10s}")
print("-" * 80)
for _, row in metrics_df.iterrows():
    print(f"{row['Name']:25s} {row['AnnualReturn']:>10.2%} {row['Volatility']:>10.2%} "
          f"{row['Sharpe']:>10.2f} {row['MaxDrawdown']:>10.2%} {row['TotalReturn']:>10.2%}")

# =============================================================================
# 八、可视化
# =============================================================================

print("\n" + "=" * 80)
print("阶段 7: 生成可视化报告")
print("=" * 80)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. 因子相关性热力图
ax1 = fig.add_subplot(gs[0, 0])
im = ax1.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax1, shrink=0.8)
ax1.set_title('Factor Return Correlation Matrix', fontsize=10)
ax1.set_xticks([])
ax1.set_yticks([])

# 2. 权重对比 - MVO vs RP
ax2 = fig.add_subplot(gs[0, 1])
x = np.arange(n_valid_factors)
width = 0.35
ax2.bar(x - width/2, mvo_weights, width, label='MVO', alpha=0.8)
ax2.bar(x + width/2, rp_weights, width, label='RiskParity', alpha=0.8)
ax2.set_xlabel('Factor Index')
ax2.set_ylabel('Weight')
ax2.set_title('Factor Weights: MVO vs RiskParity', fontsize=10)
ax2.legend(fontsize=8)
ax2.set_xticks(x[::2])
ax2.set_xticklabels([f"F{i+1}" for i in range(n_valid_factors)[::2]], fontsize=6, rotation=45)

# 3. 风险贡献
ax3 = fig.add_subplot(gs[0, 2])
ax3.bar(x, rc, alpha=0.8, color='teal')
ax3.set_xlabel('Factor Index')
ax3.set_ylabel('Risk Contribution')
ax3.set_title('Risk Parity - Factor Risk Contribution', fontsize=10)
ax3.set_xticks(x[::2])
ax3.set_xticklabels([f"F{i+1}" for i in range(n_valid_factors)[::2]], fontsize=6, rotation=45)

# 4. 验证集净值曲线
ax4 = fig.add_subplot(gs[1, :])
dates = np.arange(val_factor_returns.shape[1])

mvo_cum = np.cumprod(1 + mvo_val_returns)
rp_cum = np.cumprod(1 + rp_val_returns)
equal_cum = np.cumprod(1 + equal_val_returns)
best_cum = np.cumprod(1 + val_factor_returns[best_single_idx])

ax4.plot(dates, mvo_cum, label='MVO', linewidth=2)
ax4.plot(dates, rp_cum, label='RiskParity', linewidth=2)
ax4.plot(dates, equal_cum, label='EqualWeight', linewidth=2, alpha=0.7)
ax4.plot(dates, best_cum, label=f'BestSingle', linewidth=1.5, alpha=0.7, linestyle='--')

ax4.axhline(1, color='gray', linestyle=':', linewidth=1)
ax4.set_xlabel('Trading Day')
ax4.set_ylabel('Cumulative Return')
ax4.set_title('Validation Set: Portfolio Performance', fontsize=12)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# 5. 绩效指标对比
ax5 = fig.add_subplot(gs[2, 0])
metrics_to_plot = ['AnnualReturn', 'Volatility', 'Sharpe', 'MaxDrawdown']
x_metrics = np.arange(len(metrics_to_plot))
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

for i, metric in enumerate(metrics_to_plot):
    values = [
        mvo_metrics[metric],
        rp_metrics[metric],
        equal_metrics[metric],
        best_single_metrics[metric]
    ]
    bars = ax5.bar(np.arange(4) + i * 0.2, values, 0.2, label=metric, alpha=0.8)

ax5.set_xticks(np.arange(4) + 0.3)
ax5.set_xticklabels(['MVO', 'RP', 'EW', 'Best'], fontsize=8)
ax5.set_title('Portfolio Metrics Comparison', fontsize=10)
ax5.legend(fontsize=6, loc='best')

# 6. 因子分类权重分布
ax6 = fig.add_subplot(gs[2, 1])
category_weights_mvo = mvo_weight_df.groupby('Category')['Weight'].sum()
category_weights_rp = rp_weight_df.groupby('Category')['Weight'].sum()

cat_x = np.arange(len(category_weights_mvo))
ax6.bar(cat_x - 0.2, category_weights_mvo.values, 0.4, label='MVO', alpha=0.8)
ax6.bar(cat_x + 0.2, category_weights_rp.values, 0.4, label='RiskParity', alpha=0.8)
ax6.set_xticks(cat_x)
ax6.set_xticklabels(category_weights_mvo.index, fontsize=8, rotation=45)
ax6.set_ylabel('Total Weight')
ax6.set_title('Category Weight Distribution', fontsize=10)
ax6.legend(fontsize=8)

# 7. 滚动夏普比率对比
ax7 = fig.add_subplot(gs[2, 2])
window = 60

def rolling_sharpe(returns, window):
    sharpe = []
    for i in range(window, len(returns)):
        sub = returns[i-window:i]
        sharpe.append(np.mean(sub) / (np.std(sub) + 1e-12) * np.sqrt(252))
    return np.array(sharpe)

mvo_rs = rolling_sharpe(mvo_val_returns, window)
rp_rs = rolling_sharpe(rp_val_returns, window)
equal_rs = rolling_sharpe(equal_val_returns, window)

ax7.plot(mvo_rs, label='MVO', linewidth=1.5)
ax7.plot(rp_rs, label='RP', linewidth=1.5)
ax7.plot(equal_rs, label='EW', linewidth=1.5, alpha=0.7)
ax7.axhline(0, color='gray', linestyle=':', linewidth=1)
ax7.set_title(f'Rolling Sharpe (window={window}d)', fontsize=10)
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

plt.suptitle('Factor Combination Optimization Report', fontsize=16, y=0.98)
plt.savefig('./factor_combination_report.png', dpi=150, bbox_inches='tight')
print("组合优化报告已保存: ./factor_combination_report.png")
plt.close()

# =============================================================================
# 九、保存结果
# =============================================================================

with open('./factor_combination_report.txt', 'w') as f:
    f.write("OpenAlpha - 因子组合优化报告\n")
    f.write("=" * 80 + "\n\n")

    f.write("一、因子列表\n")
    f.write("-" * 80 + "\n")
    for i, (name, expr, category) in enumerate(TOP_FACTORS):
        f.write(f"[{i+1}] {name} ({category})\n")
        f.write(f"    {expr}\n\n")

    f.write("二、相关性分析\n")
    f.write("-" * 80 + "\n")
    if high_corr_pairs:
        f.write(f"高相关性因子对 (|corr| > 0.7):\n")
        for pair in high_corr_pairs:
            f.write(f"  {pair['factor1']} <-> {pair['factor2']} : {pair['correlation']:.3f}\n")
    else:
        f.write("未发现高相关性因子对\n")

    f.write("\n三、MVO 优化权重 (Top 10)\n")
    f.write("-" * 80 + "\n")
    for _, row in mvo_weight_df.head(10).iterrows():
        f.write(f"  {row['Factor']:35s} : {row['Weight']:>6.1%}\n")

    f.write("\n四、风险平价优化权重 (Top 10)\n")
    f.write("-" * 80 + "\n")
    for _, row in rp_weight_df.head(10).iterrows():
        f.write(f"  {row['Factor']:35s} : {row['Weight']:>6.1%}\n")

    f.write("\n五、验证集表现对比\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'组合名称':25s} {'年化收益':>10s} {'年化波动':>10s} {'夏普比率':>10s} {'最大回撤':>10s} {'累计收益':>10s}\n")
    f.write("-" * 80 + "\n")
    for _, row in metrics_df.iterrows():
        f.write(f"{row['Name']:25s} {row['AnnualReturn']:>10.2%} {row['Volatility']:>10.2%} "
                f"{row['Sharpe']:>10.2f} {row['MaxDrawdown']:>10.2%} {row['TotalReturn']:>10.2%}\n")

print("报告已保存: ./factor_combination_report.txt")

print("\n" + "=" * 80)
print("因子组合优化完成!")
print("=" * 80)
