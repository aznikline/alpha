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
# 因子稳健性检验
# =============================================================================

print("=" * 80)
print("OpenAlpha - 因子稳健性检验")
print("=" * 80)

executor = AlphaExecutor(data_dir='./data/20251231')
executor.load_all_data()
ret1 = executor.context['ret1']
datestr = executor.context['datestr']
n_dates = len(datestr)

# =============================================================================
# 一、Top因子列表
# =============================================================================

TOP_FACTORS = [
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
print(f"\n选取 Top {N_FACTORS} 个因子进行稳健性检验")

# 计算所有因子的日度收益率
print("\n" + "=" * 80)
print("阶段 1: 计算因子日度收益率")
print("=" * 80)

factor_returns = []
factor_names = []
factor_matrices = []

for idx, (name, expr, category) in enumerate(TOP_FACTORS):
    full_expr = f'at_nan2zero(cs_booksize(cs_rank(at_mask({expr},ts_fill(csi_500_weight)>0))-0.5))'
    try:
        alpha = executor.evaluate(full_expr)
        if alpha is not None and not np.all(np.isnan(alpha)):
            factor_matrices.append(alpha)
            factor_names.append(name)
            bt = executor.backtest(alpha)
            factor_returns.append(bt['net_ret'])
    except Exception as e:
        print(f"  因子 {name} 计算失败: {e}")

factor_returns_np = np.array(factor_returns)
n_valid_factors = len(factor_returns_np)
print(f"\n成功计算 {n_valid_factors} 个因子")

# =============================================================================
# 二、滚动回测 (Walk-forward Analysis)
# =============================================================================

print("\n" + "=" * 80)
print("阶段 2: 滚动回测验证")
print("=" * 80)

def risk_parity_optimize(train_returns):
    """风险平价优化"""
    cov_matrix = np.cov(train_returns) * 252
    n_factors = train_returns.shape[0]

    def risk_contribution(weights):
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights + 1e-12)
        marginal_risk = cov_matrix @ weights
        component_risk = weights * marginal_risk / portfolio_vol
        return component_risk

    def objective(weights):
        rc = risk_contribution(weights)
        return np.sum((rc - np.mean(rc)) ** 2)

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((0, 0.3) for _ in range(n_factors))
    initial_weights = np.ones(n_factors) / n_factors

    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500}
    )
    return result.x

# 滚动回测参数
train_window = 120  # 训练窗口：120个交易日
rebalance_freq = 60  # 调仓频率：每60个交易日
test_start = 200  # 从第200天开始测试

print(f"\n滚动回测参数:")
print(f"  训练窗口: {train_window} 个交易日")
print(f"  调仓频率: {rebalance_freq} 个交易日")
print(f"  测试起始日: 第 {test_start} 天")

# 初始化结果存储
rolling_weights_history = []
rolling_portfolio_returns = []
rebalance_dates = []

current_idx = test_start

while current_idx < n_dates - 1:
    # 训练窗口
    train_slice = slice(max(0, current_idx - train_window), current_idx)
    train_factor_returns = factor_returns_np[:, train_slice]

    # 优化权重（风险平价）
    weights = risk_parity_optimize(train_factor_returns)

    # 测试窗口：持有到下一次调仓
    test_end = min(current_idx + rebalance_freq, n_dates)
    test_slice = slice(current_idx, test_end)
    test_factor_returns = factor_returns_np[:, test_slice]

    # 计算组合收益率
    portfolio_returns = weights @ test_factor_returns

    # 记录结果
    rolling_weights_history.append(weights.copy())
    rolling_portfolio_returns.extend(portfolio_returns)
    rebalance_dates.append(current_idx)

    current_idx = test_end

rolling_portfolio_returns = np.array(rolling_portfolio_returns)

print(f"\n滚动回测完成:")
print(f"  调仓次数: {len(rebalance_dates)} 次")
print(f"  测试交易日: {len(rolling_portfolio_returns)} 天")

# 计算滚动回测绩效
def calculate_metrics(daily_returns, name):
    """计算组合绩效指标"""
    mean_ret = np.mean(daily_returns) * 252
    vol = np.std(daily_returns) * np.sqrt(252)
    sharpe = mean_ret / (vol + 1e-12)
    cum_ret = np.cumprod(1 + daily_returns)
    max_dd = np.max(1 - cum_ret / np.maximum.accumulate(cum_ret))

    # 胜率
    win_rate = np.mean(daily_returns > 0)

    return {
        'Name': name,
        'AnnualReturn': mean_ret,
        'Volatility': vol,
        'Sharpe': sharpe,
        'MaxDrawdown': max_dd,
        'WinRate': win_rate,
        'TotalReturn': cum_ret[-1] - 1
    }

rolling_metrics = calculate_metrics(rolling_portfolio_returns, 'Rolling_RiskParity')

# 静态权重（一次性训练，全程使用）作为对比
static_train_slice = slice(None, test_start)
static_weights = risk_parity_optimize(factor_returns_np[:, static_train_slice])
static_test_returns = factor_returns_np[:, test_start:]
static_portfolio_returns = static_weights @ static_test_returns
static_metrics = calculate_metrics(static_portfolio_returns, 'Static_RiskParity')

print("\n滚动回测 vs 静态权重 对比:")
print("-" * 80)
print(f"{'指标':20s} {'滚动回测':>15s} {'静态权重':>15s}")
print("-" * 80)
for metric in ['AnnualReturn', 'Volatility', 'Sharpe', 'MaxDrawdown', 'WinRate', 'TotalReturn']:
    rolling_val = rolling_metrics[metric]
    static_val = static_metrics[metric]
    if metric in ['AnnualReturn', 'Volatility', 'MaxDrawdown', 'TotalReturn']:
        print(f"{metric:20s} {rolling_val:>15.2%} {static_val:>15.2%}")
    elif metric == 'Sharpe':
        print(f"{metric:20s} {rolling_val:>15.2f} {static_val:>15.2f}")
    else:
        print(f"{metric:20s} {rolling_val:>15.2%} {static_val:>15.2%}")

# =============================================================================
# 三、参数敏感性分析
# =============================================================================

print("\n" + "=" * 80)
print("阶段 3: 参数敏感性分析")
print("=" * 80)

# 测试不同训练窗口和调仓频率
train_windows = [60, 90, 120, 180, 250]
rebalance_freqs = [20, 30, 60, 90, 120]

sensitivity_results = []

for tw in train_windows:
    for rf in rebalance_freqs:
        current_idx = test_start
        port_returns = []

        while current_idx < n_dates - 1:
            train_slice = slice(max(0, current_idx - tw), current_idx)
            train_factor_returns = factor_returns_np[:, train_slice]
            weights = risk_parity_optimize(train_factor_returns)

            test_end = min(current_idx + rf, n_dates)
            test_slice = slice(current_idx, test_end)
            test_factor_returns = factor_returns_np[:, test_slice]

            port_returns.extend(weights @ test_factor_returns)
            current_idx = test_end

        port_returns = np.array(port_returns)
        metrics = calculate_metrics(port_returns, f"TW{tw}_RF{rf}")
        metrics['TrainWindow'] = tw
        metrics['RebalanceFreq'] = rf
        sensitivity_results.append(metrics)

sensitivity_df = pd.DataFrame(sensitivity_results)

print("\n参数敏感性分析结果 (Top 5 by Sharpe):")
print("-" * 80)
top_sensitivity = sensitivity_df.nlargest(5, 'Sharpe')
for _, row in top_sensitivity.iterrows():
    print(f"  TrainWindow={row['TrainWindow']:3d}, RebalanceFreq={row['RebalanceFreq']:3d} | "
          f"Sharpe={row['Sharpe']:.2f}, AnnRet={row['AnnualReturn']:.2%}, Vol={row['Volatility']:.2%}")

# =============================================================================
# 四、市场环境分析
# =============================================================================

print("\n" + "=" * 80)
print("阶段 4: 市场环境分析")
print("=" * 80)

# 计算市场收益率（指数）
market_ret = np.nanmean(ret1, axis=0)  # 等权市场收益率

# 定义市场环境
def classify_market_env(market_returns, window=20):
    """基于滚动收益率划分市场环境"""
    n = len(market_returns)
    env_labels = []

    for i in range(n):
        start = max(0, i - window)
        ret_20d = np.prod(1 + market_returns[start:i+1]) - 1

        if ret_20d > 0.02:  # 上涨市：过去20天收益>2%
            env_labels.append('Bull')
        elif ret_20d < -0.02:  # 下跌市：过去20天收益<-2%
            env_labels.append('Bear')
        else:  # 震荡市
            env_labels.append('Sideways')

    return np.array(env_labels)

# 从测试起始日开始分析
test_market_ret = market_ret[test_start:]
test_portfolio_ret = rolling_portfolio_returns[:len(test_market_ret)]
env_labels = classify_market_env(test_market_ret)

# 按环境统计表现
env_stats = []
for env in ['Bull', 'Bear', 'Sideways']:
    mask = env_labels == env
    if np.sum(mask) > 0:
        env_returns = test_portfolio_ret[mask]
        metrics = calculate_metrics(env_returns, env)
        metrics['Days'] = np.sum(mask)
        env_stats.append(metrics)

print("\n不同市场环境下的表现:")
print("-" * 80)
print(f"{'市场环境':10s} {'天数':>6s} {'年化收益':>10s} {'年化波动':>10s} {'夏普比率':>10s} {'胜率':>8s}")
print("-" * 80)
for stats in env_stats:
    print(f"{stats['Name']:10s} {stats['Days']:6d} {stats['AnnualReturn']:>10.2%} "
          f"{stats['Volatility']:>10.2%} {stats['Sharpe']:>10.2f} {stats['WinRate']:>8.1%}")

# =============================================================================
# 五、换手率分析
# =============================================================================

print("\n" + "=" * 80)
print("阶段 5: 换手率分析")
print("=" * 80)

weights_history = np.array(rolling_weights_history)
weight_changes = np.abs(np.diff(weights_history, axis=0))
avg_turnover_per_rebalance = np.mean(np.sum(weight_changes, axis=1))
annual_turnover = avg_turnover_per_rebalance * (252 / rebalance_freq)

print(f"\n换手率统计:")
print(f"  单次调仓平均换手率: {avg_turnover_per_rebalance:.2%}")
print(f"  年化换手率: {annual_turnover:.1%}")

# 估算交易成本（千分之一）
transaction_cost = 0.001
annual_cost = annual_turnover * transaction_cost
print(f"\n交易成本估算 (双边千分之一):")
print(f"  年化交易成本: {annual_cost:.2%}")
print(f"  扣成本后年化收益: {rolling_metrics['AnnualReturn'] - annual_cost:.2%}")
print(f"  扣成本后夏普比率: {(rolling_metrics['AnnualReturn'] - annual_cost) / rolling_metrics['Volatility']:.2f}")

# =============================================================================
# 六、可视化报告
# =============================================================================

print("\n" + "=" * 80)
print("阶段 6: 生成可视化报告")
print("=" * 80)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. 滚动回测 vs 静态权重 净值曲线
ax1 = fig.add_subplot(gs[0, 0])
rolling_cum = np.cumprod(1 + rolling_portfolio_returns)
static_cum = np.cumprod(1 + static_portfolio_returns)
ax1.plot(rolling_cum, label='Rolling RiskParity', linewidth=2)
ax1.plot(static_cum, label='Static RiskParity', linewidth=2, alpha=0.7)
ax1.axhline(1, color='gray', linestyle=':', linewidth=1)
ax1.set_xlabel('Trading Day')
ax1.set_ylabel('Cumulative Return')
ax1.set_title('Rolling vs Static Weights', fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# 2. 参数敏感性热力图 - Sharpe
ax2 = fig.add_subplot(gs[0, 1])
sharpe_heatmap = np.zeros((len(train_windows), len(rebalance_freqs)))
for i, tw in enumerate(train_windows):
    for j, rf in enumerate(rebalance_freqs):
        row = sensitivity_df[(sensitivity_df['TrainWindow'] == tw) & (sensitivity_df['RebalanceFreq'] == rf)]
        sharpe_heatmap[i, j] = row['Sharpe'].values[0]

im2 = ax2.imshow(sharpe_heatmap, cmap='YlGnBu', aspect='auto')
plt.colorbar(im2, ax=ax2)
ax2.set_xticks(np.arange(len(rebalance_freqs)))
ax2.set_xticklabels(rebalance_freqs, fontsize=8)
ax2.set_yticks(np.arange(len(train_windows)))
ax2.set_yticklabels(train_windows, fontsize=8)
ax2.set_xlabel('Rebalance Frequency')
ax2.set_ylabel('Train Window')
ax2.set_title('Parameter Sensitivity: Sharpe', fontsize=10)

# 3. 权重变化
ax3 = fig.add_subplot(gs[0, 2])
for i in range(min(10, n_valid_factors)):
    ax3.plot(weights_history[:, i], label=f'F{i+1}', linewidth=1.5, alpha=0.7)
ax3.set_xlabel('Rebalance Period')
ax3.set_ylabel('Weight')
ax3.set_title('Factor Weight Evolution (Top 10)', fontsize=10)
ax3.legend(fontsize=6, loc='best')

# 4. 不同市场环境下的表现
ax4 = fig.add_subplot(gs[1, 0])
env_names = [s['Name'] for s in env_stats]
env_sharpe = [s['Sharpe'] for s in env_stats]
env_ret = [s['AnnualReturn'] for s in env_stats]
env_vol = [s['Volatility'] for s in env_stats]

x_env = np.arange(len(env_names))
width = 0.25
ax4.bar(x_env - width, env_sharpe, width, label='Sharpe', alpha=0.8)
ax4.bar(x_env, env_ret, width, label='AnnRet', alpha=0.8)
ax4.bar(x_env + width, env_vol, width, label='Vol', alpha=0.8)
ax4.set_xticks(x_env)
ax4.set_xticklabels(env_names)
ax4.set_title('Performance by Market Environment', fontsize=10)
ax4.legend(fontsize=8)

# 5. 滚动夏普比率
ax5 = fig.add_subplot(gs[1, 1])
window = 60
rolling_sharpe = []
for i in range(window, len(rolling_portfolio_returns)):
    sub = rolling_portfolio_returns[i-window:i]
    rolling_sharpe.append(np.mean(sub) / (np.std(sub) + 1e-12) * np.sqrt(252))
ax5.plot(rolling_sharpe, linewidth=1.5)
ax5.axhline(np.mean(rolling_sharpe), color='red', linestyle='--', linewidth=1, label='Mean')
ax5.set_title(f'Rolling Sharpe (window={window}d)', fontsize=10)
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# 6. 回撤曲线
ax6 = fig.add_subplot(gs[1, 2])
cum_ret = np.cumprod(1 + rolling_portfolio_returns)
running_max = np.maximum.accumulate(cum_ret)
drawdown = 1 - cum_ret / running_max
ax6.fill_between(range(len(drawdown)), 0, drawdown, alpha=0.5, color='red')
ax6.set_title('Drawdown Curve', fontsize=10)
ax6.set_ylabel('Drawdown')
ax6.grid(True, alpha=0.3)

# 7. 因子贡献度稳定性
ax7 = fig.add_subplot(gs[2, 0])
top_factors_idx = np.argsort(np.mean(weights_history, axis=0))[-5:][::-1]
for idx in top_factors_idx:
    ax7.plot(weights_history[:, idx], label=f'F{idx+1}', linewidth=1.5)
ax7.set_title('Top 5 Factor Weights', fontsize=10)
ax7.legend(fontsize=8)
ax7.set_ylabel('Weight')
ax7.grid(True, alpha=0.3)

# 8. 月度收益率热力图
ax8 = fig.add_subplot(gs[2, 1])
monthly_returns = []
for i in range(0, len(rolling_portfolio_returns), 21):
    month_ret = np.prod(1 + rolling_portfolio_returns[i:i+21]) - 1
    monthly_returns.append(month_ret)
monthly_returns = np.array(monthly_returns)

# 重塑为年度x月度（如果数据足够）
n_months = len(monthly_returns)
n_cols = min(12, n_months)
n_rows = (n_months + n_cols - 1) // n_cols
monthly_matrix_padded = np.zeros(n_rows * n_cols)
monthly_matrix_padded[:n_months] = monthly_returns
monthly_matrix = monthly_matrix_padded.reshape(n_rows, n_cols)

im8 = ax8.imshow(monthly_matrix, cmap='RdYlGn', aspect='auto')
plt.colorbar(im8, ax=ax8)
ax8.set_title('Monthly Returns Heatmap', fontsize=10)
ax8.set_xlabel('Month')
ax8.set_ylabel('Year')

# 9. 换手率 vs 表现散点图
ax9 = fig.add_subplot(gs[2, 2])
ax9.scatter(sensitivity_df['RebalanceFreq'], sensitivity_df['Sharpe'],
            s=sensitivity_df['TrainWindow'] * 2, alpha=0.6, c=sensitivity_df['AnnualReturn'], cmap='viridis')
ax9.set_xlabel('Rebalance Frequency')
ax9.set_ylabel('Sharpe Ratio')
ax9.set_title('Turnover vs Performance', fontsize=10)
ax9.grid(True, alpha=0.3)

plt.suptitle('Factor Robustness Test Report', fontsize=16, y=0.98)
plt.savefig('./factor_robustness_report.png', dpi=150, bbox_inches='tight')
print("稳健性检验报告已保存: ./factor_robustness_report.png")
plt.close()

# =============================================================================
# 七、保存文本报告
# =============================================================================

with open('./factor_robustness_report.txt', 'w') as f:
    f.write("OpenAlpha - 因子稳健性检验报告\n")
    f.write("=" * 80 + "\n\n")

    f.write("一、滚动回测 vs 静态权重 对比\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'指标':20s} {'滚动回测':>15s} {'静态权重':>15s}\n")
    f.write("-" * 80 + "\n")
    for metric in ['AnnualReturn', 'Volatility', 'Sharpe', 'MaxDrawdown', 'WinRate', 'TotalReturn']:
        rolling_val = rolling_metrics[metric]
        static_val = static_metrics[metric]
        if metric in ['AnnualReturn', 'Volatility', 'MaxDrawdown', 'TotalReturn']:
            f.write(f"{metric:20s} {rolling_val:>15.2%} {static_val:>15.2%}\n")
        elif metric == 'Sharpe':
            f.write(f"{metric:20s} {rolling_val:>15.2f} {static_val:>15.2f}\n")
        else:
            f.write(f"{metric:20s} {rolling_val:>15.2%} {static_val:>15.2%}\n")

    f.write("\n二、参数敏感性分析 Top 5\n")
    f.write("-" * 80 + "\n")
    for _, row in top_sensitivity.iterrows():
        f.write(f"  TrainWindow={row['TrainWindow']:3d}, RebalanceFreq={row['RebalanceFreq']:3d} | "
                f"Sharpe={row['Sharpe']:.2f}, AnnRet={row['AnnualReturn']:.2%}, Vol={row['Volatility']:.2%}\n")

    f.write("\n三、不同市场环境下的表现\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'市场环境':10s} {'天数':>6s} {'年化收益':>10s} {'年化波动':>10s} {'夏普比率':>10s} {'胜率':>8s}\n")
    f.write("-" * 80 + "\n")
    for stats in env_stats:
        f.write(f"{stats['Name']:10s} {stats['Days']:6d} {stats['AnnualReturn']:>10.2%} "
                f"{stats['Volatility']:>10.2%} {stats['Sharpe']:>10.2f} {stats['WinRate']:>8.1%}\n")

    f.write("\n四、换手率与交易成本\n")
    f.write("-" * 80 + "\n")
    f.write(f"  单次调仓平均换手率: {avg_turnover_per_rebalance:.2%}\n")
    f.write(f"  年化换手率: {annual_turnover:.1%}\n")
    f.write(f"  年化交易成本 (千分之一): {annual_cost:.2%}\n")
    f.write(f"  扣成本后年化收益: {rolling_metrics['AnnualReturn'] - annual_cost:.2%}\n")
    f.write(f"  扣成本后夏普比率: {(rolling_metrics['AnnualReturn'] - annual_cost) / rolling_metrics['Volatility']:.2f}\n")

print("文本报告已保存: ./factor_robustness_report.txt")

print("\n" + "=" * 80)
print("因子稳健性检验完成!")
print("=" * 80)
