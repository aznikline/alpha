import warnings
warnings.filterwarnings("ignore")

import sys
import os
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'simres'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from simres.expr import AlphaExecutor
import simres.operators as op

# =============================================================================
# 因子工厂 - 批量编码设计框架
# =============================================================================

print("=" * 80)
print("OpenAlpha - 因子工厂 (批量编码设计)")
print("=" * 80)

executor = AlphaExecutor(data_dir='./data/20251231')
executor.load_all_data()
ret1 = executor.context['ret1']

# 训练/验证分割
datestr = executor.context['datestr']
n_dates = len(datestr)
train_cut = int(n_dates * 0.7)
train_slice = slice(None, train_cut)
val_slice = slice(train_cut, None)

# =============================================================================
# 一、因子设计方法论
# =============================================================================

FACTOR_CATEGORIES = {
    "momentum": {
        "logic": "过去表现好的股票未来继续表现好",
        "operators": ["ts_mean", "ts_sum", "ts_ret", "ts_rank"],
        "fields": ["close", "vwap", "ret1"],
        "direction": "正",
    },
    "reversal": {
        "logic": "过去涨幅大的股票未来会回调",
        "operators": ["ts_mean", "ts_ret", "ts_rank", "ts_delta"],
        "fields": ["close", "vwap", "ret1", "high", "low"],
        "direction": "负",
    },
    "volatility": {
        "logic": "低波动股票长期表现优于高波动股票",
        "operators": ["ts_std", "ts_kurtosis", "ts_skewness", "ts_range"],
        "fields": ["close", "ret1", "vwap"],
        "direction": "负",
    },
    "volume": {
        "logic": "异常成交量预示价格变动",
        "operators": ["ts_mean", "ts_std", "ts_correlation", "ts_rank"],
        "fields": ["volume", "amount", "ret1"],
        "direction": "混合",
    },
    "value": {
        "logic": "价格相对于某种基准被低估",
        "operators": ["/", "ts_mean", "cs_rank"],
        "fields": ["close", "vwap", "volume", "amount"],
        "direction": "正",
    },
    "quality": {
        "logic": "价格走势的稳定性和趋势强度",
        "operators": ["ts_ols", "ts_regression", "ts_correlation"],
        "fields": ["close", "high", "low", "open"],
        "direction": "正",
    },
    "microstructure": {
        "logic": "盘口微观结构信号",
        "operators": ["-", "/", "ts_mean", "ts_std"],
        "fields": ["high", "low", "close", "open", "vwap"],
        "direction": "混合",
    },
}

# =============================================================================
# 二、批量编码策略
# =============================================================================

# 策略1: 模板引擎 - 定义模板，批量填充参数
FACTOR_TEMPLATES = [
    # (模板名称, 表达式模板, 参数列表, 分类)
    ("momentum_mean", "ts_mean({field},{window})", None, "momentum"),
    ("momentum_ret", "ts_ret({field},{window})", None, "momentum"),
    ("reversal_delay", "({field}-ts_delay({field},{shift}))", None, "reversal"),
    ("reversal_rank", "cs_rank(ts_delay({field},{shift}))-cs_rank({field})", None, "reversal"),
    ("volatility_std", "-ts_std({field},{window})", None, "volatility"),
    ("volume_price", "ts_correlation({field1},{field2},{window})", None, "volume"),
    ("value_ratio", "cs_rank({field1}/{field2})", None, "value"),
    ("trend_ols", "ts_ols({field1},ts_mean({field2},{window}),{window2})[2]", None, "quality"),
    ("micro_vwap", "cs_rank(vwap-close)", None, "microstructure"),
    ("micro_range", "cs_rank(high-low)", None, "microstructure"),
    ("mean_reversion", "cs_rank(ts_mean({field},{window})-{field})", None, "reversal"),
    ("momentum_drift", "ts_mean({field},{window})/ts_delay({field},{window})-1", None, "momentum"),
    ("vol_adjusted", "({field}-ts_mean({field},{window}))/ts_std({field},{window})", None, "volatility"),
    ("volume_breakout", "volume/ts_mean(volume,{window})-1", None, "volume"),
    ("price_momentum", "cs_rank(ts_mean(close,{window})/ts_delay(close,{window}))", None, "momentum"),
    ("open_gap", "cs_rank(open-ts_delay(close,1))", None, "reversal"),
    ("intra_return", "cs_rank(close/open-1)", None, "momentum"),
    ("amplitude", "cs_rank((high-low)/ts_delay(close,1))", None, "volatility"),
    ("volume_trend", "ts_ols(volume,ret1,{window})[0]", None, "volume"),
    ("skew_signal", "-ts_skewness({field},{window})", None, "volatility"),
    ("kurt_signal", "-ts_kurtosis({field},{window})", None, "volatility"),
    ("ols_residual", "ts_ols({field1},{field2},{window})[2]", None, "quality"),
    ("ols_alpha", "ts_ols({field1},{field2},{window})[0]", None, "quality"),
    ("reg_residual", "ts_regression({field1},{field2},{window},0)", None, "quality"),
    ("delay_momentum", "cs_rank(ts_delay({field},{shift})/ts_delay({field},{shift2})-1)", None, "momentum"),
    ("volume_std", "-ts_std(volume,{window})", None, "volume"),
    ("amount_momentum", "ts_mean(amount,{window})/ts_delay(amount,{window})-1", None, "volume"),
    ("price_volume", "close*volume", None, "volume"),
    ("vwap_deviation", "cs_rank(close/vwap-1)", None, "microstructure"),
    ("high_pressure", "cs_rank(high/ts_delay(close,1)-1)", None, "microstructure"),
    ("low_support", "-cs_rank(low/ts_delay(close,1)-1)", None, "microstructure"),
]

# 参数空间
PARAM_SPACE = {
    "window": [2, 3, 5, 10, 20, 40],
    "shift": [1, 2, 3, 5],
    "shift2": [2, 3, 5, 10],
    "window2": [3, 5, 10, 20],
}

# 字段映射
FIELD_MAP = {
    "field": ["close", "open", "high", "low", "vwap", "volume", "amount", "ret1"],
    "field1": ["close", "open", "high", "low", "vwap", "volume", "amount", "ret1"],
    "field2": ["close", "open", "high", "low", "vwap", "volume", "amount", "ret1"],
}


def expand_template(template_name, template_expr, category):
    """展开单个模板为多个具体表达式"""
    expressions = []

    # 解析模板中需要的参数
    needed_params = {}
    needed_fields = {}

    for key in PARAM_SPACE:
        if f"{{{key}}}" in template_expr:
            needed_params[key] = PARAM_SPACE[key]

    for key in FIELD_MAP:
        if f"{{{key}}}" in template_expr:
            needed_fields[key] = FIELD_MAP[key]

    if not needed_params and not needed_fields:
        expressions.append((template_name, template_expr, category))
        return expressions

    # 生成参数组合
    param_keys = list(needed_params.keys())
    param_values = list(needed_params.values())
    field_keys = list(needed_fields.keys())
    field_values = list(needed_fields.values())

    param_combos = list(itertools.product(*param_values)) if param_values else [()]
    field_combos = list(itertools.product(*field_values)) if field_values else [()]

    for p_combo in param_combos:
        for f_combo in field_combos:
            expr = template_expr
            name = template_name

            for k, v in zip(param_keys, p_combo):
                expr = expr.replace(f"{{{k}}}", str(v))
                name += f"_{k}{v}"

            for k, v in zip(field_keys, f_combo):
                expr = expr.replace(f"{{{k}}}", v)
                name += f"_{v}"

            expressions.append((name, expr, category))

    return expressions


# =============================================================================
# 三、批量生成所有因子
# =============================================================================

print("\n" + "=" * 80)
print("阶段 1: 批量生成因子表达式")
print("=" * 80)

all_factors = []
for template_name, template_expr, _, category in FACTOR_TEMPLATES:
    expanded = expand_template(template_name, template_expr, category)
    all_factors.extend(expanded)

print(f"模板数量: {len(FACTOR_TEMPLATES)}")
print(f"展开后因子数量: {len(all_factors)}")

# 去重
seen_exprs = set()
unique_factors = []
for name, expr, category in all_factors:
    if expr not in seen_exprs:
        seen_exprs.add(expr)
        unique_factors.append((name, expr, category))

print(f"去重后因子数量: {len(unique_factors)}")

# 随机抽样（太多因子评估很慢）
if len(unique_factors) > 200:
    import random
    random.seed(42)
    unique_factors = random.sample(unique_factors, 200)
    print(f"抽样评估: {len(unique_factors)} 个")


# =============================================================================
# 四、批量评估
# =============================================================================

def evaluate_factor_fast(expr):
    """快速评估因子"""
    full_expr = f'at_nan2zero(cs_booksize(cs_rank(at_mask({expr},ts_fill(csi_500_weight)>0))-0.5))'
    try:
        alpha = executor.evaluate(full_expr)
        if alpha is None or np.all(np.isnan(alpha)):
            return None

        bt = executor.backtest(alpha)
        net_daily = bt['net_ret']
        train_net = net_daily[train_slice]
        val_net = net_daily[val_slice]

        train_valid = train_net[~np.isnan(train_net)]
        val_valid = val_net[~np.isnan(val_net)]

        if len(train_valid) < 30 or len(val_valid) < 10:
            return None

        train_sr = np.mean(train_valid) * 252 / (np.std(train_valid) * np.sqrt(252) + 1e-12)
        val_sr = np.mean(val_valid) * 252 / (np.std(val_valid) * np.sqrt(252) + 1e-12)
        tvr = np.nanmean(bt['tvr'])

        # ICIR
        train_alpha = alpha[:, train_slice]
        train_ret = ret1[:, train_slice]
        ics = []
        for t in range(train_alpha.shape[1]):
            a = train_alpha[:, t]
            r = train_ret[:, t]
            mask = ~np.isnan(a) & ~np.isnan(r)
            if mask.sum() >= 10:
                ic, _ = stats.spearmanr(a[mask], r[mask])
                ics.append(ic)
        ics = np.array(ics)
        icir = np.mean(ics) / (np.std(ics) + 1e-12) if len(ics) > 0 else 0

        return {
            'train_sr': train_sr,
            'val_sr': val_sr,
            'tvr': tvr,
            'icir': icir,
            'fitness': 0.4 * train_sr + 0.4 * val_sr + 0.2 * icir,
        }
    except Exception:
        return None


print("\n" + "=" * 80)
print("阶段 2: 批量评估因子")
print("=" * 80)

results = []
for i, (name, expr, category) in enumerate(unique_factors):
    metrics = evaluate_factor_fast(expr)
    if metrics:
        results.append({
            'name': name,
            'expression': expr,
            'category': category,
            **metrics,
        })

    if (i + 1) % 50 == 0:
        print(f"  已评估 {i+1}/{len(unique_factors)}")

print(f"\n有效因子: {len(results)} / {len(unique_factors)}")

# =============================================================================
# 五、结果分析
# =============================================================================

print("\n" + "=" * 80)
print("阶段 3: 因子排行榜")
print("=" * 80)

results_df = pd.DataFrame(results)

# 按适应度排序
ranked = results_df.sort_values('fitness', ascending=False)

print("\n【综合Top 20】")
print("-" * 100)
print(f"{'Rank':>4} {'Fitness':>8} {'TrSR':>7} {'ValSR':>7} {'ICIR':>7} {'Tvr':>5} {'Category':>12} {'Expression'}")
print("-" * 100)

for i, (_, row) in enumerate(ranked.head(20).iterrows()):
    print(f"{i+1:>4} {row['fitness']:>+8.3f} {row['train_sr']:>+7.3f} {row['val_sr']:>+7.3f} "
          f"{row['icir']:>+7.3f} {row['tvr']:>5.2f} {row['category']:>12} {row['expression']}")

# 按分类分析
print("\n" + "=" * 80)
print("阶段 4: 分类统计")
print("=" * 80)

category_stats = results_df.groupby('category').agg({
    'fitness': ['count', 'mean', 'max'],
    'train_sr': 'mean',
    'val_sr': 'mean',
    'icir': 'mean',
}).round(3)

print("\n各分类因子统计:")
print(category_stats.to_string())

# =============================================================================
# 六、生成新因子报告
# =============================================================================

with open('./factor_factory_report.txt', 'w') as f:
    f.write("OpenAlpha - 因子工厂报告\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"总生成因子: {len(all_factors)}\n")
    f.write(f"去重后因子: {len(unique_factors)}\n")
    f.write(f"有效因子: {len(results)}\n\n")

    f.write("=" * 80 + "\n")
    f.write("Top 30 因子（按综合适应度排序）\n")
    f.write("=" * 80 + "\n\n")

    for i, (_, row) in enumerate(ranked.head(30).iterrows()):
        f.write(f"[{i+1}] {row['name']}\n")
        f.write(f"    Category: {row['category']}\n")
        f.write(f"    Fitness: {row['fitness']:+.3f} | TrSR: {row['train_sr']:+.3f} | "
                f"ValSR: {row['val_sr']:+.3f} | ICIR: {row['icir']:+.3f} | Tvr: {row['tvr']:.2f}\n")
        f.write(f"    Expression: {row['expression']}\n\n")

    f.write("=" * 80 + "\n")
    f.write("分类统计\n")
    f.write("=" * 80 + "\n\n")
    f.write(category_stats.to_string())
    f.write("\n")

print("\n报告已保存: ./factor_factory_report.txt")

# =============================================================================
# 七、可视化
# =============================================================================

print("\n" + "=" * 80)
print("阶段 5: 生成可视化")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 分类分布
ax = axes[0, 0]
cat_counts = results_df['category'].value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(cat_counts)))
ax.barh(cat_counts.index, cat_counts.values, color=colors)
ax.set_xlabel('Count')
ax.set_title('Factor Count by Category')
for i, v in enumerate(cat_counts.values):
    ax.text(v + 0.5, i, str(v), va='center')

# 2. TrSR vs ValSR 散点图
ax = axes[0, 1]
for cat in results_df['category'].unique():
    subset = results_df[results_df['category'] == cat]
    ax.scatter(subset['train_sr'], subset['val_sr'], label=cat, alpha=0.6, s=30)
ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
ax.plot([-3, 3], [-3, 3], 'r--', linewidth=1, alpha=0.5, label='y=x')
ax.set_xlabel('Train Sharpe')
ax.set_ylabel('Validation Sharpe')
ax.set_title('Train vs Validation Sharpe by Category')
ax.legend(fontsize=7, loc='best')

# 3. Fitness分布
ax = axes[1, 0]
ax.hist(results_df['fitness'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(results_df['fitness'].max(), color='red', linestyle='--', linewidth=2,
           label=f"Best={results_df['fitness'].max():.3f}")
ax.axvline(results_df['fitness'].mean(), color='green', linestyle='--', linewidth=2,
           label=f"Mean={results_df['fitness'].mean():.3f}")
ax.set_xlabel('Fitness')
ax.set_ylabel('Count')
ax.set_title('Fitness Distribution')
ax.legend()

# 4. Top 10 因子的雷达图替代为柱状图
ax = axes[1, 1]
top10 = ranked.head(10)
x = np.arange(len(top10))
width = 0.25
ax.bar(x - width, top10['train_sr'], width, label='Train SR', alpha=0.8)
ax.bar(x, top10['val_sr'], width, label='Val SR', alpha=0.8)
ax.bar(x + width, top10['icir'], width, label='ICIR', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"{i+1}" for i in range(len(top10))], fontsize=8)
ax.set_xlabel('Rank')
ax.set_ylabel('Score')
ax.set_title('Top 10 Factor Metrics')
ax.legend()

plt.suptitle('Factor Factory Dashboard', fontsize=14)
plt.tight_layout()
plt.savefig('./factor_factory_dashboard.png', dpi=150, bbox_inches='tight')
print("仪表盘已保存: ./factor_factory_dashboard.png")
plt.close()

print("\n" + "=" * 80)
print("因子工厂完成!")
print("=" * 80)
