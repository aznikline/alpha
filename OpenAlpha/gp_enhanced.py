import warnings
warnings.filterwarnings("ignore")

import sys
import os
import random
import copy
import hashlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'simres'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from simres.expr import AlphaExecutor
import simres.operators as op

# =============================================================================
# 遗传编程配置 - 增强版
# =============================================================================
random.seed(42)
np.random.seed(42)

POP_SIZE = 25
N_GENERATIONS = 20
ELITE_SIZE = 8
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.6
MAX_DEPTH = 5

# 训练/验证分割比例
TRAIN_RATIO = 0.7

# 常数范围
CONSTANT_MIN = -10.0
CONSTANT_MAX = 10.0

# 窗口参数选择
WINDOW_CHOICES = [2, 3, 5, 10, 20, 40, 60]
SHIFT_CHOICES = [1, 2, 3, 5, 10]

# 终端节点
TERMINALS = ['close', 'open', 'high', 'low', 'vwap', 'volume', 'amount', 'ret1']

# 一元算子 (无参数)
UNARY_OPS = ['cs_rank', 'cs_zscore', 'neg', 'ts_zscore']

# 一元窗口算子 (x, window)
UNARY_WINDOW_OPS = [
    'ts_mean', 'ts_std', 'ts_sum', 'ts_max', 'ts_min',
    'ts_skewness', 'ts_kurtosis', 'ts_rank', 'ts_delta'
]

# 延迟算子 (x, shift) - shift为整数
DELAY_OPS = ['ts_delay']

# 二元窗口算子 (x, y, window)
BINARY_WINDOW_OPS = ['ts_correlation']

# 二元算术算子
ARITH_OPS = ['add', 'sub', 'mul', 'div']

# OLS算子
OLS_OPS = [
    ('ts_ols', 0),
    ('ts_ols', 1),
    ('ts_ols', 2),
]

# 回归算子
REGRESSION_OPS = [
    ('ts_regression', 0),
    ('ts_regression', 1),
    ('ts_regression', 2),
    ('ts_regression', 3),
    ('ts_regression', 4),
    ('ts_regression', 5),
    ('ts_regression', 6),
    ('ts_regression', 7),
    ('ts_regression', 8),
    ('ts_regression', 9),
]

# 条件算子
WHERE_OPS = ['np_where']

# =============================================================================
# 已有因子库 - 用于正交化约束
# =============================================================================
EXISTING_FACTORS = [
    "cs_rank(ret1/volume)",
    "cs_rank(amount/ret1)",
    "ts_regression(high,ret1,40,0)",
    "ts_regression(close,ret1,10,0)",
    "ts_ols(volume,close,2)[2]",
    "ts_regression(vwap,high,3,0)",
    "(amount-ts_delay(amount,1))",
    "ts_correlation(amount,vwap,2)",
    "cs_rank(volume/vwap)",
    "ts_correlation(high,volume,2)",
]

# =============================================================================
# 表达式树节点
# =============================================================================
class Node:
    def __init__(self, node_type, value, children=None, window=None, shift=None, ols_idx=None, reg_rettype=None, const_val=None):
        self.node_type = node_type
        self.value = value
        self.children = children or []
        self.window = window
        self.shift = shift
        self.ols_idx = ols_idx
        self.reg_rettype = reg_rettype
        self.const_val = const_val

    def to_expr(self):
        if self.node_type == 'terminal':
            return self.value
        elif self.node_type == 'constant':
            return f"{self.const_val:.4f}"
        elif self.node_type == 'unary':
            if self.value == 'neg':
                return f"(-{self.children[0].to_expr()})"
            return f"{self.value}({self.children[0].to_expr()})"
        elif self.node_type == 'unary_window':
            return f"{self.value}({self.children[0].to_expr()},{self.window})"
        elif self.node_type == 'delay':
            return f"{self.value}({self.children[0].to_expr()},{self.shift})"
        elif self.node_type == 'binary_window':
            return f"{self.value}({self.children[0].to_expr()},{self.children[1].to_expr()},{self.window})"
        elif self.node_type == 'arithmetic':
            op_map = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}
            symbol = op_map[self.value]
            return f"({self.children[0].to_expr()}{symbol}{self.children[1].to_expr()})"
        elif self.node_type == 'ols':
            return f"{self.value}({self.children[0].to_expr()},{self.children[1].to_expr()},{self.window})[{self.ols_idx}]"
        elif self.node_type == 'regression':
            return f"ts_regression({self.children[0].to_expr()},{self.children[1].to_expr()},{self.window},{self.reg_rettype})"
        elif self.node_type == 'where':
            return f"np.where({self.children[0].to_expr()}>0,{self.children[1].to_expr()},{self.children[2].to_expr()})"
        return ""

    def copy(self):
        return copy.deepcopy(self)

    def depth(self):
        if self.node_type in ('terminal', 'constant'):
            return 1
        return 1 + max(c.depth() for c in self.children)

    def node_count(self):
        if self.node_type in ('terminal', 'constant'):
            return 1
        return 1 + sum(c.node_count() for c in self.children)

    def __hash__(self):
        return hash(self.to_expr())


def random_terminal():
    # 20%概率生成常数
    if random.random() < 0.2:
        return Node('constant', 'const', const_val=round(random.uniform(CONSTANT_MIN, CONSTANT_MAX), 4))
    return Node('terminal', random.choice(TERMINALS))


def random_tree(max_depth, current_depth=0):
    if current_depth >= max_depth - 1:
        return random_terminal()

    # 深度越浅，终止概率越低
    terminate_prob = 0.2 + 0.15 * current_depth
    if random.random() < terminate_prob:
        return random_terminal()

    op_type = random.random()
    cumsum = 0.0

    # 一元算子: 10%
    cumsum += 0.10
    if op_type < cumsum:
        node = Node('unary', random.choice(UNARY_OPS))
        node.children = [random_tree(max_depth, current_depth + 1)]
        return node

    # 一元窗口: 20%
    cumsum += 0.20
    if op_type < cumsum:
        node = Node('unary_window', random.choice(UNARY_WINDOW_OPS), window=random.choice(WINDOW_CHOICES))
        node.children = [random_tree(max_depth, current_depth + 1)]
        return node

    # 延迟: 5%
    cumsum += 0.05
    if op_type < cumsum:
        node = Node('delay', random.choice(DELAY_OPS), shift=random.choice(SHIFT_CHOICES))
        node.children = [random_tree(max_depth, current_depth + 1)]
        return node

    # 二元窗口: 10%
    cumsum += 0.10
    if op_type < cumsum:
        node = Node('binary_window', random.choice(BINARY_WINDOW_OPS), window=random.choice(WINDOW_CHOICES))
        node.children = [
            random_tree(max_depth, current_depth + 1),
            random_tree(max_depth, current_depth + 1)
        ]
        return node

    # 算术: 25%
    cumsum += 0.25
    if op_type < cumsum:
        node = Node('arithmetic', random.choice(ARITH_OPS))
        node.children = [
            random_tree(max_depth, current_depth + 1),
            random_tree(max_depth, current_depth + 1)
        ]
        return node

    # OLS: 10%
    cumsum += 0.10
    if op_type < cumsum:
        op_name, ols_idx = random.choice(OLS_OPS)
        node = Node('ols', op_name, window=random.choice(WINDOW_CHOICES), ols_idx=ols_idx)
        node.children = [
            random_tree(max_depth, current_depth + 1),
            random_tree(max_depth, current_depth + 1)
        ]
        return node

    # 回归: 10%
    cumsum += 0.10
    if op_type < cumsum:
        op_name, reg_rettype = random.choice(REGRESSION_OPS)
        node = Node('regression', op_name, window=random.choice(WINDOW_CHOICES), reg_rettype=reg_rettype)
        node.children = [
            random_tree(max_depth, current_depth + 1),
            random_tree(max_depth, current_depth + 1)
        ]
        return node

    # 条件: 10%
    node = Node('where', random.choice(WHERE_OPS))
    node.children = [
        random_tree(max_depth, current_depth + 1),
        random_tree(max_depth, current_depth + 1),
        random_tree(max_depth, current_depth + 1)
    ]
    return node


def get_random_node(node, exclude_root=True):
    nodes = []
    def collect(n, depth=0):
        if not (exclude_root and depth == 0):
            nodes.append((n, depth))
        for c in n.children:
            collect(c, depth + 1)
    collect(node)
    if not nodes:
        return node
    return random.choice(nodes)[0]


def crossover(parent1, parent2):
    child1 = parent1.copy()
    child2 = parent2.copy()

    node1 = get_random_node(child1)
    node2 = get_random_node(child2)

    # 交换完整结构
    temp = Node(node1.node_type, node1.value, children=[c.copy() for c in node1.children],
                window=node1.window, shift=node1.shift, ols_idx=node1.ols_idx,
                reg_rettype=node1.reg_rettype, const_val=node1.const_val)

    node1.node_type = node2.node_type
    node1.value = node2.value
    node1.children = [c.copy() for c in node2.children]
    node1.window = node2.window
    node1.shift = node2.shift
    node1.ols_idx = node2.ols_idx
    node1.reg_rettype = node2.reg_rettype
    node1.const_val = node2.const_val

    node2.node_type = temp.node_type
    node2.value = temp.value
    node2.children = [c.copy() for c in temp.children]
    node2.window = temp.window
    node2.shift = temp.shift
    node2.ols_idx = temp.ols_idx
    node2.reg_rettype = temp.reg_rettype
    node2.const_val = temp.const_val

    return child1, child2


def mutate(node, max_depth=MAX_DEPTH):
    mutant = node.copy()
    target = get_random_node(mutant)

    new_subtree = random_tree(min(max_depth - target.depth() + 1, 4))
    target.node_type = new_subtree.node_type
    target.value = new_subtree.value
    target.children = [c.copy() for c in new_subtree.children]
    target.window = new_subtree.window
    target.shift = new_subtree.shift
    target.ols_idx = new_subtree.ols_idx
    target.reg_rettype = new_subtree.reg_rettype
    target.const_val = new_subtree.const_val

    return mutant


def point_mutate(node):
    """点突变：只改一个参数值（窗口、shift、常数）"""
    mutant = node.copy()
    candidates = []

    def find_mutable(n, depth=0):
        if n.window is not None:
            candidates.append(('window', n))
        if n.shift is not None:
            candidates.append(('shift', n))
        if n.const_val is not None:
            candidates.append(('const', n))
        if n.ols_idx is not None:
            candidates.append(('ols', n))
        if n.reg_rettype is not None:
            candidates.append(('reg', n))
        for c in n.children:
            find_mutable(c, depth + 1)

    find_mutable(mutant)
    if not candidates:
        return mutant

    mtype, target = random.choice(candidates)
    if mtype == 'window':
        target.window = random.choice([w for w in WINDOW_CHOICES if w != target.window] or WINDOW_CHOICES)
    elif mtype == 'shift':
        target.shift = random.choice([s for s in SHIFT_CHOICES if s != target.shift] or SHIFT_CHOICES)
    elif mtype == 'const':
        target.const_val = round(random.uniform(CONSTANT_MIN, CONSTANT_MAX), 4)
    elif mtype == 'ols':
        target.ols_idx = random.choice([i for i in [0, 2] if i != target.ols_idx] or [0, 2])
    elif mtype == 'reg':
        rets = [0, 1, 2, 5]
        target.reg_rettype = random.choice([r for r in rets if r != target.reg_rettype] or rets)

    return mutant


# =============================================================================
# 评估引擎 - 支持样本外验证
# =============================================================================
executor = AlphaExecutor(data_dir='./data/20251231')
executor.load_all_data()

ret1 = executor.context['ret1']
datestr = executor.context['datestr']
n_dates = len(datestr)

# 训练/验证分割
train_cut = int(n_dates * TRAIN_RATIO)
train_slice = slice(None, train_cut)
val_slice = slice(train_cut, None)

eval_cache = {}

# 预计算已有因子的收益率用于正交化约束
existing_factor_returns = []
existing_factor_alphas = []


def _precompute_existing_factors():
    """预计算已有因子的收益率用于正交化约束"""
    global existing_factor_returns, existing_factor_alphas
    for expr in EXISTING_FACTORS:
        full_expr = f'at_nan2zero(cs_booksize(cs_rank(at_mask({expr},ts_fill(csi_500_weight)>0))-0.5))'
        try:
            alpha = executor.evaluate(full_expr)
            if alpha is not None and not np.all(np.isnan(alpha)):
                bt = executor.backtest(alpha)
                existing_factor_returns.append(bt['net_ret'])
                existing_factor_alphas.append(alpha)
        except Exception:
            pass
    print(f"预计算完成，共 {len(existing_factor_returns)} 个已有因子用于正交化约束")


_precompute_existing_factors()


def _compute_orthogonality_penalty(alpha):
    """计算与已有因子的正交性惩罚"""
    if len(existing_factor_alphas) == 0:
        return 0

    max_corr = 0
    alpha_flat = alpha.flatten()
    alpha_valid = ~np.isnan(alpha_flat)

    for existing_alpha in existing_factor_alphas:
        existing_flat = existing_alpha.flatten()
        existing_valid = ~np.isnan(existing_flat)
        mask = alpha_valid & existing_valid
        if mask.sum() < 100:
            continue
        corr = np.corrcoef(alpha_flat[mask], existing_flat[mask])[0, 1]
        max_corr = max(max_corr, abs(corr))

    return max_corr


def _compute_decay_ic(alpha, ret_data):
    """计算因子IC衰减"""
    n_stocks, n_dates = alpha.shape
    decay_ics = []

    for delay in [0, 1, 2, 3, 5, 10]:
        if delay >= n_dates:
            break
        ics = []
        for t in range(delay, n_dates):
            a = alpha[:, t - delay]
            r = ret_data[:, t]
            mask = ~np.isnan(a) & ~np.isnan(r)
            if mask.sum() < 10:
                continue
            ic, _ = stats.spearmanr(a[mask], r[mask])
            ics.append(ic)
        if ics:
            decay_ics.append(np.nanmean(ics))

    if len(decay_ics) >= 2:
        # 衰减率: 延迟10天IC / 当日IC
        decay_rate = decay_ics[-1] / (decay_ics[0] + 1e-12)
        return decay_rate if decay_rate > 0 else 0
    return 0


def evaluate_expression(expr, return_full=False):
    """评估因子表达式，返回多目标适应度分数

    优化目标：
    1. 验证集夏普 (权重最高)
    2. IC_IR - 信息系数稳定性
    3. Calmar比率 - 回撤控制
    4. 正交性 - 与已有因子低相关
    5. 换手率低
    6. IC衰减慢
    7. 防止过拟合
    """
    cache_key = expr
    if cache_key in eval_cache and not return_full:
        return eval_cache[cache_key]

    full_expr = f'at_nan2zero(cs_booksize(cs_rank(at_mask({expr},ts_fill(csi_500_weight)>0))-0.5))'
    try:
        alpha = executor.evaluate(full_expr)
        if alpha is None or np.all(np.isnan(alpha)):
            result = _make_bad_result()
            eval_cache[cache_key] = result
            return result

        # 回测 (全样本)
        bt = executor.backtest(alpha)
        net_daily = bt['net_ret']

        # 分割训练/验证
        train_net = net_daily[train_slice]
        val_net = net_daily[val_slice]
        train_alpha = alpha[:, train_slice]
        train_ret = ret1[:, train_slice]

        # 过滤NaN
        train_valid = train_net[~np.isnan(train_net)]
        val_valid = val_net[~np.isnan(val_net)]

        if len(train_valid) < 30 or len(val_valid) < 10:
            result = _make_bad_result()
            eval_cache[cache_key] = result
            return result

        # === 训练集指标 ===
        train_ann_ret = np.mean(train_valid) * 252
        train_ann_vol = np.std(train_valid) * np.sqrt(252)
        train_sr = train_ann_ret / train_ann_vol if train_ann_vol > 0 else -999

        train_cum = np.nancumsum(train_valid)
        train_peak = np.maximum.accumulate(train_cum)
        train_dd = np.max(train_peak - train_cum) if len(train_cum) > 0 else 1.0
        train_calmar = train_ann_ret / train_dd if train_dd > 0 else 0

        # ICIR (训练集)
        train_ic = _compute_ic(train_alpha, train_ret)
        train_ic_mean = np.mean(train_ic) if len(train_ic) > 0 else 0
        train_icir = train_ic_mean / (np.std(train_ic) + 1e-12) if np.std(train_ic) > 0 else 0

        # === 验证集指标 ===
        val_ann_ret = np.mean(val_valid) * 252
        val_ann_vol = np.std(val_valid) * np.sqrt(252)
        val_sr = val_ann_ret / val_ann_vol if val_ann_vol > 0 else -999

        # === 多目标附加指标 ===
        # 换手率
        tvr = np.nanmean(bt['tvr'][train_slice])

        # 正交性惩罚 (与已有因子的最大相关性)
        ortho_penalty = _compute_orthogonality_penalty(alpha)

        # IC衰减率
        decay_rate = _compute_decay_ic(train_alpha, train_ret)

        # === 综合适应度 - 多目标加权 ===
        # 基础收益类指标
        fitness = (
            0.35 * val_sr +           # 验证集夏普最重要（样本外）
            0.20 * train_sr +         # 训练集夏普
            0.10 * train_icir +       # IC_IR
            0.05 * train_calmar       # Calmar比率
        )

        # 惩罚项 (负权重)
        penalties = (
            0.20 * ortho_penalty +    # 正交性惩罚 (与已有因子高相关的惩罚)
            0.10 * max(0, tvr - 1.0)  # 换手率>1.0时惩罚
        )

        # 过拟合惩罚 (训练-验证差异过大)
        overfit_penalty = max(0, train_sr - val_sr) * 0.15

        # 衰减慢加分
        decay_bonus = decay_rate * 0.1

        # 最终适应度
        final_fitness = fitness - penalties - overfit_penalty + decay_bonus

        result = {
            'fitness': final_fitness,
            'train_sr': train_sr,
            'val_sr': val_sr,
            'train_icir': train_icir,
            'train_ic_mean': train_ic_mean,
            'train_calmar': train_calmar,
            'tvr': tvr,
            'overfit': train_sr - val_sr,
            'orthogonality': ortho_penalty,
            'decay_rate': decay_rate,
            'complexity': len(expr),
            'valid': True,
        }

        if return_full:
            result['alpha'] = alpha
            result['bt'] = bt

        eval_cache[cache_key] = result
        return result

    except Exception as e:
        result = _make_bad_result()
        eval_cache[cache_key] = result
        return result


def _make_bad_result():
    return {
        'fitness': -999,
        'train_sr': -999,
        'val_sr': -999,
        'train_icir': -999,
        'train_calmar': -999,
        'tvr': 999,
        'overfit': 999,
        'complexity': 999,
        'valid': False,
    }


def _compute_ic(alpha, future_ret):
    """计算每日IC"""
    n_stocks, n_dates = alpha.shape
    ics = []
    for t in range(n_dates):
        a = alpha[:, t]
        r = future_ret[:, t]
        mask = ~np.isnan(a) & ~np.isnan(r)
        if mask.sum() < 10:
            ics.append(np.nan)
            continue
        ic, _ = stats.spearmanr(a[mask], r[mask])
        ics.append(ic)
    return np.array(ics)


# =============================================================================
# 多样性保持: 语义距离
# =============================================================================
def compute_semantic_distance(expr1, expr2):
    """计算两个因子的语义距离（基于因子值相关性）"""
    if expr1 == expr2:
        return 0.0

    key = tuple(sorted([expr1, expr2]))
    if key in semantic_cache:
        return semantic_cache[key]

    try:
        r1 = evaluate_expression(expr1, return_full=True)
        r2 = evaluate_expression(expr2, return_full=True)

        if not r1.get('valid') or not r2.get('valid'):
            semantic_cache[key] = 1.0
            return 1.0

        a1 = r1['alpha'].flatten()
        a2 = r2['alpha'].flatten()
        mask = ~np.isnan(a1) & ~np.isnan(a2)
        if mask.sum() < 100:
            semantic_cache[key] = 1.0
            return 1.0

        corr = np.corrcoef(a1[mask], a2[mask])[0, 1]
        dist = 1 - abs(corr)
        semantic_cache[key] = dist
        return dist
    except Exception:
        semantic_cache[key] = 1.0
        return 1.0


semantic_cache = {}


def diversity_selection(candidates, n_select, min_dist=0.3):
    """多样性选择: 优先选择适应度好且与已选差异大的个体"""
    selected = []
    remaining = list(candidates)

    while len(selected) < n_select and remaining:
        best_score = -999
        best_idx = 0

        for i, (_, _, fitness, expr) in enumerate(remaining):
            # 多样性加分: 与已选个体的平均距离
            if selected:
                avg_dist = np.mean([compute_semantic_distance(expr, s[1]) for s in selected])
            else:
                avg_dist = 1.0

            score = fitness + 0.3 * avg_dist  # 多样性权重
            if score > best_score:
                best_score = score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected


# =============================================================================
# 遗传算法主循环
# =============================================================================
print("=" * 80)
print("OpenAlpha - 增强版遗传编程因子挖掘")
print("=" * 80)
print(f"配置: 种群={POP_SIZE}, 代数={N_GENERATIONS}, 精英={ELITE_SIZE}")
print(f"变异率={MUTATION_RATE}, 交叉率={CROSSOVER_RATE}, 最大深度={MAX_DEPTH}")
print(f"训练/验证分割: {int(TRAIN_RATIO*100)}%/{int((1-TRAIN_RATIO)*100)}% ({train_cut}/{n_dates}天)")
print(f"算子库: {len(UNARY_OPS)}一元 + {len(UNARY_WINDOW_OPS)}窗口 + {len(ARITH_OPS)}算术 + OLS/回归/条件")
print("=" * 80)

# 初始化种群
population = []
for i in range(POP_SIZE):
    tree = random_tree(MAX_DEPTH)
    expr = tree.to_expr()
    population.append((tree, expr, None))

print(f"\n初始化种群完成，共 {len(population)} 个个体")

history = []

for gen in range(N_GENERATIONS):
    print(f"\n{'=' * 80}")
    print(f"第 {gen + 1} / {N_GENERATIONS} 代")
    print(f"{'=' * 80}")

    # 评估适应度
    scored = []
    for tree, expr, _ in population:
        if expr in eval_cache:
            result = eval_cache[expr]
        else:
            result = evaluate_expression(expr)
        scored.append((tree, expr, result))
        status = "✓" if result['valid'] else "✗"
        print(f"  {status} {expr[:65]:65s} | F={result['fitness']:+.3f} "
              f"TrSR={result['train_sr']:+.3f} ValSR={result['val_sr']:+.3f} "
              f"ICIR={result['train_icir']:+.3f} OF={result['overfit']:+.3f}")

    # 按适应度排序
    scored.sort(key=lambda x: x[2]['fitness'], reverse=True)

    best = scored[0]
    valid_scores = [s[2]['fitness'] for s in scored if s[2]['valid']]
    avg_fitness = np.mean(valid_scores) if valid_scores else -999

    history.append({
        'gen': gen + 1,
        'best_fitness': best[2]['fitness'],
        'best_train_sr': best[2]['train_sr'],
        'best_val_sr': best[2]['val_sr'],
        'best_icir': best[2]['train_icir'],
        'best_ortho': best[2]['orthogonality'],
        'best_decay': best[2]['decay_rate'],
        'avg_fitness': avg_fitness,
    })

    print(f"\n  本代最佳: F={best[2]['fitness']:.3f} | TrSR={best[2]['train_sr']:.3f} | "
          f"ValSR={best[2]['val_sr']:.3f} | ICIR={best[2]['train_icir']:.3f}")
    print(f"  本代平均: F={avg_fitness:.3f}")
    print(f"  最佳表达式: {best[1]}")

    # 精英保留 (使用多样性选择)
    elites = scored[:ELITE_SIZE * 2]
    selected_elites = diversity_selection(
        [(t, e, r['fitness'], e) for t, e, r in elites],
        ELITE_SIZE
    )
    elites = [(t, e, r) for t, e, r, _ in selected_elites]

    # 生成下一代
    next_pop = [(e[0].copy(), e[1], e[2]) for e in elites]

    def tournament_select():
        contenders = random.sample(scored, min(3, len(scored)))
        return max(contenders, key=lambda x: x[2]['fitness'])[0]

    while len(next_pop) < POP_SIZE:
        r = random.random()

        if r < CROSSOVER_RATE and len(next_pop) < POP_SIZE - 1:
            p1 = tournament_select()
            p2 = tournament_select()
            c1, c2 = crossover(p1, p2)
            for c in [c1, c2]:
                if c.depth() <= MAX_DEPTH + 1 and c.node_count() <= 40:
                    next_pop.append((c, c.to_expr(), None))

        elif r < CROSSOVER_RATE + MUTATION_RATE * 0.7:
            p2 = tournament_select()
            child = mutate(p2)
            if child.depth() <= MAX_DEPTH + 1 and child.node_count() <= 40:
                next_pop.append((child, child.to_expr(), None))

        elif r < CROSSOVER_RATE + MUTATION_RATE:
            # 点突变
            p3 = tournament_select()
            child = point_mutate(p3)
            next_pop.append((child, child.to_expr(), None))

        else:
            # 新生随机个体（引入多样性）
            tree = random_tree(MAX_DEPTH)
            next_pop.append((tree, tree.to_expr(), None))

    population = next_pop[:POP_SIZE]

# =============================================================================
# 最终评估与展示
# =============================================================================
print("\n" + "=" * 80)
print("进化完成 - 最终Top因子")
print("=" * 80)

final_scored = []
for tree, expr, _ in population:
    result = evaluate_expression(expr)
    final_scored.append((tree, expr, result))

final_scored.sort(key=lambda x: x[2]['fitness'], reverse=True)

# 去重并展示
seen = set()
print("\nTop 15 独特优质因子:")
print("-" * 80)
print(f"{'Rank':>4} {'Fitness':>8} {'TrSR':>7} {'ValSR':>7} {'ICIR':>7} {'Tvr':>5} {'Expression'}")
print("-" * 80)

top_factors = []
for i, (tree, expr, result) in enumerate(final_scored):
    if expr in seen or not result['valid']:
        continue
    seen.add(expr)
    top_factors.append((tree, expr, result))
    print(f"{len(top_factors):>4} {result['fitness']:>+8.3f} {result['train_sr']:>+7.3f} "
          f"{result['val_sr']:>+7.3f} {result['train_icir']:>+7.3f} {result['tvr']:>5.2f} | {expr}")
    if len(top_factors) >= 15:
        break

# =============================================================================
# 可视化
# =============================================================================
print("\n" + "=" * 80)
print("生成进化过程图表")
print("=" * 80)

hist_df = pd.DataFrame(history)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 适应度进化
ax = axes[0, 0]
ax.plot(hist_df['gen'], hist_df['best_fitness'], 'b-o', label='Best Fitness', markersize=5)
ax.plot(hist_df['gen'], hist_df['avg_fitness'], 'g-s', label='Avg Fitness', markersize=5)
ax.set_xlabel('Generation')
ax.set_ylabel('Fitness')
ax.set_title('Fitness Evolution')
ax.legend()
ax.grid(True, alpha=0.3)

# 训练/验证夏普
ax = axes[0, 1]
ax.plot(hist_df['gen'], hist_df['best_train_sr'], 'r-o', label='Train SR', markersize=5)
ax.plot(hist_df['gen'], hist_df['best_val_sr'], 'b-s', label='Val SR', markersize=5)
ax.set_xlabel('Generation')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('Train vs Validation Sharpe')
ax.legend()
ax.grid(True, alpha=0.3)

# ICIR进化
ax = axes[1, 0]
ax.plot(hist_df['gen'], hist_df['best_icir'], 'purple', marker='o', label='ICIR', markersize=5)
ax.set_xlabel('Generation')
ax.set_ylabel('ICIR')
ax.set_title('ICIR Evolution')
ax.legend()
ax.grid(True, alpha=0.3)

# 最终种群分布
ax = axes[1, 1]
valid_fits = [s[2]['fitness'] for s in final_scored if s[2]['valid']]
if valid_fits:
    ax.hist(valid_fits, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(hist_df['best_fitness'].iloc[-1], color='red', linestyle='--', linewidth=2,
               label=f"Best={hist_df['best_fitness'].iloc[-1]:.3f}")
    ax.set_xlabel('Fitness')
    ax.set_ylabel('Count')
    ax.set_title('Final Population Distribution')
    ax.legend()

plt.suptitle('Enhanced GP Evolution Dashboard', fontsize=14)
plt.tight_layout()
plt.savefig('./gp_enhanced_dashboard.png', dpi=150, bbox_inches='tight')
print("仪表盘已保存: ./gp_enhanced_dashboard.png")
plt.close()

# 保存最佳因子
with open('./gp_best_factors.txt', 'w') as f:
    f.write("OpenAlpha - Enhanced GP Best Discovered Factors\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Config: Pop={POP_SIZE}, Gen={N_GENERATIONS}, MaxDepth={MAX_DEPTH}\n")
    f.write(f"Train/Val Split: {TRAIN_RATIO:.0%}/{1-TRAIN_RATIO:.0%}\n\n")

    for i, (tree, expr, result) in enumerate(top_factors[:20]):
        f.write(f"[{i+1}] Fitness={result['fitness']:.3f} | TrSR={result['train_sr']:.3f} | "
                f"ValSR={result['val_sr']:.3f} | ICIR={result['train_icir']:.3f} | "
                f"Tvr={result['tvr']:.2f}\n")
        f.write(f"    {expr}\n\n")

print("最佳因子已保存: ./gp_best_factors.txt")

print("\n" + "=" * 80)
print("增强版遗传编程因子挖掘完成!")
print("=" * 80)
