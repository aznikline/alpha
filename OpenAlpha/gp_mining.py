import warnings
warnings.filterwarnings("ignore")

import sys
import os
import random
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'simres'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simres.expr import AlphaExecutor
import simres.operators as op

# ============== 遗传编程配置 ==============
random.seed(42)
np.random.seed(42)

POP_SIZE = 20
N_GENERATIONS = 10
ELITE_SIZE = 5
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.7
MAX_DEPTH = 5

# 终端节点: 数据字段
TERMINALS = ['close', 'open', 'high', 'low', 'vwap', 'volume', 'amount', 'ret1']

# 一元算子 (无参数)
UNARY_OPS = ['cs_rank', 'cs_zscore']

# 一元窗口算子 (x, window)
UNARY_WINDOW_OPS = [
    ('ts_mean', 1),
    ('ts_std', 1),
    ('ts_sum', 1),
    ('ts_skewness', 1),
    ('ts_kurtosis', 1),
    ('ts_rank', 1),
]

# 二元窗口算子 (x, y, window)
BINARY_WINDOW_OPS = [
    ('ts_correlation', 2),
]

# 二元算子 (x, y) 无窗口
BINARY_OPS = [
    ('ts_delta', 1),
]

# 特殊: ts_delay 只接受 (x, shift_int)
DELAY_OPS = [
    ('ts_delay', 1),
]

# OLS 特殊算子
OLS_OPS = [
    ('ts_ols', 2, 0),  # (y, x, window)[0] = alpha
    ('ts_ols', 2, 2),  # (y, x, window)[2] = residual
]

WINDOW_CHOICES = [2, 3, 5, 10, 20, 40, 60]

# ============== 表达式树节点 ==============
class Node:
    def __init__(self, node_type, value, children=None, window=None, ols_idx=None):
        self.node_type = node_type  # 'terminal', 'unary', 'binary', 'binary_window', 'ols'
        self.value = value
        self.children = children or []
        self.window = window
        self.ols_idx = ols_idx

    def to_expr(self):
        if self.node_type == 'terminal':
            return self.value
        elif self.node_type == 'unary':
            return f"{self.value}({self.children[0].to_expr()})"
        elif self.node_type == 'unary_window':
            return f"{self.value}({self.children[0].to_expr()},{self.window})"
        elif self.node_type == 'binary':
            return f"{self.value}({self.children[0].to_expr()},{self.children[1].to_expr()})"
        elif self.node_type == 'delay':
            return f"{self.value}({self.children[0].to_expr()},{self.window})"
        elif self.node_type == 'binary_window':
            return f"{self.value}({self.children[0].to_expr()},{self.children[1].to_expr()},{self.window})"
        elif self.node_type == 'ols':
            return f"{self.value}({self.children[0].to_expr()},{self.children[1].to_expr()},{self.window})[{self.ols_idx}]"
        return ""

    def copy(self):
        return copy.deepcopy(self)

    def depth(self):
        if self.node_type == 'terminal':
            return 1
        return 1 + max(c.depth() for c in self.children)

    def node_count(self):
        if self.node_type == 'terminal':
            return 1
        return 1 + sum(c.node_count() for c in self.children)


def random_terminal():
    return Node('terminal', random.choice(TERMINALS))


def random_tree(max_depth, current_depth=0):
    if current_depth >= max_depth - 1:
        return random_terminal()

    # 50% 概率终止（避免树太深）
    if random.random() < 0.5:
        return random_terminal()

    op_type = random.random()

    if op_type < 0.15:
        # 一元算子
        node = Node('unary', random.choice(UNARY_OPS))
        node.children = [random_tree(max_depth, current_depth + 1)]
        return node

    elif op_type < 0.35:
        # 一元窗口算子 (x, window)
        op_name, n_args = random.choice(UNARY_WINDOW_OPS)
        node = Node('unary_window', op_name, window=random.choice(WINDOW_CHOICES))
        node.children = [random_tree(max_depth, current_depth + 1)]
        return node

    elif op_type < 0.45:
        # 二元算子 (无窗口) - 如 ts_delta
        op_name, _ = random.choice(BINARY_OPS)
        node = Node('binary', op_name)
        node.children = [
            random_tree(max_depth, current_depth + 1),
            random_tree(max_depth, current_depth + 1)
        ]
        return node

    elif op_type < 0.55:
        # ts_delay 特殊处理 - 第二个参数是整数shift
        op_name, _ = random.choice(DELAY_OPS)
        node = Node('delay', op_name, window=random.choice([1, 2, 3, 5, 10, 20]))
        node.children = [random_tree(max_depth, current_depth + 1)]
        return node

    elif op_type < 0.75:
        # 二元窗口算子 (x, y, window) - 如 ts_correlation
        op_name, n_args = random.choice(BINARY_WINDOW_OPS)
        node = Node('binary_window', op_name, window=random.choice(WINDOW_CHOICES))
        node.children = [
            random_tree(max_depth, current_depth + 1),
            random_tree(max_depth, current_depth + 1)
        ]
        return node

    else:
        # OLS 算子
        op_name, n_args, ols_idx = random.choice(OLS_OPS)
        node = Node('ols', op_name, window=random.choice(WINDOW_CHOICES), ols_idx=ols_idx)
        node.children = [
            random_tree(max_depth, current_depth + 1),
            random_tree(max_depth, current_depth + 1)
        ]
        return node


def get_random_node(node, target_depth=None):
    """随机获取树中的一个节点（用于交叉和变异）"""
    nodes = []

    def collect(n, depth=0):
        nodes.append((n, depth))
        for c in n.children:
            collect(c, depth + 1)

    collect(node)

    if target_depth is not None:
        filtered = [(n, d) for n, d in nodes if d == target_depth]
        if filtered:
            return random.choice(filtered)[0]

    # 避免选根节点（太破坏结构）
    non_root = [(n, d) for n, d in nodes if d > 0]
    if non_root:
        return random.choice(non_root)[0]
    return nodes[0][0]


def crossover(parent1, parent2):
    """子树交叉"""
    child1 = parent1.copy()
    child2 = parent2.copy()

    # 获取随机子树
    node1 = get_random_node(child1)
    node2 = get_random_node(child2)

    # 交换值和结构
    temp_type, temp_value, temp_children = node1.node_type, node1.value, node1.children
    temp_window, temp_ols = node1.window, node1.ols_idx

    node1.node_type, node1.value, node1.children = node2.node_type, node2.value, [c.copy() for c in node2.children]
    node1.window, node1.ols_idx = node2.window, node2.ols_idx

    node2.node_type, node2.value, node2.children = temp_type, temp_value, [c.copy() for c in temp_children]
    node2.window, node2.ols_idx = temp_window, temp_ols

    return child1, child2


def mutate(node, max_depth=MAX_DEPTH):
    """子树变异"""
    mutant = node.copy()
    target = get_random_node(mutant)

    # 用新的随机子树替换
    new_subtree = random_tree(max_depth // 2 + 1)
    target.node_type = new_subtree.node_type
    target.value = new_subtree.value
    target.children = [c.copy() for c in new_subtree.children]
    target.window = new_subtree.window
    target.ols_idx = new_subtree.ols_idx

    return mutant


# ============== 评估引擎 ==============
executor = AlphaExecutor(data_dir='./data/20251231')
executor.load_all_data()

# 缓存已评估的表达式
eval_cache = {}


def evaluate_expression(expr):
    """评估因子表达式，返回夏普比率"""
    if expr in eval_cache:
        return eval_cache[expr]

    full_expr = f'at_nan2zero(cs_booksize(cs_rank(at_mask({expr},ts_fill(csi_500_weight)>0))-0.5))'
    try:
        alpha = executor.evaluate(full_expr)
        if alpha is None or np.all(np.isnan(alpha)):
            eval_cache[expr] = -999
            return -999

        bt = executor.backtest(alpha)
        net_daily = bt['net_ret']

        # 过滤NaN
        valid = net_daily[~np.isnan(net_daily)]
        if len(valid) < 50:
            eval_cache[expr] = -999
            return -999

        ann_ret = np.mean(valid) * 252
        ann_vol = np.std(valid) * np.sqrt(252)
        sr = ann_ret / ann_vol if ann_vol > 0 else -999

        # 惩罚过长的表达式
        complexity_penalty = max(0, len(expr) - 200) * 0.001
        score = sr - complexity_penalty

        eval_cache[expr] = score
        return score

    except Exception:
        eval_cache[expr] = -999
        return -999


# ============== 遗传算法主循环 ==============
print("=" * 70)
print("OpenAlpha - 遗传编程因子挖掘")
print("=" * 70)
print(f"配置: 种群={POP_SIZE}, 代数={N_GENERATIONS}, 精英={ELITE_SIZE}")
print(f"变异率={MUTATION_RATE}, 交叉率={CROSSOVER_RATE}, 最大深度={MAX_DEPTH}")
print("=" * 70)

# 初始化种群
population = []
for i in range(POP_SIZE):
    tree = random_tree(MAX_DEPTH)
    expr = tree.to_expr()
    population.append((tree, expr, None))

print(f"\n初始化种群完成，共 {len(population)} 个个体")

# 进化历史
history = []

for gen in range(N_GENERATIONS):
    print(f"\n{'=' * 70}")
    print(f"第 {gen + 1} / {N_GENERATIONS} 代")
    print(f"{'=' * 70}")

    # 评估适应度
    scored = []
    for tree, expr, _ in population:
        if expr in eval_cache:
            score = eval_cache[expr]
        else:
            score = evaluate_expression(expr)
        scored.append((tree, expr, score))
        print(f"  评估: {expr[:70]:70s} | SR={score:+.3f}")

    # 按适应度排序
    scored.sort(key=lambda x: x[2], reverse=True)

    # 记录历史
    best_score = scored[0][2]
    best_expr = scored[0][1]
    avg_score = np.mean([s for _, _, s in scored if s > -900])
    history.append({'gen': gen + 1, 'best': best_score, 'avg': avg_score})

    print(f"\n  本代最佳: SR={best_score:.3f}")
    print(f"  本代平均: SR={avg_score:.3f}")
    print(f"  最佳表达式: {best_expr}")

    # 精英保留
    elites = scored[:ELITE_SIZE]

    # 生成下一代
    next_pop = [(e[0].copy(), e[1], e[2]) for e in elites]

    while len(next_pop) < POP_SIZE:
        # 轮盘赌选择（使用排名的倒数作为权重）
        ranked = list(enumerate(scored))
        weights = [max(0.01, 1.0 / (i + 1)) for i, _ in ranked]
        total = sum(weights)
        weights = [w / total for w in weights]

        idx1 = np.random.choice(len(scored), p=weights)
        idx2 = np.random.choice(len(scored), p=weights)

        parent1 = scored[idx1][0]
        parent2 = scored[idx2][0]

        r = random.random()
        if r < CROSSOVER_RATE and len(next_pop) < POP_SIZE - 1:
            child1, child2 = crossover(parent1, parent2)
            for c in [child1, child2]:
                if c.depth() <= MAX_DEPTH + 1 and c.node_count() <= 30:
                    next_pop.append((c, c.to_expr(), None))
        elif r < CROSSOVER_RATE + MUTATION_RATE:
            child = mutate(parent1)
            if child.depth() <= MAX_DEPTH + 1 and child.node_count() <= 30:
                next_pop.append((child, child.to_expr(), None))
        else:
            # 直接复制
            next_pop.append((parent1.copy(), parent1.to_expr(), None))

    population = next_pop[:POP_SIZE]

# ============== 最终评估 ==============
print("\n" + "=" * 70)
print("进化完成 - 最终评估")
print("=" * 70)

final_scored = []
for tree, expr, _ in population:
    score = evaluate_expression(expr)
    final_scored.append((tree, expr, score))

final_scored.sort(key=lambda x: x[2], reverse=True)

# 去重展示
seen = set()
print("\nTop 10 独特因子:")
for i, (tree, expr, score) in enumerate(final_scored):
    if expr in seen:
        continue
    seen.add(expr)
    print(f"  {len(seen):2d}. SR={score:+.3f} | {expr}")
    if len(seen) >= 10:
        break

# ============== 可视化 ==============
print("\n" + "=" * 70)
print("生成进化过程图表")
print("=" * 70)

hist_df = pd.DataFrame(history)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 进化曲线
ax1 = axes[0]
ax1.plot(hist_df['gen'], hist_df['best'], 'b-o', label='Best Sharpe', markersize=6)
ax1.plot(hist_df['gen'], hist_df['avg'], 'g-s', label='Avg Sharpe', markersize=6)
ax1.set_xlabel('Generation')
ax1.set_ylabel('Sharpe Ratio')
ax1.set_title('GP Evolution Progress')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 最终因子分布
ax2 = axes[1]
valid_scores = [s for _, _, s in final_scored if s > -900]
if valid_scores:
    ax2.hist(valid_scores, bins=15, edgecolor='black', alpha=0.7)
    ax2.axvline(np.max(valid_scores), color='red', linestyle='--', linewidth=2, label=f'Best={np.max(valid_scores):.3f}')
    ax2.set_xlabel('Sharpe Ratio')
    ax2.set_ylabel('Count')
    ax2.set_title('Final Population Score Distribution')
    ax2.legend()

plt.tight_layout()
plt.savefig('./gp_evolution.png', dpi=150, bbox_inches='tight')
print("进化过程图已保存: ./gp_evolution.png")
plt.close()

# 保存发现的因子
with open('./gp_discovered_alphas.txt', 'w') as f:
    f.write("OpenAlpha - GP Discovered Alpha Factors\n")
    f.write("=" * 70 + "\n\n")
    for i, (tree, expr, score) in enumerate(final_scored[:20]):
        if score > -900:
            f.write(f"[{i+1}] SR={score:.3f}\n")
            f.write(f"    {expr}\n\n")

print("发现的因子已保存: ./gp_discovered_alphas.txt")

print("\n" + "=" * 70)
print("遗传编程因子挖掘完成!")
print("=" * 70)
