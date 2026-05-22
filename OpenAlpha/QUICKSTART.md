# OpenAlpha 快速入门指南

## 🚀 三分钟上手

### 环境准备

```bash
cd OpenAlpha
pip install numpy pandas scipy matplotlib seaborn tqdm
```

确保数据目录存在：`./data/20251231/`

---

## 🎯 核心模块一览

| 模块 | 用途 | 核心特性 |
|------|------|----------|
| **FactorLab** | 交互式因子研究 | 20+预定义模板, 一键测试, 9维度可视化报告 |
| **FastEval** | 并行回测引擎 | 多进程5-10倍加速, 智能缓存, 参数网格搜索 |
| **FactorTracker** | 实验追踪管理 | 自动去重, 标签查询, 因子库导出, 趋势分析 |

---

## 📖 完整工作流示例

### 1. 导入模块

```python
from src.factor_lab import FactorLab
from src.fast_eval import FastEval
from src.factor_tracker import tracker

# 初始化
lab = FactorLab()                          # 研究实验室
engine = FastEval(n_workers=4)            # 并行引擎
tracker = tracker()                        # 实验追踪器
```

### 2. 一行测试单个因子

```python
# 使用预定义模板测试
result = lab.quick_test('momentum_price', w=20)

# 生成完整分析报告
lab.report(result)
```

### 3. 批量并行回测

```python
# 生成一批候选因子
factors = [
    lab.get_template('momentum_price', w=w)
    for w in [5, 10, 20, 40, 60]
]

# 并行评估 (5-10倍加速)
results = engine.evaluate(factors)

# 查看Top 5
print(results.sort_values('val_sr', ascending=False).head())
```

### 4. 记录与查询实验

```python
# 批量记录实验
for _, row in results.iterrows():
    tracker.log(
        expr=row['expr'],
        metrics={'val_sr': row['val_sr'], 'ic_ir': row['ic_ir']},
        tags=['momentum', 'batch_test'],
        source='my_research'
    )

# 查询Top 10因子
top_10 = tracker.top_k(metric='val_sr', k=10)

# 按标签筛选
vol_factors = tracker.query(tag='volume')
```

### 5. 导出因子库

```python
tracker.export_library(
    metric='val_sr',
    threshold=0.8,
    output_file='./my_factor_library.json'
)
```

---

## 🎯 可用因子模板

### 动量类 (Momentum)
| 模板名 | 描述 | 参数 |
|--------|------|------|
| `momentum_price` | 价格动量 | w |
| `momentum_vwap` | VWAP 动量 | w |
| `momentum_ret` | 收益均值动量 | w |

### 波动率类 (Volatility)
| 模板名 | 描述 | 参数 |
|--------|------|------|
| `volatility_std` | 价格波动率 | w |
| `volatility_range` | 振幅波动率 | w |
| `volatility_skew` | 收益偏度 | w |
| `volatility_kurt` | 收益峰度 | w |

### 成交量类 (Volume)
| 模板名 | 描述 | 参数 |
|--------|------|------|
| `volume_price_corr` | 量价相关性 | w |
| `amount_price_corr` | 额价相关性 | w |
| `volume_breakout` | 成交量突破 | w |
| `amount_breakout` | 成交额突破 | w |

### 回归类 (Regression)
| 模板名 | 描述 | 参数 |
|--------|------|------|
| `reg_beta_market` | 市场 Beta | w |
| `reg_resid_price` | 高低价回归残差 | w |
| `reg_alpha_volume` | 量价回归 Alpha | w |

### 截面与组合类
| 模板名 | 描述 | 参数 |
|--------|------|------|
| `cs_rank_price` | 截面价格排序 | - |
| `cs_rank_volume` | 截面成交量排序 | - |
| `cs_zscore_price` | 截面价格标准化 | - |
| `combo_price_volume` | 价量比组合 | - |
| `combo_amount_price` | 额价比组合 | - |
| `combo_high_low` | 高低价幅度组合 | - |

---

## 🔬 高级用法

### 统一 CLI 入口

```bash
# 一键自动研究流水线
python3 openalpha_cli.py auto --templates momentum_price volume_price_corr volatility_std --windows 5,10,20,40 --top 20 --artifact ./runs/volume_momentum_smoke

# 对比历史研究 run
python3 openalpha_cli.py runs --dir ./runs --top 20

# 跨 run 对比 Top 因子
python3 openalpha_cli.py runs --dir ./runs --factors --top 50

# 提升稳定因子到版本化因子库
python3 openalpha_cli.py library promote --name production --metric val_sr --min-metric 0.3 -k 50

# 查看因子库
python3 openalpha_cli.py library list
python3 openalpha_cli.py library show --name production

AutoResearch 的 `--artifact` 会保存完整研究包：

```text
runs/<run_id>/
├── config.json
├── candidates.csv
├── inspection.csv
├── evaluation.csv
├── top_factors.csv
├── diagnostics.csv
├── decay.csv
├── meta_weights.csv
├── summary.json
└── dashboard.html
```

# 静态体检表达式
python3 openalpha_cli.py validate --template momentum_price --param w=20

# 测试模板因子
python3 openalpha_cli.py test --template momentum_price --param w=20

# 扫描参数
python3 openalpha_cli.py scan momentum_price --windows 5,10,20,40

# 查看历史 Top 因子
python3 openalpha_cli.py top --metric val_sr -k 10

# Alphalens 风格分层诊断
python3 openalpha_cli.py diagnose --template momentum_price --param w=20

# 监控 Top 因子衰减
python3 openalpha_cli.py decay --metric val_sr -k 10

# 训练多因子 Ridge Ensemble
python3 openalpha_cli.py meta --metric val_sr -k 5 --ridge 1.0

# 生成一屏研究 Dashboard
python3 openalpha_cli.py dashboard --output openalpha_dashboard.html
```

### 参数网格搜索

```python
# 自动搜索最佳窗口参数
results = engine.evaluate_grid(
    "ts_correlation(close, volume, {w})",
    {'w': [5, 10, 20, 40, 60, 120]}
)

print(f"最佳参数 val_sr = {results['val_sr'].max():.3f}")
```

### 遗传编程大规模挖掘

```bash
python gp_enhanced.py
```

自动挖掘数千个因子，使用多目标适应度函数：
- 验证集夏普 (35%)
- 训练集夏普 (20%)
- IC_IR (10%)
- 正交性惩罚 (20%)
- 换手率惩罚 (10%)
- 卡玛比率 (5%)

### 因子组合优化

```bash
python factor_combination.py
```

提供两种组合方法：
1. **均值方差优化** - 最大化夏普比率
2. **风险平价** - 均衡风险贡献

输出相关性热力图、权重分布、组合回测曲线

### 因子衰减监控

```python
from src.factor_decay_monitor import FactorDecayMonitor

monitor = FactorDecayMonitor(lab=lab, tracker=tracker)
report = monitor.analyze(top_10.iloc[0]['expr'], window=60, step=20)
monitor.print_report(report)

health_df = monitor.monitor_top(metric='val_sr', k=20)
print(health_df[['status', 'health_score', 'recent_sr', 'sr_decay', 'expr']])
```

状态含义：
- `active`: 近期收益、IC 与稳定性仍满足使用条件
- `watch`: 信号仍存在，但衰减明显，需要观察
- `retired`: 近期表现失效，建议从生产因子库下架

---

## 📊 分析报告说明

`lab.report()` 生成9维度分析报告：

| 面板 | 内容 |
|------|------|
| 1 | 因子IC时间序列 + 统计摘要 |
| 2 | 累计收益曲线 (训练/验证分色) |
| 3 | 月度收益热力图 |
| 4 | 换手率时间序列 |
| 5 | 因子暴露分布直方图 |
| 6 | IC衰减分析 |
| 7 | 五分组单调性检验 |
| 8 | 回撤分析 |
| 9 | 年度收益柱状图 |

---

## 💡 最佳实践建议

### 1. 研究流程

```
模板测试 → 参数扫描 → 批量生成 → 并行回测 → 筛选记录
    ↓
生成报告 ← 组合优化 ← 去重筛选 ← 标签分类
```

### 2. 避免过拟合

- ✅ 始终使用 70%/30% 训练验证拆分
- ✅ 关注样本外表现而非训练集
- ✅ 要求 ic_ir > 0.3 作为准入门槛
- ✅ 控制换手率 tvr < 2.0
- ✅ 使用正交性惩罚避免因子冗余

### 3. 缓存利用

```bash
# 查看缓存状态
ls -la .factor_cache/ | wc -l

# 清除缓存
rm -rf .factor_cache/
```

缓存命中可节省 90%+ 的计算时间。

---

## 🔍 调试技巧

### 查看因子详情

```python
# 查看所有实验统计
tracker.print_summary()

# 标签统计
tag_stats = tracker.tag_stats()
print(tag_stats)
```

### 单个因子深度调试

```python
# 查看完整因子回测结果
result = lab.test_alpha("你的因子表达式")

# 访问原始数据
print(f"IC均值: {result.metrics['ic_mean']:.3f}")
print(f"IC_IR: {result.metrics['ic_ir']:.3f}")
print(f"换手率: {result.metrics['tvr']:.3f}")

# 可视化分析
lab.report(result)
```

---

## 📁 项目结构

```
OpenAlpha/
├── src/
│   ├── factor_lab.py          # 因子研究实验室 (核心)
│   ├── fast_eval.py           # 并行回测引擎
│   ├── factor_tracker.py      # 实验追踪器
│   └── simres/               # 底层算子与执行器
├── gp_enhanced.py            # 增强版遗传编程
├── factor_combination.py     # 因子组合优化
├── demo_complete_workflow.py # 完整工作流演示
├── QUICKSTART.md             # 本文档
├── .factor_cache/            # 回测缓存 (自动创建)
└── .factor_experiments/      # 实验记录 (自动创建)
```

---

## 🎓 下一步学习

1. **运行演示**: `python demo_complete_workflow.py`
2. **查看模板**: 编辑 `src/factor_lab.py` 添加自定义模板
3. **GP挖掘**: 运行 `python gp_enhanced.py` 进行大规模挖掘
4. **组合构建**: 使用 `factor_combination.py` 构建多因子组合
5. **交互式研究**: 在 Jupyter Notebook 中导入模块进行探索

---

## ⚡ 性能参考

| 任务 | 单进程 | 4进程并行 | 加速比 |
|------|--------|----------|--------|
| 100个因子回测 | ~120s | ~15s | 8x |
| 500个因子回测 | ~600s | ~70s | 8.6x |
| 参数网格(6点) | ~7s | ~2s | 3.5x |

*测试环境: 3.2GHz 8核CPU, 32GB RAM*

---

## 🆘 常见问题

**Q: 缓存在哪里？如何清除？**
```bash
rm -rf .factor_cache/
```

**Q: 实验记录保存在哪里？**
```bash
cat .factor_experiments/experiments.jsonl
```

**Q: 如何添加自定义因子模板？**

编辑 `src/factor_lab.py` 中的 `self.templates` 字典。

**Q: 回测太慢怎么办？**

1. 增加 `n_workers` 到 CPU 核心数
2. 启用缓存 `use_cache=True`
3. 先在小样本上验证表达式

---

## 📚 相关模块

- **gp_enhanced.py** - 遗传编程因子挖掘
- **factor_combination.py** - 因子组合与权重优化
- **src/simres/** - 底层算子库与回测引擎

祝您因子挖掘顺利！ 🎉
