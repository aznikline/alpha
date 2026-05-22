"""
FactorLab - 极致高效的因子研究平台

核心特性:
- ⚡ 单文件一键运行，零依赖配置
- 📊 因子表达式即写即测，毫秒级响应
- 🎯 内置 50+ 算子模板，灵感永不枯竭
- 📈 全维度因子分析一体化输出
- 🔄 与现有 gp_enhanced, factor_factory 无缝对接

使用方式:
    from src.factor_lab import FactorLab

    # 初始化
    lab = FactorLab()

    # 一键测试因子
    result = lab.test_alpha("ts_correlation(close, volume, 10)")

    # 批量测试因子库
    results = lab.batch_test([
        "cs_rank(close/volume)",
        "ts_mean(high-low, 5)",
    ])

    # 生成完整分析报告
    lab.report(result)
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import random
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

# 导入现有引擎
from .simres.expr import AlphaExecutor

__version__ = "1.0.0"


class FactorResult:
    """因子测试结果封装"""

    def __init__(self, expr: str, metrics: Dict, alpha_matrix: np.ndarray = None, bt_result: Dict = None):
        self.expr = expr
        self.metrics = metrics
        self.alpha_matrix = alpha_matrix
        self.bt_result = bt_result
        self.created_at = datetime.now()

    def __repr__(self):
        return f"FactorResult(expr='{self.expr[:50]}...', sr={self.metrics.get('val_sr', 0):.3f})"

    def _repr_html_(self):
        """Jupyter 友好展示"""
        m = self.metrics
        return f"""
        <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin: 5px 0;">
            <h4 style="margin: 0 0 8px 0;">{self.expr[:80]}...</h4>
            <table style="font-size: 12px; border-collapse: collapse;">
                <tr>
                    <td style="padding: 2px 8px;"><b>Train SR:</b> {m.get('train_sr', 0):.3f}</td>
                    <td style="padding: 2px 8px;"><b>Val SR:</b> {m.get('val_sr', 0):.3f}</td>
                    <td style="padding: 2px 8px;"><b>IC:</b> {m.get('ic_mean', 0):.4f}</td>
                    <td style="padding: 2px 8px;"><b>ICIR:</b> {m.get('ic_ir', 0):.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 2px 8px;"><b>Turnover:</b> {m.get('tvr', 0):.3f}</td>
                    <td style="padding: 2px 8px;"><b>Ret:</b> {m.get('ann_ret', 0):.2%}</td>
                    <td style="padding: 2px 8px;"><b>Vol:</b> {m.get('ann_vol', 0):.2%}</td>
                    <td style="padding: 2px 8px;"><b>Robust:</b> {'✅' if m.get('is_robust', False) else '⚠️'}</td>
                </tr>
            </table>
        </div>
        """


class FactorLab:
    """
    因子研究主入口

    设计理念: 零配置启动，毫秒级反馈，全维度分析
    """

    def __init__(self, data_dir: str = './data/20251231'):
        """
        初始化 FactorLab

        Args:
            data_dir: 数据目录路径
        """
        self.executor = AlphaExecutor(data_dir=data_dir)
        self.executor.load_all_data()

        # 训练/验证切分点（固定，保证可复现）
        self.n_dates = self.executor.context['datestr'].shape[0]
        self.train_cut = int(self.n_dates * 0.7)

        # 缓存机制
        self._eval_cache = {}

        # 因子模板库
        self._init_templates()

        print(f"✅ FactorLab 初始化完成")
        print(f"   数据范围: {self.n_dates} 交易日")
        print(f"   训练集: 0-{self.train_cut} | 验证集: {self.train_cut}-{self.n_dates}")
        print(f"   算子数量: 50+ | 模板: {len(self.templates)} 个")

    def _init_templates(self):
        """初始化因子模板库"""
        self.templates = {
            # 量价类
            'momentum_price': "cs_rank(ts_mean(close, {w})/ts_delay(close, {w}) - 1)",
            'momentum_vwap': "cs_rank(ts_mean(vwap, {w})/ts_delay(vwap, {w}) - 1)",
            'momentum_ret': "ts_mean(ret1, {w})",

            # 波动类
            'volatility_std': "-ts_std(close, {w})",
            'volatility_range': "-ts_std((high-low)/ts_delay(close,1), {w})",
            'volatility_skew': "-ts_skewness(ret1, {w})",
            'volatility_kurt': "-ts_kurtosis(ret1, {w})",

            # 量能类
            'volume_breakout': "volume/ts_mean(volume, {w}) - 1",
            'amount_breakout': "amount/ts_mean(amount, {w}) - 1",
            'volume_price_corr': "ts_correlation(close, volume, {w})",
            'amount_price_corr': "ts_correlation(vwap, amount, {w})",

            # 回归类
            'reg_beta_market': "ts_regression(ret1, csi_500_ret1, {w}, 2)",
            'reg_resid_price': "ts_regression(high, low, {w}, 0)",
            'reg_alpha_volume': "ts_regression(close, volume, {w}, 1)",

            # 截面类
            'cs_rank_price': "cs_rank(close)",
            'cs_rank_volume': "cs_rank(volume)",
            'cs_zscore_price': "cs_zscore(close)",

            # 组合类
            'combo_price_volume': "cs_rank(close/volume)",
            'combo_amount_price': "cs_rank(amount/vwap)",
            'combo_high_low': "cs_rank((high-low)/ts_delay(close,1))",

            # 延迟类
            'delay_price': "ts_delay(close, {d})",
            'delta_price': "ts_delta(close, {d})",
            'delta_volume': "ts_delta(volume, {d})",

            # 条件类
            'cond_breakout': "np.where(close>ts_mean(high, {w}), 1, -1)",

            # 双时间窗口
            'dual_mean': "ts_mean(close, {w1}) - ts_mean(close, {w2})",
        }

        # 默认参数集
        self.windows = [2, 3, 5, 10, 20, 40]
        self.delays = [1, 2, 3, 5]

    def get_template(self, name: str, **kwargs) -> str:
        """
        获取因子模板并填充参数

        Args:
            name: 模板名称
            **kwargs: 参数值 (如 w=5, d=2)

        Returns:
            填充后的因子表达式
        """
        if name not in self.templates:
            raise ValueError(f"模板不存在: {name}. 可用模板: {list(self.templates.keys())}")

        expr = self.templates[name]
        # 自动填充参数
        for k, v in kwargs.items():
            expr = expr.replace(f"{{{k}}}", str(v))
        return expr

    def generate_factors(self, n_samples: int = 20, seed: int = 42) -> List[str]:
        """
        随机生成一批因子

        Args:
            n_samples: 生成数量
            seed: 随机种子

        Returns:
            因子表达式列表
        """
        np.random.seed(seed)
        random.seed(seed)

        factors = []
        template_names = list(self.templates.keys())

        for _ in range(n_samples):
            tpl = random.choice(template_names)
            params = {}
            if '{w}' in self.templates[tpl]:
                params['w'] = random.choice(self.windows)
            if '{d}' in self.templates[tpl]:
                params['d'] = random.choice(self.delays)
            if '{w1}' in self.templates[tpl]:
                params['w1'] = random.choice(self.windows)
                params['w2'] = random.choice([w for w in self.windows if w > params['w1']])

            try:
                expr = self.get_template(tpl, **params)
                factors.append(expr)
            except:
                continue

        return factors

    def test_alpha(self, expr: str, use_cache: bool = True) -> FactorResult:
        """
        测试单个因子（核心功能）

        Args:
            expr: 因子表达式
            use_cache: 是否使用缓存

        Returns:
            FactorResult 封装的测试结果
        """
        # 缓存命中
        cache_key = expr
        if use_cache and cache_key in self._eval_cache:
            return self._eval_cache[cache_key]

        # 完整因子表达式
        full_expr = f'at_nan2zero(cs_booksize(cs_rank(at_mask({expr},ts_fill(csi_500_weight)>0))-0.5))'

        try:
            # 执行计算
            alpha = self.executor.evaluate(full_expr)
            if alpha is None or np.all(np.isnan(alpha)):
                return FactorResult(expr, {'error': 'invalid_factor'})

            # 回测
            bt = self.executor.backtest(alpha)
            net_ret = bt['net_ret']

            # 训练集指标
            train_ret = net_ret[:self.train_cut]
            train_valid = train_ret[~np.isnan(train_ret)]
            train_mean = np.nanmean(train_valid) * 252
            train_vol = np.nanstd(train_valid) * np.sqrt(252)
            train_sr = train_mean / (train_vol + 1e-12)

            # 验证集指标
            val_ret = net_ret[self.train_cut:]
            val_valid = val_ret[~np.isnan(val_ret)]
            val_mean = np.nanmean(val_valid) * 252
            val_vol = np.nanstd(val_valid) * np.sqrt(252)
            val_sr = val_mean / (val_vol + 1e-12)

            # IC 计算
            ret1 = self.executor.context['ret1']
            train_alpha = alpha[:, :self.train_cut]
            train_fwd_ret = ret1[:, :self.train_cut]

            ics = []
            for t in range(train_alpha.shape[1]):
                a = train_alpha[:, t]
                r = train_fwd_ret[:, t]
                mask = ~np.isnan(a) & ~np.isnan(r)
                if mask.sum() >= 10:
                    ic, _ = stats.spearmanr(a[mask], r[mask])
                    ics.append(ic)

            ics = np.array(ics)
            ic_mean = np.nanmean(ics) if len(ics) > 0 else 0
            ic_ir = np.nanmean(ics) / (np.nanstd(ics) + 1e-12) if len(ics) > 0 else 0

            # 换手率
            tvr = np.nanmean(bt['tvr'])

            # 稳健性判断
            is_robust = (
                abs(train_sr - val_sr) < 1.0 and  # 过拟合不严重
                tvr < 2.0 and  # 换手率合理
                abs(val_sr) > 0.3  # 验证集有信号
            )

            # 最大回撤
            cum = np.nancumprod(1 + val_valid)
            max_dd = np.max(1 - cum / np.maximum.accumulate(cum)) if len(cum) > 0 else 0

            metrics = {
                'train_sr': train_sr,
                'val_sr': val_sr,
                'ic_mean': ic_mean,
                'ic_ir': ic_ir,
                'tvr': tvr,
                'ann_ret': val_mean,
                'ann_vol': val_vol,
                'max_dd': max_dd,
                'is_robust': is_robust,
                'ic_len': len(ics),
            }

            result = FactorResult(expr, metrics, alpha, bt)
            self._eval_cache[cache_key] = result
            return result

        except Exception as e:
            return FactorResult(expr, {'error': str(e)})

    def batch_test(self, exprs: List[str], verbose: bool = True) -> pd.DataFrame:
        """
        批量测试因子

        Args:
            exprs: 因子表达式列表
            verbose: 是否打印进度

        Returns:
            结果 DataFrame
        """
        results = []
        for i, expr in enumerate(exprs):
            res = self.test_alpha(expr)
            row = {
                'expr': expr,
                **res.metrics
            }
            results.append(row)

            if verbose and (i + 1) % 5 == 0:
                print(f"  已测试 {i+1}/{len(exprs)} 个因子")

        df = pd.DataFrame(results)
        if 'error' in df.columns:
            df = df[df['error'].isna()]

        # 按综合得分排序
        if 'val_sr' in df.columns:
            df['score'] = df['val_sr'] * 0.5 + df['ic_ir'] * 0.3 - df['tvr'] * 0.2
            df = df.sort_values('score', ascending=False).reset_index(drop=True)

        return df

    def quick_test(self, name: str, **kwargs) -> FactorResult:
        """
        快速测试模板因子（最常用功能）

        Args:
            name: 模板名称
            **kwargs: 参数

        Returns:
            FactorResult
        """
        expr = self.get_template(name, **kwargs)
        print(f"🧪 测试因子: {expr}")
        result = self.test_alpha(expr)

        if 'error' in result.metrics:
            print(f"   ❌ 错误: {result.metrics['error']}")
        else:
            m = result.metrics
            status = "✅" if m.get('is_robust', False) else "⚠️"
            print(f"   {status} TrainSR={m['train_sr']:+.3f} | ValSR={m['val_sr']:+.3f} | "
                  f"ICIR={m['ic_ir']:+.2f} | TVR={m['tvr']:.3f}")

        return result

    def scan_window(self, name: str, windows: List[int] = None) -> pd.DataFrame:
        """
        扫描窗口参数，寻找最优值

        Args:
            name: 模板名称
            windows: 窗口列表，默认使用预设值

        Returns:
            参数扫描结果
        """
        windows = windows or self.windows
        results = []

        for w in windows:
            expr = self.get_template(name, w=w)
            res = self.test_alpha(expr)
            if 'error' not in res.metrics:
                results.append({
                    'window': w,
                    'expr': expr,
                    **res.metrics
                })

        df = pd.DataFrame(results).sort_values('val_sr', ascending=False)
        print(f"\n📊 {name} 参数扫描结果 (最优窗口={df.iloc[0]['window']})")
        print(df[['window', 'train_sr', 'val_sr', 'ic_ir', 'tvr']].to_string(index=False))
        return df

    def report(self, result: FactorResult, save_path: Optional[str] = None):
        """
        生成并展示因子完整分析报告

        Args:
            result: 因子测试结果
            save_path: 可选保存路径
        """
        if result.bt_result is None:
            print("无法生成报告：回测结果为空")
            return

        bt = result.bt_result
        m = result.metrics
        net_ret = bt['net_ret']
        dates = pd.to_datetime(bt['datestr'])

        # 切分训练/验证
        train_mask = np.arange(len(dates)) < self.train_cut
        val_mask = ~train_mask

        # 计算累计收益
        cum_train = np.nancumprod(1 + net_ret[train_mask])
        cum_val = np.nancumprod(1 + net_ret[val_mask])

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

        # 1. 累计收益曲线
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(dates[train_mask], cum_train, label='Train', color='#3498db', linewidth=1.5)
        ax1.plot(dates[val_mask], cum_val, label='Validation', color='#e74c3c', linewidth=1.5)
        ax1.axvline(dates[self.train_cut], color='gray', linestyle='--', alpha=0.7, label='Split Point')
        ax1.set_title(f'Cumulative Return\n{result.expr[:60]}...', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))

        # 2. 月度收益热力图
        ax2 = fig.add_subplot(gs[0, 2:])
        df_ret = pd.Series(net_ret, index=dates)
        monthly = df_ret.resample('ME').apply(lambda x: np.nanprod(1 + x) - 1)
        monthly_matrix = monthly.values.reshape(-1, 1) if len(monthly) < 12 else monthly.values
        im = ax2.imshow(monthly_matrix.reshape(-1, min(12, len(monthly_matrix))),
                       cmap='RdYlGn', center=0, aspect='auto')
        ax2.set_title('Monthly Returns', fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax2)

        # 3. IC 衰减
        ax3 = fig.add_subplot(gs[1, 0])
        ret1 = self.executor.context['ret1']
        alpha = result.alpha_matrix

        decay_ics = []
        for delay in [0, 1, 2, 3, 5, 10]:
            if delay >= ret1.shape[1]:
                break
            day_ics = []
            for t in range(delay, ret1.shape[1]):
                a = alpha[:, t - delay]
                r = ret1[:, t]
                mask = ~np.isnan(a) & ~np.isnan(r)
                if mask.sum() >= 10:
                    ic, _ = stats.spearmanr(a[mask], r[mask])
                    day_ics.append(ic)
            decay_ics.append(np.nanmean(day_ics) if day_ics else 0)

        ax3.bar([0, 1, 2, 3, 5, 10][:len(decay_ics)], decay_ics, color='#2ecc71', alpha=0.8)
        ax3.set_title('IC Decay', fontsize=11, fontweight='bold')
        ax3.axhline(0, color='gray', linewidth=0.8)
        ax3.set_xlabel('Days')

        # 4. 每日 IC 分布
        ax4 = fig.add_subplot(gs[1, 1])
        train_alpha = alpha[:, :self.train_cut]
        train_ret = ret1[:, :self.train_cut]
        daily_ics = []
        for t in range(train_alpha.shape[1]):
            a = train_alpha[:, t]
            r = train_ret[:, t]
            mask = ~np.isnan(a) & ~np.isnan(r)
            if mask.sum() >= 10:
                ic, _ = stats.spearmanr(a[mask], r[mask])
                daily_ics.append(ic)

        ax4.hist(daily_ics, bins=30, edgecolor='black', alpha=0.7, color='#9b59b6')
        ax4.axvline(np.mean(daily_ics), color='red', linestyle='--', linewidth=2,
                   label=f'Mean={np.mean(daily_ics):.4f}')
        ax4.set_title('Daily IC Distribution', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=9)

        # 5. 换手率分布
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.hist(bt['tvr'][~np.isnan(bt['tvr'])], bins=30, edgecolor='black', alpha=0.7, color='#f39c12')
        ax5.axvline(m['tvr'], color='red', linestyle='--', linewidth=2, label=f'Mean={m["tvr"]:.3f}')
        ax5.set_title('Daily Turnover', fontsize=11, fontweight='bold')
        ax5.legend(fontsize=9)

        # 6. 收益分布
        ax6 = fig.add_subplot(gs[1, 3])
        valid_ret = net_ret[~np.isnan(net_ret)]
        ax6.hist(valid_ret * 100, bins=50, edgecolor='black', alpha=0.7, color='#1abc9c')
        ax6.axvline(np.mean(valid_ret) * 100, color='red', linestyle='--', linewidth=2,
                    label=f'Mean={np.mean(valid_ret)*100:.2f}%')
        ax6.set_title('Daily Return (%)', fontsize=11, fontweight='bold')
        ax6.legend(fontsize=9)

        # 7. 因子暴露分布
        ax7 = fig.add_subplot(gs[2, 0])
        valid_alpha = alpha[~np.isnan(alpha)]
        ax7.hist(valid_alpha.flatten(), bins=50, edgecolor='black', alpha=0.7, color='#34495e')
        ax7.set_title('Factor Exposure Distribution', fontsize=11, fontweight='bold')

        # 8. 多空持仓数量
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(dates, bt['long_num'], label='Long', color='#27ae60', alpha=0.7)
        ax8.plot(dates, bt['short_num'], label='Short', color='#c0392b', alpha=0.7)
        ax8.set_title('Position Count', fontsize=11, fontweight='bold')
        ax8.legend(fontsize=9)
        ax8.grid(True, alpha=0.3)

        # 9. 指标汇总表
        ax9 = fig.add_subplot(gs[2, 2:])
        ax9.axis('off')

        metrics_table = [
            ['Metric', 'Train', 'Validation', ''],
            ['Annual Ret', f"{m.get('train_sr', 0)*m.get('ann_vol', 0):.2%}", f"{m.get('ann_ret', 0):.2%}", ''],
            ['Sharpe Ratio', f"{m.get('train_sr', 0):.3f}", f"{m.get('val_sr', 0):.3f}", ''],
            ['IC Mean', f"{m.get('ic_mean', 0):.4f}", '-', ''],
            ['IC IR', f"{m.get('ic_ir', 0):.2f}", '-', ''],
            ['Turnover', '-', f"{m.get('tvr', 0):.3f}", ''],
            ['Max DD', '-', f"{m.get('max_dd', 0):.2%}", ''],
            ['Robust', '✅' if m.get('is_robust', False) else '⚠️', '', ''],
        ]

        table = ax9.table(cellText=metrics_table, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)
        ax9.set_title('Performance Summary', fontsize=11, fontweight='bold')

        plt.suptitle(f'Factor Analysis Report: {result.expr[:50]}...', fontsize=14, y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 报告已保存: {save_path}")

        plt.show()

        # 打印文本摘要
        print(f"\n📋 因子摘要:")
        print(f"   表达式: {result.expr}")
        print(f"   Train SR: {m.get('train_sr', 0):+.3f} | Val SR: {m.get('val_sr', 0):+.3f}")
        print(f"   ICIR: {m.get('ic_ir', 0):.2f} | IC Mean: {m.get('ic_mean', 0):.4f}")
        print(f"   Turnover: {m.get('tvr', 0):.3f} | MaxDD: {m.get('max_dd', 0):.2%}")
        print(f"   稳健性: {'✅ 通过' if m.get('is_robust', False) else '⚠️ 需关注'}")

    def compare_factors(self, results: List[FactorResult], names: List[str] = None) -> pd.DataFrame:
        """
        多因子横向对比

        Args:
            results: 因子结果列表
            names: 可选名称列表

        Returns:
            对比 DataFrame
        """
        rows = []
        for i, res in enumerate(results):
            name = names[i] if names else f'Factor_{i+1}'
            row = {
                'Name': name,
                **res.metrics,
                'expr': res.expr[:60]
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values('val_sr', ascending=False).reset_index(drop=True)

        # 打印对比
        print(f"\n📊 因子对比 (共 {len(df)} 个)")
        print("-" * 100)
        display_cols = ['Name', 'train_sr', 'val_sr', 'ic_ir', 'tvr', 'is_robust', 'expr']
        print(df[display_cols].to_string(index=False))

        return df


# 全局快捷实例（便于交互式使用）
_lab_instance = None


def get_lab(data_dir: str = './data/20251231') -> FactorLab:
    """
    获取全局 FactorLab 实例（推荐方式）

    用法:
        from src.factor_lab import get_lab
        lab = get_lab()
        lab.quick_test('volume_price_corr', w=10)
    """
    global _lab_instance
    if _lab_instance is None:
        _lab_instance = FactorLab(data_dir=data_dir)
    return _lab_instance


# 常用算子别名（方便快速输入）
ops = {
    'ts_mean': 'ts_mean',
    'ts_std': 'ts_std',
    'ts_corr': 'ts_correlation',
    'ts_reg': 'ts_regression',
    'ts_ols': 'ts_ols',
    'ts_delay': 'ts_delay',
    'ts_delta': 'ts_delta',
    'ts_rank': 'ts_rank',
    'cs_rank': 'cs_rank',
    'cs_zscore': 'cs_zscore',
}


if __name__ == '__main__':
    # 快速演示
    lab = FactorLab()

    # 测试几个因子
    print("\n=== 快速测试演示 ===\n")
    for w in [5, 10, 20]:
        lab.quick_test('volume_price_corr', w=w)

    # 生成报告
    print("\n=== 生成完整报告 ===\n")
    res = lab.quick_test('volume_price_corr', w=10)
    if res.bt_result:
        lab.report(res)
