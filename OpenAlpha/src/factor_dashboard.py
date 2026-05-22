"""
FactorDashboard - 静态因子研究 Dashboard 生成器

生成一个可直接打开的 HTML 文件，汇总实验记录、Top 因子、标签分布和衰减监控状态。
"""
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Optional

import pandas as pd

from .factor_decay_monitor import FactorDecayMonitor
from .factor_lab import FactorLab
from .factor_tracker import FactorTracker, tracker as get_tracker


class FactorDashboard:
    """生成静态因子研究总览报告"""

    def __init__(self, lab: Optional[FactorLab] = None, tracker: Optional[FactorTracker] = None):
        self.lab = lab or FactorLab()
        self.tracker = tracker or get_tracker()

    def build(self, output_file: str = './openalpha_dashboard.html', top_k: int = 20, include_decay: bool = True) -> str:
        experiments = self.tracker.to_df()
        stats = self.tracker.stats()
        top = self.tracker.top_k(metric='val_sr', k=top_k)
        tags = self.tracker.tag_stats()
        decay = self._decay_table(top_k=top_k) if include_decay and len(top) > 0 else pd.DataFrame()

        html = self._render(
            experiments=experiments,
            stats=stats,
            top=top,
            tags=tags,
            decay=decay,
            top_k=top_k,
        )
        Path(output_file).write_text(html, encoding='utf-8')
        print(f"📊 Dashboard 已生成: {output_file}")
        return output_file

    def _decay_table(self, top_k: int) -> pd.DataFrame:
        monitor = FactorDecayMonitor(lab=self.lab, tracker=self.tracker)
        return monitor.monitor_top(metric='val_sr', k=min(top_k, 10), window=60, step=20)

    def _render(
        self,
        experiments: pd.DataFrame,
        stats: dict,
        top: pd.DataFrame,
        tags: pd.DataFrame,
        decay: pd.DataFrame,
        top_k: int,
    ) -> str:
        total = stats.get('total', 0)
        best_sr = top['val_sr'].max() if len(top) > 0 and 'val_sr' in top.columns else 0
        avg_ic = experiments['ic_ir'].mean() if len(experiments) > 0 and 'ic_ir' in experiments.columns else 0
        avg_tvr = experiments['tvr'].mean() if len(experiments) > 0 and 'tvr' in experiments.columns else 0

        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenAlpha Factor Dashboard</title>
  <style>
    :root {{
      --bg: #0d1117;
      --panel: #151b23;
      --panel-2: #1f2630;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #58a6ff;
      --good: #3fb950;
      --warn: #d29922;
      --bad: #f85149;
      --border: #30363d;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: radial-gradient(circle at top left, #172033, var(--bg) 40%); color: var(--text); font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header {{ padding: 32px 40px 20px; border-bottom: 1px solid var(--border); }}
    h1 {{ margin: 0; font-size: 34px; letter-spacing: -0.04em; }}
    h2 {{ margin: 0 0 16px; font-size: 18px; }}
    .subtitle {{ color: var(--muted); margin-top: 8px; }}
    main {{ padding: 28px 40px 48px; display: grid; gap: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; }}
    .card {{ background: linear-gradient(180deg, var(--panel), #111720); border: 1px solid var(--border); border-radius: 18px; padding: 18px; box-shadow: 0 20px 60px rgba(0,0,0,.25); }}
    .metric {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
    .value {{ margin-top: 8px; font-size: 28px; font-weight: 700; }}
    .section {{ background: rgba(21, 27, 35, .78); border: 1px solid var(--border); border-radius: 20px; padding: 22px; overflow: hidden; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid var(--border); padding: 9px 8px; text-align: left; vertical-align: top; }}
    th {{ color: var(--accent); background: var(--panel-2); position: sticky; top: 0; }}
    td {{ color: #d1d7de; }}
    .table-wrap {{ max-height: 540px; overflow: auto; border: 1px solid var(--border); border-radius: 12px; }}
    .expr {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: #a5d6ff; max-width: 560px; word-break: break-all; }}
    .pill {{ display: inline-block; padding: 3px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; }}
    .active {{ background: rgba(63,185,80,.18); color: var(--good); }}
    .watch {{ background: rgba(210,153,34,.18); color: var(--warn); }}
    .retired, .error {{ background: rgba(248,81,73,.18); color: var(--bad); }}
    .empty {{ color: var(--muted); padding: 18px; border: 1px dashed var(--border); border-radius: 12px; }}
    @media (max-width: 1000px) {{ .grid {{ grid-template-columns: repeat(2, 1fr); }} header, main {{ padding-left: 20px; padding-right: 20px; }} }}
    @media (max-width: 640px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>OpenAlpha Factor Dashboard</h1>
    <div class="subtitle">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · Top K = {top_k}</div>
  </header>
  <main>
    <section class="grid">
      {self._metric_card('Total Experiments', total)}
      {self._metric_card('Best Val SR', f'{best_sr:.3f}')}
      {self._metric_card('Avg ICIR', f'{avg_ic:.3f}')}
      {self._metric_card('Avg TVR', f'{avg_tvr:.3f}')}
    </section>

    <section class="section">
      <h2>Top Factors</h2>
      {self._table(top, ['val_sr', 'train_sr', 'ic_ir', 'tvr', 'source', 'tags', 'expr'])}
    </section>

    <section class="section">
      <h2>Decay Monitor</h2>
      {self._decay_html(decay)}
    </section>

    <section class="section">
      <h2>Tag Distribution</h2>
      {self._table(tags, ['tag', 'count'])}
    </section>

    <section class="section">
      <h2>Template Library</h2>
      {self._template_table()}
    </section>
  </main>
</body>
</html>"""

    def _metric_card(self, label: str, value) -> str:
        return f"<article class=\"card\"><div class=\"metric\">{escape(str(label))}</div><div class=\"value\">{escape(str(value))}</div></article>"

    def _table(self, df: pd.DataFrame, cols: list[str]) -> str:
        if df is None or len(df) == 0:
            return '<div class="empty">暂无数据</div>'

        table = df[[col for col in cols if col in df.columns]].copy()
        if 'expr' in table.columns:
            table['expr'] = table['expr'].map(lambda value: f'<span class="expr">{escape(str(value))}</span>')
        if 'tags' in table.columns:
            table['tags'] = table['tags'].map(lambda value: escape(', '.join(value) if isinstance(value, list) else str(value)))

        return '<div class="table-wrap">' + table.to_html(index=False, escape=False) + '</div>'

    def _decay_html(self, decay: pd.DataFrame) -> str:
        if decay is None or len(decay) == 0:
            return '<div class="empty">暂无衰减监控数据</div>'

        table = decay[['status', 'health_score', 'recent_sr', 'sr_decay', 'positive_window_ratio', 'reason', 'expr']].copy()
        table['status'] = table['status'].map(lambda value: f'<span class="pill {escape(str(value))}">{escape(str(value))}</span>')
        table['expr'] = table['expr'].map(lambda value: f'<span class="expr">{escape(str(value))}</span>')
        table['health_score'] = table['health_score'].map(lambda value: f'{value:.1f}')
        table['positive_window_ratio'] = table['positive_window_ratio'].map(lambda value: f'{value:.1%}')
        return '<div class="table-wrap">' + table.to_html(index=False, escape=False) + '</div>'

    def _template_table(self) -> str:
        rows = [{'template': name, 'expression': expr} for name, expr in self.lab.templates.items()]
        df = pd.DataFrame(rows)
        df['expression'] = df['expression'].map(lambda value: f'<span class="expr">{escape(str(value))}</span>')
        return '<div class="table-wrap">' + df.to_html(index=False, escape=False) + '</div>'


if __name__ == '__main__':
    FactorDashboard().build()
