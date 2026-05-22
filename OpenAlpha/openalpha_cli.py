#!/usr/bin/env python3
"""OpenAlpha 高频因子研究操作统一入口"""
import argparse
import warnings
warnings.filterwarnings("ignore")

from src.auto_research import AutoResearchPipeline
from src.factor_dashboard import FactorDashboard
from src.factor_decay_monitor import FactorDecayMonitor
from src.factor_diagnostics import FactorDiagnostics
from src.factor_expression_inspector import FactorExpressionInspector
from src.factor_lab import FactorLab
from src.factor_library_manager import FactorLibraryManager
from src.factor_meta_model import FactorMetaModel
from src.factor_tracker import tracker as get_tracker
from src.fast_eval import FastEval
from src.run_artifacts import RunArtifacts


def cmd_test(args: argparse.Namespace) -> None:
    lab = FactorLab()
    if args.template:
        result = lab.quick_test(args.template, **parse_params(args.param))
    else:
        result = lab.test_alpha(args.expr)
        print(result)

    if args.report:
        lab.report(result, save_path=args.save)


def cmd_scan(args: argparse.Namespace) -> None:
    lab = FactorLab()
    windows = [int(value) for value in args.windows.split(',')]
    lab.scan_window(args.template, windows=windows)


def cmd_eval(args: argparse.Namespace) -> None:
    lab = FactorLab()
    engine = FastEval(n_workers=args.workers, use_cache=True, enable_ic=not args.no_ic)
    exprs = [lab.get_template(args.template, w=w) for w in [int(value) for value in args.windows.split(',')]]
    df = engine.evaluate(exprs, show_progress=True)
    print(df[['val_sr', 'train_sr', 'ic_ir', 'tvr', 'score', 'expr']].head(args.top).to_string(index=False))


def cmd_top(args: argparse.Namespace) -> None:
    experiment_tracker = get_tracker()
    df = experiment_tracker.top_k(metric=args.metric, k=args.k)
    if len(df) == 0:
        print("暂无实验记录")
        return
    cols = [col for col in ['val_sr', 'train_sr', 'ic_ir', 'tvr', 'tags', 'expr'] if col in df.columns]
    print(df[cols].to_string(index=False))


def cmd_decay(args: argparse.Namespace) -> None:
    lab = FactorLab()
    experiment_tracker = get_tracker()
    monitor = FactorDecayMonitor(lab=lab, tracker=experiment_tracker)

    if args.expr:
        report = monitor.analyze(args.expr, window=args.window, step=args.step)
        monitor.print_report(report)
        return

    df = monitor.monitor_top(metric=args.metric, k=args.k, window=args.window, step=args.step)
    if len(df) == 0:
        print("暂无可监控因子")
        return
    print(df[['status', 'health_score', 'recent_sr', 'sr_decay', 'positive_window_ratio', 'expr']].to_string(index=False))


def cmd_library(args: argparse.Namespace) -> None:
    lab = FactorLab()
    experiment_tracker = get_tracker()
    manager = FactorLibraryManager(library_dir=args.dir, lab=lab, tracker=experiment_tracker)

    if args.action == 'promote':
        manager.promote(
            name=args.name,
            metric=args.metric,
            k=args.k,
            min_metric=args.min_metric,
            allowed_status=args.status,
            include_decay=not args.no_decay,
        )
    elif args.action == 'list':
        df = manager.list_libraries()
        print(df.to_string(index=False) if len(df) else "暂无因子库")
    elif args.action == 'show':
        manager.print_library(name=args.name, version=args.version, top=args.k)


def cmd_runs(args: argparse.Namespace) -> None:
    artifacts = RunArtifacts(runs_dir=args.dir)
    if args.factors:
        df = artifacts.compare_top_factors(top=args.top)
        if df.empty:
            print("暂无 top factor 产物")
            return
        cols = [col for col in ['run_id', 'val_sr', 'score', 'ic_ir', 'tvr', 'expr'] if col in df.columns]
        print(df[cols].head(args.top).to_string(index=False))
        return

    artifacts.print_leaderboard(top=args.top)
    if args.export:
        artifacts.export_leaderboard(args.export)


def cmd_auto(args: argparse.Namespace) -> None:
    lab = FactorLab()
    experiment_tracker = get_tracker()
    engine = FastEval(n_workers=args.workers, use_cache=True, enable_ic=not args.no_ic)
    pipeline = AutoResearchPipeline(lab=lab, tracker=experiment_tracker, engine=engine)
    pipeline.run(
        templates=args.templates,
        windows=[int(value) for value in args.windows.split(',')],
        top_k=args.top,
        workers=args.workers,
        enable_ic=not args.no_ic,
        dashboard_file=args.dashboard,
        artifact_dir=args.artifact,
    )


def cmd_validate(args: argparse.Namespace) -> None:
    lab = FactorLab() if args.template else None
    expr = args.expr or lab.get_template(args.template, **parse_params(args.param))
    inspector = FactorExpressionInspector()
    inspector.print_report(inspector.inspect(expr))


def cmd_meta(args: argparse.Namespace) -> None:
    lab = FactorLab()
    experiment_tracker = get_tracker()
    model = FactorMetaModel(lab=lab, tracker=experiment_tracker)
    if args.expr:
        report = model.fit(args.expr, ridge=args.ridge)
    else:
        report = model.fit_from_tracker(metric=args.metric, k=args.k, ridge=args.ridge)
    model.print_report(report)


def cmd_diagnose(args: argparse.Namespace) -> None:
    lab = FactorLab()
    diagnostics = FactorDiagnostics(lab=lab)
    expr = args.expr or lab.get_template(args.template, **parse_params(args.param))
    report = diagnostics.analyze(expr, quantiles=args.quantiles)
    diagnostics.print_report(report)


def cmd_dashboard(args: argparse.Namespace) -> None:
    lab = FactorLab()
    experiment_tracker = get_tracker()
    dashboard = FactorDashboard(lab=lab, tracker=experiment_tracker)
    dashboard.build(output_file=args.output, top_k=args.top, include_decay=not args.no_decay)


def parse_params(raw_params: list[str]) -> dict:
    params = {}
    for raw in raw_params:
        key, value = raw.split('=', 1)
        try:
            params[key] = int(value)
        except ValueError:
            params[key] = float(value)
    return params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='OpenAlpha 因子研究 CLI')
    subparsers = parser.add_subparsers(dest='command', required=True)

    library = subparsers.add_parser('library', help='版本化因子库管理')
    library.add_argument('action', choices=['promote', 'list', 'show'])
    library.add_argument('--name', default='default')
    library.add_argument('--version', default='latest')
    library.add_argument('--dir', default='./factor_libraries')
    library.add_argument('--metric', default='val_sr')
    library.add_argument('-k', type=int, default=50)
    library.add_argument('--min-metric', type=float, default=0.0)
    library.add_argument('--status', nargs='+', default=['active', 'watch'])
    library.add_argument('--no-decay', action='store_true')
    library.set_defaults(func=cmd_library)

    runs = subparsers.add_parser('runs', help='对比 AutoResearch artifact')
    runs.add_argument('--dir', default='./runs')
    runs.add_argument('--top', type=int, default=20)
    runs.add_argument('--export', help='导出 leaderboard CSV')
    runs.add_argument('--factors', action='store_true', help='跨 run 对比 Top 因子')
    runs.set_defaults(func=cmd_runs)

    auto = subparsers.add_parser('auto', help='一键自动因子研究流水线')
    auto.add_argument('--templates', nargs='+', default=['momentum_price', 'volume_price_corr', 'volatility_std', 'reg_beta_market'])
    auto.add_argument('--windows', default='5,10,20,40,60')
    auto.add_argument('--top', type=int, default=20)
    auto.add_argument('--workers', type=int, default=4)
    auto.add_argument('--no-ic', action='store_true')
    auto.add_argument('--dashboard', default='./openalpha_auto_dashboard.html')
    auto.add_argument('--artifact', help='保存完整研究产物的目录')
    auto.set_defaults(func=cmd_auto)

    test = subparsers.add_parser('test', help='测试单个因子表达式或模板')
    test.add_argument('--expr', help='因子表达式')
    test.add_argument('--template', help='FactorLab 模板名')
    test.add_argument('--param', action='append', default=[], help='模板参数，如 w=20')
    test.add_argument('--report', action='store_true', help='生成分析报告')
    test.add_argument('--save', help='报告保存路径')
    test.set_defaults(func=cmd_test)

    validate = subparsers.add_parser('validate', help='静态体检因子表达式')
    validate.add_argument('--expr', help='因子表达式')
    validate.add_argument('--template', help='FactorLab 模板名')
    validate.add_argument('--param', action='append', default=[], help='模板参数，如 w=20')
    validate.set_defaults(func=cmd_validate)

    scan = subparsers.add_parser('scan', help='扫描单模板窗口参数')
    scan.add_argument('template', help='FactorLab 模板名')
    scan.add_argument('--windows', default='2,3,5,10,20,40,60', help='逗号分隔窗口列表')
    scan.set_defaults(func=cmd_scan)

    eval_cmd = subparsers.add_parser('eval', help='并行评估模板窗口组合')
    eval_cmd.add_argument('template', help='FactorLab 模板名')
    eval_cmd.add_argument('--windows', default='2,3,5,10,20,40,60', help='逗号分隔窗口列表')
    eval_cmd.add_argument('--workers', type=int, default=4)
    eval_cmd.add_argument('--top', type=int, default=10)
    eval_cmd.add_argument('--no-ic', action='store_true')
    eval_cmd.set_defaults(func=cmd_eval)

    top = subparsers.add_parser('top', help='查看历史 Top 因子')
    top.add_argument('--metric', default='val_sr')
    top.add_argument('-k', type=int, default=10)
    top.set_defaults(func=cmd_top)

    decay = subparsers.add_parser('decay', help='衰减监控')
    decay.add_argument('--expr', help='指定因子表达式；不指定则监控历史 Top 因子')
    decay.add_argument('--metric', default='val_sr')
    decay.add_argument('-k', type=int, default=10)
    decay.add_argument('--window', type=int, default=60)
    decay.add_argument('--step', type=int, default=20)
    decay.set_defaults(func=cmd_decay)

    diagnose = subparsers.add_parser('diagnose', help='Alphalens 风格分层诊断')
    diagnose.add_argument('--expr', help='因子表达式')
    diagnose.add_argument('--template', help='FactorLab 模板名')
    diagnose.add_argument('--param', action='append', default=[], help='模板参数，如 w=20')
    diagnose.add_argument('--quantiles', type=int, default=5)
    diagnose.set_defaults(func=cmd_diagnose)

    meta = subparsers.add_parser('meta', help='训练多因子 Ridge Ensemble')
    meta.add_argument('--expr', action='append', help='指定因子表达式，可重复；不指定则使用历史 Top 因子')
    meta.add_argument('--metric', default='val_sr')
    meta.add_argument('-k', type=int, default=5)
    meta.add_argument('--ridge', type=float, default=1.0)
    meta.set_defaults(func=cmd_meta)

    dashboard = subparsers.add_parser('dashboard', help='生成静态 HTML Dashboard')
    dashboard.add_argument('--output', default='./openalpha_dashboard.html')
    dashboard.add_argument('--top', type=int, default=20)
    dashboard.add_argument('--no-decay', action='store_true')
    dashboard.set_defaults(func=cmd_dashboard)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command in {'test', 'validate', 'diagnose'} and not args.expr and not args.template:
        parser.error(f'{args.command} requires --expr or --template')
    args.func(args)


if __name__ == '__main__':
    main()
