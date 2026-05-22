"""
FactorExpressionInspector - 因子表达式静态体检

在执行昂贵回测前检查表达式语法、未知变量/函数、窗口参数和复杂度。
"""
import ast
import re
from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass(frozen=True)
class InspectionIssue:
    severity: str
    message: str


class FactorExpressionInspector:
    """因子表达式静态检查器"""

    DEFAULT_FIELDS = {
        'open', 'high', 'low', 'close', 'volume', 'amount', 'vwap', 'ret1',
        'csi_500_open', 'csi_500_high', 'csi_500_low', 'csi_500_close',
        'csi_500_volume', 'csi_500_amount', 'csi_500_weight', 'csi_500_ret1',
    }
    DEFAULT_FUNCTIONS = {
        'abs', 'min', 'max', 'np.where',
        'at_mask', 'at_nan2zero', 'at_zero2nan',
        'cs_booksize', 'cs_group_quantile', 'cs_indneut', 'cs_rank', 'cs_zscore',
        'ts_correlation', 'ts_delay', 'ts_delta', 'ts_fill', 'ts_kurtosis',
        'ts_max', 'ts_mean', 'ts_min', 'ts_ols', 'ts_rank', 'ts_regression',
        'ts_ret', 'ts_skewness', 'ts_std', 'ts_sum', 'ts_zscore',
    }
    WINDOW_FUNCTIONS = {
        'ts_correlation', 'ts_delay', 'ts_delta', 'ts_kurtosis', 'ts_max',
        'ts_mean', 'ts_min', 'ts_ols', 'ts_rank', 'ts_regression', 'ts_ret',
        'ts_skewness', 'ts_std', 'ts_sum', 'ts_zscore',
    }

    def __init__(self, fields: Set[str] = None, functions: Set[str] = None):
        self.fields = fields or self.DEFAULT_FIELDS
        self.functions = functions or self.DEFAULT_FUNCTIONS

    def inspect(self, expr: str) -> Dict:
        issues: List[InspectionIssue] = []
        tree = self._parse(expr, issues)
        if tree is None:
            return self._result(expr, issues, complexity=0, names=set(), calls=[])

        names = self._names(tree)
        calls = self._calls(tree)
        complexity = self._complexity(tree, calls)

        self._check_unknown_names(names, calls, issues)
        self._check_unknown_calls(calls, issues)
        self._check_windows(calls, issues)
        self._check_complexity(complexity, calls, issues)
        self._check_constants(expr, issues)

        return self._result(expr, issues, complexity=complexity, names=names, calls=calls)

    def print_report(self, report: Dict) -> None:
        print("\n" + "=" * 70)
        print("🔎 因子表达式体检")
        print("=" * 70)
        print(f"表达式: {report['expr']}")
        print(f"状态: {report['status']} | 复杂度: {report['complexity']} | 函数调用: {len(report['calls'])}")
        print(f"字段: {', '.join(sorted(report['fields'])) or '-'}")
        print(f"函数: {', '.join(report['calls']) or '-'}")
        if report['issues']:
            print("\n问题:")
            for issue in report['issues']:
                print(f"  [{issue['severity']}] {issue['message']}")
        else:
            print("\n未发现静态问题，可进入回测。")
        print("=" * 70 + "\n")

    def _parse(self, expr: str, issues: List[InspectionIssue]):
        try:
            return ast.parse(expr, mode='eval')
        except SyntaxError as error:
            issues.append(InspectionIssue('error', f'语法错误: {error.msg}'))
            return None

    def _names(self, tree: ast.AST) -> Set[str]:
        return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    def _calls(self, tree: ast.AST) -> List[str]:
        calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                calls.append(f'{node.func.value.id}.{node.func.attr}')
            else:
                calls.append('<dynamic>')
        return calls

    def _complexity(self, tree: ast.AST, calls: List[str]) -> int:
        binary_ops = sum(isinstance(node, ast.BinOp) for node in ast.walk(tree))
        comparisons = sum(isinstance(node, ast.Compare) for node in ast.walk(tree))
        conditionals = sum(isinstance(node, ast.IfExp) for node in ast.walk(tree))
        return len(calls) * 3 + binary_ops * 2 + comparisons * 2 + conditionals * 4

    def _check_unknown_names(self, names: Set[str], calls: List[str], issues: List[InspectionIssue]) -> None:
        call_roots = {call.split('.')[0] for call in calls if call != '<dynamic>'}
        allowed_roots = self.fields | {'np'} | call_roots
        unknown = sorted(name for name in names if name not in allowed_roots)
        for name in unknown:
            issues.append(InspectionIssue('error', f'未知字段或变量: {name}'))

    def _check_unknown_calls(self, calls: List[str], issues: List[InspectionIssue]) -> None:
        for call in calls:
            if call == '<dynamic>':
                issues.append(InspectionIssue('error', '不支持动态函数调用'))
            elif call not in self.functions:
                issues.append(InspectionIssue('error', f'未知函数: {call}'))

    def _check_windows(self, calls: List[str], issues: List[InspectionIssue]) -> None:
        expr_calls = set(calls)
        if expr_calls & self.WINDOW_FUNCTIONS:
            return
        issues.append(InspectionIssue('warning', '表达式未使用时序窗口函数，可能缺少时间维度信息'))

    def _check_complexity(self, complexity: int, calls: List[str], issues: List[InspectionIssue]) -> None:
        if complexity > 80:
            issues.append(InspectionIssue('warning', f'表达式复杂度较高({complexity})，建议拆分或降低嵌套'))
        if len(calls) > 20:
            issues.append(InspectionIssue('warning', '函数调用过多，可能增加过拟合和计算成本'))

    def _check_constants(self, expr: str, issues: List[InspectionIssue]) -> None:
        constants = [float(value) for value in re.findall(r'(?<![A-Za-z_])-?\d+\.\d+|-?\b\d+\b', expr)]
        large = [value for value in constants if abs(value) > 252]
        if large:
            issues.append(InspectionIssue('warning', f'发现较大常数 {large[:5]}，请确认是否合理'))
        tiny = [value for value in constants if 0 < abs(value) < 1e-6]
        if tiny:
            issues.append(InspectionIssue('warning', '发现极小常数，可能导致数值不稳定'))

    def _result(self, expr: str, issues: List[InspectionIssue], complexity: int, names: Set[str], calls: List[str]) -> Dict:
        serialized = [{'severity': issue.severity, 'message': issue.message} for issue in issues]
        status = 'fail' if any(issue.severity == 'error' for issue in issues) else 'warn' if issues else 'pass'
        return {
            'expr': expr,
            'status': status,
            'complexity': complexity,
            'fields': sorted(name for name in names if name in self.fields),
            'calls': calls,
            'issues': serialized,
        }


if __name__ == '__main__':
    inspector = FactorExpressionInspector()
    inspector.print_report(inspector.inspect('ts_correlation(close, volume, 20)'))
