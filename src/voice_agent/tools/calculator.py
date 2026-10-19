"""Calculations in the Windows Calculator: open it, press the keys so the user sees them, read the result.

The expression is also worked out here, to check what the calculator shows.
"""

from __future__ import annotations

import ast
import operator
import re
import time
from functools import partial

from voice_agent.tools.base import Tool, ToolResult

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_KEYS = {"+": "k:add", "-": "k:subtract", "*": "k:multiply", "/": "k:divide", ".": "k:decimal"}
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def normalize(expression: str) -> str:
    """'٢٥ × ٤', '25 x 4', '10%' -> '25*4', '25*4', '10/100'. Spaces and thousands commas go.
    Note: a comma is read as a thousands separator ("1,500" -> 1500)."""
    e = expression.translate(_DIGITS).replace("٫", ".").replace("٬", "").replace(",", "")
    e = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", e)
    for a, b in (("×", "*"), ("÷", "/"), ("−", "-"), ("–", "-"), ("%", "/100"), (" ", "")):
        e = e.replace(a, b)
    # a minus right after an operator ("2*-3") would replace that operator in the calculator
    return re.sub(r"(?<=[*/+\-(])-(\d+(?:\.\d+)?)", r"(0-\1)", e.lstrip("+"))


def evaluate(expr: str) -> float:
    """Safe arithmetic: numbers, + - * /, brackets. Raises ValueError for anything else."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -ev(n.operand) if isinstance(n.op, ast.USub) else ev(n.operand)
        raise ValueError("only numbers, + - * / and brackets")
    if not re.fullmatch(r"[0-9.+\-*/()]+", expr):
        raise ValueError("only numbers, + - * / and brackets")
    return float(ev(ast.parse(expr, mode="eval")))


def key_tokens(expr: str) -> list[str]:
    """Keys for the calculator: clear, scientific mode when the order of operations matters, the
    expression, then Enter. Standard mode works left to right (2+3*4 would give 20)."""
    add_sub = any(c in "+-" and i > 0 and expr[i - 1] not in "*/(" for i, c in enumerate(expr))
    scientific = "(" in expr or (add_sub and any(c in "*/" for c in expr))
    tokens = ["k:escape"] + (["k:alt+2"] if scientific else [])
    for ch in expr:
        tokens.append(f"k:{ch}" if ch.isdigit() else _KEYS.get(ch, f"c:{ch}"))
    return tokens + ["k:enter"]


def parse_display(name: str | None) -> float | None:
    """'Display is 1,234.5' / 'العرض هو ١٬٢٣٤٫٥' -> 1234.5."""
    if not name:
        return None
    s = re.sub(r"[‎‏‪-‮⁦-⁩]", "", name.translate(_DIGITS))
    s = s.replace("٫", ".").replace("٬", "").replace(",", "").replace("−", "-")
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", s)
    return float(nums[-1]) if nums else None


def fmt(x: float) -> str:
    return str(int(x)) if x == int(x) and abs(x) < 1e15 else f"{x:.10g}"


def calculator_window(tb):
    # This function finds the open Calculator, or opens it.
    from voice_agent.tools.launch_app import open_app

    wins = tb.find_windows("calculator")
    if wins:
        return wins[0]
    res = open_app(tb, "calculator")
    if not res.ok:
        return None
    wins = tb.find_windows("calculator")
    return wins[0] if wins else tb.desktop.foreground()


def calculate(tb, expression: str) -> ToolResult:
    # This function types the expression into Windows Calculator and reads the result.
    expr = normalize(expression)
    try:
        expected = evaluate(expr)
    except ZeroDivisionError:
        return ToolResult(False, {"expression": expr}, error="division by zero")
    except (ValueError, SyntaxError) as e:
        return ToolResult(False, {"expression": expression}, error=f"not a calculation: {e}")
    win = calculator_window(tb)
    if win is None:
        return ToolResult(False, {"expression": expr}, error="could not open the Calculator")
    tb.desktop.calculator_input(win, key_tokens(expr))
    time.sleep(0.3)
    shown = parse_display(tb.desktop.calculator_display(win))
    data = {"expression": expr, "result": fmt(shown if shown is not None else expected),
            "read_from_calculator": shown is not None}
    if shown is not None and abs(shown - expected) > 1e-9 * max(1.0, abs(expected)):
        data["check"] = f"Calculator shows {fmt(shown)}, but {expr} = {fmt(expected)}"
    return ToolResult(True, data)


def tools(tb) -> list[Tool]:
    return [Tool("calculate", "Do a calculation in the Windows Calculator (the user sees it typed in). "
                 "Use it for every math question. expression: digits and + - * / ( ), e.g. 25*4+3.",
                 {"expression": {"type": "string"}}, partial(calculate, tb), ["expression"], timeout_s=30)]
