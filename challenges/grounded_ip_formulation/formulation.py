"""Parser, evaluator and solver for the grounded-IP formulation grammar (see CHALLENGE.md).

This module is a *tool* (grader re-implementation), not a model: it is used for
  * the local metric (metric.py),
  * the weak-supervision verification of candidate formulations during training
    (`matches_optimum`), and
  * validity filtering of sampled candidates at inference (`analyze`).

Grammar implemented (strict reading of the description):
  program     := statement (';' statement)*            (empty statements are skipped)
  objective   := ('max'|'min'|'maximize'|'minimize') [':'] expr     (must be the FIRST statement)
  constraint  := expr ('<=' | '>=' | '=') expr
  declaration := ('int'|'bin') name ((',')? name)*
  expr        := term (('+'|'-') term)* ; term := factor (('*'|'/') factor)* ;
  factor      := ('+'|'-') factor | number | ref | name | '(' expr ')'
  number      := digits with optional thousands commas and decimals, optional trailing '%' (/100)
  ref         := x<k>   -> value taken from numbers_json
  name        := [A-Za-z_][A-Za-z0-9_]*, not of the form x<k> or q<k>
A product may have at most one side containing variables; a quotient needs a variable-free,
non-zero denominator. All variables are >= 0; 'bin' adds the bound <= 1 and integrality.
Limits: <= 60 variables, <= 150 constraints.

Assumptions that the description does not pin down (flagged for the metric_spec open questions):
  * keywords are matched case-insensitively; implicit multiplication ("2x", "3(x+y)") is a syntax error;
  * '<', '>', '==' are not comparators (syntax error); chained comparisons are errors;
  * an identifier such as `x4cake_a` (as in the example in the description) is the VARIABLE name
    "x4cake_a", because only the exact form x<k> is a reference;
  * variables that are only declared (int/bin) count as variables (and toward the 60 limit);
  * a constant constraint (no variables) is kept as a row; if it is violated the model is infeasible.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

import numpy as np

MAX_VARS = 60
MAX_CONS = 150

# optimality gap used for MILP solves. scipy's default (1e-4) is looser than the 1e-6 counterfactual tolerance.
MIP_REL_GAP = 1e-9
# deterministic work cap per MILP (a node count, never a clock) so a degenerate sampled program cannot stall the run
MIP_NODE_LIMIT = 20000


class FormulationError(ValueError):
    """Raised for anything the grader would reject: syntax, nonlinearity, unknown reference, limits."""


# --------------------------------------------------------------------------- tokenizer
_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?%?|\.\d+%?"
_TOKEN_RE = re.compile(
    r"\s*(?:(?P<num>" + _NUM + r")|(?P<id>[A-Za-z_][A-Za-z0-9_]*)|(?P<op><=|>=|=|\+|-|\*|/|\(|\)|,|:))"
)
_REF_RE = re.compile(r"x\d+\Z")
_Q_RE = re.compile(r"q\d+\Z")
_CMP = ("<=", ">=", "=")


def _tokenize(stmt: str):
    toks, pos, n = [], 0, len(stmt)
    while pos < n:
        if stmt[pos:].strip() == "":
            break
        m = _TOKEN_RE.match(stmt, pos)
        if not m:
            raise FormulationError(f"cannot tokenize near {stmt[pos:pos + 20]!r}")
        if m.group("num") is not None:
            toks.append(("num", m.group("num")))
        elif m.group("id") is not None:
            toks.append(("id", m.group("id")))
        else:
            toks.append(("op", m.group("op")))
        pos = m.end()
    return toks


def _num_value(s: str) -> float:
    pct = s.endswith("%")
    if pct:
        s = s[:-1]
    v = float(s.replace(",", ""))
    return v / 100.0 if pct else v


# --------------------------------------------------------------------------- linear expressions
@dataclass
class Lin:
    coefs: dict = field(default_factory=dict)  # variable -> coefficient
    const: float = 0.0

    def is_const(self) -> bool:
        return not self.coefs

    def add(self, other: "Lin", sign: float = 1.0) -> "Lin":
        out = Lin(dict(self.coefs), self.const + sign * other.const)
        for v, c in other.coefs.items():
            out.coefs[v] = out.coefs.get(v, 0.0) + sign * c
        return out

    def scale(self, k: float) -> "Lin":
        return Lin({v: c * k for v, c in self.coefs.items()}, self.const * k)


class _Parser:
    def __init__(self, toks, numbers, var_registry):
        self.t, self.i, self.numbers, self.vars = toks, 0, numbers, var_registry

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def expr(self) -> Lin:
        acc = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()[1]
            acc = acc.add(self.term(), 1.0 if op == "+" else -1.0)
        return acc

    def term(self) -> Lin:
        acc = self.factor()
        while self.peek() in (("op", "*"), ("op", "/")):
            op = self.take()[1]
            rhs = self.factor()
            if op == "*":
                if acc.coefs and rhs.coefs:
                    raise FormulationError("nonlinear product (variables on both sides)")
                acc = rhs.scale(acc.const) if not acc.coefs else acc.scale(rhs.const)
            else:
                if rhs.coefs:
                    raise FormulationError("nonlinear quotient (variable in denominator)")
                if rhs.const == 0:
                    raise FormulationError("division by zero")
                acc = acc.scale(1.0 / rhs.const)
        return acc

    def factor(self) -> Lin:
        kind, val = self.take()
        if kind is None:
            raise FormulationError("unexpected end of expression")
        if kind == "op":
            if val in ("+", "-"):
                f = self.factor()
                return f.scale(1.0 if val == "+" else -1.0)
            if val == "(":
                e = self.expr()
                if self.take() != ("op", ")"):
                    raise FormulationError("missing ')'")
                return e
            raise FormulationError(f"unexpected operator {val!r}")
        if kind == "num":
            return Lin({}, _num_value(val))
        # identifier: reference or variable
        if _REF_RE.match(val):
            if val not in self.numbers:
                raise FormulationError(f"reference {val} does not exist")
            return Lin({}, float(self.numbers[val]))
        _check_var_name(val)
        self.vars.setdefault(val, len(self.vars))
        return Lin({val: 1.0}, 0.0)


def _check_var_name(name: str):
    if _REF_RE.match(name) or _Q_RE.match(name):
        raise FormulationError(f"illegal variable name {name!r}")


# --------------------------------------------------------------------------- model
@dataclass
class Model:
    sense: str                      # 'max' | 'min'
    obj: dict                       # variable -> coefficient
    obj_const: float
    rows: list                      # list of (coefs dict, sense in {'<=','>=','='}, rhs float)
    int_vars: set
    bin_vars: set
    variables: list                 # in order of first appearance


def normalize_numbers(numbers) -> dict:
    """Accept numbers_json as a JSON string, a list of [ref, value] or a dict; return {ref: float}."""
    if isinstance(numbers, str):
        numbers = json.loads(numbers)
    if isinstance(numbers, dict):
        return {str(k): float(v) for k, v in numbers.items()}
    return {str(k): float(v) for k, v in numbers}


def parse(text: str, numbers) -> Model:
    """Parse a formulation against `numbers` (numbers_json). Raises FormulationError if the grader would reject it."""
    try:
        return _parse(text, numbers)
    except RecursionError:      # absurdly deep nesting in a sampled program: just another invalid program
        raise FormulationError("expression nested too deeply")


def _parse(text: str, numbers) -> Model:
    if not isinstance(text, str):
        raise FormulationError("formulation is not a string")
    numbers = normalize_numbers(numbers)
    stmts = [s for s in text.split(";") if s.strip() != ""]
    if not stmts:
        raise FormulationError("empty program")
    registry: dict = {}
    sense = obj = None
    obj_const = 0.0
    rows, int_vars, bin_vars = [], set(), set()
    for si, stmt in enumerate(stmts):
        toks = _tokenize(stmt)
        if not toks:
            continue
        head = toks[0][1].lower() if toks[0][0] == "id" else None
        if si == 0:
            if head not in ("max", "maximize", "min", "minimize"):
                raise FormulationError("first statement must be the objective (max/min)")
            sense = "max" if head.startswith("max") else "min"
            rest = toks[1:]
            if rest and rest[0] == ("op", ":"):
                rest = rest[1:]
            if not rest:
                raise FormulationError("empty objective")
            p = _Parser(rest, numbers, registry)
            lin = p.expr()
            if p.i != len(rest):
                raise FormulationError("trailing tokens in objective")
            obj, obj_const = lin.coefs, lin.const
            continue
        if head in ("int", "bin") and not any(t[0] == "op" and t[1] in _CMP for t in toks):
            names = []
            expect_name = True
            for kind, val in toks[1:]:
                if kind == "id" and (expect_name or True):
                    _check_var_name(val)
                    names.append(val)
                    expect_name = False
                elif (kind, val) == ("op", ","):
                    expect_name = True
                else:
                    raise FormulationError("bad token in declaration")
            if not names:
                raise FormulationError("empty declaration")
            for nme in names:
                registry.setdefault(nme, len(registry))
                (int_vars if head == "int" else bin_vars).add(nme)
            continue
        cmp_idx = [k for k, t in enumerate(toks) if t[0] == "op" and t[1] in _CMP]
        if len(cmp_idx) != 1:
            raise FormulationError("constraint needs exactly one comparator")
        k = cmp_idx[0]
        lt, rt = toks[:k], toks[k + 1:]
        if not lt or not rt:
            raise FormulationError("empty side in constraint")
        pl, pr = _Parser(lt, numbers, registry), _Parser(rt, numbers, registry)
        left, right = pl.expr(), pr.expr()
        if pl.i != len(lt) or pr.i != len(rt):
            raise FormulationError("trailing tokens in constraint")
        diff = left.add(right, -1.0)           # left - right  (sense) 0
        rows.append((diff.coefs, toks[k][1], -diff.const))
        if len(rows) > MAX_CONS:
            raise FormulationError("too many constraints")
    if len(registry) > MAX_VARS:
        raise FormulationError("too many variables")
    bin_vars -= set()  # bin overrides int below
    int_vars = int_vars - bin_vars
    for row in rows:
        for v, c in row[0].items():
            if not math.isfinite(c):
                raise FormulationError("non-finite coefficient")
    return Model(sense, obj, obj_const, rows, int_vars, bin_vars, list(registry))


# --------------------------------------------------------------------------- solver
def solve(model: Model, mip_rel_gap: float = MIP_REL_GAP):
    """Return the optimal objective value (float) or None if infeasible / unbounded / solver failure."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    vars_ = model.variables
    n = len(vars_)
    idx = {v: i for i, v in enumerate(vars_)}
    sgn = -1.0 if model.sense == "max" else 1.0
    if n == 0:
        for coefs, s, rhs in model.rows:
            if (s == "<=" and 0 > rhs + 1e-9) or (s == ">=" and 0 < rhs - 1e-9) or (s == "=" and abs(rhs) > 1e-9):
                return None
        return float(model.obj_const)
    c = np.zeros(n)
    for v, k in model.obj.items():
        c[idx[v]] = sgn * k
    lb = np.zeros(n)
    ub = np.full(n, np.inf)
    integrality = np.zeros(n)
    for v in model.int_vars:
        integrality[idx[v]] = 1
    for v in model.bin_vars:
        integrality[idx[v]] = 1
        ub[idx[v]] = 1.0
    cons = []
    if model.rows:
        A = np.zeros((len(model.rows), n))
        lo = np.full(len(model.rows), -np.inf)
        hi = np.full(len(model.rows), np.inf)
        for r, (coefs, s, rhs) in enumerate(model.rows):
            for v, k in coefs.items():
                A[r, idx[v]] += k
            if s in ("<=", "="):
                hi[r] = rhs
            if s in (">=", "="):
                lo[r] = rhs
        cons = [LinearConstraint(A, lo, hi)]
    try:
        res = milp(c, constraints=cons, integrality=integrality, bounds=Bounds(lb, ub),
                   options={"mip_rel_gap": mip_rel_gap, "disp": False, "node_limit": MIP_NODE_LIMIT})
    except Exception:
        return None
    if res.status != 0 or res.x is None or not np.isfinite(res.fun):
        return None
    return float(sgn * res.fun + model.obj_const)


def optimum(text, numbers):
    """Optimal value of the program or None for unparsable / nonlinear / bad reference / infeasible / unbounded."""
    try:
        return solve(parse(text, numbers))
    except FormulationError:
        return None


def analyze(text, numbers) -> dict:
    """Validity report for candidate filtering: {'parsed','error','optimum','n_vars','n_rows'}."""
    try:
        m = parse(text, numbers)
    except FormulationError as e:
        return {"parsed": False, "error": str(e), "optimum": None, "n_vars": 0, "n_rows": 0}
    return {"parsed": True, "error": None, "optimum": solve(m), "n_vars": len(m.variables), "n_rows": len(m.rows)}


def rel_close(a, b, rel_tol: float) -> bool:
    """|a-b| <= rel_tol * |b| (b = reference), with a tiny absolute floor for reference values near zero."""
    return abs(a - b) <= rel_tol * abs(b) + 1e-12


def matches_optimum(text, numbers, optimal_value: float, rel_tol: float = 1e-4) -> bool:
    """Weak-supervision verifier: does the program's optimum on the real numbers equal the label?

    This is exactly the value term of the metric; use it as the reward / filter when searching for
    formulations of training cases that have only an optimal_value label.
    """
    v = optimum(text, numbers)
    return v is not None and rel_close(v, float(optimal_value), rel_tol)
