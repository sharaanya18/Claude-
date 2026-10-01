"""Local re-implementation of the grounded-formulation metric (see CHALLENGE.md, "Evaluation").

case score = (value + counterfactual + structure) / 3, metric = mean over cases.

EXACT parts (follow the description verbatim):
  * value term: pred optimum on the real numbers within rel 1e-4 of the reference optimum;
  * counterfactual term semantics: share of variants (all numbers scaled by per-value factors in
    [0.55,0.95] U [1.05,1.45], same-valued numbers share a factor, whole years 1990..2030 fixed,
    typed-in constants of the program NOT scaled) on which pred and reference optima agree to rel 1e-6;
    variants whose reference has no finite optimum are skipped and replaced by the next one;
  * invalid program (unparsable / nonlinear / unknown reference) -> 0 on all terms; parseable but
    infeasible/unbounded -> value 0, counterfactual 0, structure still scored;
  * row canonicalisation (<= form or equality up to sign), (variable, coefficient) pairs, RHS, objective
    direction + pairs, multiset F1.

APPROXIMATIONS (the hidden grader code is not available):
  * the hidden counterfactual factors are unknown; we draw our own (seeded) ones. Counterfactual values
    therefore agree with the grader only in expectation (3 variants -> noisy, steps of 1/3);
  * the renaming: initial pairing by colour refinement (occurrence signature = rows, coefficients, then 3
    rounds of co-occurring-variable colours; coarser rounds pair leftovers), then pairwise-exchange hill
    climbing on the number of matched items. The grader's exact signature / tie-breaking differs, but both
    aim at the maximum-agreement renaming, so scores agree whenever the optimum is found;
  * the int/bin item is a single item, emitted for a model only if it has at least one int or bin
    variable (deduced: an always-present item would lift the one-line baseline well above its reported 0.3368);
  * equality rows are compared up to overall sign after renaming (instead of 'first coefficient positive');
  * a pred row/objective coefficient tolerance of 9 significant digits is used for equality of numbers.
"""
from __future__ import annotations

import math
import random
from collections import Counter

import formulation as F

VALUE_TOL = 1e-4
CF_TOL = 1e-6
N_VARIANTS = 3
MAX_VARIANT_TRIES = 60
# Open question: is the int/bin item emitted when a model has no int/bin variable at all? Default False
# (see module docstring). Flip to test the alternative reading.
INT_ITEM_ALWAYS = False
FACTOR_LO, FACTOR_HI, EXCL_LO, EXCL_HI = 0.55, 1.45, 0.95, 1.05


# ----------------------------------------------------------------------------- counterfactual variants
def draw_factor(rng: random.Random) -> float:
    """Uniform on [0.55,1.45] excluding (0.95,1.05)."""
    lo_len, hi_len = EXCL_LO - FACTOR_LO, FACTOR_HI - EXCL_HI
    u = rng.random() * (lo_len + hi_len)
    return FACTOR_LO + u if u < lo_len else EXCL_HI + (u - lo_len)


def is_year(v: float) -> bool:
    return float(v) == int(v) and 1990 <= int(v) <= 2030


def make_variant(numbers: dict, rng: random.Random) -> dict:
    """Scale every number by its own factor; equal values share a factor; years are unchanged."""
    factors: dict = {}
    out = {}
    for ref, v in sorted(numbers.items()):
        v = float(v)
        if is_year(v):
            out[ref] = v
            continue
        if v not in factors:
            factors[v] = draw_factor(rng)
        out[ref] = v * factors[v]
    return out


def make_variants(ref_text: str, numbers: dict, seed: int = 0, n: int = N_VARIANTS):
    """First n seeded variants on which the reference has a finite optimum, with that optimum."""
    rng = random.Random(seed)
    out = []
    for _ in range(MAX_VARIANT_TRIES):
        vnum = make_variant(numbers, rng)
        ro = F.optimum(ref_text, vnum)
        if ro is not None:
            out.append((vnum, ro))
            if len(out) == n:
                break
    return out


# ----------------------------------------------------------------------------- structure term
def _fmt(x: float) -> str:
    x = float(x)
    if x == 0:
        return "0"
    return "%.9g" % x


def _items(model: F.Model):
    """Canonical items of a model: list of (kind, sense/dir, {var: coef str}, rhs str, signs-needed).

    kind 'obj' : objective (dir, pairs); 'le' : sum <= rhs; 'eq' : sum = rhs (sign resolved after renaming);
    'int' : the integrality item (sets of vars).
    """
    items = []
    items.append(("obj", model.sense, {v: float(c) for v, c in model.obj.items() if c != 0}, 0.0))
    for coefs, s, rhs in model.rows:
        cc = {v: float(c) for v, c in coefs.items() if c != 0}
        if s == ">=":
            cc = {v: -c for v, c in cc.items()}
            rhs = -rhs
            s = "<="
        items.append(("le" if s == "<=" else "eq", s, cc, float(rhs)))
    if INT_ITEM_ALWAYS or model.int_vars or model.bin_vars:
        items.append(("int", "", {v: 1.0 for v in model.int_vars}, 0.0, {v: 1.0 for v in model.bin_vars}))
    return items


def _item_vars(it):
    vs = set(it[2])
    if it[0] == "int":
        vs |= set(it[4])
    return vs


def _item_key(it, f, unp):
    """Key of an item under renaming f (var -> name or None); unpaired variables get unique names."""
    def nm(v):
        t = f.get(v)
        return t if t is not None else ("~", unp[v])

    kind = it[0]
    if kind == "int":
        return ("int", tuple(sorted(repr(nm(v)) for v in it[2])), tuple(sorted(repr(nm(v)) for v in it[4])))
    pairs = tuple(sorted((repr(nm(v)), _fmt(c)) for v, c in it[2].items()))
    if kind == "obj":
        return ("obj", it[1], pairs)
    if kind == "le":
        return ("le", pairs, _fmt(it[3]))
    neg = tuple(sorted((repr(nm(v)), _fmt(-c)) for v, c in it[2].items()))
    return ("eq",) + min((pairs, _fmt(it[3])), (neg, _fmt(-it[3])))


def _colour_rounds(items, vars_, table, rounds=3):
    """Colour refinement; returns list (len rounds+1) of {var: colour int}. `table` is shared by both models."""
    def intern(t):
        return table.setdefault(t, len(table))

    occ = {v: [] for v in vars_}
    for k, it in enumerate(items):
        for v in _item_vars(it):
            occ[v].append(k)

    def rowsig(it):
        return (it[0], it[1], _fmt(it[3]) if it[0] in ("le", "eq") else "")

    def coef(it, v):
        return _fmt(it[2].get(v, 0.0)) if it[0] != "int" else ("b" if v in it[4] else "i")

    col0 = {v: intern(("v0", tuple(sorted((rowsig(items[k]), coef(items[k], v)) for k in occ[v])))) for v in vars_}
    cols = [col0]
    for _ in range(rounds):
        prev = cols[-1]
        nxt = {}
        for v in vars_:
            sig = []
            for k in occ[v]:
                it = items[k]
                nb = tuple(sorted(prev[u] for u in _item_vars(it) if u != v))
                sig.append((rowsig(it), coef(it, v), nb))
            nxt[v] = intern((prev[v], tuple(sorted(sig))))
        cols.append(nxt)
    return cols


def structure_score(ref_model: F.Model, pred_model: F.Model, return_details: bool = False):
    """F1 between canonical items under the best one-to-one variable renaming (approximate search)."""
    ri, pi = _items(ref_model), _items(pred_model)
    n_r, n_p = len(ri), len(pi)
    rv = list(ref_model.variables)
    pv = list(pred_model.variables)
    ident = {v: v for v in rv}
    rc = Counter(_item_key(it, ident, {}) for it in ri)

    # --- initial pairing by colour refinement (coarser rounds pair what finer rounds could not)
    table: dict = {}
    rcols = _colour_rounds(ri, rv, table)
    pcols = _colour_rounds(pi, pv, table)
    mapping = {v: None for v in pv}
    used = set()
    for r in range(len(rcols) - 1, -1, -1):
        groups_r, groups_p = {}, {}
        for v in rv:
            if v not in used:
                groups_r.setdefault(rcols[r][v], []).append(v)
        for v in pv:
            if mapping[v] is None:
                groups_p.setdefault(pcols[r][v], []).append(v)
        for c, plist in groups_p.items():
            for a, b in zip(plist, groups_r.get(c, [])):
                mapping[a] = b
                used.add(b)
    left_p = [v for v in pv if mapping[v] is None]
    left_r = [v for v in rv if v not in used]
    for a, b in zip(left_p, left_r):
        mapping[a] = b
        used.add(b)
    unp = {v: i for i, v in enumerate(pv)}

    # --- incremental matching count
    var_items = {v: [] for v in pv}
    for k, it in enumerate(pi):
        for v in _item_vars(it):
            var_items[v].append(k)
    keys = [_item_key(it, mapping, unp) for it in pi]
    pc = Counter(keys)
    m = sum(min(c, rc[k]) for k, c in pc.items())

    def remove(k):
        nonlocal m
        key = keys[k]
        if pc[key] <= rc[key]:
            m -= 1
        pc[key] -= 1

    def add(k, key):
        nonlocal m
        keys[k] = key
        pc[key] += 1
        if pc[key] <= rc[key]:
            m += 1

    def apply(changes):
        """changes: {pred var: new target}. Returns the old targets for reverting."""
        old = {a: mapping[a] for a in changes}
        aff = sorted({k for a in changes for k in var_items[a]})
        for k in aff:
            remove(k)
        mapping.update(changes)
        for k in aff:
            add(k, _item_key(pi[k], mapping, unp))
        return old

    best_possible = min(n_r, n_p)
    for _ in range(20):
        if m >= best_possible:
            break
        improved = False
        # pairwise exchanges between pred variables (a target may be None = unpaired)
        for i in range(len(pv)):
            for j in range(i + 1, len(pv)):
                a, b = pv[i], pv[j]
                if mapping[a] is None and mapping[b] is None:
                    continue
                before = m
                old = apply({a: mapping[b], b: mapping[a]})
                if m > before:
                    improved = True
                else:
                    apply(old)
            if m >= best_possible:
                break
        # re-assign to a reference variable currently unused
        free = [v for v in rv if v not in set(mapping.values())]
        for a in pv:
            if not free or m >= best_possible:
                break
            for u in list(free):
                before = m
                old_t = mapping[a]
                apply({a: u})
                if m > before:
                    free.remove(u)
                    if old_t is not None:
                        free.append(old_t)
                    improved = True
                    break
                apply({a: old_t})
        if not improved:
            break
    f1 = 0.0 if (n_r == 0 or n_p == 0 or m == 0) else 2.0 * m / (n_r + n_p)
    return (f1, m, n_p, n_r) if return_details else f1


# ----------------------------------------------------------------------------- case / corpus score
def _agree(a, b, tol):
    return a is not None and b is not None and F.rel_close(a, b, tol)


def case_score(ref_text: str, pred_text: str, numbers, ref_opt=None, variants=None, seed: int = 0) -> dict:
    """Score one case. `variants` (optional) = list of (numbers_dict, ref_optimum) to override our own draw."""
    nums = F.normalize_numbers(numbers)
    zero = {"value": 0.0, "cf": 0.0, "structure": 0.0, "score": 0.0, "valid": False}
    try:
        pm = F.parse(pred_text, nums)
    except F.FormulationError:
        return zero
    rm = F.parse(ref_text, nums)  # a broken reference is a bug in the caller
    if ref_opt is None:
        ref_opt = F.solve(rm)
    p_opt = F.solve(pm)
    value = 1.0 if _agree(p_opt, ref_opt, VALUE_TOL) else 0.0
    if variants is None:
        variants = make_variants(ref_text, nums, seed=seed)
    hits = 0
    for vnum, ro in variants:
        try:
            po = F.optimum(pred_text, vnum)
        except F.FormulationError:
            po = None
        hits += _agree(po, ro, CF_TOL)
    if variants:
        cf = hits / len(variants)
    else:
        # LOCAL FALLBACK (not in the description): none of our own random variants kept the reference finite
        # (about 3% of seed cases have equality/bound chains that need correlated factors). The grader's hidden
        # factors are guaranteed feasible for evaluation cases, so use the value term as a neutral stand-in.
        cf = value
    st = structure_score(rm, pm)
    return {"value": value, "cf": cf, "structure": st, "score": (value + cf + st) / 3.0,
            "valid": True, "pred_opt": p_opt, "n_variants": len(variants)}


def metric(refs, preds, numbers_list, ref_opts=None, seed: int = 0) -> float:
    """Mean case score over aligned lists (ref formulation, predicted formulation, numbers_json)."""
    tot = 0.0
    for i, (r, p, n) in enumerate(zip(refs, preds, numbers_list)):
        tot += case_score(r, p, n, None if ref_opts is None else ref_opts[i], seed=seed + i)["score"]
    return tot / max(1, len(refs))
