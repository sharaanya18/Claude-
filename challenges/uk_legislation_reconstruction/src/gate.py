"""Edit gate: features, and the sequential walk that both trains and runs it.

The reconstruction is a sequence of decisions, not a set: whether an edit is worth applying
depends on what has already been applied (its anchor may only exist afterwards, or an earlier
edit may already have made the same change).  So the gate is trained by imitation of an
oracle walk - at each edit the oracle applies it iff the exact challenge metric improves, and
the features the model sees are the ones available at that moment, on the text as the oracle
has left it.  At inference the same walk runs with the model's decision in the oracle's place.
"""
import re, sys
sys.path.insert(0, '/home/user/Claude-/challenges/uk_legislation_reconstruction/src')
from apply import markers, span_of, _find, _bounds, apply_one
from metric import mtok, edits, f1

OPS = ["sub", "omit", "after", "before", "omitspan", "subspan", "omitnear", "omitlast", "subprov",
       "insprov", "omitprov", "subdefn", "insdefn", "omitdefn", "approp", "endins"]
GFEATS = (["op_" + o for o in OPS] +
          ["n_old", "n_new", "found_scope", "found_glob", "n_occ", "depth", "resolved",
           "every", "has_defn", "defn_found", "prov_score", "prov_days", "prov_tbl",
           "prov_nins", "prov_len", "ins_idx", "ins_n", "en_len", "n_prov", "chg_frac",
           "new_marker", "new_marker_next", "cur_len_ratio", "applied", "osec",
           "quoted_old", "new_in_cur", "del_frac", "ins_frac",
           # sequence state: what the walk has already done
           "seq_idx", "n_done", "dup_exact", "dup_old", "new_already", "old_gone",
           "prov_rank", "plan_thr"])


def gate_row(e, s, ctx, done):
    """Feature vector for instruction `e` against the current text `s`.

    `done` is the list of (op, old, new) already applied in this walk, which is what makes the
    duplicate and already-made-this-change signals available.
    """
    f = {k: 0.0 for k in GFEATS}
    if "op_" + e["op"] in f:
        f["op_" + e["op"]] = 1.0
    old = (e.get("old") or e.get("a") or "")
    new = (e.get("new") or "")
    f["n_old"] = len(old.split()); f["n_new"] = len(new.split())
    mk = markers(s)
    chain = e.get("chain") or []
    f["depth"] = len(chain)
    f["resolved"] = 1.0 if (not chain or span_of(s, chain, mk) is not None) else 0.0
    lo, hi = _bounds(s, chain, mk)
    if old:
        f["found_scope"] = 1.0 if _find(s[lo:hi], old) >= 0 else 0.0
        f["found_glob"] = 1.0 if _find(s, old) >= 0 else 0.0
        f["n_occ"] = min(20.0, float(s.count(old)))
        f["old_gone"] = 0.0 if f["found_glob"] else 1.0
    f["every"] = 1.0 if e.get("every") else 0.0
    if e.get("defn"):
        f["has_defn"] = 1.0
        f["defn_found"] = 1.0 if ('"%s"' % e["defn"]) in s else 0.0
    if e.get("tags"):
        f["resolved"] = 1.0 if span_of(s, chain + [e["tags"][-1]], mk) is not None else 0.0
    for k in ("prov_score", "prov_days", "prov_tbl", "prov_nins", "prov_len",
              "ins_idx", "ins_n", "en_len", "n_prov", "prov_rank", "plan_thr"):
        f[k] = ctx.get(k, 0.0)
    f["chg_frac"] = (len(old.split()) + len(new.split())) / max(1.0, ctx["en_len"])
    m = re.match(r"\(([0-9A-Za-z]{1,4})\)", new)
    if m:
        f["new_marker"] = 1.0
        f["new_marker_next"] = 0.0 if any(t == m.group(1) for t, _, _, _ in mk) else 1.0
    f["cur_len_ratio"] = len(s) / max(1.0, ctx["en_chars"])
    f["osec"] = float(e.get("osec") or 0.0)
    f["quoted_old"] = 1.0 if old else 0.0
    f["new_in_cur"] = 1.0 if (new and new[:40] in s) else 0.0
    f["seq_idx"] = float(ctx.get("seq_idx", 0.0))
    f["n_done"] = float(len(done))
    key = (e["op"], old, new)
    f["dup_exact"] = 1.0 if key in done else 0.0
    f["dup_old"] = 1.0 if (old and any(d[1] == old for d in done)) else 0.0
    f["new_already"] = 1.0 if (len(new) > 8 and new[:40] in s) else 0.0
    s2, ok = apply_one(s, e)
    f["applied"] = 1.0 if (ok and s2 != s) else 0.0
    d = len(s2) - len(s)
    f["ins_frac"] = max(0.0, d) / max(1.0, ctx["en_chars"])
    f["del_frac"] = max(0.0, -d) / max(1.0, ctx["en_chars"])
    return [f[k] for k in GFEATS], s2, ok


def _tidy(s):
    return re.sub(r"\s+", " ", s).strip()


def walk(enacted, instr, ctxs, decide):
    """Run the plan, asking `decide(row)` whether to keep each edit.

    Returns (feature rows, decisions, final text).  Rejected edits leave the text untouched,
    so later features describe the text the accepted edits actually produced.
    """
    s = enacted
    done = []
    X, took = [], []
    for j, (e, ctx) in enumerate(zip(instr, ctxs)):
        c = dict(ctx); c["seq_idx"] = float(j)
        row, s2, ok = gate_row(e, s, c, done)
        X.append(row)
        keep = decide(row, e, s, s2, ok)
        took.append(bool(keep))
        if keep and ok and s2 != s:
            s = s2
            done.append((e["op"], (e.get("old") or e.get("a") or ""), (e.get("new") or "")))
    return X, took, _tidy(s)


def oracle_walk(enacted, instr, ctxs, true_text):
    """Teacher walk: keep an edit iff it improves the exact metric at its turn."""
    en = mtok(enacted)
    et, kt = edits(en, mtok(true_text))

    def score(txt):
        ep, kp = edits(en, mtok(txt))
        return 0.0 if kp < 0.5 * kt else f1(ep, et)

    cur = [score(enacted)]

    def decide(row, e, s, s2, ok):
        if not ok or s2 == s:
            return False
        v = score(_tidy(s2))
        if v > cur[0] + 1e-12:
            cur[0] = v
            return True
        return False

    X, took, out = walk(enacted, instr, ctxs, decide)
    return X, [1 if t else 0 for t in took], out, cur[0]


def model_walk(enacted, instr, ctxs, predict, thr):
    """Inference walk: the model stands in for the oracle."""
    def decide(row, e, s, s2, ok):
        if not ok or s2 == s:
            return False
        return predict(row) >= thr
    return walk(enacted, instr, ctxs, decide)
