"""Features and labels for the learned edit gate."""
import re, sys
sys.path.insert(0, '/home/user/Claude-/challenges/uk_legislation_reconstruction/src')
from apply import markers, span_of, _find, _bounds, apply_one
from metric import mtok, edits, f1

OPS = ["sub", "omit", "after", "before", "omitspan", "subspan", "omitnear", "subprov",
       "insprov", "omitprov", "subdefn", "insdefn", "omitdefn", "approp", "endins"]
GFEATS = (["op_" + o for o in OPS] +
          ["n_old", "n_new", "found_scope", "found_glob", "n_occ", "depth", "resolved",
           "every", "has_defn", "defn_found", "prov_score", "prov_days", "prov_tbl",
           "prov_nins", "prov_len", "ins_idx", "ins_n", "en_len", "n_prov", "chg_frac",
           "new_marker", "new_marker_next", "cur_len_ratio", "applied", "osec",
           "quoted_old", "new_in_cur", "del_frac", "ins_frac"])


def gate_row(e, s, ctx):
    """Feature vector for instruction `e` against running text `s`."""
    f = {k: 0.0 for k in GFEATS}
    if "op_" + e["op"] in f:
        f["op_" + e["op"]] = 1.0
    old = e.get("old") or e.get("a") or ""
    new = e.get("new") or ""
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
    f["every"] = 1.0 if e.get("every") else 0.0
    if e.get("defn"):
        f["has_defn"] = 1.0
        f["defn_found"] = 1.0 if ('"%s"' % e["defn"]) in s else 0.0
    if e.get("tags"):
        f["resolved"] = 1.0 if span_of(s, chain + [e["tags"][-1]], mk) is not None else 0.0
    f["prov_score"] = ctx["prov_score"]; f["prov_days"] = ctx["prov_days"]
    f["prov_tbl"] = ctx["prov_tbl"]; f["prov_nins"] = ctx["prov_nins"]
    f["prov_len"] = ctx["prov_len"]; f["ins_idx"] = ctx["ins_idx"]; f["ins_n"] = ctx["ins_n"]
    f["en_len"] = ctx["en_len"]; f["n_prov"] = ctx["n_prov"]
    f["chg_frac"] = (len(old.split()) + len(new.split())) / max(1.0, ctx["en_len"])
    m = re.match(r"\(([0-9A-Za-z]{1,4})\)", new)
    if m:
        f["new_marker"] = 1.0
        f["new_marker_next"] = 0.0 if any(t == m.group(1) for t, _, _, _ in mk) else 1.0
    f["cur_len_ratio"] = len(s) / max(1.0, ctx["en_chars"])
    f["osec"] = float(e.get("osec") or 0.0)
    f["quoted_old"] = 1.0 if old else 0.0
    f["new_in_cur"] = 1.0 if (new and new[:40] in s) else 0.0
    s2, ok = apply_one(s, e)
    f["applied"] = 1.0 if (ok and s2 != s) else 0.0
    d = len(s2) - len(s)
    f["ins_frac"] = max(0.0, d) / max(1.0, ctx["en_chars"])
    f["del_frac"] = max(0.0, -d) / max(1.0, ctx["en_chars"])
    return [f[k] for k in GFEATS], s2, ok


def plan_rows(enacted, plan_instr, ctxs, true_text=None):
    """Walk the plan once, collecting gate features and (optionally) incremental gains."""
    s = enacted
    en = mtok(enacted)
    et, kt = (edits(en, mtok(true_text)) if true_text is not None else (None, None))

    def sc(txt):
        ep, kp = edits(en, mtok(txt))
        return 0.0 if kp < 0.5 * kt else f1(ep, et)

    cur = sc(s) if et is not None else 0.0
    X, Y = [], []
    for e, ctx in zip(plan_instr, ctxs):
        row, s2, ok = gate_row(e, s, ctx)
        X.append(row)
        if et is not None:
            nxt = sc(re.sub(r"\s+", " ", s2).strip())
            Y.append(nxt - cur)
            cur = nxt
        s = s2
    return X, Y, re.sub(r"\s+", " ", s).strip()
