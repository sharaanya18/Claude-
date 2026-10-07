"""Local re-implementation of the challenge metric.

score = 100 * mean_q [ 0.40 * setF1(amending_ids) + 0.60 * editF1(text_at_date) ]

Edit representation (per the description): both the submitted and the true text are
diffed against the ENACTED text; an edit is either an enacted token deleted, or a new
token inserted at a position of the enacted text.  The term is the F1 between the two
edit sets.  Guard: a text keeping < half as many enacted tokens as the true text keeps
scores 0 (so throwing the section away cannot harvest its deletions).
"""
import re, json
from difflib import SequenceMatcher

_D = {ord(ch): "-" for ch in "‐‑‒–—―−⁃"}
_Q = {ord(ch): '"' for ch in "“”„«»″"}
_Q.update({ord(ch): "'" for ch in "‘’‚′"})
_Q.update({0xa0: " ", 0x2007: " ", 0x202f: " ", 0xad: ""})
_TR = {**_D, **_Q}

def mnorm(s):
    return re.sub(r"\s+", " ", str(s).translate(_TR)).strip()

def mtok(s):
    return mnorm(s).split()

# Two readings of "new tokens inserted at a position of the enacted text".
#   "pos"   -> an edit is (position, token): a token-level slip costs only that token.
#   "index" -> an edit is (position, offset-in-block, token): the whole block must match.
# The description's wording ("a set of edits ... new tokens inserted at a position") reads
# as the first, which is also the kinder of the two; both are reported in experiments and
# decisions are only taken where they agree.
EDIT_MODE = "pos"   # fixed reading used for every decision in this script


def edits(enacted_toks, other_toks, mode=None):
    """Edit set of `other` relative to `enacted`, plus the count of enacted tokens kept."""
    mode = EDIT_MODE if mode is None else mode
    sm = SequenceMatcher(a=enacted_toks, b=other_toks, autojunk=False)
    E, kept = set(), 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            kept += i2 - i1
        else:
            for i in range(i1, i2):
                E.add(("D", i))
            if mode == "pos":
                for j in range(j1, j2):
                    E.add(("I", i1, other_toks[j]))
            else:
                for k, j in enumerate(range(j1, j2)):
                    E.add(("I", i1, k, other_toks[j]))
    return E, kept

def f1(pred, true):
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 2 * p * r / (p + r)

def text_score(enacted, pred_text, true_text, mode=None):
    en = mtok(enacted)
    tp, kp = edits(en, mtok(pred_text), mode)
    tt, kt = edits(en, mtok(true_text), mode)
    if kp < 0.5 * kt:
        return 0.0
    return f1(tp, tt)

def id_score(pred_ids, true_ids):
    return f1(set(int(x) for x in pred_ids), set(int(x) for x in true_ids))

def query_score(enacted, pred_ids, pred_text, true_ids, true_text):
    return 0.40 * id_score(pred_ids, true_ids) + 0.60 * text_score(enacted, pred_text, true_text)

def score_frame(queries, pred, truth):
    """queries: item_id -> enacted_text ; pred/truth: item_id -> (ids, text)."""
    tot = idt = txt = 0.0
    for q, en in queries.items():
        pi, px = pred.get(q, ([], en))
        ti, tx = truth[q]
        a = id_score(pi, ti); b = text_score(en, px, tx)
        idt += a; txt += b; tot += 0.40 * a + 0.60 * b
    n = len(queries)
    return 100 * tot / n, 100 * idt / n, 100 * txt / n
