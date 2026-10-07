"""End-to-end pipeline: retrieval reranker + in-force selection + gated reconstruction."""
import sys, json, time, collections, datetime
import numpy as np
sys.path.insert(0, '/home/user/Claude-/challenges/uk_legislation_reconstruction/src')
from lib import norm
from feats import FeatureBuilder, FEATS
from apply import all_instructions, apply_one, looks_tabular
from gate import GFEATS, oracle_walk, model_walk
import re


def dnum(d):
    y, m, dd = (int(x) for x in str(d)[:10].split("-"))
    return float(datetime.date(y, m, dd).toordinal())


def make_query(r):
    en = norm(r.enacted_text)
    sec = norm(r.section_label).replace("s. ", "").upper()
    at = norm(r.act_title)
    return dict(sec=sec, en=en, enl=en.lower(), enset=set(en.lower().split()),
                atitle=at, atitle_n=at, atset=set(at.lower().split()),
                acit=r.act_citation, cno=r.act_citation.split("c.")[-1].strip(),
                qd=dnum(r.date), qlen=float(np.log1p(len(en))),
                qsecnum=float("".join(ch for ch in sec if ch.isdigit()) or 0), ncand=0.0)


def retrieval_matrix(C, FB, queries):
    """Candidate feature matrix for a list of query dicts."""
    X, qi, cu = [], [], []
    for k, Q in enumerate(queries):
        cand = C.candidates(Q["acit"], Q["sec"])
        Q["ncand"] = float(len(cand))
        for i in cand:
            X.append(FB.row(Q, int(i))); qi.append(k); cu.append(int(i))
    return (np.asarray(X, dtype=np.float32).reshape(-1, len(FEATS)),
            np.asarray(qi, dtype=np.int32), np.asarray(cu, dtype=np.int32))


_INSTR_CACHE = {}


def instructions_for(C, Q, i):
    """Parsed edits of corpus provision `i` for this query's act and section (memoised:
    the same provision is parsed again for every plan width and every CV pass)."""
    key = (i, Q["sec"], Q["acit"])
    v = _INSTR_CACHE.get(key)
    if v is None:
        t = C.text[i]
        ap = [p for c, p in C.arefs[i] if c == Q["acit"]]
        v = (all_instructions(t, Q["sec"], ap, looks_tabular(t)), looks_tabular(t), len(t))
        _INSTR_CACHE[key] = v
    return v


def build_plan(C, Q, rows, scores, plan_thr=0.0):
    """Chronologically ordered instructions from the selected provisions, with context."""
    order = sorted(range(len(rows)), key=lambda k: (C.date[rows[k]], C.label[rows[k]]))
    rank = {k: r for r, k in enumerate(sorted(range(len(rows)), key=lambda k: -scores[k]))}
    instr, ctxs = [], []
    for k in order:
        i = rows[k]
        es, tbl, tlen = instructions_for(C, Q, i)
        for j, e in enumerate(es):
            instr.append(e)
            ctxs.append(dict(prov_score=float(scores[k]), prov_days=min(20000.0, Q["qd"] - C.dnum[i]),
                             prov_tbl=1.0 if tbl else 0.0, prov_nins=float(len(es)),
                             prov_len=float(np.log1p(tlen)), ins_idx=float(j), ins_n=float(len(es)),
                             en_len=float(len(Q["en"].split())), n_prov=float(len(rows)),
                             en_chars=float(len(Q["en"])), prov_rank=float(rank[k]),
                             plan_thr=float(plan_thr)))
    return instr, ctxs


def select_ids(C, p, uids, thr, min_one=True):
    o = np.argsort(-p, kind="stable")
    keep = [int(uids[j]) for j in o if p[j] >= thr]
    if min_one and not keep and len(o):
        keep = [int(uids[o[0]])]
    return keep


