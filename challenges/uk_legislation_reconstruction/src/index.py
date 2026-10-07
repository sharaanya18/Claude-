"""Corpus index: act-reference resolution + (act, section) -> units inverted index."""
import re, sys, time, collections, pickle
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/Claude-/challenges/uk_legislation_reconstruction/src')
from lib import *

T0 = time.time()
def log(m): print("[%6.0fs] %s" % (time.time()-T0, m), flush=True)

RE_AMEND_VERB = re.compile(r"\b(is|are|shall be|has|have)\s+(amended|effect|have effect)|"
                           r"\bis amended\b|\bare amended\b|\bhas effect\b", re.I)
RE_YEARACT = re.compile(r"\b(?:the\s+)?((?:1[6-9]|20)\d\d)\s+Act\b")

def _dnum(s):
    """Date string -> day number (proleptic, no calendar lib needed in hot loops)."""
    import datetime
    y, m, d = (int(x) for x in str(s)[:10].split('-'))
    return float(datetime.date(y, m, d).toordinal())


class Corpus:
    """Parsed corpus with act/section reference resolution and an inverted index."""

    def __init__(self, path):
        df = pd.read_csv(path, keep_default_na=False)
        self.uid = df.unit_id.to_numpy()
        self.pos = {u: i for i, u in enumerate(self.uid)}
        self.title = df.document_title.tolist()
        self.cit = df.citation.tolist()
        self.date = df.document_date.tolist()
        self.dnum = np.array([_dnum(x) for x in self.date], dtype=np.float64)
        self.label = df.label.tolist()
        self.ctx = [norm(x) for x in df.context]
        self.text = [norm(x) for x in df.text]
        self.n = len(df)
        self.doc = {}
        for i, cc in enumerate(self.cit):
            self.doc.setdefault(cc, []).append(i)
        self.title2cit = {}
        for i in range(self.n):
            self.title2cit.setdefault(act_title_key(self.title[i]), self.cit[i])
        log("corpus n=%d docs=%d" % (self.n, len(self.doc)))

    # ---- abbreviations, document-scoped ----
    def build_abbrev(self):
        self.abbr = {}
        self.yearact = {}
        for cc, idxs in self.doc.items():
            mp = {}
            yr = collections.Counter()
            for i in idxs:
                t = self.text[i]
                for m in RE_ABBR_MEANS.finditer(t):
                    k = act_title_key(m.group(2))
                    if k in self.title2cit: mp[m.group(1).strip()] = self.title2cit[k]
                for m in RE_ABBR_PAREN.finditer(t):
                    k = act_title_key(m.group(1))
                    if k in self.title2cit: mp[m.group(2).strip()] = self.title2cit[k]
                for m in RE_FULLCIT.finditer(t):
                    yr["%s c. %s" % (m.group(1), m.group(2))] += 1
                for m in RE_ACT.finditer(t):
                    k = act_title_key(m.group(1))
                    c2 = self.title2cit.get(k)
                    if c2: yr[c2] += 1
            self.abbr[cc] = mp
            # "the 2002 Act" -> the most-referenced act of that year in this document
            best = {}
            for c2, nref in yr.most_common():
                y = c2.split()[0]
                if y.isdigit() and y not in best: best[y] = c2
            self.yearact[cc] = best
        log("abbrev maps (%d non-empty) + year-act maps" % sum(1 for v in self.abbr.values() if v))

    def _resolve_string(self, s, cc):
        """All act citations referenced by string `s`, with char positions."""
        out = [(c, p) for c, p, _ in find_act_refs(s, self.title2cit)]
        for ab, c2 in self.abbr.get(cc, {}).items():
            st = s.find(ab)
            k = 0
            while st >= 0 and k < 50:
                out.append((c2, st)); st = s.find(ab, st + 1); k += 1
        ya = self.yearact.get(cc, {})
        for m in RE_YEARACT.finditer(s):
            c2 = ya.get(m.group(1))
            if c2: out.append((c2, m.start()))
        return out

    def build_refs(self):
        self.arefs = [None]*self.n
        self.srefs = [None]*self.n
        self.ranges = [None]*self.n
        self.decl = [None]*self.n
        for i in range(self.n):
            t = self.text[i]; cc = self.cit[i]
            # total sort key: equal positions must order identically on every run,
            # otherwise set iteration order (PYTHONHASHSEED) leaks into the index
            ar = sorted(set(self._resolve_string(t, cc)), key=lambda x: (x[1], x[0]))
            self.arefs[i] = ar
            self.srefs[i] = section_refs(t)
            self.ranges[i] = expand_ranges(t)
            # declaration: an act named in the opening of a unit whose opening says "is amended"
            head = t[:300]
            if RE_AMEND_VERB.search(head):
                vb = RE_AMEND_VERB.search(head).start()
                before = [c for c, p in ar if p < vb + 12]
                if before: self.decl[i] = before[-1]
        log("refs extracted")

    def build_scope_acts(self):
        """Propagate a schedule/part declaration forward in label order within its scope."""
        self.scope_act = [None]*self.n
        for cc, idxs in self.doc.items():
            by = collections.defaultdict(list)
            for i in sorted(idxs, key=lambda i: parse_label(self.label[i])[1]):
                by[sch_scope(self.label[i])].append(i)
            for _, lst in by.items():
                cur = None; curctx = ""
                for i in lst:
                    d = self.decl[i]
                    if d is not None:
                        cur, curctx = d, self.ctx[i]
                    elif cur is not None:
                        a = self.ctx[i].split(" > "); b = curctx.split(" > ")
                        share = sum(1 for x, y in zip(a, b) if x == y)
                        if share < min(2, len(b)): cur = None
                    self.scope_act[i] = cur
        log("scope acts: %d units" % sum(1 for x in self.scope_act if x))

    def build_ctx_acts(self):
        """Acts named in the context path, including abbreviations and 'the YYYY Act'."""
        cache = {}; self.ctx_act = [None]*self.n
        for i in range(self.n):
            k = (self.ctx[i], self.cit[i])
            if k not in cache:
                cache[k] = sorted(set(c for c, _ in self._resolve_string(self.ctx[i], self.cit[i])))
            self.ctx_act[i] = cache[k]
        log("context acts extracted")

    def build_index(self):
        """(citation, section) -> unit rows.  Positional pairing plus unit-dominant acts."""
        idx = collections.defaultdict(list)
        self.dom_act = [None]*self.n
        for i in range(self.n):
            ar = self.arefs[i]; sr = self.srefs[i]
            secs = set(s for s, _, _ in sr) | self.ranges[i]
            cnt = collections.Counter(c for c, _ in ar)
            dom = [c for c, _ in cnt.most_common(4)]
            self.dom_act[i] = dom
            scoped = set(x for x in ([self.scope_act[i]] + list(self.ctx_act[i]) + dom) if x)
            scoped.add(self.cit[i])
            pairs = set()
            if ar:
                apos = [p for _, p in ar]; acit = [c for c, _ in ar]
                for s, sp, _ in sr:
                    j = np.searchsorted(apos, sp)
                    for k in (j-2, j-1, j, j+1):
                        if 0 <= k < len(acit): pairs.add((acit[k], s))
                for s in sorted(self.ranges[i]):
                    for c in acit[:6]: pairs.add((c, s))
            for c in sorted(scoped):
                for s in sorted(secs): pairs.add((c, s))
            for p in sorted(pairs): idx[p].append(i)
        # posting lists sorted by row index: candidate order, and therefore the order of
        # the feature rows the boosters bag over, must not vary between runs
        self.index = {k: np.array(sorted(set(v)), dtype=np.int32) for k, v in idx.items()}
        log("index: %d keys, %d postings" % (len(self.index), sum(len(v) for v in self.index.values())))

    def candidates(self, citation, sec):
        v = self.index.get((citation, sec))
        return v if v is not None else np.empty(0, dtype=np.int32)

def build(path):
    C = Corpus(path)
    C.build_abbrev(); C.build_refs(); C.build_scope_acts(); C.build_ctx_acts(); C.build_index()
    return C

if __name__ == "__main__":
    import json
    D = '/home/user/Claude-/challenges/uk_legislation_reconstruction/dataset/public/'
    C = build(D + 'corpus.csv')
    tr = pd.read_csv(D+'train.csv', keep_default_na=False)
    tt = pd.read_csv(D+'train_targets.csv', keep_default_na=False)
    g = dict(zip(tt.item_id, tt.amending_ids.map(json.loads)))
    rec, nc, missed = [], [], collections.Counter()
    for r in tr.itertuples():
        sec = norm(r.section_label).replace('s. ', '').upper()
        cands = set(C.candidates(r.act_citation, sec).tolist())
        gold = set(C.pos[u] for u in g[r.item_id])
        nc.append(len(cands)); rec.append(len(gold & cands)/max(1, len(gold)))
        for u in gold - cands: missed[(C.cit[u], C.label[u])] += 1
    print("mean recall %.4f  all-found %.3f  mean cands %.1f  p90 %.0f  max %d" %
          (np.mean(rec), np.mean([x == 1 for x in rec]), np.mean(nc), np.percentile(nc, 90), max(nc)))
    print("misses:", sum(missed.values()), missed.most_common(8))
    pickle.dump(C, open('/home/user/Claude-/challenges/uk_legislation_reconstruction/working/corpus.pkl', 'wb'), protocol=4)
