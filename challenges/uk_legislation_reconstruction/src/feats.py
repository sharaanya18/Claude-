"""Reranker features for (query, candidate provision) pairs."""
import re, sys, math, collections
import numpy as np
sys.path.insert(0, '/home/user/Claude-/challenges/uk_legislation_reconstruction/src')
from lib import norm, parse_label, section_refs, RE_FULLCIT
from apply import all_instructions, apply_one

RE_QUOTE = re.compile(r'"([^"]{1,400})"')
RE_SUBST = re.compile(r'for\s+"([^"]{1,300})"[^"]{0,80}?substitut\w+\s+"([^"]{0,300})"', re.I)
RE_SUBST2 = re.compile(r'for\s+"([^"]{1,300})"[^"]{0,80}?there\s+is\s+substituted\s+"([^"]{0,300})"', re.I)
RE_OMITQ = re.compile(r'(?:omit|repeal\w*|revoke\w*)\s+(?:the\s+words?\s+)?"([^"]{1,300})"', re.I)
RE_INSQ = re.compile(r'(?:insert|add)\w*\s+"([^"]{1,300})"', re.I)
RE_SUBSEC = re.compile(r'\b(?:sub-?section|subsection)s?\s*\(([0-9A-Za-z]{1,6})\)', re.I)
RE_PARA = re.compile(r'\bparagraphs?\s*\(([0-9a-z]{1,5})\)', re.I)
AMEND_W = re.compile(r'\b(insert|substitut|omit|repeal|revoke|amend|add|delete|cease)', re.I)

FEATS = [
 'days','neg_date','cyear','is_si','lkind','ltxt','nsec','nsecd','nacts',
 'tgt_cnt','tgt_pos','sec_share','f_in','f_omit','f_after','f_subject','f_insafter','f_ofact','f_amendnear',
 'a_title_txt','a_cit_txt','a_ctx','a_scope','a_dom','a_domrank','a_self','a_any','min_dist','n_between',
 'nq','nq_found','q_found_frac','max_found','max_qlen','n_sub','n_sub_found','n_omit','n_omit_found',
 'n_anch','n_anch_found','anch_frac','idf_ov','ctx_ov','is_repeal','ncand','qlen','qsecnum','jacc',
 # parser-derived: does this provision actually carry a textual edit to THIS section?
 'p_nins','p_napp','p_chg','p_sub','p_ins','p_omit','p_tbl','p_other',
]

def _quotes(t):
    return [m.group(1) for m in RE_QUOTE.finditer(t)]

class FeatureBuilder:
    def __init__(self, C):
        self.C = C
        # document frequency over corpus for IDF-weighted overlap
        df = collections.Counter()
        step = max(1, C.n // 20000)
        nd = 0
        for i in range(0, C.n, step):
            nd += 1
            for w in set(C.text[i].lower().split()[:400]):
                df[w] += 1
        self.idf = {w: math.log(nd / (1 + c)) for w, c in df.items()}
        self.dflt = math.log(nd)
        self._cache = {}
        self._pcache = {}

    def _unit(self, i):
        """Per-unit features that do not depend on the query (cached)."""
        c = self._cache.get(i)
        if c is None:
            C = self.C; t = C.text[i]
            sr = C.srefs[i]
            qs = _quotes(t)
            c = dict(
                sr=sr, nsec=len(sr), nsecd=len(set(s for s, _, _ in sr)),
                nacts=len(set(x for x, _ in C.arefs[i])), ltxt=len(t),
                quotes=qs, nq=len(qs),
                subs=[m.group(1) for m in RE_SUBST.finditer(t)] + [m.group(1) for m in RE_SUBST2.finditer(t)],
                omits=RE_OMITQ.findall(t), inss=RE_INSQ.findall(t),
                anch=[m.group(1) for m in RE_SUBSEC.finditer(t)] + [m.group(1) for m in RE_PARA.finditer(t)],
                is_repeal=1.0 if ('Extent of repeal' in t or 'Extent of revocation' in t) else 0.0,
                low=set(t.lower().split()[:600]),
            )
            self._cache[i] = c
        return c

    def parse_feats(self, Q, i):
        """Run the amendment parser on a candidate: real edits to this section are the
        strongest evidence that the provision is one of the section's amending provisions."""
        key = (i, Q['sec'], Q['acit'])
        v = self._pcache.get(key)
        if v is None:
            C = self.C; t = C.text[i]
            tbl = ('Extent of repeal' in t) or ('Extent of revocation' in t)
            ap = [p for c, p in C.arefs[i] if c == Q['acit']]
            try:
                ins = all_instructions(t, Q['sec'], ap, tbl)
            except Exception:
                ins = []
            napp = 0; chg = 0
            s0 = Q['en']
            for e in ins[:40]:
                s1, ok = apply_one(s0, e)
                if ok and s1 != s0:
                    napp += 1; chg += abs(len(s1) - len(s0)) + len(e.get('old') or '')
                    s0 = s1
            fam = collections.Counter(e['op'] for e in ins)
            v = (float(len(ins)), float(napp), math.log1p(chg),
                 float(sum(fam[k] for k in ('sub', 'subprov', 'subdefn', 'subspan'))),
                 float(sum(fam[k] for k in ('after', 'before', 'insprov', 'insdefn', 'approp', 'endins'))),
                 float(sum(fam[k] for k in ('omit', 'omitprov', 'omitdefn', 'omitspan', 'omitnear'))),
                 1.0 if tbl else 0.0)
            self._pcache[key] = v
        return v

    def row(self, Q, i):
        """Feature vector for query dict Q and candidate unit row i."""
        C = self.C; u = self._unit(i); t = C.text[i]
        sec = Q['sec']; en = Q['en']; enl = Q['enl']
        f = {k: 0.0 for k in FEATS}
        f['days'] = min(20000.0, max(-8000.0, (Q['qd'] - C.dnum[i])))
        f['neg_date'] = 1.0 if C.dnum[i] > Q['qd'] else 0.0
        f['cyear'] = float(C.date[i][:4])
        f['is_si'] = 1.0 if C.cit[i].startswith('S.I.') else 0.0
        f['lkind'] = {'s': 0, 'sch': 1, 'art': 2, 'other': 3}[parse_label(C.label[i])[0]]
        f['ltxt'] = math.log1p(u['ltxt'])
        f['nsec'] = math.log1p(u['nsec']); f['nsecd'] = math.log1p(u['nsecd'])
        f['nacts'] = math.log1p(u['nacts'])
        tg = [(p, fr) for s, p, fr in u['sr'] if s == sec]
        f['tgt_cnt'] = len(tg)
        f['tgt_pos'] = (tg[0][0] / max(1, u['ltxt'])) if tg else 1.0
        f['sec_share'] = len(tg) / max(1, u['nsec'])
        for p, fr in tg:
            frl = fr.lower()
            if re.search(r'(^|[.;:] |\) |, )in $', frl) or frl.endswith('in '): f['f_in'] = 1.0
            if 'omit' in frl or 'repeal' in frl or 'revoke' in frl: f['f_omit'] = 1.0
            if frl.rstrip().endswith('after'): f['f_after'] = 1.0
            if re.search(r'(^|\(\d+\) |\) )$', fr) and re.match(r'[Ss]ection', t[p:p+7]): f['f_subject'] = 1.0
            if AMEND_W.search(t[p:p+160]): f['f_amendnear'] = 1.0
        if re.search(r'[Aa]fter\s+section\s+' + re.escape(sec) + r'\b', t): f['f_insafter'] = 1.0
        if re.search(r'section\s+' + re.escape(sec) + r'[^.]{0,30}?\bof\s+(the\s+)?' + re.escape(Q['atitle'][:40]), t, re.I):
            f['f_ofact'] = 1.0
        # --- act provenance ---
        ac = Q['acit']
        f['a_title_txt'] = 1.0 if Q['atitle_n'] in t else 0.0
        f['a_cit_txt'] = 1.0 if (ac in t or ('(c. %s)' % Q['cno']) in t) else 0.0
        f['a_ctx'] = 1.0 if ac in C.ctx_act[i] else 0.0
        f['a_scope'] = 1.0 if C.scope_act[i] == ac else 0.0
        dom = C.dom_act[i]
        f['a_dom'] = 1.0 if ac in dom else 0.0
        f['a_domrank'] = float(dom.index(ac)) if ac in dom else 9.0
        f['a_self'] = 1.0 if C.cit[i] == ac else 0.0
        apos = [p for c2, p in C.arefs[i] if c2 == ac]
        f['a_any'] = 1.0 if apos else 0.0
        if apos and tg:
            d = min(abs(p - q) for p in apos for q, _ in tg)
            f['min_dist'] = math.log1p(d)
            lo = min(min(apos), tg[0][0]); hi = max(max(apos), tg[0][0])
            f['n_between'] = sum(1 for c2, p in C.arefs[i] if lo < p < hi and c2 != ac)
        else:
            f['min_dist'] = math.log1p(30000); f['n_between'] = 50.0
        # --- quoted-text evidence against the enacted section ---
        qs = u['quotes']; f['nq'] = math.log1p(len(qs))
        found = [q for q in qs if len(q) > 2 and q.lower() in enl]
        f['nq_found'] = math.log1p(len(found))
        f['q_found_frac'] = len(found) / max(1, len(qs))
        f['max_found'] = math.log1p(max((len(q) for q in found), default=0))
        f['max_qlen'] = math.log1p(max((len(q) for q in qs), default=0))
        f['n_sub'] = len(u['subs']); f['n_sub_found'] = sum(1 for q in u['subs'] if q.lower() in enl)
        f['n_omit'] = len(u['omits']); f['n_omit_found'] = sum(1 for q in u['omits'] if q.lower() in enl)
        an = u['anch']; f['n_anch'] = len(an)
        fa = sum(1 for a in an if ('(%s)' % a) in en)
        f['n_anch_found'] = fa; f['anch_frac'] = fa / max(1, len(an))
        # --- lexical overlap ---
        ov = u['low'] & Q['enset']
        # summed in sorted order: float addition is not associative, so an unordered set
        # walk could change the last bits of this feature between runs
        f['idf_ov'] = sum(self.idf.get(w, self.dflt) for w in sorted(ov)) / 50.0
        f['jacc'] = len(ov) / max(1, len(u['low'] | Q['enset']))
        cw = set(C.ctx[i].lower().split())
        f['ctx_ov'] = len(cw & Q['atset']) / max(1, len(Q['atset']))
        f['is_repeal'] = u['is_repeal']
        f['ncand'] = Q['ncand']; f['qlen'] = Q['qlen']; f['qsecnum'] = Q['qsecnum']
        (f['p_nins'], f['p_napp'], f['p_chg'], f['p_sub'], f['p_ins'], f['p_omit'],
         f['p_tbl']) = self.parse_feats(Q, i)
        f['p_other'] = 1.0 if (f['p_nins'] == 0 and f['tgt_cnt'] > 0) else 0.0
        return [f[k] for k in FEATS]
