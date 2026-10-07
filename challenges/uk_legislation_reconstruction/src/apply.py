"""Amendment-operation parser and text applier.

Parses an amending provision into a list of scoped edit instructions, then applies them
to the running text of the queried section.  Everything works on the normalised flat
rendering used by the corpus: sub-provision markers "(1)", "(a)", "(i)" in reading order.
"""
import re

MARK = re.compile(r"\(([0-9]{1,3}[A-Z]{0,3}|[a-z]{1,4}|[ivxl]{1,6})\)")
_CONJ = re.compile(r"(?:^|[.;:,\-])\s*(?:and|or|but|also)\s*$", re.I)
_ROMAN = re.compile(r"^[ivxl]+$")


def _strong_start(s, p):
    """Could the marker at char p open a sub-provision (vs. being a cross-reference)?"""
    j = p - 1
    while j >= 0 and s[j] == " ":
        j -= 1
    if j < 0:
        return True
    ch = s[j]
    if ch in ".;:,-?!\"'":
        return True
    if ch == ")":
        return p - j >= 2          # "(1) (a)" chains, not "section 130(1)(a)"
    return bool(_CONJ.search(s[max(0, p - 14):p]))


def _key(tag):
    """Legislative sort key within a level: 1 < 1A < 2 ; a < aa < ab < b ; za sorts first."""
    if tag[0].isdigit():
        n = re.match(r"\d+", tag).group(0)
        return (int(n), tag[len(n):])
    if tag.startswith("z") and len(tag) > 1:
        return (-1, tag)           # (za), (zb) are inserted ahead of (a)
    return (0, tag)


def markers(s):
    """Sub-provision markers in reading order: (tag, level, start, end).

    A candidate is kept only if it can open a sub-provision *and* continues the running
    sequence at its level, which is what separates real markers from cross-references
    such as "subsection (7)" or "section 130(1)(a)".
    """
    out = []
    last = {}                      # level -> last accepted key
    prev_lv = 0
    for m in MARK.finditer(s):
        p = m.start()
        if not _strong_start(s, p):
            continue
        tag = m.group(1)
        if tag[0].isdigit():
            lv = 1
        elif _ROMAN.fullmatch(tag) and (
                (3 in last and _key(tag) > last[3]) or (tag == "i" and prev_lv == 2)):
            lv = 3
        else:
            lv = 2
        k = _key(tag)
        if lv in last and not k > last[lv]:
            continue               # out of sequence -> a reference, not a marker
        last[lv] = k
        for deeper in [x for x in last if x > lv]:
            del last[deeper]       # a new outer item restarts inner numbering
        prev_lv = lv
        out.append((tag, lv, p, m.end()))
    return out


def span_of(s, chain, mk=None, lo=0, hi=None):
    """Char span of the sub-provision addressed by `chain` (e.g. ['4','b'])."""
    mk = markers(s) if mk is None else mk
    hi = len(s) if hi is None else hi
    for tag in chain:
        sub = [x for x in mk if lo <= x[2] < hi]
        hit = None
        for k, (t, lv, a, b) in enumerate(sub):
            if t == tag:
                hit = (k, lv, a)
                break
        if hit is None:
            return None
        k, lv, a = hit
        end = hi
        for t2, lv2, a2, _ in sub[k + 1:]:
            if lv2 <= lv:
                end = a2
                break
        lo, hi = a, end
    return (lo, hi)


# ---------------------------------------------------------------- instruction parsing
RE_CHAIN = re.compile(r"\(([0-9A-Za-z]{1,4})\)")
SCOPE_RX = re.compile(
    r"\bin\s+(?:the\s+)?(sub-?paragraphs?|subsections?|paragraphs?|sub-?sections?)\s+((?:\([0-9A-Za-z]{1,4}\))+)",
    re.I)
SEC_SCOPE_RX = re.compile(r"\bsections?\s+([0-9]{1,4}[A-Z]{0,3})((?:\([0-9A-Za-z]{1,4}\))*)", re.I)
DEFN_RX = re.compile(r"\bthe\s+definition\s+of\s+\"([^\"]{1,120})\"", re.I)

_SUBST = r"(?:substitute[ds]?|there\s+(?:is|are|shall\s+be)\s+substituted|there\s+shall\s+be\s+substituted)"
_INSERT = r"(?:insert(?:ed)?|there\s+(?:is|are|shall\s+be)\s+inserted|add(?:ed)?)"
_PROV = r"(?:sub-?sections?|subsections?|paragraphs?|sub-?paragraphs?)"

RE_FOR_SUB = re.compile(
    r"for\s+(?:the\s+\w+\s+)?\"(?P<old>[^\"]{1,600})\"(?P<mid>[^\"]{0,140}?)" + _SUBST +
    r"[^\"]{0,40}?\"(?P<new>[^\"]{0,600})\"", re.I)
RE_AFTER_INS = re.compile(
    r"after\s+\"(?P<anchor>[^\"]{1,400})\"[^\"]{0,90}?" + _INSERT + r"[^\"]{0,30}?\"(?P<new>[^\"]{0,600})\"", re.I)
RE_BEFORE_INS = re.compile(
    r"before\s+\"(?P<anchor>[^\"]{1,400})\"[^\"]{0,90}?" + _INSERT + r"[^\"]{0,30}?\"(?P<new>[^\"]{0,600})\"", re.I)
RE_OMIT_Q = re.compile(r"(?:omit|omits|leave\s+out)\s+(?:the\s+words?\s+)?\"(?P<old>[^\"]{1,600})\"", re.I)
RE_REPEAL_Q = re.compile(r"the\s+words?\s+\"(?P<old>[^\"]{1,600})\"\s+(?:are|is)\s+(?:repealed|omitted|revoked)", re.I)
RE_WORDS_FROM = re.compile(
    r"the\s+words\s+from\s+\"(?P<a>[^\"]{1,300})\"\s+to\s+(?:\"(?P<b>[^\"]{1,300})\"|the\s+end)", re.I)

RE_SUB_PROV = re.compile(
    r"for\s+(?:the\s+)?" + _PROV + r"\s+(?P<tags>\([0-9A-Za-z]{1,4}\)(?:\s*(?:,|and|to)\s*\([0-9A-Za-z]{1,4}\))*)"
    r"[^\"]{0,90}?" + _SUBST + r"\s*[-:]\s*(?P<new>.+)$", re.I | re.S)
RE_SUB_DEFN = re.compile(
    r"for\s+the\s+definition\s+of\s+\"(?P<name>[^\"]{1,120})\"[^\"]{0,60}?" + _SUBST + r"\s*[-:]\s*(?P<new>.+)$",
    re.I | re.S)
RE_INS_PROV = re.compile(
    r"after\s+(?:the\s+)?" + _PROV + r"\s+(?P<tags>\([0-9A-Za-z]{1,4}\))[^\"]{0,70}?" + _INSERT +
    r"\s*[-:]\s*(?P<new>.+)$", re.I | re.S)
RE_INS_PROV_Q = re.compile(
    r"after\s+(?:the\s+)?" + _PROV + r"\s+(?P<tags>\([0-9A-Za-z]{1,4}\))[^\"]{0,70}?" + _INSERT +
    r"\s*[-:]?\s*\"(?P<new>[^\"]{1,600})\"", re.I)
RE_INS_DEFN = re.compile(
    r"after\s+the\s+definition\s+of\s+\"(?P<name>[^\"]{1,120})\"[^\"]{0,60}?" + _INSERT +
    r"\s*[-:]?\s*(?P<new>.+)$", re.I | re.S)
RE_OMIT_PROV = re.compile(
    r"(?:omit|omits)\s+(?:the\s+)?" + _PROV +
    r"\s+(?P<tags>\([0-9A-Za-z]{1,4}\)(?:\s*(?:,|and|to)\s*\([0-9A-Za-z]{1,4}\))*)", re.I)
RE_PROV_REPEALED = re.compile(
    _PROV + r"\s+(?P<tags>(?:\([0-9A-Za-z]{1,4}\))+)\s+(?:is|are)\s+(?:repealed|omitted|revoked)", re.I)
RE_OMIT_DEFN = re.compile(r"omit\s+the\s+definition\s+of\s+\"(?P<name>[^\"]{1,120})\"", re.I)
RE_OMIT_SECS = re.compile(
    r"(?:omit|repeal)\w*\s+(?P<lst>sections?\s+[0-9]{1,4}[A-Z]{0,3}(?:\([0-9A-Za-z]{1,4}\))*"
    r"(?:\s*(?:,|and)\s*(?:sections?\s+)?[0-9]{1,4}[A-Z]{0,3}(?:\([0-9A-Za-z]{1,4}\))*)*)", re.I)
RE_SUBLIST = re.compile(
    r"[Ff]or\s+(?:the\s+words?\s+)?\"(?P<old>[^\"]{1,300})\"[^\"]{0,60}?" + _SUBST +
    r"[^\"]{0,30}?\"(?P<new>[^\"]{0,300})\"[^.]{0,80}?in\s+the\s+following\s+"
    r"(?:provisions|enactments|sections)[^-:]{0,60}[-:]\s*(?P<lst>.{0,2500})", re.I | re.S)
RE_APPROP = re.compile(r"at\s+the\s+appropriate\s+places?\s+" + _INSERT + r"\s*[-:]?\s*(?P<new>.+)$", re.I | re.S)
RE_END_INS = re.compile(r"at\s+the\s+end\s+(?:of\s+[^,]{0,40}\s+)?" + _INSERT + r"\s*[-:]?\s*(?P<new>.+)$", re.I | re.S)

_DUP_TAIL = re.compile(r"([.;,])[\s.]*\.\s*$")
RE_EVERY = re.compile(
    r"each\s+place|both\s+places|every\s+place|wherever\s+(?:it\s+)?(?:occur|appear)|"
    r"in\s+each\s+case|where(?:ver)?\s+those\s+words|each\s+time", re.I)
INSTR_HEAD = re.compile(
    r"^(in|for|after|before|omit|insert|add|at|section|sections|subsection|paragraph|"
    r"the\s+words|the\s+word|the\s+definition|repeal|revoke|substitute|leave)\b", re.I)


def _find(hay, needle, frm=0):
    """Find `needle` respecting word boundaries, so "it" does not match inside "condition"."""
    n = len(needle)
    if n == 0:
        return -1
    p = hay.find(needle, frm)
    while p >= 0:
        okl = p == 0 or not (needle[0].isalnum() and hay[p - 1].isalnum())
        okr = p + n >= len(hay) or not (needle[-1].isalnum() and hay[p + n].isalnum())
        if okl and okr:
            return p
        p = hay.find(needle, p + 1)
    return -1


def _replace_all(hay, old, new):
    out, i, k = [], 0, 0
    while True:
        p = _find(hay, old, i)
        if p < 0:
            break
        out.append(hay[i:p]); out.append(new); i = p + len(old); k += 1
        if k > 200:
            break
    out.append(hay[i:])
    return "".join(out), k


def _clean_new(s):
    """Tidy inserted content: drop the lead-in dash and the amending sentence's own
    full stop, but keep the punctuation that belongs to the inserted text itself."""
    s = re.sub(r"^[-:\s]+", "", s.strip()).strip()
    if s.startswith('"') and s.count('"') == 2:
        q = s.rfind('"')
        if q > 0:
            return s[1:q].strip()
    s = _DUP_TAIL.sub(r"\1", s)                 # ".." -> "." ,  ";." -> ";"
    return s.strip()


def _seq_bounds(t, mk):
    """Positions of the amending provision's own numbered sequence (1), (2), (3)..."""
    seq, want = [], 1
    for tag, lv, a, b in mk:
        if lv == 1 and tag == str(want):
            seq.append((a, b)); want += 1
    return seq


def split_clauses(t):
    """Split a provision into instruction clauses.

    Only markers that both continue a strict sequence *and* are followed by instruction
    language start a clause; that keeps inserted sub-provisions (which look just like
    markers) inside the instruction that inserts them.
    """
    t = t.strip()
    cand = [(m.group(1), m.start(), m.end()) for m in MARK.finditer(t) if _strong_start(t, m.start())]
    for first, nxt, isnum in (("1", lambda x: str(int(x) + 1), True),
                              ("a", lambda x: chr(ord(x) + 1), False)):
        hard, want = [], first
        for tag, a, b in cand:
            if tag != want or tag[0].isdigit() != isnum:
                continue
            if INSTR_HEAD.match(t[b:b + 24].strip()):
                hard.append((a, tag)); want = nxt(want)
        if len(hard) >= 2:
            head = t[:hard[0][0]]
            parts = []
            for k, (a, tag) in enumerate(hard):
                stop = hard[k + 1][0] if k + 1 < len(hard) else len(t)
                # a bare continuation of the amending provision's own numbering ends the clause
                parts.append(t[a:stop])
            return head, parts
    return "", [t]


def parse_scope(clause, head, sec):
    """Scope chain of markers addressed by this clause (outermost first)."""
    chain = []
    for src in (head, clause):
        if not src:
            continue
        pre = re.split(r"\b(?:insert|substitut|omit|repeal)", src, flags=re.I)[0]
        for m in SEC_SCOPE_RX.finditer(pre):
            if m.group(1).upper() == sec and m.group(2):
                chain = RE_CHAIN.findall(m.group(2))
        for m in SCOPE_RX.finditer(pre):
            c = RE_CHAIN.findall(m.group(2))
            if not c:
                continue
            if m.group(1).lower().startswith(("subsection", "sub-section")):
                chain = c
            elif chain and chain[-1] != c[0]:
                chain = chain + c
            else:
                chain = c
    out, seen = [], set()
    for x in chain:
        if x not in seen:
            out.append(x); seen.add(x)
    return out


def _tags(s):
    return RE_CHAIN.findall(s)


def _prefix(s):
    return re.split(r"\b(?:insert|substitut|omit|repeal|revoke|cease)", s, flags=re.I)[0]


def other_section(clause, head, sec):
    """True when the clause explicitly amends some other section of the act."""
    cs = set(m.group(1).upper() for m in SEC_SCOPE_RX.finditer(_prefix(clause)))
    if sec in cs:
        return False
    if cs:
        return True
    hs = set(m.group(1).upper() for m in SEC_SCOPE_RX.finditer(_prefix(head or "")))
    return bool(hs) and sec not in hs


def parse_instructions(text, sec):
    """Parse one amending provision (already focused on the target act) into edits."""
    head, clauses = split_clauses(text)
    ins = []
    for cl in clauses:
        # a clause that addresses a different section is kept, with that fact recorded as
        # a feature for the edit gate; discarding such clauses outright lost more than it saved
        osec = 1.0 if other_section(cl, head, sec) else 0.0
        chain = parse_scope(cl, head, sec)
        defn = None
        md = DEFN_RX.search(re.split(r"\b(?:insert|substitut)", cl, flags=re.I)[0])
        if md:
            defn = md.group(1)
        base = dict(chain=chain, defn=defn, osec=osec)
        got = len(ins)
        m = RE_SUB_DEFN.search(cl)
        if m:
            ins.append(dict(base, op="subdefn", name=m.group("name"), new=_clean_new(m.group("new"))))
        m = RE_INS_DEFN.search(cl)
        if m:
            ins.append(dict(base, op="insdefn", name=m.group("name"), new=_clean_new(m.group("new"))))
        m = RE_SUB_PROV.search(cl)
        if m:
            ins.append(dict(base, op="subprov", tags=_tags(m.group("tags")), new=_clean_new(m.group("new"))))
        m = RE_INS_PROV.search(cl) or RE_INS_PROV_Q.search(cl)
        if m:
            ins.append(dict(base, op="insprov", tags=_tags(m.group("tags")), new=_clean_new(m.group("new"))))
        for m in RE_FOR_SUB.finditer(cl):
            ins.append(dict(base, op="sub", old=m.group("old"), new=m.group("new"),
                            every=bool(RE_EVERY.search(m.group("mid"))) or
                                  (not chain and cl.lower().count(m.group("old").lower()) == 1
                                   and len(m.group("old")) > 6)))
        for m in RE_AFTER_INS.finditer(cl):
            ins.append(dict(base, op="after", old=m.group("anchor"), new=m.group("new")))
        for m in RE_BEFORE_INS.finditer(cl):
            ins.append(dict(base, op="before", old=m.group("anchor"), new=m.group("new")))
        for m in RE_WORDS_FROM.finditer(cl):
            tail = cl[m.end():m.end() + 90]
            sm2 = re.search(_SUBST + r"[^\"]{0,30}?\"(?P<new>[^\"]{0,400})\"", tail, re.I)
            if sm2:
                ins.append(dict(base, op="subspan", a=m.group("a"), b=m.group("b") or "",
                                new=sm2.group("new")))
            elif re.search(r"\b(omit|repeal|revoke|leave\s+out)", cl, re.I) or "op_table" in base:
                ins.append(dict(base, op="omitspan", a=m.group("a"), b=m.group("b") or ""))
        for rx in (RE_OMIT_Q, RE_REPEAL_Q):
            for m in rx.finditer(cl):
                ins.append(dict(base, op="omit", old=m.group("old")))
        for rx in (RE_OMIT_PROV, RE_PROV_REPEALED):
            for m in rx.finditer(cl):
                ins.append(dict(base, op="omitprov", tags=_tags(m.group("tags"))))
        for m in RE_OMIT_SECS.finditer(cl):
            for sm in RE_TBL_SECLIST.finditer(m.group("lst")):
                if sm.group(1).upper() == sec:
                    tg = _tags(sm.group(2) or "")
                    if tg:
                        ins.append(dict(op="omitprov", tags=[tg[-1]], chain=tg[:-1], defn=None))
        for m in RE_SUBLIST.finditer(cl):
            for sm in RE_TBL_SECLIST.finditer(m.group("lst")):
                if sm.group(1).upper() == sec:
                    ins.append(dict(op="sub", old=m.group("old"), new=m.group("new"),
                                    chain=_tags(sm.group(2) or ""), defn=None, every=True))
        for m in RE_OMIT_DEFN.finditer(cl):
            ins.append(dict(base, op="omitdefn", name=m.group("name")))
        m = RE_APPROP.search(cl)
        if m:
            ins.append(dict(base, op="approp", new=_clean_new(m.group("new"))))
        if len(ins) == got:
            m = RE_END_INS.search(cl)
            if m:
                ins.append(dict(base, op="endins", new=_clean_new(m.group("new"))))
    return ins


# ---------------------------------------------------------------- repeal / revocation tables
RE_TBL_ENTRY = re.compile(
    r"(?:^|(?<=[.;]) )(?P<body>(?:In\s+)?[Ss]ections?\s+[0-9]{1,4}[A-Z]{0,3}[^.]{0,400}?)(?=\.(?:\s|$))")
RE_TBL_SEC = re.compile(r"^(?P<in>In\s+)?[Ss]ections?\s+(?P<sec>[0-9]{1,4}[A-Z]{0,3})(?P<chain>(?:\([0-9A-Za-z]{1,4}\))*)",
                        re.I)
RE_WORD_AT_END = re.compile(
    r"(?:the\s+)?words?\s+\"(?P<w>[^\"]{1,200})\"\s+(?:at\s+the\s+end\s+of|preceding|before|after)\s+"
    r"(?:paragraph|sub-?paragraph|subsection)\s+\((?P<tag>[0-9A-Za-z]{1,4})\)", re.I)


RE_TBL_WORDS_IN = re.compile(
    r"[Tt]he\s+words?\s+\"(?P<w>[^\"]{1,200})\"\s+in\s*[-:]?\s*(?P<lst>(?:[^.]{0,400}))")
RE_TBL_SECLIST = re.compile(r"sections?\s+([0-9]{1,4}[A-Z]{0,3})((?:\([0-9A-Za-z]{1,4}\))*)", re.I)


def parse_table(text, sec):
    """Parse a repeal/revocation table segment into omissions of the target section."""
    ins = []
    for m in RE_TBL_WORDS_IN.finditer(text):
        for sm in RE_TBL_SECLIST.finditer(m.group("lst")):
            if sm.group(1).upper() == sec:
                ins.append(dict(op="omit", old=m.group("w"), chain=_tags(sm.group(2) or ""), defn=None))
    for em in RE_TBL_ENTRY.finditer(text):
        body = em.group("body").strip()
        sm = RE_TBL_SEC.match(body)
        if not sm or sm.group("sec").upper() != sec:
            continue
        base_chain = _tags(sm.group("chain") or "")
        rest = body[sm.end():].strip(" ,-")
        if not rest:
            if base_chain:
                ins.append(dict(op="omitprov", tags=[base_chain[-1]], chain=base_chain[:-1], defn=None))
            continue
        for part in re.split(r";\s*", rest):
            part = part.strip(" ,-")
            if not part:
                continue
            chain = list(base_chain)
            m = SCOPE_RX.search(part)
            if m:
                chain = chain + [c for c in _tags(m.group(2)) if c not in chain]
            b = dict(chain=chain, defn=None)
            m = RE_WORDS_FROM.search(part)
            if m:
                ins.append(dict(b, op="omitspan", a=m.group("a"), b=m.group("b") or "")); continue
            m = RE_WORD_AT_END.search(part)
            if m:
                ins.append(dict(b, op="omitnear", old=m.group("w"), tag=m.group("tag"))); continue
            m = DEFN_RX.search(part)
            if m:
                ins.append(dict(b, op="omitdefn", name=m.group(1))); continue
            qs = re.findall(r"\"([^\"]{1,300})\"", part)
            if qs:
                for q in qs:
                    ins.append(dict(b, op="omit", old=q))
                continue
            m = re.search(r"(?:sub-?sections?|subsections?|paragraphs?|sub-?paragraphs?)\s+"
                          r"((?:\([0-9A-Za-z]{1,4}\))(?:\s*(?:,|and)\s*\([0-9A-Za-z]{1,4}\))*)", part, re.I)
            if m:
                for tg in _tags(m.group(1)):
                    ins.append(dict(op="omitprov", tags=[tg], chain=base_chain, defn=None))
    return ins


# ---------------------------------------------------------------- applying
def _bounds(s, chain, mk):
    """Span for a scope chain, backing off to the deepest resolvable prefix."""
    if not chain:
        return 0, len(s)
    sp = span_of(s, chain, mk)
    if sp is None:
        for k in range(len(chain) - 1, 0, -1):
            sp = span_of(s, chain[:k], mk)
            if sp:
                return sp
        return 0, len(s)
    return sp


def _defn_span(s, name, lo, hi):
    """Span of a defined term's entry: from its opening quote to the closing ';'."""
    p = s.find('"%s"' % name, lo, hi)
    if p < 0:
        return None
    q = s.find(";", p)
    return (p, (q + 1) if 0 <= q < hi else hi)


_LEAD_PUNCT = ",;:.)!?"


def _join(left, new):
    """Concatenate keeping legislative spacing: no space before leading punctuation."""
    left = left.rstrip()
    if new[:1] in _LEAD_PUNCT:
        return left + new
    return left + " " + new


def _join_r(new, right):
    if right[:1] in _LEAD_PUNCT or not right:
        return new + right
    return new + " " + right.lstrip() if not new.endswith(" ") else new + right.lstrip()


_NEEDS_NEW = {"sub", "subspan", "after", "before", "subprov", "insprov",
              "subdefn", "insdefn", "approp", "endins"}


def apply_one(s, e):
    """Apply one instruction to text s; return (new_s, applied?)."""
    op = e["op"]
    if op in _NEEDS_NEW and not (e.get("new") or "").strip():
        return s, False          # empty replacement means the parse failed, not a deletion
    mk = markers(s)
    lo, hi = _bounds(s, e.get("chain") or [], mk)
    if e.get("defn") and op in ("sub", "omit", "after", "before", "omitspan"):
        d = _defn_span(s, e["defn"], lo, hi)
        if d:
            lo, hi = d
    seg = s[lo:hi]

    def put(new_seg):
        return s[:lo] + new_seg + s[hi:], True

    if op in ("sub", "omit", "after", "before"):
        old = (e.get("old") or "").strip()
        if not old:
            return s, False
        p = _find(seg, old)
        if p < 0:
            p = _find(s, old)
            if p < 0:
                return s, False
            lo, hi, seg = 0, len(s), s
        if op == "sub":
            if e.get("every"):
                return put(_replace_all(seg, old, e["new"])[0])
            return put(seg[:p] + e["new"] + seg[p + len(old):])
        if op == "omit":
            return put(re.sub(r"\s{2,}", " ", seg[:p] + seg[p + len(old):]))
        if op == "after":
            q = p + len(old)
            return put(_join(seg[:q], e["new"]) + seg[q:])
        return put(_join_r(seg[:p] + e["new"], seg[p:]))
    if op in ("omitspan", "subspan"):
        a = seg.find(e["a"])
        if a < 0:
            return s, False
        if e.get("b"):
            b = seg.find(e["b"], a)
            end = (b + len(e["b"])) if b >= 0 else len(seg)
        else:
            end = len(seg)
        rep = e.get("new", "")
        return put(re.sub(r"\s{2,}", " ", (_join_r(seg[:a] + rep, seg[end:]) if rep
                                           else seg[:a] + seg[end:])))
    if op == "omitnear":
        sp = span_of(s, (e.get("chain") or []) + [e["tag"]], mk)
        a, b = (max(0, sp[0] - 60), min(len(s), sp[1] + 10)) if sp else (lo, hi)
        w = e["old"]
        p = s.find(w, a, b)
        if p < 0:
            return s, False
        return s[:p] + s[p + len(w):], True
    if op == "subprov":
        tags = e["tags"]
        a = span_of(s, (e.get("chain") or []) + [tags[0]], mk)
        if a is None:
            return s, False
        b = span_of(s, (e.get("chain") or []) + [tags[-1]], mk) if len(tags) > 1 else a
        end = b[1] if b else a[1]
        return _join_r(s[:a[0]] + e["new"], s[end:]), True
    if op == "insprov":
        a = span_of(s, (e.get("chain") or []) + [e["tags"][-1]], mk)
        if a is None:
            return s, False
        return _join_r(_join(s[:a[1]], e["new"]), s[a[1]:]), True
    if op == "omitprov":
        ok = False
        for tg in e["tags"]:
            a = span_of(s, (e.get("chain") or []) + [tg], markers(s))
            if a:
                s = re.sub(r"\s{2,}", " ", s[:a[0]] + s[a[1]:]); ok = True
        return s, ok
    if op == "subdefn":
        d = _defn_span(s, e["name"], lo, hi)
        if d is None:
            return s, False
        return _join_r(s[:d[0]] + e["new"], s[d[1]:]), True
    if op == "insdefn":
        d = _defn_span(s, e["name"], lo, hi)
        if d is None:
            return s, False
        return _join_r(_join(s[:d[1]], e["new"]), s[d[1]:]), True
    if op == "omitdefn":
        d = _defn_span(s, e["name"], lo, hi)
        if d is None:
            return s, False
        return s[:d[0]] + s[d[1]:], True
    if op == "approp":
        return put(_insert_in_order(seg, e["new"]))
    if op == "endins":
        return _join_r(_join(s[:hi].rstrip().rstrip("."), e["new"]), s[hi:]), True
    return s, False


_ITEM_KEY = re.compile(r"(\d+)([A-Z]*)")


def _order_key(item):
    m = _ITEM_KEY.search(item)
    return (int(m.group(1)), m.group(2)) if m else (10 ** 9, "")


def _insert_in_order(seg, new):
    """Insert a list entry ("section 25B(1);") at its ordered place in a ';'-separated list."""
    items = [x for x in re.split(r";", seg)]
    if len(items) < 2:
        return _join(seg.rstrip().rstrip("."), new)
    k = _order_key(new)
    if k[0] == 10 ** 9:
        return _join(seg.rstrip().rstrip("."), new)
    pos = None
    for j, it in enumerate(items):
        if it.strip() and _order_key(it)[0] != 10 ** 9 and _order_key(it) > k:
            pos = j; break
    piece = new.rstrip().rstrip(";")
    if pos is None:
        return _join(seg.rstrip().rstrip("."), new)
    items.insert(pos, " " + piece)
    return ";".join(items)


RE_ACTREF_ANY = re.compile(r"\b[A-Z][\w'\-]*(?:\s+[\w'()\-,&]+){0,12}?\s+Act\s+(?:1[6-9]|20)\d\d")


def focus(text, sec, act_positions, limit=4000):
    """For long provisions, keep only the stretch belonging to the target act/section."""
    if len(text) <= limit:
        return text
    pats = [m.start() for m in re.finditer(r"\bs(?:ection|\.)?\s*" + re.escape(sec) + r"\b", text, re.I)]
    if not pats:
        return text[:limit]
    acts = sorted(act_positions) if act_positions else []
    best = None
    for p in pats:
        a = max([x for x in acts if x <= p + 80], default=None)
        d = (p - a) if a is not None else 10 ** 6
        if best is None or d < best[0]:
            best = (d, p, a)
    _, p, a = best
    lo = a if a is not None else max(0, p - 300)
    nxt = [m.start() for m in RE_ACTREF_ANY.finditer(text) if m.start() > p + 40]
    hi = min(nxt[0] if nxt else len(text), lo + limit)
    return text[lo:max(hi, p + 400)]


def all_instructions(text, sec, act_positions, is_table):
    """Edits from one provision: clause parsing, plus table parsing when it looks tabular."""
    ins = [] if is_table else parse_instructions(text[:60000], sec)
    if is_table or not ins:
        ins = ins + parse_table(focus(text, sec, act_positions, 6000), sec)
        if is_table and not ins:
            ins = parse_instructions(focus(text, sec, act_positions, 6000), sec)
    return ins


def reconstruct(enacted, plan, sec):
    """plan: list of (text, act_positions, is_table) in chronological order."""
    s = enacted
    nap = 0
    for t, ap, tbl in plan:
        for e in all_instructions(t, sec, ap, tbl):
            s, ok = apply_one(s, e)
            nap += ok
    return re.sub(r"\s+", " ", s).strip(), nap
