"""Shared library: normalisation, label parsing, act-reference resolution, candidate generation."""
import re, collections, numpy as np

# ---------- normalisation ----------
_D = {ord(ch): "-" for ch in "‐‑‒–—―−⁃"}
_Q = {ord(ch): '"' for ch in "“”„«»″"}
_Q.update({ord(ch): "'" for ch in "‘’‚′"})
_NB = {0xa0: " ", 0x2007: " ", 0x202f: " ", 0xad: ""}
_TR = {**_D, **_Q, **_NB}

def norm(s):
    return re.sub(r"\s+", " ", str(s).translate(_TR)).strip()

def toks(s):
    """Token list used by the scorer: whitespace split of normalised text."""
    return norm(s).split()

# ---------- labels ----------
RE_L_SCH = re.compile(r"^Sch\.\s*([0-9A-Z]+)(?:\s+para\.\s*([0-9A-Za-z.]+))?")
RE_L_SEC = re.compile(r"^s\.\s*([0-9]+)([A-Z]*)")
RE_L_ART = re.compile(r"^(art|reg|rule|r|para|Pt)\.?\s*([0-9]+)([A-Z]*)")

def parse_label(lab):
    """Return (kind, order_tuple). kind in {'s','sch','art','other'}."""
    lab = norm(lab)
    m = RE_L_SEC.match(lab)
    if m:
        return "s", (0, int(m.group(1)), m.group(2), 0.0)
    m = RE_L_SCH.match(lab)
    if m:
        sch = m.group(1)
        schn = int(sch) if sch.isdigit() else 900
        para = m.group(2)
        pn = 0.0
        if para:
            parts = re.findall(r"\d+", para)
            pn = float(parts[0]) + (float(parts[1]) / 1000 if len(parts) > 1 else 0)
        return "sch", (2, schn, "", pn)
    m = RE_L_ART.match(lab)
    if m:
        return "art", (1, int(m.group(2)), m.group(3), 0.0)
    return "other", (3, 0, "", 0.0)

def sch_scope(lab):
    """Schedule identity for scoped propagation ('' for body provisions)."""
    m = RE_L_SCH.match(norm(lab))
    return m.group(1) if m else ""

# ---------- act identity ----------
RE_ACT = re.compile(
    r"\b((?:[A-Z][\w'’\-]*|and|of|the|for|in|to|\(Northern|Ireland\)|&|,)"
    r"(?:\s+(?:[A-Z][\w'’\-]*|and|of|the|for|in|to|\(Northern|Ireland\)|&|,)){0,14}?"
    r"\s+Act\s+((?:1[6-9]|20)\d\d))")
RE_CHAP = re.compile(r"^\s*\(?\s*c\.\s*(\d+)\)?")
RE_FULLCIT = re.compile(r"\b((?:1[6-9]|20)\d\d)\s+c\.\s*(\d+)\b")
RE_ABBR_MEANS = re.compile(r'"([A-Za-z0-9 ()\-\'’.]{2,40})"\s+means\s+(?:the\s+)?([^;.]{5,120}?Act\s+(?:19|20)\d\d)')
RE_ABBR_PAREN = re.compile(r'([A-Z][^;.("]{4,110}?Act\s+(?:19|20)\d\d)\s*(?:\(c\.\s*\d+\)\s*)?\(\s*"([^"]{2,40})"\s*\)')

def act_title_key(t):
    """Canonical key for an Act title string."""
    t = norm(t).lower()
    t = re.sub(r"^(the)\s+", "", t)
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def find_act_refs(text, title2cit):
    """Yield (citation_or_titlekey, start, end) for every Act reference in `text`."""
    out = []
    for m in RE_ACT.finditer(text):
        tk = act_title_key(m.group(1))
        if tk not in title2cit:
            # strip leading filler words until it resolves (handles "of the X Act 1992")
            parts = tk.split()
            for i in range(1, min(6, len(parts))):
                cand = " ".join(parts[i:])
                if cand in title2cit:
                    tk = cand
                    break
        cit = title2cit.get(tk)
        if cit is None:
            cm = RE_CHAP.match(text[m.end():m.end() + 12])
            if cm:
                cit = "%s c. %s" % (m.group(2), cm.group(1))
            else:
                cit = "T:" + tk
        out.append((cit, m.start(), m.end()))
    for m in RE_FULLCIT.finditer(text):
        out.append(("%s c. %s" % (m.group(1), m.group(2)), m.start(), m.end()))
    return out

# ---------- section references ----------
SN = r"\d{1,4}[A-Z]{0,3}"
RE_SECREF = re.compile(r"\bs(?:ection|ections)?\s+(" + SN + r")", re.I)
RE_SECREF2 = re.compile(r"\bss?\.\s*(" + SN + r")", re.I)
RE_SECRANGE = re.compile(r"\bsections\s+(" + SN + r")\s+to\s+(" + SN + r")", re.I)

def section_refs(text):
    """Yield (secnum, start, frame) for section mentions. frame is the 24 chars before."""
    out = []
    for rx in (RE_SECREF, RE_SECREF2):
        for m in rx.finditer(text):
            out.append((m.group(1).upper(), m.start(), text[max(0, m.start() - 26):m.start()]))
    return out

def expand_ranges(text):
    """Section numbers covered by 'sections A to B' ranges (numeric part only)."""
    got = set()
    for m in RE_SECRANGE.finditer(text):
        a, b = m.group(1), m.group(2)
        if a.isdigit() and b.isdigit() and int(b) - int(a) < 60:
            got.update(str(x) for x in range(int(a), int(b) + 1))
    return got
