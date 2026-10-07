"""Per-document commencement evidence, mined from each amending document's own provisions.

Separate commencement orders are almost absent from the corpus, so the only in-force
evidence available is what the amending document itself says.  These are learned features,
not a rule: the reranker decides how much weight an explicit date or a "two months" clause
deserves against the plain document date.
"""
import re, datetime, collections

RE_CIF = re.compile(r"com(?:es?|ing)\s+into\s+force|shall\s+come\s+into\s+force", re.I)
MONTHS = {m: k+1 for k, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"])}
RE_DATE = re.compile(r"\b([0-9]{1,2})(?:st|nd|rd|th)?\s+([A-Z][a-z]+)\s+((?:19|20)\d\d)\b")
RE_TWOMO = re.compile(r"end\s+of\s+the\s+period\s+of\s+(\w+)\s+months?\s+beginning", re.I)
RE_APPOINT = re.compile(r"such\s+day\s+as|by\s+order\s+appoint|may\s+by\s+(?:order|regulations)\s+appoint", re.I)
WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "six": 6, "twelve": 12}
RE_SCHREF = re.compile(r"\bSchedule\s+([0-9]{1,3})", re.I)
RE_SECREF = re.compile(r"\bsections?\s+([0-9]{1,4}[A-Z]{0,3})", re.I)


def _ord(y, m, d):
    try:
        return float(datetime.date(y, m, d).toordinal())
    except ValueError:
        return None


def parse_dates(t):
    out = []
    for m in RE_DATE.finditer(t):
        mm = MONTHS.get(m.group(2).lower())
        if mm:
            o = _ord(int(m.group(3)), mm, int(m.group(1)))
            if o:
                out.append((o, m.start()))
    return out


def build_cif(C):
    """citation -> commencement summary for the document."""
    out = {}
    for cc, idxs in C.doc.items():
        dates = []          # (ordinal, {sections}, {schedules})
        twomo = 0.0
        appoint = 0.0
        for i in idxs:
            t = C.text[i]
            if not RE_CIF.search(t):
                continue
            if RE_APPOINT.search(t):
                appoint = 1.0
            mt = RE_TWOMO.search(t)
            if mt:
                twomo = float(WORDNUM.get(mt.group(1).lower(), 2))
            for o, p in parse_dates(t):
                w = t[max(0, p - 500):p + 500]
                dates.append((o, set(x.upper() for x in RE_SECREF.findall(w)),
                              set(RE_SCHREF.findall(w))))
        out[cc] = dict(dates=dates, twomo=twomo, appoint=appoint)
    return out


def cif_feats(cif, cc, label, qd, docd):
    """Features for one candidate provision: evidence about when it came into force."""
    s = cif.get(cc) or dict(dates=[], twomo=0.0, appoint=0.0)
    f = dict(cif_n=float(len(s["dates"])), cif_twomo=s["twomo"], cif_appoint=s["appoint"],
             cif_lbl_days=0.0, cif_lbl_hit=0.0, cif_min_days=0.0, cif_max_days=0.0,
             cif_est_days=qd - docd)
    ms = re.match(r"Sch\.\s*([0-9]+)", label)
    sch = ms.group(1) if ms else None
    ms = re.match(r"s\.\s*([0-9]+[A-Z]*)", label)
    sec = ms.group(1).upper() if ms else None
    if s["dates"]:
        os_ = [o for o, _, _ in s["dates"]]
        f["cif_min_days"] = qd - min(os_)
        f["cif_max_days"] = qd - max(os_)
        hit = [o for o, secs, schs in s["dates"]
               if (sch and sch in schs) or (sec and sec in secs)]
        if hit:
            f["cif_lbl_hit"] = 1.0
            f["cif_lbl_days"] = qd - min(hit)
            f["cif_est_days"] = f["cif_lbl_days"]
        else:
            f["cif_est_days"] = qd - min(os_)
    elif s["twomo"]:
        f["cif_est_days"] = qd - (docd + 30.44 * s["twomo"])
    return f


CIF_FEATS = ["cif_n", "cif_twomo", "cif_appoint", "cif_lbl_days", "cif_lbl_hit",
             "cif_min_days", "cif_max_days", "cif_est_days"]
