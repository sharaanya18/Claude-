"""Marginal Role Coverage@2 metric and metric-aware decode for complementary_entity_evidence.

Pure standard library (no numpy / pandas), so it can be pasted into solution.py unchanged.

Part A  - exact re-implementation of the description's grade():
    slate_score, marginal_role_coverage, grade_rows
Part B  - metric-aware decode (training-free, uses only model probabilities for ONE slate at a time):
    pair_tables_mc / choose_pair_mc_ratio       Monte-Carlo expected-ratio decode (fixed seed)
    choose_pair_expected_union                  closed-form expected-union decode
    pair_to_scores / decode_slate / decode_all  convert the chosen pair into within-slate ranking scores
Part C  - toy slate generator used by the tests and the decode comparison.

Compliance note: the decode operates on one slate at a time from that slate's own model outputs (per-sample
inference style); it never aggregates across test slates and never branches on wall-clock time.
"""
import json
import math
import random
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Part A: exact metric
# --------------------------------------------------------------------------------------


def _numeric(value) -> float:
    """Same coercion as the grader's _numeric (oracle JSON payloads map to their oracle_rank)."""
    if isinstance(value, str) and value.lstrip().startswith("{"):
        value = json.loads(value)["oracle_rank"]
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise Exception("Predictions must be finite real ranking scores.") from exc
    if not math.isfinite(value):
        raise Exception("Predictions must be finite real ranking scores.")
    return value


def best_pair_union(role_sets: Mapping[str, Iterable[str]]) -> int:
    """Largest union attainable by any pair in the slate (the denominator). Needs >= 2 candidates."""
    sets = {k: set(v) for k, v in role_sets.items()}
    ids = list(sets)
    if len(ids) < 2:
        raise Exception("A slate needs at least two candidates.")
    return max(len(sets[a] | sets[b]) for a, b in combinations(ids, 2))


def select_top2(scores: Mapping[str, float]) -> List[str]:
    """Two highest scores; score ties broken by ascending candidate id (string order), as in grade()."""
    return sorted(scores, key=lambda rid: (-scores[rid], rid))[:2]


def slate_score(selected_ids_or_scores, role_sets: Mapping[str, Iterable[str]]) -> float:
    """Slate score = |union(selected pair)| / max_pair |union|.

    selected_ids_or_scores is either
      * a mapping id -> score (pair chosen exactly like grade(): -score, then ascending id), or
      * a sequence of exactly two candidate ids (the selected pair).
    role_sets maps every candidate id of the slate to its hidden new-role set.
    Raises if the slate optimum is 0 (grade() raises "Scored slate has no new supported roles.").
    """
    sets = {k: set(v) for k, v in role_sets.items()}
    optimum = best_pair_union(sets)
    if optimum == 0:
        raise Exception("Scored slate has no new supported roles.")
    if isinstance(selected_ids_or_scores, Mapping):
        if set(selected_ids_or_scores) != set(sets):
            raise Exception("Scores must cover exactly the slate candidates.")
        selected = select_top2({k: _numeric(v) for k, v in selected_ids_or_scores.items()})
    else:
        selected = list(selected_ids_or_scores)
        if len(selected) != 2 or selected[0] == selected[1] or any(s not in sets for s in selected):
            raise Exception("Selected pair must be two distinct candidates of the slate.")
    return len(sets[selected[0]] | sets[selected[1]]) / optimum


def parse_gold_rows(rows: Iterable[Tuple[str, str]]) -> Dict[str, dict]:
    """rows = (id, target_json_string). Mirrors the grader's _answers() payload validation."""
    result: Dict[str, dict] = {}
    for rid, payload in rows:
        item = json.loads(payload)
        if set(item) != {"query_id", "roles", "slate_size", "oracle_rank"} or not isinstance(item["query_id"], str):
            raise Exception("Invalid answer payload.")
        roles = item["roles"]
        if not isinstance(roles, list) or any(not isinstance(r, str) or not r for r in roles) or len(roles) != len(set(roles)):
            raise Exception("Invalid answer role set.")
        if str(rid) in result:
            raise Exception("Answers must be non-empty with unique non-missing IDs.")
        result[str(rid)] = item
    if not result:
        raise Exception("Answers must be non-empty with unique non-missing IDs.")
    return result


def marginal_role_coverage(predictions: Mapping[str, float], gold: Mapping[str, dict],
                           return_per_slate: bool = False):
    """Marginal Role Coverage@2, identical to grade(): mean over slates (grouped by query_id) of slate scores.

    predictions: id -> finite score (larger = earlier).  gold: id -> {query_id, roles, slate_size, ...}.
    Validation copied from grade(): IDs must match exactly, every slate complete (len == slate_size, all members
    agree on slate_size, int in [2, 8]), optimum must be > 0.  Slate order of averaging is irrelevant (plain mean).
    """
    if not predictions:
        raise Exception("Submission must be non-empty without missing values.")
    if len(predictions) != len(gold) or set(map(str, predictions)) != set(gold):
        raise Exception("Submission IDs must match scored answer IDs exactly.")
    preds = {str(k): _numeric(v) for k, v in predictions.items()}
    queries: Dict[str, List[str]] = {}
    for rid, item in gold.items():
        queries.setdefault(item["query_id"], []).append(rid)
    scores = []
    for members in queries.values():
        expected = {gold[rid]["slate_size"] for rid in members}
        if (len(expected) != 1 or any(type(n) is not int or n < 2 or n > 8 for n in expected)
                or len(members) != next(iter(expected))):
            raise Exception("A scoring partition must retain complete candidate slates.")
        role_sets = {rid: set(gold[rid]["roles"]) for rid in members}
        scores.append(slate_score({rid: preds[rid] for rid in members}, role_sets))
    score = sum(scores) / len(scores)
    if not 0 <= score <= 1:
        raise Exception("Score outside [0,1].")
    return (score, scores) if return_per_slate else score


def grade_rows(submission_rows: Sequence[Tuple[str, object]], answer_rows: Sequence[Tuple[str, str]]) -> float:
    """List-of-tuples front end: submission_rows=(id, prediction), answer_rows=(id, target_json)."""
    ids = [str(r[0]) for r in submission_rows]
    if len(ids) != len(set(ids)):
        raise Exception("Duplicate submission IDs.")
    gold = parse_gold_rows(answer_rows)
    return marginal_role_coverage({str(i): p for i, p in submission_rows}, gold)


# --------------------------------------------------------------------------------------
# Part B: decode
# --------------------------------------------------------------------------------------
# Model: candidate c holds role r independently with probability p[c][r] (known seed roles already masked to 0).
# Closed form: E|A u B| = sum_r 1 - (1-pA_r)(1-pB_r).
# MC expected ratio: E[ |A u B| / max_pair |.| | optimum > 0 ].  Conditioning on optimum>0 divides every pair's
# value by the same constant P(opt>0), so it never changes the argmax; it only matters for the reported value.
# Closed-form == argmax of ratio-of-expectations E|A u B| / E[opt]  (same denominator for all pairs).
# The two differ because 1/opt varies by draw: the MC decode prefers pairs that are good in worlds where the
# slate optimum is small (ratio is then sensitive), i.e. it hedges toward reliable coverage.

PAIR_TOL = 1e-12


def mask_known_roles(role_probs: Mapping[str, Mapping[str, float]], known_roles: Iterable[str]) -> Dict[str, Dict[str, float]]:
    """Hard-mask the seed's known roles to probability zero (they are never 'new' by construction)."""
    known = set(known_roles)
    return {cid: {r: (0.0 if r in known else float(p)) for r, p in d.items()} for cid, d in role_probs.items()}


def _clean_probs(role_probs: Mapping[str, Mapping[str, float]]) -> Dict[str, Dict[str, float]]:
    out = {}
    for cid, d in role_probs.items():
        row = {}
        for r, p in d.items():
            p = float(p)
            if not math.isfinite(p) or p < 0.0 or p > 1.0:
                raise ValueError(f"probability out of [0,1] for {cid}/{r}: {p}")
            if p > 0.0:
                row[r] = p
        out[str(cid)] = row
    if len(out) < 2:
        raise ValueError("A slate needs at least two candidates.")
    return out


def expected_union_table(role_probs: Mapping[str, Mapping[str, float]]) -> Dict[Tuple[str, str], float]:
    """Closed-form E|union| for every pair (keys are id pairs in ascending id order)."""
    probs = _clean_probs(role_probs)
    ids = sorted(probs)
    table = {}
    for a, b in combinations(ids, 2):
        pa, pb = probs[a], probs[b]
        total = 0.0
        for r in set(pa) | set(pb):
            total += 1.0 - (1.0 - pa.get(r, 0.0)) * (1.0 - pb.get(r, 0.0))
        table[(a, b)] = total
    return table


def _argmax_pair(table: Mapping[Tuple[str, str], float]) -> Tuple[str, str]:
    """Pair with the largest value; near-ties (<= PAIR_TOL) go to the lexicographically smallest id pair."""
    best_pair, best_val = None, -math.inf
    for pair in sorted(table):
        if table[pair] > best_val + PAIR_TOL:
            best_pair, best_val = pair, table[pair]
    return best_pair


def choose_pair_expected_union(role_probs: Mapping[str, Mapping[str, float]]) -> Tuple[Tuple[str, str], Dict]:
    """Closed-form decode: argmax_pair E|union|."""
    table = expected_union_table(role_probs)
    return _argmax_pair(table), table


def pair_tables_mc(role_probs: Mapping[str, Mapping[str, float]], n_draws: int = 2000,
                   seed: int = 20240601) -> Tuple[Dict[Tuple[str, str], float], float]:
    """Monte-Carlo E[ratio | opt>0] for every pair, with common random numbers across pairs.

    Returns (table, p_opt_positive). Deterministic: a fresh random.Random(seed) per call and candidates processed
    in ascending id order, so the result does not depend on dict order or on which other slates are decoded.
    If all probabilities are 0/1 (e.g. gold oracle) a single draw is exact.
    """
    probs = _clean_probs(role_probs)
    ids = sorted(probs)
    roles = sorted({r for d in probs.values() for r in d})
    bit = {r: 1 << i for i, r in enumerate(roles)}
    fixed = []       # mask of roles with p == 1
    uncertain = []   # (bit, p) for 0 < p < 1
    for cid in ids:
        m = 0
        unc = []
        for r, p in probs[cid].items():
            if p >= 1.0:
                m |= bit[r]
            else:
                unc.append((bit[r], p))
        fixed.append(m)
        uncertain.append(unc)
    deterministic = not any(uncertain)
    draws = 1 if deterministic else int(n_draws)
    rng = random.Random(seed)
    rnd = rng.random
    idx_pairs = list(combinations(range(len(ids)), 2))
    sums = [0.0] * len(idx_pairs)
    n_pos = 0
    n = len(ids)
    for _ in range(draws):
        masks = list(fixed)
        for i in range(n):
            m = masks[i]
            for b, p in uncertain[i]:
                if rnd() < p:
                    m |= b
            masks[i] = m
        unions = [(masks[i] | masks[j]).bit_count() for i, j in idx_pairs]
        opt = max(unions)
        if opt == 0:
            continue
        n_pos += 1
        inv = 1.0 / opt
        for k, u in enumerate(unions):
            sums[k] += u * inv
    if n_pos == 0:
        return {(ids[i], ids[j]): 0.0 for i, j in idx_pairs}, 0.0
    table = {(ids[i], ids[j]): sums[k] / n_pos for k, (i, j) in enumerate(idx_pairs)}
    return table, n_pos / draws


def choose_pair_mc_ratio(role_probs: Mapping[str, Mapping[str, float]], n_draws: int = 2000,
                         seed: int = 20240601) -> Tuple[Tuple[str, str], Dict]:
    """MC expected-ratio decode. Falls back to the closed form when every draw has optimum 0 (all-zero probs)."""
    table, p_pos = pair_tables_mc(role_probs, n_draws, seed)
    if p_pos == 0.0:
        return choose_pair_expected_union(role_probs)
    return _argmax_pair(table), table


def pair_to_scores(role_probs: Mapping[str, Mapping[str, float]], pair: Tuple[str, str]) -> Dict[str, float]:
    """Within-slate scores: chosen pair on top (member with larger own expected new-role count first, ties by id),
    remaining candidates ordered by marginal expected new roles given the pair (sum_r p_c,r * prod_{s in pair}(1-p_s,r)),
    ties by ascending id.  Scores are distinct integers n, n-1, ..., 1, so the grader's tie rule is never exercised.
    """
    probs = _clean_probs(role_probs)
    ids = sorted(probs)
    a, b = pair
    own = {c: sum(probs[c].values()) for c in ids}
    first, second = sorted(pair, key=lambda c: (-own[c], c))
    rest = [c for c in ids if c not in pair]

    def marginal(c):
        total = 0.0
        for r, p in probs[c].items():
            miss = 1.0
            for s in (a, b):
                miss *= 1.0 - probs[s].get(r, 0.0)
            total += p * miss
        return total

    rest.sort(key=lambda c: (-marginal(c), c))
    order = [first, second] + rest
    n = len(order)
    return {c: float(n - i) for i, c in enumerate(order)}


def decode_slate(role_probs: Mapping[str, Mapping[str, float]], known_roles: Optional[Iterable[str]] = None,
                 method: str = "mc_ratio", n_draws: int = 2000, seed: int = 20240601) -> Dict[str, float]:
    """role_probs (one slate) -> id -> ranking score.  method in {'mc_ratio', 'expected_union'}."""
    if known_roles is not None:
        role_probs = mask_known_roles(role_probs, known_roles)
    if method == "mc_ratio":
        pair, _ = choose_pair_mc_ratio(role_probs, n_draws, seed)
    elif method == "expected_union":
        pair, _ = choose_pair_expected_union(role_probs)
    else:
        raise ValueError(method)
    return pair_to_scores(role_probs, pair)


def decode_all(role_probs: Mapping[str, Mapping[str, float]], slate_of: Mapping[str, str],
               known_by_slate: Optional[Mapping[str, Iterable[str]]] = None, method: str = "mc_ratio",
               n_draws: int = 2000, seed: int = 20240601) -> Dict[str, float]:
    """Decode every slate independently (grouping by slate key only) and merge the scores."""
    groups: Dict[str, Dict[str, Mapping[str, float]]] = {}
    for cid, d in role_probs.items():
        groups.setdefault(slate_of[cid], {})[cid] = d
    out: Dict[str, float] = {}
    for sk, members in groups.items():
        known = None if known_by_slate is None else known_by_slate.get(sk)
        out.update(decode_slate(members, known, method, n_draws, seed))
    return out


# --------------------------------------------------------------------------------------
# Part C: toy slates (tests / decode comparison only)
# --------------------------------------------------------------------------------------


def make_toy_slate(rng: random.Random, n_roles: int = 12, informative: float = 1.0, slate_key: str = "s0"):
    """One synthetic slate with calibrated uncertainty.

    Latent per-(candidate, role) probability q ~ Beta(0.35, 2.0) scaled by a Zipf-like role prior; the gold role set is
    Bernoulli(q); the model's probability is q (calibrated). informative in (0,1] shrinks q toward the role prior
    (less informative model).  Rejection-samples until the optimum > 0 and pair utilities differ, like the real prep.
    Returns (probs, gold_roles, ids) with probs: id -> {role: p}.
    """
    roles = [f"{'in' if i % 2 else 'out'}:rel{i // 2}" for i in range(n_roles)]
    prior = [min(0.9, 0.5 / (1 + i)) for i in range(n_roles)]
    while True:
        n = rng.randint(2, 8)
        ids = [f"ev_{slate_key}_{rng.getrandbits(40):010x}" for _ in range(n)]
        if len(set(ids)) != n:
            continue
        probs, gold = {}, {}
        for cid in ids:
            d, g = {}, set()
            for i, r in enumerate(roles):
                q = rng.betavariate(0.35, 2.0) * min(1.0, 2.5 * prior[i] * 2)
                p = informative * q + (1 - informative) * prior[i] * 0.5
                d[r] = p
                if rng.random() < q:
                    g.add(r)
            probs[cid], gold[cid] = d, g
        utils = {len(gold[a] | gold[b]) for a, b in combinations(ids, 2)}
        if max(utils) > 0 and len(utils) > 1:
            return probs, gold, ids
