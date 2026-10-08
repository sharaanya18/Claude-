"""
metric.py — exact implementation of the assembly_amendments evaluation metric.

Verbatim metric text (challenge description, "Evaluation" section):

    Each test board receives up to three terms, each corrected so that an
    uninformed answer scores zero.

    carried, weight 0.35: adopted or not. If the fates list has the wrong
    length, an unreadable value, or uses any code a different number of times
    than the counts say, this term and the next are -1. Otherwise the share
    of amendments put on the right side of adopted / not adopted is
    chance-corrected against a random settlement that respects the counts:
    (earned - expected) / (1 - expected). A board with no adopted amendment
    is not scored on this term.

    disposal, weight 0.35: for the amendments that were not adopted, how they
    ended (rejected, fell, withdrawn, not moved). The share placed on the
    right fate, where placing one among the adopted counts as wrong, is
    chance-corrected the same way.

    joint, weight 0.30: the adjusted Rand index between your grouping and the
    recorded one, so leaving every amendment alone scores 0, and so does
    putting them all in one group unless that is the recorded grouping. A
    list of the wrong length or one that cannot be read scores 0. Boards
    where no two amendments were called together (227 of the 622 test
    boards) are not scored on this term.

    A board's score is the weighted mean of the terms defined on it, each
    floored at -1. The final score is 100 times the mean board score,
    floored at 0.

Derivation of "expected" (the chance-correction baseline), confirmed by
brute-force exhaustive enumeration over small n in dev (see
reports/metric_spec.md for the full derivation and the enumeration check):

The "random settlement that respects the counts" is a uniformly random
permutation of the FULL true fate multiset (all n items, respecting the full
[a, r, f, w, m] counts) laid back down over the n board positions. For any
fixed position i whose true fate class has size c (one of a, r, f, w, m out
of n total items), the marginal probability that the random settlement
reproduces a specific fate value at position i is c / n (standard marginal
of a uniform random permutation / sampling-without-replacement argument —
true regardless of correlations between positions, since we only need
linearity of expectation over per-position indicators).

    carried:  expected = (a^2 + (n-a)^2) / n^2
    disposal: expected = (r^2 + f^2 + w^2 + m^2) / (n * (n-a))   [n-a = # non-adopted]

Both formulas were cross-checked against brute-force exhaustive enumeration
of all distinct permutations of the true fate multiset for many small (n<=5)
boards and matched to float precision in every case tried — see
reports/metric_spec.md and tests_metric.py.

Degenerate edge case (expected == 1, i.e. 1 - expected == 0): this happens
for `carried` iff a == 0 or a == n, and for `disposal` iff a == 0 AND all
non-adopted items share the single same true fate. In every such case the
counts-validity constraint on a VALID submission forces earned == 1 as well
(there is only one multiset-respecting arrangement), so we define the term
as 1.0 in that branch rather than dividing 0/0. This is common in train
(no size/variety guarantee) but never occurs in test (test boards are
guaranteed >=2 distinct true fates). See reports/metric_spec.md.
"""
import json
import numpy as np
from sklearn.metrics import adjusted_rand_score

FATE_CODES = (0, 1, 2, 3, 4)  # adopted, rejected, fell, withdrawn, not moved
WEIGHTS = {"carried": 0.35, "disposal": 0.35, "joint": 0.30}
_EPS = 1e-9


# --------------------------------------------------------------------------
# Parsing / validity
# --------------------------------------------------------------------------

def _parse_json_list(raw):
    """Parse `raw` (a JSON-list string, or already a python list/tuple) into
    a python list. Returns (list_or_None, ok_bool)."""
    if isinstance(raw, (list, tuple, np.ndarray)):
        return list(raw), True
    if not isinstance(raw, str):
        return None, False
    try:
        val = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, False
    if not isinstance(val, list):
        return None, False
    return val, True


def parse_fates(raw, n, counts):
    """Parse + validate a `fates` cell against the board's true counts.

    Invalid (returns (None, False)) on: unparseable / not a list, wrong
    length, a value that is not one of the integer codes 0..4, or a
    multiplicity of any code that differs from `counts`.
    """
    val, ok = _parse_json_list(raw)
    if not ok:
        return None, False
    if len(val) != n:
        return None, False
    arr = np.empty(n, dtype=np.int64)
    for i, x in enumerate(val):
        # strict-ish int check: accept python/numpy ints, reject bool/float/str
        if isinstance(x, bool):
            return None, False
        if isinstance(x, (int, np.integer)):
            code = int(x)
        else:
            return None, False
        if code not in FATE_CODES:
            return None, False
        arr[i] = code
    counts = np.asarray(counts, dtype=np.int64)
    for code in FATE_CODES:
        if int(np.sum(arr == code)) != int(counts[code]):
            return None, False
    return arr, True


def parse_joint(raw, n):
    """Parse + validate a `joint` cell. Invalid on unparseable/not-a-list or
    wrong length. Label *values* are unconstrained integers (only which
    amendments share a label matters)."""
    val, ok = _parse_json_list(raw)
    if not ok:
        return None, False
    if len(val) != n:
        return None, False
    out = []
    for x in val:
        if isinstance(x, bool):
            return None, False
        if isinstance(x, (int, np.integer)):
            out.append(int(x))
        else:
            return None, False
    return np.asarray(out, dtype=np.int64), True


# --------------------------------------------------------------------------
# Chance-correction baselines
# --------------------------------------------------------------------------

def carried_expected(a, n):
    """Expected hit-rate of a uniform random counts-respecting settlement on
    the adopted-vs-not split."""
    return (a ** 2 + (n - a) ** 2) / (n ** 2)


def disposal_expected(r, f, w, m, n, na):
    """Expected hit-rate (as a share of the `na` = n-a non-adopted items) of
    a uniform random counts-respecting settlement on the 4-way disposal
    fate, restricted to the true-non-adopted subset."""
    return (r * r + f * f + w * w + m * m) / (n * na)


# --------------------------------------------------------------------------
# Per-term scoring (called only when the relevant preconditions hold; the
# caller in score_board() handles validity / definedness / exclusion)
# --------------------------------------------------------------------------

def carried_term(true_fates, pred_fates, a, n):
    true_adopted = true_fates == 0
    pred_adopted = pred_fates == 0
    earned = float(np.mean(true_adopted == pred_adopted))
    expected = carried_expected(a, n)
    denom = 1.0 - expected
    if denom <= _EPS:
        # degenerate: a==0 (excluded before this is ever called) or a==n.
        # a==n forces every valid submission to earn exactly 1.
        return 1.0 if earned >= expected - _EPS else -1.0
    return (earned - expected) / denom


def disposal_term(true_fates, pred_fates, counts, n):
    a = int(counts[0])
    na = n - a
    nonadopted_mask = true_fates != 0
    matches = pred_fates[nonadopted_mask] == true_fates[nonadopted_mask]
    earned = float(np.mean(matches))
    r, f, w, m = int(counts[1]), int(counts[2]), int(counts[3]), int(counts[4])
    expected = disposal_expected(r, f, w, m, n, na)
    denom = 1.0 - expected
    if denom <= _EPS:
        # degenerate: only when a==0 AND all non-adopted items share one true
        # fate, in which case any valid submission is forced to earn exactly 1.
        return 1.0 if earned >= expected - _EPS else -1.0
    return (earned - expected) / denom


def joint_term(true_joint, pred_joint_raw, n):
    """Returns (score_or_None, defined_bool).

    defined=False iff the true partition is singleton-only (no two
    amendments called together) -- the term is excluded from the board's
    weighted mean in that case, regardless of the prediction.
    Malformed predicted joint (when defined) scores 0, NOT -1 (asymmetric
    with the fates/-1 rule -- verbatim: "A list of the wrong length or one
    that cannot be read scores 0").
    """
    true_joint = np.asarray(true_joint)
    if len(np.unique(true_joint)) == n:
        return None, False  # all-singleton truth: excluded
    pred_arr, ok = parse_joint(pred_joint_raw, n)
    if not ok:
        return 0.0, True
    score = float(adjusted_rand_score(true_joint, pred_arr))
    return score, True


# --------------------------------------------------------------------------
# Board-level and submission-level scoring
# --------------------------------------------------------------------------

def score_board(n, counts, true_fates_raw, true_joint_raw, pred_fates_raw, pred_joint_raw):
    """Score one board. `counts` = [a, r, f, w, m] (list/array of 5 ints
    summing to n). Ground truth (`true_*_raw`) is assumed well-formed (it is
    asserted). Predictions (`pred_*_raw`) may be malformed JSON-list-shaped
    strings or already-parsed python lists/arrays.

    Returns a dict with raw (unfloored, pre-exclusion) term values, defined
    flags, fates_valid flag, and the board's final weighted-mean board_score
    (None if literally no term is definable on this board -- does not occur
    given the data's constraints, see reports/metric_spec.md).
    """
    counts = [int(c) for c in counts]
    a = counts[0]
    na = n - a

    true_fates, true_fates_ok = parse_fates(true_fates_raw, n, counts)
    assert true_fates_ok, "ground truth fates must be well-formed"
    true_joint, true_joint_ok = parse_joint(true_joint_raw, n)
    assert true_joint_ok, "ground truth joint must be well-formed"

    pred_fates, fates_valid = parse_fates(pred_fates_raw, n, counts)

    if not fates_valid:
        # Verbatim: "this term and the next are -1" -- applied unconditionally
        # on malformation, regardless of a==0 / a==n (judgment call, see spec).
        carried_raw, carried_defined = -1.0, True
        disposal_raw, disposal_defined = -1.0, True
    else:
        if a == 0:
            carried_raw, carried_defined = None, False
        else:
            carried_raw, carried_defined = carried_term(true_fates, pred_fates, a, n), True
        if na == 0:
            disposal_raw, disposal_defined = None, False
        else:
            disposal_raw, disposal_defined = disposal_term(true_fates, pred_fates, counts, n), True

    joint_raw, joint_defined = joint_term(true_joint, pred_joint_raw, n)

    terms = {}
    if carried_defined:
        terms["carried"] = max(carried_raw, -1.0)
    if disposal_defined:
        terms["disposal"] = max(disposal_raw, -1.0)
    if joint_defined:
        terms["joint"] = max(joint_raw, -1.0)

    if terms:
        wsum = sum(WEIGHTS[k] for k in terms)
        board_score = sum(WEIGHTS[k] * terms[k] for k in terms) / wsum
    else:
        board_score = None  # should not occur; see reports/metric_spec.md

    return {
        "carried": carried_raw, "carried_defined": carried_defined,
        "disposal": disposal_raw, "disposal_defined": disposal_defined,
        "joint": joint_raw, "joint_defined": joint_defined,
        "fates_valid": fates_valid,
        "board_score": board_score,
    }


def score_submission(meta_df, true_df, pred_df, return_details=False):
    """Score a full submission.

    meta_df: DataFrame with columns item_id, counts (JSON-string or list) --
             e.g. train.csv/test.csv.
    true_df: DataFrame with columns item_id, fates, joint (ground truth) --
             e.g. train_targets.csv.
    pred_df: DataFrame with columns item_id, fates, joint (submission /
             predictions to be scored).
    All three must cover exactly the same set of item_id values.

    Returns final_score (float): 100 * mean_over_boards(board_score), floored
    at 0 (the floor is applied once, at the very end, not per board).
    If return_details, also returns a per-board pandas DataFrame of the
    score_board() outputs (useful for CV debugging / error analysis).
    """
    import pandas as pd

    meta = meta_df.set_index("item_id")
    true = true_df.set_index("item_id")
    pred = pred_df.set_index("item_id")
    ids = list(meta.index)
    assert set(ids) == set(true.index) == set(pred.index), \
        "item_id sets must match exactly across meta_df/true_df/pred_df"

    rows = []
    for item_id in ids:
        counts_raw = meta.loc[item_id, "counts"]
        counts = json.loads(counts_raw) if isinstance(counts_raw, str) else list(counts_raw)
        n = int(sum(counts))
        r = score_board(
            n, counts,
            true.loc[item_id, "fates"], true.loc[item_id, "joint"],
            pred.loc[item_id, "fates"], pred.loc[item_id, "joint"],
        )
        r["item_id"] = item_id
        rows.append(r)

    board_scores = [row["board_score"] for row in rows if row["board_score"] is not None]
    mean_score = float(np.mean(board_scores)) if board_scores else 0.0
    final = max(0.0, 100.0 * mean_score)

    if return_details:
        return final, pd.DataFrame(rows)
    return final
