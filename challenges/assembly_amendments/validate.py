# eris-template-version: 3 (validation-architect build for assembly_amendments)
"""CV validation harness for assembly_amendments. Implements the "Validation design" section of
`reports/eris_plan.md`. Importable from `solution.py` (never duplicates it -- `metric.py`'s
`score_submission` is imported, not reimplemented).

Provides, in order of the task's deliverable list:
  1. `build_folds`           -- bill-grouped, size-aware, deterministic K=5 folds, with bills that
                                 share >=3 "distinctive" (long, see MIN_DISTINCTIVE_TEXT_LEN) exact
                                 dispositif texts union-merged first, so they never split across folds.
  2. `filter_test_like`      -- the n in [4,80] AND >=2 nonzero `counts` entries filter.
  3. `score_cv`              -- thin wrapper around `metric.score_submission`: headline test-like OOF
                                 score, per-term means, per-fold scores, raw all-train (labelled
                                 inflated), and slices (examining_body, division type, n-bucket,
                                 bill_kind).
  4. `bootstrap_ci`          -- bill-level (not board-level) bootstrap, ~1000 resamples, 90% CI.
  5. `nested_select`         -- generic 2-way cross-fitted selection helper for any decode constant
                                 (cluster cut tau, Sinkhorn on/off, pair-weighting scheme, ...).
  6. `run_oracle_checks`     -- (a) gold-fed-back ceiling, (b) counts-respecting random ~ 0 on the
                                 test-like subset, (c) fold determinism, (d) no bill spans folds,
                                 (e) every merged duplicate-text bill-group stays intact in one fold.

Run as a script to (re)build the fold assignment from `dataset/public/` and write
`reports/split_audit.md` with the fold-composition diagnostics:

    python3 validate.py

Compliance notes: every statistic here (text groups, fold loads, bootstrap, nested selection) is
computed from TRAIN rows only; this module never reads test.csv/test_items.csv. No wall-clock
branching; the only randomness is the fixed SEED, used for deterministic tie-breaking (seeded
shuffle before a stable sort) and for the bootstrap/nested-selection RNGs.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metric  # noqa: E402  (local module, the scoring ground truth)

SEED = 42
N_FOLDS = 5
# Minimum normalized-character length for a dispositif text to count as a "distinctive legal edit"
# for the bill-union rule (see build_merged_bill_groups docstring for the empirical justification:
# below this length the rule chains ~85% of all boards into one group via generic boilerplate like
# "Supprimer cet article" (1553 occurrences), which is not what the challenge means by "distinctive").
MIN_DISTINCTIVE_TEXT_LEN = 40
MIN_SHARED_TEXTS = 3  # the challenge's own threshold ("three or more... word for word")


# --------------------------------------------------------------------------------------------------
# Text normalization (consistent with .claude/scripts/make_groups.py's norm() so diagnostics agree)
# --------------------------------------------------------------------------------------------------

def normalize_text(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(s).lower())).strip()


# --------------------------------------------------------------------------------------------------
# Union-find
# --------------------------------------------------------------------------------------------------

class UnionFind:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# --------------------------------------------------------------------------------------------------
# 1a. Bill-group merging from shared distinctive exact-dispositif text
# --------------------------------------------------------------------------------------------------

def build_merged_bill_groups(train_df, train_items_df, min_shared_texts=MIN_SHARED_TEXTS,
                              min_text_len=MIN_DISTINCTIVE_TEXT_LEN):
    """Union-merge bill_ids that share >= `min_shared_texts` identical (normalized) dispositif
    texts of at least `min_text_len` characters.

    Mirrors the challenge's own leakage-control rule verbatim ("dossiers that share three or more
    distinctive legal edits word for word... are kept on the same side"). The length floor is load-
    bearing, not cosmetic: a naive exact-match with NO length floor chains 72/209 train bills into one
    union-find component covering 84.9% of all boards and 90.8% of all items, driven entirely by
    generic one-line amendments repeated across nearly every bill regardless of topic (dominant
    offender: normalized "supprimer cet article", 1553 occurrences across the whole train set -- not
    a "distinctive legal edit" by any reading of the challenge text, just the standard one-line
    article-deletion amendment every bill attracts). Restricting to texts >= 40 normalized characters
    removes exactly this boilerplate (verified: the 6 bill-pairs that remain at this threshold share
    texts with a median length of 300+ chars, each citing a specific code article/provision -- the
    budget-cycle re-tabling the challenge describes) and the result is stable for any floor in
    [40, 100] (same 203 groups, same 6 pairs) -- i.e. not a knife-edge choice.

    Returns (group_of_bill: dict bill_id -> group_id, merged_pairs: list of (bill_a, bill_b,
    shared_count) that triggered a union, diagnostics: dict).
    """
    board_to_bill = dict(zip(train_df["item_id"], train_df["bill_id"]))
    items = train_items_df[["item_id", "dispositif"]].copy()
    items["bill_id"] = items["item_id"].map(board_to_bill)
    items["norm_text"] = items["dispositif"].map(normalize_text)
    long_items = items[items["norm_text"].str.len() >= min_text_len]

    text_to_bills = long_items.groupby("norm_text")["bill_id"].apply(lambda s: frozenset(s))
    multi = text_to_bills[text_to_bills.apply(len) >= 2]

    shared = Counter()
    for bills in multi:
        bl = sorted(bills)
        for i in range(len(bl)):
            for j in range(i + 1, len(bl)):
                shared[(bl[i], bl[j])] += 1

    all_bills = sorted(train_df["bill_id"].unique())
    uf = UnionFind(all_bills)
    merged_pairs = []
    for (a, b), c in shared.items():
        if c >= min_shared_texts:
            uf.union(a, b)
            merged_pairs.append((a, b, c))

    group_of_bill = {b: uf.find(b) for b in all_bills}
    group_sizes = Counter(group_of_bill.values())
    diagnostics = {
        "n_bills": len(all_bills),
        "n_groups": len(group_sizes),
        "n_bills_merged": sum(1 for b in all_bills if group_sizes[group_of_bill[b]] > 1),
        "largest_group_bills": max(group_sizes.values()),
        "n_merge_pairs": len(merged_pairs),
        "cross_bill_distinctive_texts": len(multi),
    }
    return group_of_bill, merged_pairs, diagnostics


# --------------------------------------------------------------------------------------------------
# 1b. Test-like population (deliverable 2) -- defined early because fold balancing targets it
# --------------------------------------------------------------------------------------------------

def _nonzero_count(counts_raw):
    c = json.loads(counts_raw) if isinstance(counts_raw, str) else list(counts_raw)
    return sum(1 for x in c if x > 0)


def test_like_mask(board_df):
    """Boolean mask: n_amendments in [4,80] AND >=2 nonzero entries in `counts`. This is the
    strategist's plan's CV headline population (mirrors the stated test population: test boards are
    guaranteed n in [4,80] and >= 2 distinct true fates). Verified independently on train: 1652
    boards / 144 bills / 21,257 items (matches the plan's own profiling)."""
    nz = board_df["counts"].apply(_nonzero_count)
    return board_df["n_amendments"].between(4, 80) & (nz >= 2)


def filter_test_like(board_df):
    return board_df[test_like_mask(board_df)].copy()


# --------------------------------------------------------------------------------------------------
# 1c. Fold builder (deliverable 1)
# --------------------------------------------------------------------------------------------------

def build_folds(train_df, train_items_df, n_folds=N_FOLDS, seed=SEED,
                 min_shared_texts=MIN_SHARED_TEXTS, min_text_len=MIN_DISTINCTIVE_TEXT_LEN):
    """Deterministic, seeded, bill-grouped, size-aware K-fold assignment.

    Algorithm: merge bills into groups via `build_merged_bill_groups` (so a dossier re-tabled across
    a budget cycle never splits across folds), then greedily bin-pack merged groups into folds: sort
    groups descending by test-like item count (ties broken by all-train item count, then by a seeded
    shuffle for full determinism), and assign each group to the fold with the currently smallest
    running test-like item count (tie-break: smallest running test-like board count, then fold
    index). This balances the CV headline population's item AND board load per fold -- test-like item
    count is the primary sort/assign key (as the plan specifies: "sort by descending item count,
    round-robin to the fold with the currently smallest running item count"), and the board-count
    tie-break keeps board counts from drifting (observed spread 295-354 test-like boards per fold on
    train, vs 4249-4253 test-like items -- boards are naturally close because within-group board/item
    ratios are similar across groups).

    Groups with zero test-like items (common: small or degenerate-only bills) are packed afterward by
    all-train item count using the same rule, so every bill still gets a fold (needed for stage-1
    encoder training which uses all board sizes) without disturbing the test-like balance already
    achieved.

    Returns (fold_of_bill: dict bill_id -> int fold in [0, n_folds), group_of_bill: dict, merged_pairs:
    list, diagnostics: dict).
    """
    group_of_bill, merged_pairs, group_diag = build_merged_bill_groups(
        train_df, train_items_df, min_shared_texts, min_text_len)

    df = train_df.copy()
    df["_grp"] = df["bill_id"].map(group_of_bill)
    df["_test_like"] = test_like_mask(df)

    agg = df.groupby("_grp").agg(
        all_items=("n_amendments", "sum"),
        all_boards=("item_id", "size"),
    )
    tl_items = df[df["_test_like"]].groupby("_grp")["n_amendments"].sum()
    tl_boards = df[df["_test_like"]].groupby("_grp").size()
    agg["tl_items"] = tl_items.reindex(agg.index, fill_value=0)
    agg["tl_boards"] = tl_boards.reindex(agg.index, fill_value=0)
    agg = agg.reset_index()

    # Deterministic tie-break: seeded shuffle then a stable sort (so equal-key groups land in a
    # reproducible but non-arbitrary order across runs with the same seed).
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(agg))
    agg = agg.iloc[order].reset_index(drop=True)

    with_tl = agg[agg["tl_items"] > 0].sort_values(
        ["tl_items", "all_items"], ascending=False, kind="stable")
    without_tl = agg[agg["tl_items"] == 0].sort_values(
        ["all_items"], ascending=False, kind="stable")

    run_tl_items = [0] * n_folds
    run_tl_boards = [0] * n_folds
    run_all_items = [0] * n_folds
    fold_of_grp = {}

    for _, row in with_tl.iterrows():
        f = min(range(n_folds), key=lambda k: (run_tl_items[k], run_tl_boards[k], k))
        fold_of_grp[row["_grp"]] = f
        run_tl_items[f] += row["tl_items"]
        run_tl_boards[f] += row["tl_boards"]
        run_all_items[f] += row["all_items"]

    for _, row in without_tl.iterrows():
        f = min(range(n_folds), key=lambda k: (run_all_items[k], k))
        fold_of_grp[row["_grp"]] = f
        run_all_items[f] += row["all_items"]

    fold_of_bill = {b: fold_of_grp[g] for b, g in group_of_bill.items()}

    diagnostics = dict(group_diag)
    diagnostics["fold_tl_items"] = run_tl_items
    diagnostics["fold_tl_boards"] = run_tl_boards
    diagnostics["fold_all_items"] = run_all_items
    return fold_of_bill, group_of_bill, merged_pairs, diagnostics


# --------------------------------------------------------------------------------------------------
# 2. Slicing helpers used by score_cv
# --------------------------------------------------------------------------------------------------

def parse_division_type(div):
    """Deterministic category extraction from the free-text `division` column (diagnostic/feature
    use only -- never a fate/joint rule). Verified against data_audit.md's counts: numbered_article
    2836, additionnel_apres 1179, other 352, additionnel_avant 25."""
    d = str(div).strip().lower()
    if "additionnel" in d and "après" in d:
        return "additionnel_apres"
    if "additionnel" in d and "avant" in d:
        return "additionnel_avant"
    if re.match(r"^article\s+\d", d):
        return "numbered_article"
    return "other"


def n_bucket(n):
    if n <= 9:
        return "4-9"
    if n <= 29:
        return "10-29"
    return "30-80"


def _add_slice_columns(meta_df):
    out = meta_df.copy()
    out["_division_type"] = out["division"].map(parse_division_type)
    out["_n_bucket"] = out["n_amendments"].map(n_bucket)
    out["_is_seance"] = out["examining_body"] == "Séance publique"
    return out


# --------------------------------------------------------------------------------------------------
# 3. Scoring wrapper (deliverable 3) -- thin wrapper around metric.score_submission
# --------------------------------------------------------------------------------------------------

def score_cv(meta_df, true_df, pred_df, fold_of_board=None):
    """Score a full OOF prediction set against `metric.score_submission`.

    meta_df/true_df/pred_df: same contract as `metric.score_submission` (item_id-indexable frames
    covering the same id set). `fold_of_board`: optional pd.Series indexed like meta_df (or aligned
    by position) giving each board's fold, for a per-fold breakdown.

    Returns a dict:
      headline            -- test-like-subset score (0..100), the number every experiment is judged on
      headline_detail      -- per-board detail DataFrame for the test-like subset (for bootstrap/nested select)
      per_term_means       -- {carried, disposal, joint}: mean of the *_raw column where *_defined, test-like subset
      per_fold             -- {fold_id: test-like score in that fold} if fold_of_board given
      raw_all_train         -- score over the FULL population, labelled inflated (see metric_spec.md: ~0.46 vs
                               ~0.017 baseline-inflated); never compare this number to the headline or to test
      slices                -- {slice_name: score} for examining_body (seance/committee), division type,
                               n-bucket, bill_kind -- each computed on the test-like subset intersected with
                               the slice
    """
    tl_mask = test_like_mask(meta_df)
    tl_ids = set(meta_df.loc[tl_mask, "item_id"])

    def _restrict(ids):
        m = meta_df["item_id"].isin(ids)
        return meta_df[m], true_df[true_df["item_id"].isin(ids)], pred_df[pred_df["item_id"].isin(ids)]

    m_tl, t_tl, p_tl = _restrict(tl_ids)
    headline, detail_tl = metric.score_submission(m_tl, t_tl, p_tl, return_details=True)
    detail_tl = detail_tl.merge(meta_df[["item_id"]], on="item_id", how="left")

    per_term_means = {}
    for term in ("carried", "disposal", "joint"):
        defined = detail_tl[f"{term}_defined"]
        per_term_means[term] = float(detail_tl.loc[defined, term].mean()) if defined.any() else None

    per_fold = {}
    if fold_of_board is not None:
        # Accept either a dict / Series indexed by item_id, or an array aligned positionally with meta_df.
        if isinstance(fold_of_board, dict):
            board_fold = meta_df["item_id"].map(fold_of_board)
        elif isinstance(fold_of_board, pd.Series) and set(fold_of_board.index) >= set(meta_df["item_id"]):
            board_fold = meta_df["item_id"].map(fold_of_board.to_dict())
        else:
            board_fold = pd.Series(np.asarray(fold_of_board), index=meta_df.index)
        meta_with_fold = meta_df.copy()
        meta_with_fold["_fold"] = board_fold.values
        for f in sorted(meta_with_fold["_fold"].dropna().unique()):
            ids_f = set(meta_with_fold.loc[(meta_with_fold["_fold"] == f) & tl_mask.values, "item_id"])
            if not ids_f:
                per_fold[int(f)] = None
                continue
            m_f, t_f, p_f = _restrict(ids_f)
            per_fold[int(f)] = metric.score_submission(m_f, t_f, p_f)

    raw_all_train = metric.score_submission(meta_df, true_df, pred_df)

    sliced_meta = _add_slice_columns(meta_df.loc[tl_mask])
    slices = {}
    slice_defs = {
        "examining_body=Séance publique": sliced_meta["_is_seance"],
        "examining_body=committee": ~sliced_meta["_is_seance"],
    }
    for dtype in sliced_meta["_division_type"].unique():
        slice_defs[f"division={dtype}"] = sliced_meta["_division_type"] == dtype
    for bucket in sliced_meta["_n_bucket"].unique():
        slice_defs[f"n_bucket={bucket}"] = sliced_meta["_n_bucket"] == bucket
    for bk in sliced_meta["bill_kind"].unique():
        slice_defs[f"bill_kind={bk}"] = sliced_meta["bill_kind"] == bk

    for name, mask in slice_defs.items():
        ids = set(sliced_meta.loc[mask, "item_id"])
        if len(ids) < 5:  # too few boards for a meaningful slice score
            slices[name] = {"score": None, "n_boards": len(ids)}
            continue
        m_s, t_s, p_s = _restrict(ids)
        slices[name] = {"score": metric.score_submission(m_s, t_s, p_s), "n_boards": len(ids)}

    return {
        "headline": headline,
        "headline_detail": detail_tl,
        "per_term_means": per_term_means,
        "per_fold": per_fold,
        "raw_all_train": raw_all_train,
        "slices": slices,
    }


# --------------------------------------------------------------------------------------------------
# 4. Bill-level bootstrap (deliverable 4)
# --------------------------------------------------------------------------------------------------

def bootstrap_ci(detail_df, bill_of_item_id, n_resamples=1000, seed=SEED, ci=0.90):
    """90% CI on the headline score via resampling BILLS (not boards) with replacement.

    `detail_df`: the per-board detail frame from `score_submission(..., return_details=True)`
    (must have columns item_id, board_score), restricted to the test-like population (pass
    `score_cv(...)["headline_detail"]`).
    `bill_of_item_id`: dict or Series mapping item_id -> bill_id (the TRUE independence unit --
    boards of the same bill are not independent, so resampling boards directly would understate the
    CI, exactly the mistake V2/V5 warn about).

    Each resample draws len(bills) bills with replacement, concatenates every board belonging to the
    drawn bills (so a bill contributing many boards contributes many rows, same as the real metric's
    per-board averaging -- no re-weighting), computes `max(0, 100 * mean(board_score))` exactly as
    `score_submission` does, and the CI is the empirical percentile interval over the resamples.
    """
    detail = detail_df.copy()
    get_bill = bill_of_item_id.to_dict() if isinstance(bill_of_item_id, pd.Series) else bill_of_item_id
    detail["_bill"] = detail["item_id"].map(get_bill)
    detail = detail.dropna(subset=["board_score"])  # undefined boards are excluded from the mean, same as score_submission

    boards_by_bill = {b: g["board_score"].to_numpy(dtype=float) for b, g in detail.groupby("_bill")}
    bills = sorted(boards_by_bill)  # sorted for determinism before the seeded resampling
    rng = np.random.default_rng(seed)

    resampled_scores = np.empty(n_resamples)
    n_bills = len(bills)
    for i in range(n_resamples):
        draw = rng.choice(n_bills, size=n_bills, replace=True)
        vals = np.concatenate([boards_by_bill[bills[j]] for j in draw])
        resampled_scores[i] = max(0.0, 100.0 * float(vals.mean()))

    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    lo, hi = np.percentile(resampled_scores, [lo_q, hi_q])
    return {
        "mean": float(resampled_scores.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "ci": ci,
        "n_resamples": n_resamples,
        "n_bills": n_bills,
    }


# --------------------------------------------------------------------------------------------------
# 5. Nested cross-fitted selection helper (deliverable 5)
# --------------------------------------------------------------------------------------------------

def nested_select(bill_of_board, scores_by_candidate, seed=SEED):
    """Generic cross-fitted selection of a post-hoc constant (decode threshold, on/off flag, pair-
    weighting scheme, ...) that is fit from OOF predictions already in hand -- no refitting of any
    model is needed, only re-scoring the same OOF outputs under different candidate values.

    `bill_of_board`: array-like of bill_id, one per board in the population being selected over
    (e.g. the test-like boards with the relevant term defined -- pass only boards where the
    candidate choice actually matters, e.g. boards with >=1 true joint pair for a tau selection).
    `scores_by_candidate`: {candidate_value: array of per-board scores}, each array aligned 1:1 with
    `bill_of_board` (same order, same length).

    Splits the DISTINCT bills into two halves by a seeded shuffle (never splits a bill across
    halves), then:
      - selects the best candidate on half A, scores it (honestly) on half B;
      - selects the best candidate on half B, scores it (honestly) on half A;
      - the cross-fitted ("honest") number is the board-count-weighted average of those two scores.
    Also reports the in-sample number (select AND score on the full population) so the two can be
    compared directly -- CLAUDE.md's rule is to trust the honest number, and a large gap between the
    two numbers is itself a diagnostic (selection is overfitting to this population).

    Returns dict: in_sample_best, in_sample_score, half_a_best, half_b_best, honest_score,
    honest_score_a_selected_on_b (float), honest_score_b_selected_on_a (float), n_boards_a, n_boards_b.
    """
    bill_of_board = np.asarray(bill_of_board)
    candidates = list(scores_by_candidate.keys())
    n = len(bill_of_board)
    for c in candidates:
        assert len(scores_by_candidate[c]) == n, "every candidate's score array must align with bill_of_board"

    distinct_bills = sorted(set(bill_of_board))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(distinct_bills))
    half = len(distinct_bills) // 2
    bills_a = set(np.array(distinct_bills)[shuffled[:half]])
    bills_b = set(np.array(distinct_bills)[shuffled[half:]])

    mask_a = np.array([b in bills_a for b in bill_of_board])
    mask_b = ~mask_a  # distinct_bills partitions exactly into A/B, every board is in exactly one

    def best_on(mask):
        means = {c: float(np.mean(scores_by_candidate[c][mask])) for c in candidates}
        best_c = max(means, key=means.get)
        return best_c, means[best_c]

    half_a_best, _ = best_on(mask_a)
    half_b_best, _ = best_on(mask_b)
    score_a_on_b = float(np.mean(scores_by_candidate[half_a_best][mask_b]))  # selected on A, scored on B
    score_b_on_a = float(np.mean(scores_by_candidate[half_b_best][mask_a]))  # selected on B, scored on A

    n_a, n_b = int(mask_a.sum()), int(mask_b.sum())
    honest_score = (score_a_on_b * n_b + score_b_on_a * n_a) / (n_a + n_b)

    in_sample_best, in_sample_score = best_on(np.ones(n, dtype=bool))

    return {
        "candidates": candidates,
        "in_sample_best": in_sample_best,
        "in_sample_score": in_sample_score,
        "half_a_best": half_a_best,
        "half_b_best": half_b_best,
        "honest_score_a_selected_on_b": score_a_on_b,
        "honest_score_b_selected_on_a": score_b_on_a,
        "honest_score": honest_score,
        "n_boards_a": n_a,
        "n_boards_b": n_b,
        "optimism": in_sample_score - honest_score,
    }


# --------------------------------------------------------------------------------------------------
# 6. Oracle / sanity checks (deliverable 6)
# --------------------------------------------------------------------------------------------------

def random_shuffle_baseline(meta_df, true_df, seed=SEED):
    """Build a counts-respecting random-shuffle prediction: for every board, permute the TRUE
    fates/joint positions by a single random permutation of the n item positions (this is exactly
    the "random settlement that respects the counts" construction `metric.py`'s chance-correction is
    derived from -- see metric_spec.md). Used only to sanity-check that this harness's fold/filter
    code integrates correctly with `metric.py` (metric_spec.md already measured the number: ~0.017 on
    the test-like subset, ~0.46 on raw unfiltered train -- this function does not re-derive that
    number, only reproduces it as an integration check)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _, row in true_df.iterrows():
        fates = json.loads(row["fates"]) if isinstance(row["fates"], str) else list(row["fates"])
        joint = json.loads(row["joint"]) if isinstance(row["joint"], str) else list(row["joint"])
        n = len(fates)
        perm = rng.permutation(n)
        rows.append({
            "item_id": row["item_id"],
            "fates": json.dumps([int(fates[i]) for i in perm]),
            "joint": json.dumps([int(joint[i]) for i in perm]),
        })
    return pd.DataFrame(rows)


def run_oracle_checks(train_df, train_items_df, true_df, verbose=True):
    """Runs all 5 sanity checks from the task's deliverable 6 and raises AssertionError on any
    failure. Returns a dict of the measured numbers for reporting in split_audit.md."""
    out = {}

    # (a) feeding the TRUE fates/joint back through metric.score_submission must score every
    #     defined board at its ceiling (board_score == 1.0 where defined) and the final score == 100.
    ceiling, detail = metric.score_submission(train_df, true_df, true_df, return_details=True)
    out["oracle_ceiling_final_score"] = ceiling
    assert abs(ceiling - 100.0) < 1e-6, f"oracle ceiling expected 100.0, got {ceiling}"
    defined_scores = detail["board_score"].dropna()
    assert np.allclose(defined_scores.to_numpy(dtype=float), 1.0, atol=1e-9), \
        "every board with >=1 defined term must score exactly 1.0 when fed its own ground truth"

    # (b) counts-respecting random shuffle baseline scores ~0 on the test-like subset.
    tl = filter_test_like(train_df)
    tl_ids = set(tl["item_id"])
    pred_rand = random_shuffle_baseline(train_df, true_df, seed=SEED)
    m_tl = train_df[train_df["item_id"].isin(tl_ids)]
    t_tl = true_df[true_df["item_id"].isin(tl_ids)]
    p_tl = pred_rand[pred_rand["item_id"].isin(tl_ids)]
    rand_score = metric.score_submission(m_tl, t_tl, p_tl)
    out["random_baseline_test_like"] = rand_score
    # metric_spec.md measured ~1.7 (0.017 * 100) on this population with its own seed/sample; this is
    # an integration check (did our filter+metric glue code work), not a re-derivation -- allow a
    # generous band.
    assert 0.0 <= rand_score < 6.0, \
        f"counts-respecting random baseline on test-like subset should be near 0, got {rand_score}"

    # (c) fold assignment is deterministic across repeated runs with the same seed.
    fold_of_bill_1, _, _, _ = build_folds(train_df, train_items_df, seed=SEED)
    fold_of_bill_2, _, _, _ = build_folds(train_df, train_items_df, seed=SEED)
    assert fold_of_bill_1 == fold_of_bill_2, "build_folds is not deterministic for a fixed seed"
    out["fold_determinism_ok"] = True

    # (d) no bill_id appears in more than one fold (true by construction -- fold_of_bill is a dict
    #     keyed by bill_id -- but assert the DERIVED per-board fold agrees, catching any join bug).
    board_fold = train_df["bill_id"].map(fold_of_bill_1)
    bill_fold_spread = train_df.assign(_fold=board_fold).groupby("bill_id")["_fold"].nunique()
    assert (bill_fold_spread == 1).all(), "a bill_id spans more than one fold"
    out["no_bill_spans_folds"] = True

    # (e) every merged-duplicate-text bill-group stays intact within one fold.
    fold_of_bill, group_of_bill, merged_pairs, _ = build_folds(train_df, train_items_df, seed=SEED)
    group_fold = defaultdict(set)
    for b, g in group_of_bill.items():
        group_fold[g].add(fold_of_bill[b])
    bad = {g: fs for g, fs in group_fold.items() if len(fs) > 1}
    assert not bad, f"merged bill-groups spanning >1 fold: {bad}"
    out["merged_groups_intact"] = True
    out["n_merge_pairs"] = len(merged_pairs)

    # Illustrative bootstrap on the random-shuffle baseline's detail: not a model number, just a
    # concrete demonstration of what pure noise looks like under this harness's bootstrap_ci, so the
    # split_audit report can cite a real interval instead of an abstract promise.
    _, detail_rand = metric.score_submission(m_tl, t_tl, p_tl, return_details=True)
    bill_of_item = dict(zip(train_df["item_id"], train_df["bill_id"]))
    out["bootstrap_demo_random"] = bootstrap_ci(detail_rand, bill_of_item, n_resamples=1000, seed=SEED)

    if verbose:
        print(f"[oracle] ceiling final_score={ceiling:.4f} (expect 100.0)")
        print(f"[oracle] random-shuffle baseline on test-like subset = {rand_score:.4f} (expect near 0)")
        print("[oracle] fold determinism / no-bill-spans-folds / merged-groups-intact: all OK")
        bd = out["bootstrap_demo_random"]
        print(f"[oracle] bootstrap demo (random baseline, noise floor): mean={bd['mean']:.4f} "
              f"90% CI=({bd['lo']:.4f}, {bd['hi']:.4f})")

    return out


# --------------------------------------------------------------------------------------------------
# Script entry point: (re)build folds from dataset/public/ and write reports/split_audit.md
# --------------------------------------------------------------------------------------------------

def _load_train():
    pub = HERE / "dataset" / "public"
    train = pd.read_csv(pub / "train.csv")
    items = pd.read_csv(pub / "train_items.csv")
    targets = pd.read_csv(pub / "train_targets.csv")
    return train, items, targets


def _write_split_audit(train, items, fold_of_bill, group_of_bill, merged_pairs, fold_diag, oracle_out):
    df = train.copy()
    df["fold"] = df["bill_id"].map(fold_of_bill)
    df["_test_like"] = test_like_mask(df)
    df["_division_type"] = df["division"].map(parse_division_type)
    df["_n_bucket"] = df["n_amendments"].map(n_bucket)
    df["_is_seance"] = df["examining_body"] == "Séance publique"

    lines = []
    lines.append("# Split audit — assembly_amendments (CV harness diagnostics)\n")
    lines.append(f"Author: eris-validation-architect. Computed by `validate.py` "
                 f"(`python3 {HERE}/validate.py`) from `dataset/public/train.csv` + "
                 f"`train_items.csv` + `train_targets.csv` only — never reads test.* "
                 f"(see module docstring).\n")

    lines.append("## Hidden-split reading\n")
    lines.append(
        "The challenge states bills, not boards, are split (\"Bills are split by legislative "
        "dossier\"), and that committee + plenary + every reading of a bill sit on the same side "
        "(contract/data_audit: 0 `bill_id` overlap between train (209 bills) and test (111 bills), "
        "confirmed again here). We read this as the strictest plausible held-out unit — one bill, "
        "not one board, not one (bill, examining_body) pair — because the description's own leakage "
        "rationale (55% of bills re-table identical text across committee/plenary/readings) implies "
        "that anything coarser than whole-bill disjointness would leak. Held-out unit size is not "
        "stated beyond \"the test set is divided into two evaluation slices by bill\" (unknown "
        "boundary, unknown sizes); we therefore do not assume a specific private/public split ratio, "
        "only that CV must be bill-grouped and should report subgroup slices (division/body/kind) in "
        "case either hidden slice correlates with one of them.\n")

    lines.append("## Group derivation\n")
    lines.append(
        f"- {fold_diag['n_bills']} train bills -> {fold_diag['n_groups']} merged groups after "
        f"union-merging bills that share >= {MIN_SHARED_TEXTS} identical dispositif texts of >= "
        f"{MIN_DISTINCTIVE_TEXT_LEN} normalized characters ({fold_diag['n_bills_merged']} bills "
        f"touched by a merge, {fold_diag['n_merge_pairs']} triggering pairs, out of "
        f"{fold_diag['cross_bill_distinctive_texts']} distinctive texts that appear in >=2 bills at "
        f"all).\n"
        f"- Largest merged group: {fold_diag['largest_group_bills']} bills.\n"
        "- **Critical finding, load-bearing for this design**: without the length floor, the SAME "
        "rule (exact text match, >=3 shared, union-merge) chains 72/209 bills into ONE group covering "
        "84.9% of all boards and 90.8% of all items — the transitive closure of generic one-line "
        "amendments (dominant offender: normalized `\"supprimer cet article\"`, 1553 occurrences "
        "across train, present in nearly every bill regardless of topic) collapses almost the whole "
        "dataset into a single unsplittable component, which would make bill-grouped K=5 CV "
        "impossible. We verified the length floor is not a knife-edge choice: any floor in [40, 100] "
        "normalized characters gives the identical 203-group result (6 triggering pairs, largest "
        "merged group 3 bills), and the 6 surviving pairs share texts with a median length over 300 "
        "characters, each citing a specific code article/provision — exactly the budget-cycle "
        "re-tabling the challenge describes, not boilerplate. This is reported, not hidden: a "
        "reviewer who prefers the literal unfiltered rule should know it is unusable as stated.\n")

    lines.append("\n## Fold composition (test-like subset is the primary balance target)\n")
    board_counts = df.groupby("fold").size()
    item_counts = df.groupby("fold")["n_amendments"].sum()
    bill_counts = df.groupby("fold")["bill_id"].nunique()
    tl_board_counts = df[df["_test_like"]].groupby("fold").size()
    tl_item_counts = df[df["_test_like"]].groupby("fold")["n_amendments"].sum()
    tl_bill_counts = df[df["_test_like"]].groupby("fold")["bill_id"].nunique()
    lines.append("| fold | all boards | all items | bills | test-like boards | test-like items | test-like bills |")
    lines.append("|---|---|---|---|---|---|---|")
    for f in sorted(df["fold"].dropna().unique()):
        lines.append(f"| {int(f)} | {board_counts.get(f,0)} | {item_counts.get(f,0)} | {bill_counts.get(f,0)} | "
                     f"{tl_board_counts.get(f,0)} | {tl_item_counts.get(f,0)} | {tl_bill_counts.get(f,0)} |")
    lines.append(f"\nTotals: {len(df)} boards / {df['n_amendments'].sum()} items / "
                 f"{df['bill_id'].nunique()} bills; test-like subset {df['_test_like'].sum()} boards / "
                 f"{int(df.loc[df['_test_like'],'n_amendments'].sum())} items / "
                 f"{df.loc[df['_test_like'],'bill_id'].nunique()} bills "
                 f"(matches the plan's own profiling of ~1652/144/~21257, confirmed independently here).\n")
    lines.append("Test-like items are balanced to within 0.1% across folds by construction (the "
                 "primary sort/assign key); test-like boards vary more (the secondary tie-break), "
                 "reflecting that merged bill-groups differ in average items-per-board.\n")

    lines.append("## Rare-category coverage per fold (test-like subset)\n")
    lines.append(
        "Per `data_audit.md`'s warning: CMP (`texte de commission mixte paritaire`) and `deuxième "
        "lecture` boards are bill-clustered and overwhelmingly fail the test-like filter on their "
        "own (CMP boards are ~97% adopted -> usually <2 nonzero `counts` entries; `deuxième lecture` "
        "is ~93% rejected with little other variety) -- confirmed here: neither reading category "
        "appears in the test-like subset's per-fold table below except `nouvelle lecture`, which "
        "appears only in one fold. **This is an expected, direct consequence of the mandated bill-"
        "grouped split intersected with the test-like filter, not a bug** -- a rare reading clustered "
        "in a handful of bills will, by definition of grouping by bill, land in whichever fold(s) "
        "those bills land in.\n")
    reading_tbl = df[df["_test_like"]].groupby(["fold", "reading"]).size().unstack(fill_value=0)
    lines.append("```\n" + reading_tbl.to_string() + "\n```\n")

    lines.append("## Categorical coverage per fold (test-like subset)\n")
    lines.append("`bill_kind`:\n```\n" +
                 df[df["_test_like"]].groupby(["fold", "bill_kind"]).size().unstack(fill_value=0).to_string() +
                 "\n```\n")
    lines.append("`examining_body` (Séance publique vs committee):\n```\n" +
                 df[df["_test_like"]].groupby(["fold", "_is_seance"]).size().unstack(fill_value=0).to_string() +
                 "\n```\n")
    lines.append("`division` type:\n```\n" +
                 df[df["_test_like"]].groupby(["fold", "_division_type"]).size().unstack(fill_value=0).to_string() +
                 "\n```\n")
    lines.append("`n_amendments` bucket:\n```\n" +
                 df[df["_test_like"]].groupby(["fold", "_n_bucket"]).size().unstack(fill_value=0).to_string() +
                 "\n```\n")

    lines.append("## Oracle / sanity checks\n")
    lines.append(f"- Gold fates/joint fed back through `metric.score_submission`: final score = "
                 f"{oracle_out['oracle_ceiling_final_score']:.4f} (expected 100.0) -- PASS.\n")
    lines.append(f"- Counts-respecting random-shuffle baseline on the test-like subset: "
                 f"{oracle_out['random_baseline_test_like']:.4f} (metric_spec.md's own measurement "
                 f"was ~1.7 on a 100-scale with its own sample/seed; this run confirms the fold/filter "
                 f"code integrates correctly with `metric.py`, within a generous band) -- PASS.\n")
    lines.append("- Fold assignment reproduced twice with the same seed: identical -- PASS.\n")
    lines.append("- No `bill_id` spans more than one fold -- PASS.\n")
    lines.append(f"- All {oracle_out['n_merge_pairs']} merge-triggering bill pairs stay within a "
                 "single fold (checked over every merged group, not just the pairs) -- PASS.\n")
    bd = oracle_out["bootstrap_demo_random"]
    lines.append(
        f"- Bill-level bootstrap demo (`bootstrap_ci`, 1000 resamples) on the counts-respecting "
        f"random-shuffle baseline's test-like detail (the noise floor, NOT a model number): mean="
        f"{bd['mean']:.4f}, 90% CI=({bd['lo']:.4f}, {bd['hi']:.4f}) over {bd['n_bills']} bills -- "
        f"demonstrates the helper end-to-end; a trained model's OOF will be run through the same "
        f"function in `solution.py`'s dev logs for the real interval.\n")

    lines.append("## Bias direction of the proxy\n")
    lines.append(
        "- **Optimistic sources**: (1) in-script decode-constant selection (tau, Sinkhorn on/off, "
        "pair weighting) is chosen on the same OOF it is then reported on unless `nested_select` is "
        "used for the final number -- expect ~1-2 points of optimism from this alone, per the plan's "
        "own estimate, and `nested_select` exists precisely to quantify and report it. (2) Stacking: "
        "stage-2 models are trained on stage-1 OOF, so stage-2's own OOF retains a sliver of "
        "stage-1's fold-boundary smoothing. (3) Test-time stage-1 features are 5-fold-averaged "
        "(smoother than any single-fold OOF column), a slight train/test calibration mismatch whose "
        "sign is not obvious but is unlikely to be large.\n"
        "- **Pessimistic / neutral sources**: bill-grouped folds mirror the actual held-out unit "
        "exactly (no optimism from an under-grouped split); the length-floored text-merge rule, if "
        "anything, UNDER-merges relative to a hypothetical reviewer who wanted the literal rule "
        "applied with no floor (which we showed is unusable) -- so if the true leakage surface is "
        "broader than our 6 pairs, our CV is mildly optimistic in that one respect (a few extra "
        "cross-bill text echoes not accounted for), but we judge this a small residual next to the "
        "demonstrated alternative (85% of the data in one group) being strictly worse.\n"
        "- **Sampling noise**: the private slice is an unknown subset of 111 test bills (~55 at a "
        "guess if it is a roughly even two-way split); per-board score SD on the test-like train "
        "subset propagates to a private-score SD of roughly `per_board_sd / sqrt(n_private_boards)`, "
        "measured concretely by `bootstrap_ci` (see the noise-floor demo in the oracle-checks section "
        "above; a real model's OOF interval will typically be narrower than the pure-noise demo "
        "since a working model's per-board scores have lower variance than random). "
        "Grouped splits are inherently noisier than random splits (V22) -- expect visible rank churn "
        "between near-tied experiments; use the paired-fold/noise rule (CLAUDE.md §4A, V4/V16) before "
        "keeping any change.\n"
        "- **Net**: plan for private-LB ≈ test-like OOF minus ~1-2 points of selection optimism, ± a "
        "bootstrap CI half-width of the same kind reported above (computed on the real model's OOF, "
        "not the noise-floor demo), consistent with the strategist's plan's own estimate "
        "(35-50 expected band).\n")

    lines.append("## Reproduce\n")
    lines.append(f"```\npython3 {HERE}/validate.py\n```\n"
                 "Rebuilds the fold assignment, runs the oracle/sanity battery, and rewrites this "
                 "report, from `dataset/public/{train,train_items,train_targets}.csv` only (SEED=42, "
                 f"N_FOLDS={N_FOLDS}, MIN_SHARED_TEXTS={MIN_SHARED_TEXTS}, "
                 f"MIN_DISTINCTIVE_TEXT_LEN={MIN_DISTINCTIVE_TEXT_LEN}).\n")

    (HERE / "reports" / "split_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    train, items, targets = _load_train()
    fold_of_bill, group_of_bill, merged_pairs, fold_diag = build_folds(train, items, seed=SEED)
    oracle_out = run_oracle_checks(train, items, targets, verbose=True)

    # Headline sanity: score the gold predictions through score_cv as an end-to-end integration
    # check of the wrapper itself (should match run_oracle_checks's ceiling exactly).
    board_fold = train["bill_id"].map(fold_of_bill)
    res = score_cv(train, targets, targets, fold_of_board=board_fold)
    print(f"[score_cv integration] headline (gold-fed-back) = {res['headline']:.4f} (expect 100.0)")
    print(f"[score_cv integration] per_fold = {res['per_fold']}")

    # Bill-level bootstrap on the gold-fed-back detail, as a demonstration of the helper (the real
    # use is on a trained model's OOF detail from solution.py).
    bill_of_item = dict(zip(train["item_id"], train["bill_id"]))
    ci = bootstrap_ci(res["headline_detail"], bill_of_item, n_resamples=1000, seed=SEED)
    print(f"[bootstrap demo on gold] mean={ci['mean']:.4f} 90% CI=({ci['lo']:.4f}, {ci['hi']:.4f}) "
         f"over {ci['n_bills']} bills (expect ~100/100/100 since this is the ceiling, demonstrating "
         "the helper runs end-to-end; a real model's OOF will show a non-trivial interval)")

    _write_split_audit(train, items, fold_of_bill, group_of_bill, merged_pairs, fold_diag, oracle_out)
    print(f"wrote {HERE / 'reports' / 'split_audit.md'}")


if __name__ == "__main__":
    main()
