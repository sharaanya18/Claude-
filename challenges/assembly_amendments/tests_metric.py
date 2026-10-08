"""
tests_metric.py — unit tests for metric.py (assembly_amendments).

Covers: perfect prediction, reversed/all-wrong prediction, Monte-Carlo chance
baseline (~0 in expectation), malformed fates (-1/-1), malformed joint (0,
not -1), a==0 exclusion, a==n exclusion, singleton-only joint exclusion, a
hand-enumerated n<=5 worked example cross-checked against brute-force
exhaustive enumeration, and an oracle-decode sanity check against real
train.csv / train_targets.csv data (ground truth scores ~100; a
counts-ignoring majority-vote baseline is heavily penalized; a
counts-respecting random shuffle baseline averages close to 0).

Run: python3 tests_metric.py   (or: python3 -m unittest tests_metric -v)
"""
import itertools
import json
import os
import random
import unittest

import numpy as np
import pandas as pd

import metric as M

HERE = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(HERE, "dataset", "public")

RNG = np.random.default_rng(12345)


def make_true_fates(counts):
    """[a,r,f,w,m] -> true fate vector [0]*a + [1]*r + ... (un-shuffled; the
    metric does not care about physical order between calls, only that the
    same fixed order is used for both the "true" array and any array we
    compare it against)."""
    out = []
    for code, c in enumerate(counts):
        out += [code] * c
    return np.array(out, dtype=np.int64)


class TestCarriedDisposalFormula(unittest.TestCase):
    """Cross-check the closed-form expected-value formulas against brute
    force exhaustive enumeration over the true fate multiset's distinct
    permutations, for every small board tried."""

    def brute_force_expected(self, counts):
        n = sum(counts)
        true = list(make_true_fates(counts))
        perms = set(itertools.permutations(true))
        N = len(perms)
        na = n - counts[0]
        nonadopted_idx = [i for i, t in enumerate(true) if t != 0]
        carried_sum, disposal_sum = 0.0, 0.0
        for p in perms:
            carried_sum += sum((p[i] == 0) == (true[i] == 0) for i in range(n)) / n
            if nonadopted_idx:
                disposal_sum += sum(p[i] == true[i] for i in nonadopted_idx) / len(nonadopted_idx)
        carried_exp = carried_sum / N
        disposal_exp = disposal_sum / N if nonadopted_idx else None
        return carried_exp, disposal_exp

    def test_formula_matches_brute_force(self):
        cases = [
            [1, 1, 1, 1, 0], [1, 2, 1, 0, 0], [2, 1, 1, 0, 0],
            [1, 1, 1, 1, 1], [2, 2, 0, 0, 0], [3, 1, 1, 0, 0],
            [0, 2, 1, 1, 0], [4, 0, 0, 0, 0],
        ]
        for counts in cases:
            n = sum(counts)
            a = counts[0]
            na = n - a
            be_c, be_d = self.brute_force_expected(counts)
            fc = M.carried_expected(a, n)
            self.assertAlmostEqual(fc, be_c, places=12, msg=f"carried mismatch for {counts}")
            if na > 0:
                fd = M.disposal_expected(counts[1], counts[2], counts[3], counts[4], n, na)
                self.assertAlmostEqual(fd, be_d, places=12, msg=f"disposal mismatch for {counts}")

    def test_hand_worked_n4_example(self):
        """n=4, counts=[2,1,1,0,0]: a=2,r=1,f=1,w=0,m=0.
        carried expected = (4+4)/16 = 0.5
        disposal (na=2): expected = (1+1+0+0)/(4*2) = 2/8 = 0.25
        Confirmed against brute force above (test_formula_matches_brute_force);
        here we also hand-verify via score_board with a concrete submission.
        """
        counts = [2, 1, 1, 0, 0]
        n = 4
        true_fates = [0, 0, 1, 2]  # item order: adopted, adopted, rejected, fell
        true_joint = [0, 1, 2, 3]  # no pairing (irrelevant to this test)
        # perfect prediction
        res = M.score_board(n, counts, json.dumps(true_fates), json.dumps(true_joint),
                             json.dumps(true_fates), json.dumps(true_joint))
        self.assertAlmostEqual(res["carried"], 1.0, places=12)
        self.assertAlmostEqual(res["disposal"], 1.0, places=12)
        # one carried swap: predict item0 as rejected(1), item2 as adopted(0) (still valid counts)
        pred_fates = [1, 0, 0, 2]
        res2 = M.score_board(n, counts, json.dumps(true_fates), json.dumps(true_joint),
                              json.dumps(pred_fates), json.dumps(true_joint))
        # earned carried: matches at positions where (pred==0)==(true==0):
        # true_adopted=[T,T,F,F], pred_adopted=[F,T,T,F] -> matches at idx1,idx3 -> earned=0.5
        expected_c = M.carried_expected(2, 4)  # 0.5
        self.assertAlmostEqual(expected_c, 0.5, places=12)
        self.assertAlmostEqual(res2["carried"], (0.5 - 0.5) / (1 - 0.5), places=12)  # == 0.0
        # disposal: true non-adopted idx = [2,3] with true fates [1,2]; pred fates there = [0,2]
        # match only at idx3 -> earned = 0.5; expected = 0.25 -> (0.5-0.25)/0.75
        self.assertAlmostEqual(res2["disposal"], (0.5 - 0.25) / 0.75, places=12)


class TestPerfectAndWorst(unittest.TestCase):
    def test_perfect_prediction_scores_max(self):
        counts = [3, 2, 1, 2, 1]
        n = sum(counts)
        true_fates = make_true_fates(counts).tolist()
        true_joint = [0, 0, 1, 2, 2, 3, 4, 5, 6]  # some pairs, some singles
        res = M.score_board(n, counts, json.dumps(true_fates), json.dumps(true_joint),
                             json.dumps(true_fates), json.dumps(true_joint))
        self.assertAlmostEqual(res["carried"], 1.0, places=9)
        self.assertAlmostEqual(res["disposal"], 1.0, places=9)
        self.assertAlmostEqual(res["joint"], 1.0, places=9)
        self.assertAlmostEqual(res["board_score"], 1.0, places=9)

    def test_full_submission_perfect_scores_100(self):
        # small synthetic submission of several boards, all predicted perfectly
        meta = pd.DataFrame({
            "item_id": ["b1", "b2", "b3"],
            "counts": [json.dumps([3, 2, 1, 2, 1]), json.dumps([0, 2, 2, 0, 0]), json.dumps([5, 0, 0, 0, 0])],
        })
        true_fates = {
            "b1": make_true_fates([3, 2, 1, 2, 1]).tolist(),
            "b2": make_true_fates([0, 2, 2, 0, 0]).tolist(),
            "b3": make_true_fates([5, 0, 0, 0, 0]).tolist(),
        }
        true_joint = {
            "b1": [0, 0, 1, 2, 2, 3, 4, 5, 6],
            "b2": [0, 1, 2, 3],
            "b3": [0, 1, 2, 3, 4],
        }
        true_df = pd.DataFrame({
            "item_id": ["b1", "b2", "b3"],
            "fates": [json.dumps(true_fates[k]) for k in ["b1", "b2", "b3"]],
            "joint": [json.dumps(true_joint[k]) for k in ["b1", "b2", "b3"]],
        })
        pred_df = true_df.copy()  # perfect prediction
        score = M.score_submission(meta, true_df, pred_df)
        self.assertAlmostEqual(score, 100.0, places=6)

    def test_reversed_all_wrong_carried(self):
        # a==n-a case (balanced), predicting exactly the complement side for carried
        counts = [2, 2, 0, 0, 0]
        n = 4
        true_fates = [0, 0, 1, 1]
        true_joint = [0, 1, 2, 3]
        pred_fates = [1, 1, 0, 0]  # fully reversed adopted/not
        res = M.score_board(n, counts, json.dumps(true_fates), json.dumps(true_joint),
                             json.dumps(pred_fates), json.dumps(true_joint))
        expected_c = M.carried_expected(2, 4)  # 0.5
        # earned = 0 (nothing matches)
        self.assertAlmostEqual(res["carried"], (0.0 - expected_c) / (1 - expected_c), places=9)
        self.assertLess(res["carried"], 0.0)


class TestMonteCarloChanceBaseline(unittest.TestCase):
    """A uniformly random permutation of the TRUE fate (resp. joint) vector,
    fed back in as the "prediction", must score ~0 in expectation for every
    term (that's the whole point of the chance correction)."""

    def test_carried_disposal_chance_baseline_near_zero(self):
        # Single random permutation draws are high-variance for small n, so
        # we average several independent permutation draws per board (still
        # an unbiased MC estimate of the same per-board expectation) to get
        # a tight enough aggregate estimate that a real formula bug (as
        # opposed to MC noise) would be clearly visible.
        rng = np.random.default_rng(777)
        carried_vals, disposal_vals = [], []
        for _ in range(500):
            n = rng.integers(4, 30)
            # random counts summing to n, at least 2 nonzero classes to avoid
            # the fully-degenerate "only one valid arrangement" boards which
            # trivially score exactly 1.0 (handled/tested separately in
            # TestMalformedAndExclusions) and would bias a "near zero" check.
            while True:
                cuts = sorted(rng.choice(range(1, n), size=4, replace=True))
                counts = [cuts[0], cuts[1] - cuts[0], cuts[2] - cuts[1], cuts[3] - cuts[2], n - cuts[3]]
                if sum(c > 0 for c in counts) >= 2:
                    break
            true_fates = make_true_fates(counts)
            a = counts[0]
            na = n - a
            cvs, dvs = [], []
            for _ in range(20):
                perm = rng.permutation(n)
                pred_fates = true_fates[perm]
                if a > 0:
                    cvs.append(M.carried_term(true_fates, pred_fates, a, n))
                if na > 0:
                    dvs.append(M.disposal_term(true_fates, pred_fates, counts, n))
            if cvs:
                carried_vals.append(np.mean(cvs))
            if dvs:
                disposal_vals.append(np.mean(dvs))
        self.assertAlmostEqual(np.mean(carried_vals), 0.0, delta=0.02)
        self.assertAlmostEqual(np.mean(disposal_vals), 0.0, delta=0.02)

    def test_joint_ari_chance_baseline_near_zero(self):
        rng = np.random.default_rng(999)
        vals = []
        for _ in range(300):
            n = rng.integers(4, 30)
            n_groups = rng.integers(2, n)  # ensure not all-singleton
            true_joint = rng.integers(0, n_groups, size=n)
            if len(np.unique(true_joint)) == n:
                continue  # skip accidental all-singleton draw
            perm = rng.permutation(n)
            pred_joint = true_joint[perm]
            score, defined = M.joint_term(true_joint, json.dumps(pred_joint.tolist()), n)
            if defined:
                vals.append(score)
        self.assertAlmostEqual(np.mean(vals), 0.0, delta=0.05)


class TestMalformedAndExclusions(unittest.TestCase):
    def test_malformed_fates_wrong_length(self):
        counts = [1, 1, 0, 0, 0]
        n = 2
        true_fates = [0, 1]
        true_joint = [0, 1]
        res = M.score_board(n, counts, json.dumps(true_fates), json.dumps(true_joint),
                             json.dumps([0]), json.dumps(true_joint))
        self.assertFalse(res["fates_valid"])
        self.assertEqual(res["carried"], -1.0)
        self.assertEqual(res["disposal"], -1.0)

    def test_malformed_fates_wrong_multiplicity(self):
        counts = [1, 1, 0, 0, 0]
        n = 2
        true_fates = [0, 1]
        true_joint = [0, 1]
        res = M.score_board(n, counts, json.dumps(true_fates), json.dumps(true_joint),
                             json.dumps([1, 1]), json.dumps(true_joint))  # uses code1 twice, code0 zero times
        self.assertFalse(res["fates_valid"])
        self.assertEqual(res["carried"], -1.0)
        self.assertEqual(res["disposal"], -1.0)

    def test_malformed_fates_unreadable(self):
        counts = [1, 1, 0, 0, 0]
        n = 2
        true_fates = [0, 1]
        true_joint = [0, 1]
        res = M.score_board(n, counts, json.dumps(true_fates), json.dumps(true_joint),
                             "not json", json.dumps(true_joint))
        self.assertFalse(res["fates_valid"])
        self.assertEqual(res["carried"], -1.0)
        self.assertEqual(res["disposal"], -1.0)

    def test_malformed_joint_scores_zero_not_minus_one(self):
        counts = [1, 1, 0, 1, 0]
        n = 3
        true_fates = [0, 1, 3]
        true_joint = [0, 1, 1]  # a real pair exists -> joint term is defined
        res = M.score_board(n, counts, json.dumps(true_fates), json.dumps(true_joint),
                             json.dumps(true_fates), "garbage")
        self.assertTrue(res["joint_defined"])
        self.assertEqual(res["joint"], 0.0)
        res2 = M.score_board(n, counts, json.dumps(true_fates), json.dumps(true_joint),
                              json.dumps(true_fates), json.dumps([0, 1]))  # wrong length
        self.assertTrue(res2["joint_defined"])
        self.assertEqual(res2["joint"], 0.0)

    def test_a_zero_excludes_carried(self):
        counts = [0, 2, 1, 0, 0]
        n = 3
        true_fates = [1, 1, 2]
        true_joint = [0, 1, 2]
        res = M.score_board(n, counts, json.dumps(true_fates), json.dumps(true_joint),
                             json.dumps(true_fates), json.dumps(true_joint))
        self.assertFalse(res["carried_defined"])
        self.assertIsNone(res["carried"])
        self.assertTrue(res["disposal_defined"])  # na=3>0, still scored

    def test_a_equals_n_excludes_disposal(self):
        counts = [3, 0, 0, 0, 0]
        n = 3
        true_fates = [0, 0, 0]
        true_joint = [0, 1, 2]
        res = M.score_board(n, counts, json.dumps(true_fates), json.dumps(true_joint),
                             json.dumps(true_fates), json.dumps(true_joint))
        self.assertFalse(res["disposal_defined"])
        self.assertIsNone(res["disposal"])
        self.assertTrue(res["carried_defined"])
        self.assertAlmostEqual(res["carried"], 1.0, places=9)  # degenerate-but-forced-perfect case

    def test_singleton_only_joint_excludes_joint_term(self):
        counts = [1, 1, 0, 0, 0]
        n = 2
        true_fates = [0, 1]
        true_joint = [0, 1]  # singleton-only (no pair)
        res = M.score_board(n, counts, json.dumps(true_fates), json.dumps(true_joint),
                             json.dumps(true_fates), json.dumps([5, 5]))  # pred says "grouped" -- doesn't matter
        self.assertFalse(res["joint_defined"])
        self.assertIsNone(res["joint"])
        # board score should be the mean of carried/disposal only... but here a==1==n so disposal
        # undefined too (na=1>0 actually: a=1,n=2,na=1 -> disposal defined). carried: a=1 -> defined.
        self.assertTrue(res["carried_defined"])
        self.assertTrue(res["disposal_defined"])


class TestOracleDecodeOnRealData(unittest.TestCase):
    """Sanity-check metric.py against real train.csv / train_targets.csv."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(os.path.join(PUBLIC_DIR, "train.csv")):
            cls.skip_all = True
            return
        cls.skip_all = False
        cls.train = pd.read_csv(os.path.join(PUBLIC_DIR, "train.csv"))
        cls.targets = pd.read_csv(os.path.join(PUBLIC_DIR, "train_targets.csv"))
        # use a reproducible sample for speed
        rng = np.random.default_rng(0)
        sample_ids = rng.choice(cls.train["item_id"].values, size=400, replace=False)
        cls.meta = cls.train[cls.train["item_id"].isin(sample_ids)][["item_id", "counts"]].reset_index(drop=True)
        cls.true_df = cls.targets[cls.targets["item_id"].isin(sample_ids)][["item_id", "fates", "joint"]].reset_index(drop=True)

    def setUp(self):
        if self.skip_all:
            self.skipTest("dataset/public not found locally")

    def test_ground_truth_scores_max(self):
        score = M.score_submission(self.meta, self.true_df, self.true_df)
        self.assertGreater(score, 99.999)  # should be (numerically) exactly 100

    def test_counts_ignoring_majority_baseline_is_heavily_penalized(self):
        # predict the globally most common single fate code for every item on
        # every board, ignoring the board's own counts. Most boards (all but
        # the small boards where this happens to equal the true single-code
        # count) end up with mismatched multiplicities -> carried=disposal=-1
        # -> the submission-level mean is driven negative -> final score is
        # floored at 0 (the documented floor), i.e. the worst possible score.
        all_fates = []
        for s in self.targets["fates"]:
            all_fates += json.loads(s)
        majority_code = int(pd.Series(all_fates).mode()[0])
        pred_rows = []
        for _, row in self.meta.iterrows():
            n = int(sum(json.loads(row["counts"])))
            pred_rows.append({"item_id": row["item_id"],
                               "fates": json.dumps([majority_code] * n),
                               "joint": json.dumps(list(range(n)))})
        pred_df = pd.DataFrame(pred_rows)
        score, details = M.score_submission(self.meta, self.true_df, pred_df, return_details=True)
        # the majority of boards (mostly n>1) have invalid fates; only tiny
        # n==1 (or boards whose single fate happens to equal majority_code)
        # boards slip through as "valid" by coincidence.
        frac_invalid = (~details["fates_valid"]).mean()
        self.assertGreater(frac_invalid, 0.5)
        self.assertAlmostEqual(score, 0.0, places=6)  # hits the documented floor-at-0

    def test_counts_respecting_random_shuffle_baseline_near_zero(self):
        # IMPORTANT finding (see reports/metric_spec.md): unlike test boards
        # (guaranteed >=2 distinct true fates per board), ~1/5 of TRAIN
        # boards have a==n_amendments (all adopted) and a further ~1/4 have
        # a==0 with every non-adopted item sharing one single true fate.
        # Both are the "expected==1" degenerate case in metric.py, where the
        # counts-validity constraint forces EVERY valid submission (random
        # or not) to score exactly 1.0 on that term -- a free, zero-variance
        # point unrelated to prediction quality. We therefore check the
        # "near zero" chance-baseline property on the subset of boards that
        # mirror the test guarantee (>=2 distinct true fates), and merely
        # *report* (not assert near-zero on) the unfiltered full-train
        # number to document the inflation this causes if CV is run on raw
        # train boards without this filter.
        rng = random.Random(42)
        pred_rows = []
        for _, row in self.true_df.iterrows():
            true_fates = json.loads(row["fates"])
            true_joint = json.loads(row["joint"])
            shuffled_fates = true_fates[:]
            rng.shuffle(shuffled_fates)
            shuffled_joint = true_joint[:]
            rng.shuffle(shuffled_joint)
            pred_rows.append({"item_id": row["item_id"],
                               "fates": json.dumps(shuffled_fates),
                               "joint": json.dumps(shuffled_joint)})
        pred_df = pd.DataFrame(pred_rows)
        score, details = M.score_submission(self.meta, self.true_df, pred_df, return_details=True)
        self.assertTrue((details["fates_valid"]).all())  # shuffle always respects counts

        unfiltered_mean = details["board_score"].mean()
        print(f"\n  [report] unfiltered (raw train) mean board_score under random-shuffle "
              f"baseline = {unfiltered_mean:.4f}  (inflated by degenerate boards, see comment)")

        n_distinct = self.true_df["fates"].apply(lambda s: len(set(json.loads(s))))
        test_like_mask = (n_distinct >= 2).values
        filtered_mean = details.loc[test_like_mask, "board_score"].mean()
        print(f"  [report] filtered (>=2 distinct true fates, test-like) mean board_score "
              f"under random-shuffle baseline = {filtered_mean:.4f}")
        self.assertLess(abs(filtered_mean), 0.08)


if __name__ == "__main__":
    unittest.main(verbosity=2)
