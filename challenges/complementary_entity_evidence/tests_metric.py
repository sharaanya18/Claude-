"""Unit tests for metric.py (stdlib unittest).  Run:  python3 tests_metric.py -v"""
import json
import random
import unittest
from itertools import combinations

import metric as M

try:  # the verbatim grader needs numpy/pandas; tests that use it are skipped if they are missing
    import numpy as np
    import pandas as pd
except Exception:  # pragma: no cover
    np = pd = None

# ---- verbatim copy of the description's grader (reference for differential tests) ----------------------------
if pd is not None:
    def _ref_answers(answers):
        if list(answers.columns) not in (["id", "prediction"], ["id", "prediction", "visibility"]):
            raise Exception("Answers must contain id,prediction and optional visibility.")
        if answers.empty or answers.isna().any().any() or answers.id.astype(str).duplicated().any():
            raise Exception("Answers must be non-empty with unique non-missing IDs.")
        result = {}
        for row in answers.itertuples(index=False):
            item = json.loads(row.prediction)
            if set(item) != {"query_id", "roles", "slate_size", "oracle_rank"} or not isinstance(item["query_id"], str):
                raise Exception("Invalid answer payload.")
            roles = item["roles"]
            if not isinstance(roles, list) or any(not isinstance(r, str) or not r for r in roles) or len(roles) != len(set(roles)):
                raise Exception("Invalid answer role set.")
            result[str(row.id)] = item
        return result

    def _ref_numeric(value):
        if isinstance(value, str) and value.lstrip().startswith("{"):
            value = json.loads(value)["oracle_rank"]
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise Exception("Predictions must be finite real ranking scores.") from exc
        if not np.isfinite(value):
            raise Exception("Predictions must be finite real ranking scores.")
        return value

    def ref_grade(submission, answers):
        if list(submission.columns) != ["id", "prediction"]:
            raise Exception("Submission columns must be exactly id,prediction.")
        if submission.empty or submission.isna().any().any():
            raise Exception("Submission must be non-empty without missing values.")
        ids = submission.id.astype(str)
        if ids.duplicated().any():
            raise Exception("Duplicate submission IDs.")
        gold = _ref_answers(answers)
        if len(submission) != len(gold) or set(ids) != set(gold):
            raise Exception("Submission IDs must match scored answer IDs exactly.")
        predictions = {str(r.id): _ref_numeric(r.prediction) for r in submission.itertuples(index=False)}
        queries = {}
        for rid, item in gold.items():
            queries.setdefault(item["query_id"], []).append(rid)
        scores = []
        for members in queries.values():
            expected = {gold[rid]["slate_size"] for rid in members}
            if len(expected) != 1 or any(type(n) is not int or n < 2 or n > 8 for n in expected) or len(members) != next(iter(expected)):
                raise Exception("A scoring partition must retain complete candidate slates.")
            role_sets = {rid: set(gold[rid]["roles"]) for rid in members}
            optimum = max(len(role_sets[a] | role_sets[b]) for a, b in combinations(members, 2))
            if optimum == 0:
                raise Exception("Scored slate has no new supported roles.")
            selected = sorted(members, key=lambda rid: (-predictions[rid], rid))[:2]
            scores.append(len(role_sets[selected[0]] | role_sets[selected[1]]) / optimum)
        score = float(np.mean(scores))
        if not 0 <= score <= 1:
            raise Exception("Score outside [0,1].")
        return score


def gold_entry(query_id, roles, slate_size, oracle_rank=0.0):
    return {"query_id": query_id, "roles": list(roles), "slate_size": slate_size, "oracle_rank": oracle_rank}


def frames(gold, preds):
    sub = pd.DataFrame({"id": list(preds), "prediction": [preds[k] for k in preds]})
    ans = pd.DataFrame({"id": list(gold), "prediction": [json.dumps(gold[k]) for k in gold]})
    return sub, ans


# One hand-built slate used by several tests.  Role universe {a,b,c,d}.
#   A={a,b}  B={a,b}  C={c}  D={a}  E={}   (ids sort as ev_s_A < ... < ev_s_E)
SLATE = {
    "ev_s_A": ["a", "b"],
    "ev_s_B": ["a", "b"],
    "ev_s_C": ["c"],
    "ev_s_D": ["a"],
    "ev_s_E": [],
}
SLATE_GOLD = {k: gold_entry("q1", v, 5) for k, v in SLATE.items()}
# pair unions: AB=2 AC=3 AD=2 AE=2 BC=3 BD=2 BE=2 CD=2 CE=1 DE=1 -> optimum 3 (pairs AC, BC)


class TestSlateScore(unittest.TestCase):
    def test_perfect_pair_is_one(self):
        preds = {"ev_s_A": 9, "ev_s_C": 8, "ev_s_B": 1, "ev_s_D": 0, "ev_s_E": -1}
        self.assertEqual(M.marginal_role_coverage(preds, SLATE_GOLD), 1.0)

    def test_equally_optimal_alternative_pair_is_one(self):
        preds = {"ev_s_B": 9, "ev_s_C": 8, "ev_s_A": 1, "ev_s_D": 0, "ev_s_E": -1}
        self.assertEqual(M.marginal_role_coverage(preds, SLATE_GOLD), 1.0)

    def test_worst_pair(self):
        # worst pair is (C,E) or (D,E): union 1 / 3
        preds = {"ev_s_C": 9, "ev_s_E": 8, "ev_s_A": 1, "ev_s_B": 0, "ev_s_D": -1}
        self.assertAlmostEqual(M.marginal_role_coverage(preds, SLATE_GOLD), 1 / 3)

    def test_redundancy_case(self):
        # Pointwise-best two candidates A,B are redundant: union 2 of optimum 3.
        preds = {"ev_s_A": 9, "ev_s_B": 8, "ev_s_C": 5, "ev_s_D": 1, "ev_s_E": 0}
        self.assertAlmostEqual(M.marginal_role_coverage(preds, SLATE_GOLD), 2 / 3)

    def test_ties_broken_by_ascending_id(self):
        # All scores equal -> pair (ev_s_A, ev_s_B) -> union 2 -> 2/3
        preds = {k: 0.5 for k in SLATE}
        self.assertAlmostEqual(M.marginal_role_coverage(preds, SLATE_GOLD), 2 / 3)
        # Three-way tie for the second slot: B, C, D tie at 0.5 behind A; ascending id picks B -> A+B = 2/3
        preds = {"ev_s_A": 1.0, "ev_s_B": 0.5, "ev_s_C": 0.5, "ev_s_D": 0.5, "ev_s_E": 0.0}
        self.assertAlmostEqual(M.marginal_role_coverage(preds, SLATE_GOLD), 2 / 3)
        # Make C the unique 2nd -> 1.0
        preds["ev_s_C"] = 0.6
        self.assertEqual(M.marginal_role_coverage(preds, SLATE_GOLD), 1.0)
        # tie between C and D at top: ids C<D -> (C,D) union {c,a} = 2 -> 2/3
        preds = {"ev_s_C": 1.0, "ev_s_D": 1.0, "ev_s_A": 0.0, "ev_s_B": 0.0, "ev_s_E": 0.0}
        self.assertAlmostEqual(M.marginal_role_coverage(preds, SLATE_GOLD), 2 / 3)

    def test_slate_score_pair_and_dict_forms(self):
        sets = {k: set(v) for k, v in SLATE.items()}
        self.assertEqual(M.slate_score(["ev_s_A", "ev_s_C"], sets), 1.0)
        self.assertAlmostEqual(M.slate_score(("ev_s_D", "ev_s_E"), sets), 1 / 3)
        self.assertAlmostEqual(M.slate_score({"ev_s_A": 2, "ev_s_B": 1, "ev_s_C": 0, "ev_s_D": 0, "ev_s_E": 0}, sets), 2 / 3)
        with self.assertRaises(Exception):
            M.slate_score(["ev_s_A", "ev_s_A"], sets)
        with self.assertRaises(Exception):
            M.slate_score(["ev_s_A", "nope"], sets)

    def test_hand_computed_mid_case_two_slates(self):
        gold = {}
        gold.update({k: gold_entry("q1", v, 5) for k, v in SLATE.items()})
        # slate q2 (size 3): X={r1,r2,r3}, Y={r3,r4}, Z={r1}; pairs XY=4 XZ=3 YZ=3 -> optimum 4 (XY)
        q2 = {"ev_t_X": ["r1", "r2", "r3"], "ev_t_Y": ["r3", "r4"], "ev_t_Z": ["r1"]}
        gold.update({k: gold_entry("q2", v, 3) for k, v in q2.items()})
        preds = {"ev_s_A": 9, "ev_s_B": 8, "ev_s_C": 5, "ev_s_D": 1, "ev_s_E": 0,   # 2/3
                 "ev_t_X": 5, "ev_t_Z": 4, "ev_t_Y": 1}                              # XZ = 3/4
        self.assertAlmostEqual(M.marginal_role_coverage(preds, gold), (2 / 3 + 3 / 4) / 2)
        # per-slate list is returned in order of first appearance of each query
        score, per = M.marginal_role_coverage(preds, gold, return_per_slate=True)
        self.assertAlmostEqual(per[0], 2 / 3)
        self.assertAlmostEqual(per[1], 3 / 4)

    def test_constant_scores_baseline_is_id_order(self):
        # documented baseline: constant prediction == pick the two smallest ids
        preds = {k: 0.0 for k in SLATE}
        self.assertAlmostEqual(M.marginal_role_coverage(preds, SLATE_GOLD), 2 / 3)


class TestInvalid(unittest.TestCase):
    def test_zero_optimum_raises(self):
        sets = {"x": set(), "y": set(), "z": set()}
        with self.assertRaises(Exception) as cm:
            M.slate_score(["x", "y"], sets)
        self.assertIn("no new supported roles", str(cm.exception))
        gold = {k: gold_entry("q", [], 3) for k in sets}
        with self.assertRaises(Exception):
            M.marginal_role_coverage({k: 0.0 for k in sets}, gold)

    def test_partial_slate_rejected(self):
        gold = {k: gold_entry("q1", v, 6) for k, v in SLATE.items()}  # claims 6 but has 5
        with self.assertRaises(Exception) as cm:
            M.marginal_role_coverage({k: 0.0 for k in SLATE}, gold)
        self.assertIn("complete candidate slates", str(cm.exception))

    def test_slate_size_range_and_type(self):
        for bad in (1, 9, 5.0, True):
            gold = {k: gold_entry("q1", v, bad) for k, v in SLATE.items()}
            with self.assertRaises(Exception):
                M.marginal_role_coverage({k: 0.0 for k in SLATE}, gold)
        # size 9 even if the slate really has 9 members
        ids9 = [f"ev_z_{i}" for i in range(9)]
        gold9 = {k: gold_entry("q9", ["r%d" % i], 9) for i, k in enumerate(ids9)}
        with self.assertRaises(Exception):
            M.marginal_role_coverage({k: 0.0 for k in ids9}, gold9)

    def test_inconsistent_slate_size_inside_slate(self):
        gold = {k: gold_entry("q1", v, 5) for k, v in SLATE.items()}
        gold["ev_s_A"]["slate_size"] = 4
        with self.assertRaises(Exception):
            M.marginal_role_coverage({k: 0.0 for k in SLATE}, gold)

    def test_id_mismatch_and_nonfinite(self):
        preds = {k: 0.0 for k in SLATE}
        missing = dict(preds); missing.pop("ev_s_E")
        with self.assertRaises(Exception):
            M.marginal_role_coverage(missing, SLATE_GOLD)
        extra = dict(preds); extra["ev_s_F"] = 0.0
        with self.assertRaises(Exception):
            M.marginal_role_coverage(extra, SLATE_GOLD)
        for bad in (float("nan"), float("inf"), "abc", None):
            p = dict(preds); p["ev_s_A"] = bad
            with self.assertRaises(Exception):
                M.marginal_role_coverage(p, SLATE_GOLD)
        with self.assertRaises(Exception):
            M.marginal_role_coverage({}, {})

    def test_duplicate_ids_in_rows(self):
        rows = [("a", 1.0), ("a", 2.0)]
        ans = [("a", json.dumps(gold_entry("q", ["r"], 2)))]
        with self.assertRaises(Exception):
            M.grade_rows(rows, ans)

    def test_oracle_payload_prediction_is_decoded(self):
        # grade() accepts the answer JSON as a prediction and uses its oracle_rank
        self.assertEqual(M._numeric(json.dumps(gold_entry("q", ["r"], 2, oracle_rank=3.5))), 3.5)


@unittest.skipIf(pd is None, "numpy/pandas not available for the reference grader")
class TestAgainstReferenceGrader(unittest.TestCase):
    def test_differential_fuzz(self):
        rng = random.Random(7)
        for trial in range(60):
            gold, preds = {}, {}
            for s in range(rng.randint(1, 6)):
                probs, g, ids = M.make_toy_slate(rng, slate_key=f"t{trial}_{s}")
                for cid in ids:
                    gold[cid] = gold_entry(f"q{trial}_{s}", sorted(g[cid]), len(ids))
                    preds[cid] = rng.choice([0.0, 1.0, 2.0, rng.random()])  # many exact ties
            sub, ans = frames(gold, preds)
            self.assertAlmostEqual(M.marginal_role_coverage(preds, gold), ref_grade(sub, ans), places=12)
            rows = [(k, preds[k]) for k in preds]
            arows = [(k, json.dumps(gold[k])) for k in gold]
            self.assertAlmostEqual(M.grade_rows(rows, arows), ref_grade(sub, ans), places=12)

    def test_hand_slates_match_reference(self):
        preds = {"ev_s_A": 9, "ev_s_B": 8, "ev_s_C": 5, "ev_s_D": 1, "ev_s_E": 0}
        sub, ans = frames(SLATE_GOLD, preds)
        self.assertAlmostEqual(ref_grade(sub, ans), 2 / 3)
        self.assertAlmostEqual(M.marginal_role_coverage(preds, SLATE_GOLD), ref_grade(sub, ans))


class TestDecode(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(123)

    def test_oracle_decode_is_perfect(self):
        """Gold role sets as 0/1 probabilities -> both decodes score exactly 1.0 on every toy slate."""
        for method in ("mc_ratio", "expected_union"):
            tot = []
            for i in range(400):
                probs, gold, ids = M.make_toy_slate(self.rng, slate_key=f"o{i}")
                oracle = {c: {r: 1.0 for r in gold[c]} for c in ids}
                scores = M.decode_slate(oracle, method=method)
                tot.append(M.slate_score(scores, gold))
            self.assertEqual(min(tot), 1.0, method)
            self.assertEqual(sum(tot) / len(tot), 1.0)

    def test_oracle_through_global_score_path(self):
        gold_all, probs_all, slate_of = {}, {}, {}
        for i in range(50):
            probs, gold, ids = M.make_toy_slate(self.rng, slate_key=f"g{i}")
            for c in ids:
                gold_all[c] = gold_entry(f"q{i}", sorted(gold[c]), len(ids))
                probs_all[c] = {r: 1.0 for r in gold[c]}
                slate_of[c] = f"q{i}"
        scores = M.decode_all(probs_all, slate_of)
        self.assertEqual(M.marginal_role_coverage(scores, gold_all), 1.0)

    def test_known_roles_are_masked(self):
        probs = {"ev_s_A": {"a": 0.9, "b": 0.1}, "ev_s_B": {"a": 0.8, "c": 0.5}, "ev_s_C": {"b": 0.4}}
        masked = M.mask_known_roles(probs, ["a"])
        self.assertEqual(masked["ev_s_A"]["a"], 0.0)
        self.assertEqual(masked["ev_s_B"]["a"], 0.0)
        self.assertEqual(masked["ev_s_B"]["c"], 0.5)
        self.assertEqual(probs["ev_s_A"]["a"], 0.9)  # input untouched

    def test_closed_form_hand_value(self):
        probs = {"x": {"r": 0.5, "s": 0.2}, "y": {"r": 0.5}, "z": {"s": 1.0}}
        table = M.expected_union_table(probs)
        self.assertAlmostEqual(table[("x", "y")], (1 - 0.25) + 0.2)
        self.assertAlmostEqual(table[("x", "z")], 0.5 + 1.0)
        self.assertAlmostEqual(table[("y", "z")], 0.5 + 1.0)
        pair, _ = M.choose_pair_expected_union(probs)
        self.assertEqual(pair, ("x", "z"))  # (x,z) and (y,z) tie at 1.5 -> ascending id

    def test_mc_is_deterministic_and_order_invariant(self):
        probs, _, ids = M.make_toy_slate(self.rng)
        s1 = M.decode_slate(probs)
        s2 = M.decode_slate({c: probs[c] for c in reversed(ids)})
        self.assertEqual(s1, s2)
        self.assertEqual(M.decode_slate(probs, seed=20240601), s1)

    def test_scores_are_distinct_finite_and_pair_on_top(self):
        probs, _, ids = M.make_toy_slate(self.rng)
        scores = M.decode_slate(probs)
        self.assertEqual(len(set(scores.values())), len(ids))
        pair, _ = M.choose_pair_mc_ratio(probs)
        self.assertEqual(set(M.select_top2(scores)), set(pair))

    def test_mc_converges_to_closed_form_union_scale(self):
        # With a single candidate pair and no competition the MC ratio is >= 0 and <= 1; pair values lie in [0,1].
        probs, _, _ = M.make_toy_slate(self.rng)
        table, p_pos = M.pair_tables_mc(probs, n_draws=500)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in table.values()))
        self.assertTrue(0.0 < p_pos <= 1.0)

    def test_all_zero_probabilities_fall_back_gracefully(self):
        probs = {"a": {"r": 0.0}, "b": {}, "c": {"r": 0.0}}
        scores = M.decode_slate(probs)
        self.assertEqual(set(scores), {"a", "b", "c"})
        self.assertEqual(M.select_top2(scores), ["a", "b"])  # ascending-id fallback

    def test_bad_probabilities_rejected(self):
        for bad in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                M.decode_slate({"a": {"r": bad}, "b": {"r": 0.5}})
        with self.assertRaises(ValueError):
            M.decode_slate({"a": {"r": 0.5}})

    def test_mc_beats_or_matches_closed_form_in_expectation_on_calibrated_toys(self):
        """On calibrated toy slates the MC decode maximises the expected ratio, so its mean gold score should not
        be materially below the closed form (loose bound: statistical noise only)."""
        rng = random.Random(99)
        mc, cf = [], []
        for i in range(300):
            probs, gold, ids = M.make_toy_slate(rng, slate_key=f"c{i}")
            mc.append(M.slate_score(M.decode_slate(probs, method="mc_ratio", n_draws=800), gold))
            cf.append(M.slate_score(M.decode_slate(probs, method="expected_union"), gold))
        self.assertGreater(sum(mc) / len(mc), sum(cf) / len(cf) - 0.03)


if __name__ == "__main__":
    unittest.main(verbosity=2)
