"""Tests for formulation.py and metric.py using hand-made toy problems (fixtures only, never training data).

Toy problem (all hand-solved):
  products A, B; profit 40 / 30; material use 2 / 3 with cap 100; labour use 4 / 1 with cap 120.
  max 40A + 30B  s.t. 2A + 3B <= 100, 4A + B <= 120  -> A = 26, B = 16, optimum 1520.
References x0..x5 hold 40, 30, 2, 3, 100, 4 ... (distinct values so same-value sharing does not interfere).

Run: python3 tests_metric.py   (or python3 -m unittest tests_metric -v)
"""
import os
import random
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import formulation as F  # noqa: E402
import metric as M  # noqa: E402

NUMS = {"x0": 40.0, "x1": 30.0, "x2": 2.0, "x3": 3.0, "x4": 100.0, "x5": 4.0, "x6": 1.0 + 0.5, "x7": 120.0}
# x6 (=1.5) is unused by the reference except in some tests; labour use of B is the typed constant 1.
REF = "max x0*A + x1*B; x2*A + x3*B <= x4; x5*A + B <= x7"
OPT = 1520.0


def typed_in(text, numbers):
    """Replace every reference x<k> by its literal value (the 'typed-in numbers' baseline)."""
    return re.sub(r"\bx(\d+)\b", lambda m: "(%r)" % numbers["x" + m.group(1)], text)


class ParserTests(unittest.TestCase):
    def test_toy_optimum(self):
        self.assertAlmostEqual(F.optimum(REF, NUMS), OPT, places=6)

    def test_objective_keywords_and_colon(self):
        for kw in ("max", "maximize", "max:", "maximize:", "MAX"):
            self.assertAlmostEqual(F.optimum(kw + " 3*a; a <= 4", {}), 12.0, places=7)
        for kw in ("min", "minimize", "min:", "minimize:"):
            self.assertAlmostEqual(F.optimum(kw + " 3*a; a >= 4", {}), 12.0, places=7)

    def test_numbers_commas_percent(self):
        self.assertAlmostEqual(F.optimum("max a; a <= 1,250", {}), 1250.0)
        self.assertAlmostEqual(F.optimum("max a; a <= 1,250,000.5", {}), 1250000.5)
        self.assertAlmostEqual(F.optimum("max a; a <= 25%", {}), 0.25)
        self.assertAlmostEqual(F.optimum("max a; a <= 50% * 8", {}), 4.0)
        self.assertAlmostEqual(F.optimum("max a; a <= 200 * 2.5%", {}), 5.0)

    def test_linear_arithmetic(self):
        # products / quotients with one variable side, constants folded, unary minus, parentheses
        self.assertAlmostEqual(F.optimum("max (x0 + (-x2))*a + b/2; a <= 3; b <= 4*(2+1)", NUMS), 38 * 3 + 6.0)
        self.assertAlmostEqual(F.optimum("max a; 2*a + 3 <= x4 - 1", NUMS), 48.0)  # constants move to RHS
        self.assertAlmostEqual(F.optimum("max a; a/4 <= 5", NUMS), 20.0)

    def test_nonlinear_rejected(self):
        for bad in ("max a*b; a <= 1; b <= 1", "max a; a/b <= 1", "max a; a*a <= 4", "max a; 3/a <= 1",
                    "max a; a <= 1/0"):
            with self.assertRaises(F.FormulationError, msg=bad):
                F.parse(bad, NUMS)
            self.assertIsNone(F.optimum(bad, NUMS))

    def test_variable_name_rules(self):
        for bad in ("max a; int x5", "max q3; q3 <= 1", "max a; int q12"):
            self.assertIsNone(F.optimum(bad, {"x5": 1.0}), bad)
        for good in ("max x5a; x5a <= 2", "max q; q <= 2", "max Q3; Q3 <= 2", "max _v; _v <= 2", "max xx1; xx1 <= 2"):
            self.assertAlmostEqual(F.optimum(good, {}), 2.0, msg=good)
        # example in the task text: 'x4cake_a' is a legal VARIABLE name, not ref x4 times cake_a
        m = F.parse("max x4cake_a; x4cake_a <= 3", {"x4": 7.0})
        self.assertEqual(m.variables, ["x4cake_a"])

    def test_unknown_reference_and_syntax(self):
        for bad in ("max x99*a; a <= 1", "a <= 1", "max ; a<=1", "max a; a < 1", "max a; a == 1", "max a; a <= 1 <= 2",
                    "max 2a; a <= 1", "max a; a <= ", "max a +; a <= 1", "max a; int", "", "max a; (a <= 1"):
            self.assertIsNone(F.optimum(bad, NUMS), bad)

    def test_limits(self):
        many = "max " + " + ".join("v%d" % i for i in range(61)) + "; " + "v0 <= 1"
        self.assertIsNone(F.optimum(many, {}))
        ok = "max " + " + ".join("v%d" % i for i in range(60)) + "; " + "; ".join("v%d <= 1" % i for i in range(60))
        self.assertAlmostEqual(F.optimum(ok, {}), 60.0)
        cons = "max a; " + "; ".join("a <= %d" % (i + 1) for i in range(151))
        self.assertIsNone(F.optimum(cons, {}))
        cons150 = "max a; " + "; ".join("a <= %d" % (i + 1) for i in range(150))
        self.assertAlmostEqual(F.optimum(cons150, {}), 1.0)

    def test_milp_and_declarations(self):
        lp = "max 5*a + 4*b; 6*a + 4*b <= 24; a + 2*b <= 6"
        self.assertAlmostEqual(F.optimum(lp, {}), 21.0, places=7)           # a=3, b=1.5
        self.assertAlmostEqual(F.optimum(lp + "; int a, b", {}), 20.0, places=7)  # a=4, b=0
        self.assertAlmostEqual(F.optimum(lp + "; int a; int b", {}), 20.0, places=7)
        knap = "max 6*a + 5*b + 4*c; 3*a + 2*b + 2*c <= 4; bin a, b, c"       # best: b + c = 9
        self.assertAlmostEqual(F.optimum(knap, {}), 9.0, places=7)
        self.assertAlmostEqual(F.optimum("max a; a <= 5.5; bin a", {}), 1.0)  # bin caps at 1

    def test_infeasible_unbounded(self):
        self.assertIsNone(F.optimum("max a; a <= 1; a >= 2", {}))
        self.assertIsNone(F.optimum("max a; a >= 1", {}))
        self.assertIsNone(F.optimum("min a - b; a >= 1; b >= 1", {}))      # unbounded below
        self.assertAlmostEqual(F.optimum("min a; a >= -5", {}), 0.0)         # nonnegativity is implicit
        self.assertAlmostEqual(F.optimum("max a + 5; a <= 1", {}), 6.0)      # objective constant

    def test_equality_constraint(self):
        self.assertAlmostEqual(F.optimum("max a; a + b = 10; b >= 4", {}), 6.0)

    def test_numbers_json_string_with_string_values(self):
        s = '[["x0", "40"], ["x1", "30"], ["x2", "2"], ["x3", "3"], ["x4", "100"], ["x5", "4"], ["x7", "120"]]'
        self.assertAlmostEqual(F.optimum(REF, s), OPT, places=6)

    def test_matches_optimum(self):
        self.assertTrue(F.matches_optimum(REF, NUMS, OPT))
        self.assertTrue(F.matches_optimum(REF, NUMS, OPT * (1 + 5e-5)))
        self.assertFalse(F.matches_optimum(REF, NUMS, OPT * (1 + 2e-4)))
        self.assertFalse(F.matches_optimum("garbage", NUMS, OPT))


class MetricTests(unittest.TestCase):
    def test_perfect_model_scores_one(self):
        s = M.case_score(REF, REF, NUMS, seed=1)
        self.assertEqual((s["value"], s["cf"], s["structure"]), (1.0, 1.0, 1.0))
        self.assertAlmostEqual(s["score"], 1.0)

    def test_typed_in_numbers_two_thirds(self):
        pred = typed_in(REF, NUMS)
        self.assertAlmostEqual(F.optimum(pred, {}), OPT, places=6)  # same optimum at the real numbers
        s = M.case_score(REF, pred, NUMS, seed=1)
        self.assertEqual(s["value"], 1.0)
        self.assertEqual(s["structure"], 1.0)   # rows are computed with the real numbers
        self.assertEqual(s["cf"], 0.0)          # numbers moved, literals did not
        self.assertAlmostEqual(s["score"], 2.0 / 3.0)

    def test_one_line_optimum_value_only(self):
        s = M.case_score(REF, "max v; v <= 1520", NUMS, seed=1)
        self.assertEqual((s["value"], s["cf"], s["structure"]), (1.0, 0.0, 0.0))
        self.assertAlmostEqual(s["score"], 1.0 / 3.0)
        # minimisation flavour: reference "min 2a + 3b; a + b >= 100" has optimum 200, one-liner "min v; v >= 200"
        mref, mnums = "min x2*a + x3*b; a + b >= x4", NUMS
        ms = M.case_score(mref, "min v; v >= 200", mnums, seed=1)
        self.assertEqual((ms["value"], ms["cf"], ms["structure"]), (1.0, 0.0, 0.0))
        # sample_submission style ("max v; v <= x0") -> value only if optimum coincides, here 40 != 1520
        s = M.case_score(REF, "max v; v <= x0", NUMS, seed=1)
        self.assertEqual(s["score"], 0.0)

    def test_renamed_variables_score_one(self):
        pred = "max x0*zebra + x1*apple; x2*zebra + x3*apple <= x4; x5*zebra + apple <= x7"
        s = M.case_score(REF, pred, NUMS, seed=3)
        self.assertEqual((s["value"], s["cf"], s["structure"]), (1.0, 1.0, 1.0))

    def test_row_order_and_sides_invariant(self):
        pred = "max x1*B + x0*A; x7 >= x5*A + B; x4 >= x3*B + x2*A"   # reordered rows, flipped senses, reordered terms
        s = M.case_score(REF, pred, NUMS, seed=3)
        self.assertAlmostEqual(s["score"], 1.0)

    def test_swapped_coefficients_inside_row_lose_structure(self):
        pred = "max x0*A + x1*B; x3*A + x2*B <= x4; x5*A + B <= x7"   # 3,2 swapped between A and B in row 1
        f1, m, n_p, n_r = M.structure_score(F.parse(REF, NUMS), F.parse(pred, NUMS), return_details=True)
        # best renaming keeps A->A, B->B: objective and row 2 match, row 1 does not: m=2 of 3 items each
        self.assertEqual((m, n_p, n_r), (2, 3, 3))
        self.assertAlmostEqual(f1, 2.0 / 3.0)
        s = M.case_score(REF, pred, NUMS, seed=3)
        self.assertLess(s["structure"], 1.0)
        # the optimum changes too, so value and counterfactual are not "free" either
        self.assertEqual(s["value"], 0.0)

    def test_unparsable_nonlinear_unknown_ref_score_zero(self):
        for bad in ("this is not a formulation", "max A*B; A <= 1", "max x99*A; A <= 1", "", "max A; A <= 1 <= 2"):
            s = M.case_score(REF, bad, NUMS, seed=1)
            self.assertEqual(s["score"], 0.0, bad)
            self.assertFalse(s["valid"])

    def test_parsable_but_unbounded_keeps_structure_only(self):
        pred = "max x0*A + x1*B; x2*A + x3*B >= x4"   # unbounded; objective item matches, row does not
        s = M.case_score(REF, pred, NUMS, seed=1)
        self.assertEqual((s["value"], s["cf"]), (0.0, 0.0))
        f1, m, n_p, n_r = M.structure_score(F.parse(REF, NUMS), F.parse(pred, NUMS), return_details=True)
        self.assertEqual((m, n_p, n_r), (1, 2, 3))
        self.assertAlmostEqual(s["structure"], 2.0 / 5.0)

    def test_extra_row_precision_loss(self):
        pred = REF + "; A <= 1000"
        f1, m, n_p, n_r = M.structure_score(F.parse(REF, NUMS), F.parse(pred, NUMS), return_details=True)
        self.assertEqual((m, n_p, n_r), (3, 4, 3))
        self.assertAlmostEqual(f1, 6.0 / 7.0)

    def test_slack_extra_constraint_is_caught_by_counterfactual(self):
        pred = REF + "; A <= 27"          # slack at A=26 -> same real optimum, but binds when numbers move
        s0 = M.case_score(REF, pred, NUMS, variants=[], seed=0)
        self.assertEqual(s0["value"], 1.0)
        # hand-made variant: material cap x4 doubled (100 -> 200): A=30 is then optimal region; true optimum:
        # max 40A+30B s.t. 2A+3B<=200, 4A+B<=120 -> A=(360-200)/10=16? solve: B=120-4A -> 2A+360-12A=200 -> A=16, B=56
        vnum = dict(NUMS, x4=200.0)
        true = F.optimum(REF, vnum)
        self.assertAlmostEqual(true, 40 * 16 + 30 * 56, places=6)         # 2320
        self.assertAlmostEqual(F.optimum(pred, vnum), 2320.0, places=6)    # A=16 <= 27: still slack
        # variant with labour cap x7 doubled (120 -> 240): REF -> A=(720-100)/10=62 ... B<0 so A=50,B=0 -> 2000
        vnum2 = dict(NUMS, x7=240.0)
        self.assertAlmostEqual(F.optimum(REF, vnum2), 40 * 50, places=6)
        self.assertAlmostEqual(F.optimum(pred, vnum2), 40 * 27 + 30 * ((100 - 54) / 3.0), places=6)  # capped A
        s = M.case_score(REF, pred, NUMS, variants=[(vnum, 2320.0), (vnum2, 2000.0), (dict(NUMS, x4=150.0), None)][:2])
        self.assertAlmostEqual(s["cf"], 0.5)

    def test_value_tolerance(self):
        near = "max v; v <= %r" % (OPT * (1 + 5e-5))
        far = "max v; v <= %r" % (OPT * (1 + 2e-4))
        self.assertEqual(M.case_score(REF, near, NUMS, seed=1)["value"], 1.0)
        self.assertEqual(M.case_score(REF, far, NUMS, seed=1)["value"], 0.0)

    def test_integer_item(self):
        ref = "max x0*A + x1*B; x2*A + x3*B <= x4; x5*A + B <= x7; int A, B"
        s = M.case_score(ref, ref, NUMS, seed=2)
        self.assertAlmostEqual(s["score"], 1.0)
        no_int = ref.replace("; int A, B", "")
        f1, m, n_p, n_r = M.structure_score(F.parse(ref, NUMS), F.parse(no_int, NUMS), return_details=True)
        self.assertEqual((m, n_p, n_r), (3, 3, 4))
        self.assertAlmostEqual(f1, 6.0 / 7.0)
        as_bin = ref.replace("int A, B", "int A; bin B")
        f1b = M.structure_score(F.parse(ref, NUMS), F.parse(as_bin, NUMS))
        self.assertAlmostEqual(f1b, 3.0 / 4.0)   # 3 of 4 items; the int/bin item differs

    def test_equality_row_sign_and_ge_normalisation(self):
        ref = F.parse("max a + b; a - b = 0; a + b >= 2; a <= 9", {})
        pred = F.parse("max b + a; b - a = 0; -a - b <= -2; a <= 9", {})
        self.assertAlmostEqual(M.structure_score(ref, pred), 1.0)

    def test_variant_generator(self):
        nums = {"x0": 10.0, "x1": 10.0, "x2": 2020.0, "x3": 0.25, "x4": 7.0, "x5": 1990.5}
        rng = random.Random(0)
        for _ in range(300):
            v = M.make_variant(nums, rng)
            self.assertEqual(v["x0"], v["x0"])
            self.assertAlmostEqual(v["x0"] / 10.0, v["x1"] / 10.0)          # same value -> same factor
            self.assertEqual(v["x2"], 2020.0)                                # year unchanged
            for k in ("x0", "x3", "x4", "x5"):
                f = v[k] / nums[k]
                self.assertTrue(0.55 - 1e-12 <= f <= 1.45 + 1e-12)
                self.assertFalse(0.95 < f < 1.05)
        # 2021 is outside 1990..2030? no: 2030 is the last year; 2031 and 1989 are scaled
        v = M.make_variant({"a": 2031.0, "b": 1989.0, "c": 1990.0, "d": 2030.0}, random.Random(1))
        self.assertNotEqual(v["a"], 2031.0)
        self.assertNotEqual(v["b"], 1989.0)
        self.assertEqual((v["c"], v["d"]), (1990.0, 2030.0))

    def test_variants_skip_unbounded_reference(self):
        # reference bounded only through a number: always finite, so check skipping by a reference that is
        # infeasible when the factor on x0 exceeds that of x1 (x0 <= a <= x1 requires x0 <= x1).
        ref = "max a; a >= x0; a <= x1"
        nums = {"x0": 10.0, "x1": 12.0}
        vs = M.make_variants(ref, nums, seed=5, n=3)
        self.assertEqual(len(vs), 3)
        for vnum, ro in vs:
            self.assertLessEqual(vnum["x0"], vnum["x1"] + 1e-12)   # only feasible variants are kept
            self.assertAlmostEqual(ro, vnum["x1"], places=7)

    def test_renaming_random_model(self):
        """Random 14-variable model vs a shuffled, renamed copy of itself: structure must be exactly 1."""
        rng = random.Random(7)
        nv, nr = 14, 12
        names = ["v%d" % i for i in range(nv)]
        coefs = {}
        parts = []
        nums = {}
        ctr = [0]

        def ref(value):
            k = "x%d" % ctr[0]
            ctr[0] += 1
            nums[k] = float(value)
            return k

        obj = " + ".join("%s*%s" % (ref(rng.randint(1, 9) + i / 100.0), v) for i, v in enumerate(names))
        rows = []
        for r in range(nr):
            vs = rng.sample(names, rng.randint(2, 6))
            rows.append(" + ".join("%s*%s" % (ref(rng.randint(1, 9) + rng.random()), v) for v in vs)
                        + " <= " + ref(rng.randint(50, 99) + rng.random()))
        text = "max " + obj + "; " + "; ".join(rows) + "; int " + ", ".join(names[:5])
        perm = names[:]
        rng.shuffle(perm)
        mapping = dict(zip(names, ["w_%s" % p for p in perm]))
        renamed = re.sub(r"\bv\d+\b", lambda m: mapping[m.group(0)], text)
        stmts = renamed.split("; ")
        head, body, tail = stmts[0], stmts[1:-1], stmts[-1]
        rng.shuffle(body)
        shuffled = "; ".join([head] + body + [tail])
        s = M.structure_score(F.parse(text, nums), F.parse(shuffled, nums))
        self.assertAlmostEqual(s, 1.0)

    def test_metric_mean(self):
        refs = [REF, REF]
        preds = [REF, "max v; v <= 1520"]
        got = M.metric(refs, preds, [NUMS, NUMS], seed=0)
        self.assertAlmostEqual(got, (1.0 + 1.0 / 3.0) / 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
