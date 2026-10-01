"""Regression tests for the stdlib helper scripts. Run: python3 -m unittest discover -s .claude/scripts/tests"""
import csv, subprocess, sys, tempfile, unittest
from pathlib import Path

S = Path(__file__).resolve().parent.parent


def run(*args):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)


class ValidateSubmission(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        write(self.d / "sample.csv", [["id", "y", "tags"], ["a", "0", "[]"], ["b", "0", "[]"]])

    def test_pass(self):
        write(self.d / "s.csv", [["id", "y", "tags"], ["a", "0.2", "[1]"], ["b", "0.9", "[]"]])
        r = run(S / "validate_submission.py", self.d / "s.csv", self.d / "sample.csv", "--prob-cols", "y", "--json-cols", "tags")
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_fail_modes(self):
        cases = {
            "order": [["id", "y", "tags"], ["b", "0.2", "[]"], ["a", "0.2", "[]"]],
            "empty": [["id", "y", "tags"], ["a", "", "[]"], ["b", "0.2", "[]"]],
            "rows": [["id", "y", "tags"], ["a", "0.2", "[]"]],
            "json": [["id", "y", "tags"], ["a", "0.2", "[oops"], ["b", "0.2", "[]"]],
            "prob": [["id", "y", "tags"], ["a", "1.5", "[]"], ["b", "0.2", "[]"]],
            "header": [["id", "z", "tags"], ["a", "0.2", "[]"], ["b", "0.2", "[]"]],
            "dup": [["id", "y", "tags"], ["a", "0.2", "[]"], ["a", "0.2", "[]"]],
        }
        for name, rows in cases.items():
            write(self.d / f"{name}.csv", rows)
            r = run(S / "validate_submission.py", self.d / f"{name}.csv", self.d / "sample.csv", "--prob-cols", "y", "--json-cols", "tags")
            self.assertEqual(r.returncode, 1, f"{name}: {r.stdout}")


class MakeGroups(unittest.TestCase):
    def test_keys_and_duplicates(self):
        d = Path(tempfile.mkdtemp())
        rows = [["id", "site", "text", "label"]]
        for i in range(20):
            rows.append([f"r{i}", f"S{i // 4}", f"unique text number {i} about subject {i * 37}", "A" if i % 2 else "B"])
        rows.append(["dup", "Sx", "unique text number 3 about subject 111", "A"])      # exact dup of row 3 (differs only in site)
        write(d / "t.csv", rows)
        r = run(S / "make_groups.py", d / "t.csv", "--keys", "site", "--text", "text", "--jaccard", "1.0", "--folds", "4",
                "--label", "label", "--out", d / "g.csv")
        self.assertEqual(r.returncode, 0, r.stderr)
        g = list(csv.DictReader(open(d / "g.csv")))
        self.assertEqual(len(g), 21)
        # the duplicate row must share its group with row 3 and therefore its fold
        self.assertEqual(g[20]["group"], g[3]["group"])
        self.assertEqual(g[20]["fold"], g[3]["fold"])
        # groups never split across folds
        by = {}
        for x in g:
            by.setdefault(x["group"], set()).add(x["fold"])
        self.assertTrue(all(len(v) == 1 for v in by.values()))


class TemplateScan(unittest.TestCase):
    def test_template_only_missing_training(self):
        r = run(S / "compliance_scan.py", S / "solution_template.py", "--json")
        import json
        rules = {f["rule"] for f in json.loads(r.stdout) if f["level"] == "ERROR"}
        self.assertEqual(rules, {"no-training"})


if __name__ == "__main__":
    unittest.main()


class HalfRows(unittest.TestCase):
    def _public(self):
        d = Path(tempfile.mkdtemp()) / "public"
        d.mkdir()
        write(d / "train.csv", [["id", "x", "y"]] + [[f"t{i}", str(i), str(i % 2)] for i in range(10)])
        write(d / "test.csv", [["id", "x"]] + [[f"e{i}", str(i * 3)] for i in range(10)])
        write(d / "sample_submission.csv", [["id", "y"]] + [[f"e{i}", "0"] for i in range(10)])
        return d

    def _solution(self, body):
        f = Path(tempfile.mkdtemp()) / "solution.py"
        f.write_text("import csv, sys\nfrom pathlib import Path\npub, out = Path(sys.argv[1]), Path(sys.argv[2])\n"
                     "test = list(csv.DictReader(open(pub / 'test.csv')))\n" + body +
                     "out.parent.mkdir(parents=True, exist_ok=True)\n"
                     "w = csv.writer(open(out, 'w', newline='')); w.writerow(['id', 'y'])\n"
                     "[w.writerow([r['id'], p]) for r, p in zip(test, preds)]\n")
        return f

    def test_per_row_passes(self):
        sol = self._solution("preds = [float(r['x']) * 2 for r in test]\n")
        r = run(S / "half_rows_test.py", sol, self._public())
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_whole_test_normalisation_fails(self):
        sol = self._solution("m = sum(float(r['x']) for r in test) / len(test)\npreds = [float(r['x']) - m for r in test]\n")
        r = run(S / "half_rows_test.py", sol, self._public())
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)


class SchemaAudit(unittest.TestCase):
    def test_prints_schema_only(self):
        d = Path(tempfile.mkdtemp())
        write(d / "train.csv", [["id", "text", "label"], ["a1", "hello secret", "0"]])
        write(d / "test.csv", [["id", "text"], ["b7", "TOPSECRETVALUE"], ["b8", "x"]])
        write(d / "sample_submission.csv", [["id", "label"], ["b7", "0"], ["b8", "0"]])
        r = run(S / "test_schema_audit.py", d)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("TOPSECRETVALUE", r.stdout)
        self.assertIn("2 rows", r.stdout)


class TokenKeys(unittest.TestCase):
    def test_token_key_groups(self):
        d = Path(tempfile.mkdtemp())
        write(d / "t.csv", [["id", "text"], ["1", "a proj1_3 b"], ["2", "proj1_9 c"], ["3", "proj2_1 d"], ["4", "x proj2_5"]])
        r = run(S / "make_groups.py", d / "t.csv", "--token-key", r"text:proj(\d+)_", "--out", d / "g.csv")
        self.assertEqual(r.returncode, 0, r.stderr)
        g = [x["group"] for x in csv.DictReader(open(d / "g.csv"))]
        self.assertEqual(g[0], g[1])
        self.assertEqual(g[2], g[3])
        self.assertNotEqual(g[0], g[2])


class ScannerFalsePositive(unittest.TestCase):
    def test_bare_fit_predict_helper_is_not_flagged(self):
        d = Path(tempfile.mkdtemp())
        f = d / "solution.py"
        f.write_text("import sys\ndef fit_predict(train, y, test):\n    return test\nimport sklearn\n"
                     "p = fit_predict([1], [1], [2])\nclass M:\n    def fit(self, x): return self\nM().fit([1])\n"
                     "import random; random.seed(0)\nopen(sys.argv[2], 'w').write(str(sys.argv[1]))\n")
        import json
        r = run(S / "compliance_scan.py", f, "--json")
        rules = {x["rule"] for x in json.loads(r.stdout)}
        self.assertNotIn("test-fit", rules)


class Corpus(unittest.TestCase):
    def test_blind_first_protocol(self):
        import shutil, time
        root = S.parent.parent / "corpus"
        slug = "unittest_tmp"
        shutil.rmtree(root / slug, ignore_errors=True)
        try:
            self.assertEqual(run(S / "corpus.py", "new", slug).returncode, 0)
            self.assertEqual(run(S / "corpus.py", "check", slug).returncode, 1)       # no blind plan yet
            (root / slug / "blind" / "eris_plan.md").write_text("plan")
            time.sleep(0.05)
            (root / slug / "digests" / "rank1.md").write_text("digest")
            self.assertEqual(run(S / "corpus.py", "check", slug).returncode, 0)
            time.sleep(0.05)
            (root / slug / "blind" / "eris_plan.md").write_text("rewritten later")    # blind plan newer than digest: protocol broken
            self.assertEqual(run(S / "corpus.py", "check", slug).returncode, 1)
        finally:
            shutil.rmtree(root / slug, ignore_errors=True)
            if root.exists() and not any(root.iterdir()):
                root.rmdir()
