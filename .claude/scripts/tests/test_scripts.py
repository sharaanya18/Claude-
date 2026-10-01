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
