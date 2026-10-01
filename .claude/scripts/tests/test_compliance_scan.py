import subprocess, sys, json, unittest
from pathlib import Path
HERE = Path(__file__).parent
SCAN = HERE.parent / "compliance_scan.py"

def scan(fixture):
    r = subprocess.run([sys.executable, str(SCAN), str(HERE / "fixtures" / fixture), "--json"],
                       capture_output=True, text=True)
    return r.returncode, {f["rule"] for f in json.loads(r.stdout)}

class T(unittest.TestCase):
    def test_good_is_clean(self):
        rc, rules = scan("good_solution.py")
        self.assertEqual(rc, 0, rules)
        self.assertNotIn("time-branch", rules)
        self.assertNotIn("external-source", rules)      # URL only in the docstring
        self.assertNotIn("subprocess", rules)

    def test_bad_trips_everything(self):
        rc, rules = scan("bad_solution.py")
        self.assertEqual(rc, 2)
        for r in ["time-branch", "env-branch", "import-fallback", "test-fit", "subprocess",
                  "forbidden-import", "external-source", "time-limit-arg", "silent-fallback-write",
                  "argv-contract", "whole-test-aggregation", "abs-path"]:
            self.assertIn(r, rules, f"missing {r}: got {sorted(rules)}")

    def test_inference_only_flagged(self):
        rc, rules = scan("inference_only.py")
        self.assertEqual(rc, 2)
        self.assertIn("no-training", rules)

if __name__ == "__main__":
    unittest.main()
