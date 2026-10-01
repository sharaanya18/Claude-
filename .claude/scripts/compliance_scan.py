#!/usr/bin/env python3
"""Static compliance scan for an Eris/Shipd `solution.py` (stdlib only, AST based).

Usage:  python3 compliance_scan.py path/to/solution.py [--json] [--strict]

Exit code: 2 if any ERROR finding, 1 if --strict and any WARN, else 0.

Why AST and not regex: docstrings and comments may legitimately mention URLs,
"pip", "time" etc. An AST walk only looks at executable code, so those do not
cause false alarms.  Heuristics that need a human decision are WARN, not ERROR.
Every finding names the rule it comes from (CLAUDE.md section) so the author
can look it up.  This scanner is a checklist, not a verdict: verify by hand.
"""
import ast
import json
import re
import sys
from pathlib import Path

TIME_FUNCS = {"time", "perf_counter", "monotonic", "process_time", "time_ns",
              "perf_counter_ns", "monotonic_ns"}
TIME_MODULES = {"time", "datetime"}
TIME_KWARGS = {"timeout", "time_limit", "timelimit", "time_budget", "max_time",
               "max_seconds", "deadline", "time_limit_s", "n_jobs_timeout"}
TESTISH = re.compile(r"(^|_)(test|tst|te|eval|holdout|private|unseen)(_|$)|^X_?te|^df_?te", re.I)
AGG_ATTRS = {"mean", "std", "var", "median", "sum", "rank", "quantile", "value_counts",
             "describe", "corr", "cov", "min", "max", "nunique", "mode", "skew", "kurt",
             "cumsum", "transform", "groupby", "pct_change", "zscore", "idxmax", "argsort"}
FIT_ATTRS = {"fit", "fit_transform", "partial_fit", "fit_predict"}
BAD_IMPORTS = {"openai", "anthropic", "google.generativeai", "cohere", "mistralai",
               "requests", "urllib3", "selenium", "bs4", "scrapy", "httpx", "aiohttp"}
BAD_URL = re.compile(r"(github\.com|githubusercontent|gitlab\.com|kaggle\.com/(datasets|c/)|"
                     r"drive\.google|dropbox|s3\.amazonaws|storage\.googleapis)", re.I)
HELDOUT = re.compile(r"(^|[/_.\\s-])(private|answers?|held[-_ ]?out|ground[-_ ]?truth|solution\\.csv|labels?_test|test_labels?)([/_.\\s-]|$)", re.I)
ABS_PATH = re.compile(r"^(/home/|/kaggle/|/content/|/mnt/|/Users/|C:\\\\|/tmp/)")
SEED_CALLS = {"seed", "manual_seed", "manual_seed_all", "set_seed", "seed_everything"}


class Finding:
    def __init__(self, level, rule, msg, line=None, ref=""):
        self.level, self.rule, self.msg, self.line, self.ref = level, rule, msg, line, ref

    def as_dict(self):
        return dict(level=self.level, rule=self.rule, line=self.line, msg=self.msg, ref=self.ref)


def dotted(node):
    """Return dotted name for Name/Attribute chains, else ''."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def is_time_call(node):
    if not isinstance(node, ast.Call):
        return False
    name = dotted(node.func)
    last = name.split(".")[-1]
    head = name.split(".")[0]
    if last in TIME_FUNCS and (head in TIME_MODULES or "." not in name):
        return True
    if name.endswith("datetime.now") or name.endswith("datetime.utcnow") or name == "datetime.now":
        return True
    return False


def names_in(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def calls_in(node):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)]


class Scanner(ast.NodeVisitor):
    def __init__(self, src, tree):
        self.src, self.tree = src, tree
        self.findings = []
        self.time_names = set()      # variables derived from clock reads
        self.time_funcs = set()      # user functions whose return derives from clock reads
        self.imports = set()
        self.seen = dict(argv1=False, argv2=False, seed=False, fit=False, to_csv=False,
                         torch=False, det_algos=False, cudnn_det=False, from_pretrained=[],
                         placeholder=False)
        self.fn_stack = []

    def add(self, level, rule, msg, node=None, ref=""):
        self.findings.append(Finding(level, rule, msg, getattr(node, "lineno", None), ref))

    # ---- pass 1: collect clock-derived names -------------------------------------------
    def collect_time_taint(self):
        for _ in range(4):  # propagate a few rounds
            changed = False
            for node in ast.walk(self.tree):
                if isinstance(node, ast.FunctionDef):
                    for r in [n for n in ast.walk(node) if isinstance(n, ast.Return) and n.value is not None]:
                        if self.expr_has_time(r.value) and node.name not in self.time_funcs:
                            self.time_funcs.add(node.name)
                            changed = True
                if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                    value = node.value
                    if value is None or not self.expr_has_time(value):
                        continue
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for t in targets:
                        for n in ast.walk(t):
                            if isinstance(n, ast.Name) and n.id not in self.time_names:
                                self.time_names.add(n.id)
                                changed = True
            if not changed:
                break

    def expr_has_time(self, expr):
        for n in ast.walk(expr):
            if is_time_call(n):
                return True
            if isinstance(n, ast.Call) and dotted(n.func).split(".")[-1] in self.time_funcs:
                return True
            if isinstance(n, ast.Name) and n.id in self.time_names:
                return True
        return False

    # ---- pass 2: rules --------------------------------------------------------------------
    def run(self):
        self.collect_time_taint()
        self.check_source_level()
        self.visit(self.tree)
        self.post_checks()
        return self.findings

    def check_source_level(self):
        size = len(self.src.encode("utf-8"))
        if size >= 512_000:
            self.add("ERROR", "source-size", f"source is {size} bytes (limit 512,000)", ref="CLAUDE.md §1")
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if len(node.value) > 20000:
                    self.add("WARN", "embedded-blob", f"string literal of {len(node.value)} chars "
                             "(embedded data/weights fail the readability check)", node, "CLAUDE.md §1")
                if BAD_URL.search(node.value) and not self.is_docstring(node):
                    self.add("ERROR", "external-source",
                             f"URL/host in executable string: {node.value[:80]!r} (only HF/timm weights allowed)",
                             node, "CLAUDE.md §2.3")
                if HELDOUT.search(node.value) and not self.is_docstring(node) and len(node.value) < 200:
                    self.add("WARN", "heldout-reference", f"string {node.value[:70]!r} refers to private/answer/held-out data: "
                             "the platform's Held-out Answer Ingestion check fails any solution that reads held-out answers; "
                             "remove it or justify it in a comment", node, "CLAUDE.md §7 / platform checks")
                if ABS_PATH.match(node.value) and not self.is_docstring(node):
                    self.add("WARN", "abs-path", f"hardcoded absolute path {node.value[:60]!r}", node, "CLAUDE.md §1")

    def is_docstring(self, const):
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                body = node.body
                if body and isinstance(body[0], ast.Expr) and body[0].value is const:
                    return True
        return False

    def visit_Import(self, node):
        for a in node.names:
            self.note_import(a.name, node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        self.note_import(node.module or "", node)
        self.generic_visit(node)

    def note_import(self, name, node):
        self.imports.add(name)
        if name.split(".")[0] == "torch":
            self.seen["torch"] = True
        for bad in BAD_IMPORTS:
            if name == bad or name.startswith(bad + "."):
                self.add("ERROR", "forbidden-import", f"import {name}: network/LLM-API libraries are not allowed",
                         node, "CLAUDE.md §2.3")

    def visit_Try(self, node):
        has_import = any(isinstance(n, (ast.Import, ast.ImportFrom)) for b in node.body for n in ast.walk(b))
        for h in node.handlers:
            htype = dotted(h.type) if h.type is not None else "bare"
            if has_import:
                self.add("ERROR", "import-fallback", "try/except around an import: environment-dependent "
                         "fallback is rejected by the Deterministic Execution check", node, "CLAUDE.md §3.3")
            writes = [c for s in h.body for c in calls_in(s)
                      if dotted(c.func).split(".")[-1] in {"to_csv", "write", "write_text", "save", "replace", "copy", "copyfile"}]
            if writes:
                self.add("ERROR", "silent-fallback-write",
                         "except-block writes output files: a failure would silently ship a fallback "
                         "submission. Fail loudly instead", node, "CLAUDE.md §7 / rejections log")
            elif htype in {"Exception", "BaseException", "bare"}:
                swallow = all(isinstance(s, (ast.Pass, ast.Continue)) or
                              (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and dotted(s.value.func) == "print")
                              for s in h.body)
                if swallow:
                    self.add("WARN", "swallowed-exception", "broad except that swallows the error; a training "
                             "failure would pass silently", node, "CLAUDE.md §9")
        self.generic_visit(node)

    def check_test_expr(self, test, node, kind):
        for n in ast.walk(test):
            if is_time_call(n):
                self.add("ERROR", "time-branch", f"wall-clock value inside {kind} condition: training/inference "
                         "plan depends on runtime", node, "CLAUDE.md §3 (Deterministic Execution)")
                return
            if isinstance(n, ast.Call) and dotted(n.func).split(".")[-1] in self.time_funcs:
                self.add("ERROR", "time-branch", f"call to clock-derived helper inside {kind} condition",
                         node, "CLAUDE.md §3")
                return
            if isinstance(n, ast.Name) and n.id in self.time_names:
                self.add("ERROR", "time-branch", f"clock-derived variable '{n.id}' inside {kind} condition",
                         node, "CLAUDE.md §3")
                return
            if isinstance(n, ast.Call):
                name = dotted(n.func)
                if name in {"torch.cuda.is_available", "os.cpu_count", "multiprocessing.cpu_count",
                            "torch.cuda.device_count", "psutil.cpu_count", "psutil.virtual_memory",
                            "torch.cuda.mem_get_info", "shutil.disk_usage"}:
                    self.add("ERROR", "env-branch", f"{name}() in a condition: hardware-dependent control flow",
                             node, "CLAUDE.md §3.3")
                    return
            if isinstance(n, ast.Call) and dotted(n.func) in {"os.environ.get", "os.getenv"}:
                self.add("ERROR", "env-branch", "environment variable decides control flow; the recipe must be "
                         "fixed by the code alone", node, "CLAUDE.md §3.3 / patterns F8")
                return

    def visit_If(self, node):
        self.check_test_expr(node.test, node, "if")
        self.generic_visit(node)

    def visit_While(self, node):
        self.check_test_expr(node.test, node, "while")
        self.generic_visit(node)

    def visit_IfExp(self, node):
        self.check_test_expr(node.test, node, "conditional expression")
        self.generic_visit(node)

    def visit_Assert(self, node):
        self.generic_visit(node)

    def visit_Subscript(self, node):
        # sys.argv[1] / sys.argv[2]
        if dotted(node.value) == "sys.argv":
            sl = node.slice
            if isinstance(sl, ast.Constant) and sl.value == 1:
                self.seen["argv1"] = True
            if isinstance(sl, ast.Constant) and sl.value == 2:
                self.seen["argv2"] = True
        self.generic_visit(node)

    def visit_Call(self, node):
        name = dotted(node.func)
        last = node.func.attr if isinstance(node.func, ast.Attribute) else name
        # time used as limit argument
        for kw in node.keywords:
            if kw.arg in TIME_KWARGS and not (isinstance(kw.value, ast.Constant) and kw.value.value in (None, 0)):
                self.add("ERROR", "time-limit-arg", f"{name or 'call'}({kw.arg}=...): wall-clock limited work "
                         "changes the plan with machine speed (no Optuna timeout / time_limit)", node, "CLAUDE.md §3.2")
        if last in {"min", "max"} and name in {"min", "max"}:
            if any(self.expr_has_time(a) for a in node.args):
                self.add("ERROR", "time-branch", "min()/max() over a clock-derived value", node, "CLAUDE.md §3")
        if name in {"os.system"} or name.startswith("subprocess."):
            self.add("ERROR", "subprocess", f"{name}: no shelling out (pip install, git, curl are all banned)",
                     node, "CLAUDE.md §1/§2.3")
        if last in {"exec", "eval", "compile"} and name in {"exec", "eval", "compile"}:
            self.add("ERROR", "dynamic-code", f"{name}(): runtime-generated code fails the source-inspection check",
                     node, "CLAUDE.md §1")
        if last in {"b64decode", "decompress", "loads"} and name.split(".")[0] in {"base64", "zlib", "marshal", "lzma", "bz2", "gzip", "codecs"}:
            self.add("WARN", "encoded-blob", f"{name}: encoded blobs make source opaque", node, "CLAUDE.md §1")
        if name in {"torch.manual_seed", "random.seed", "np.random.seed", "numpy.random.seed", "torch.cuda.manual_seed_all"} \
                or last in SEED_CALLS:
            self.seen["seed"] = True
        if name.endswith("use_deterministic_algorithms"):
            self.seen["det_algos"] = True
        if last in {"fit", "train", "fit_transform", "backward", "train_on_batch"} or last == "step":
            self.seen["fit"] = True
        if last == "to_csv":
            self.seen["to_csv"] = True
            if not any(k.arg == "index" for k in node.keywords):
                self.add("WARN", "to-csv-index", "to_csv without index=False (an index column breaks the format)",
                         node, "CLAUDE.md §5")
        if last == "from_pretrained":
            model = None
            if node.args and isinstance(node.args[0], ast.Constant):
                model = node.args[0].value
            self.seen["from_pretrained"].append((model, any(k.arg == "revision" for k in node.keywords), node.lineno))
        if name == "timm.create_model":
            for kw in node.keywords:
                if kw.arg == "pretrained":
                    self.seen["from_pretrained"].append(("timm:" + str(getattr(node.args[0], "value", "?")) if node.args else "timm", True, node.lineno))
        if name.endswith("cudnn.deterministic") or name == "torch.backends.cudnn.deterministic":
            self.seen["cudnn_det"] = True
        # fits on test-ish data
        if last in FIT_ATTRS:
            for a in list(node.args) + [k.value for k in node.keywords]:
                hit = self.mentions_test(a)
                if hit:
                    self.add("ERROR", "test-fit", f".{last}() on data that mentions test-like name '{hit}': "
                             "scalers/vocab/TF-IDF/PCA/clustering may be fit on train only", node, "CLAUDE.md §2.3.5")
                    break
        # whole-test aggregation (heuristic)
        if isinstance(node.func, ast.Attribute) and last in AGG_ATTRS:
            base = node.func.value
            hit = self.mentions_test(base)
            if hit and not self.is_per_row_axis(node):
                self.add("WARN", "whole-test-aggregation",
                         f"{hit}.{last}(...) reduces over test rows; if the result feeds predictions of the same "
                         "rows it is banned whole-test calibration. Verify each use", node, "CLAUDE.md §2.3.5 / rejections R1")
        if name in {"np.concatenate", "numpy.concatenate", "pd.concat", "np.vstack", "np.hstack", "numpy.vstack"}:
            if node.args and any(self.mentions_test(a) for a in node.args[:1]) and self.mentions_train(node.args[0]):
                self.add("WARN", "train-test-concat", "train and test combined in one array/frame; make sure nothing "
                         "is fit on the combined object", node, "CLAUDE.md §2.3.5 / rejections Q2")
        self.generic_visit(node)

    def is_per_row_axis(self, call):
        for kw in call.keywords:
            if kw.arg == "axis" and isinstance(kw.value, ast.Constant) and kw.value.value in (1, -1, "columns"):
                return True
        return False

    def mentions_test(self, node):
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and TESTISH.search(n.id) and n.id not in {"test_size", "latest"}:
                return n.id
            if isinstance(n, ast.Attribute) and TESTISH.search(n.attr) and n.attr not in {"test_size"}:
                return n.attr
        return None

    def mentions_train(self, node):
        return any(isinstance(n, ast.Name) and re.search(r"train|tr_|X_tr", n.id, re.I) for n in ast.walk(node)) or \
            any(isinstance(n, ast.Name) for n in ast.walk(node))

    def visit_Name(self, node):
        if re.search(r"pseudo", node.id, re.I):
            self.add("WARN", "pseudo-label", f"name '{node.id}': pseudo-labelling on test rows is banned; "
                     "confirm it is train-only", node, "CLAUDE.md §2.3.5")

    def visit_FunctionDef(self, node):
        self.fn_stack.append(node.name)
        self.generic_visit(node)
        self.fn_stack.pop()

    def visit_ClassDef(self, node):
        self.generic_visit(node)

    # ---- global checks ---------------------------------------------------------------------
    def post_checks(self):
        if not (self.seen["argv1"] and self.seen["argv2"]):
            self.add("ERROR", "argv-contract", "solution must read sys.argv[1] (public_dir) and sys.argv[2] "
                     "(submission_out); the platform runs `python3 solution.py <public_dir> <submission_out>`",
                     ref="CLAUDE.md §1")
        if not self.seen["seed"]:
            self.add("WARN", "no-seed", "no seeding call found (random/numpy/torch)", ref="CLAUDE.md §3.4")
        if not self.seen["fit"]:
            self.add("ERROR", "no-training", "no .fit/.train/.backward/.step call: inference-only solutions are rejected",
                     ref="CLAUDE.md §2.1")
        if not self.seen["to_csv"]:
            self.add("WARN", "no-to-csv", "no to_csv call found; make sure the submission is written", ref="CLAUDE.md §5")
        if self.seen["torch"] and not (self.seen["det_algos"] or self.seen["cudnn_det"]):
            self.add("WARN", "gpu-determinism", "torch imported without cudnn.deterministic / "
                     "use_deterministic_algorithms(True, warn_only=True)", ref="CLAUDE.md §3.6")
        for model, pinned, line in self.seen["from_pretrained"]:
            lvl = "INFO"
            msg = f"pretrained load {model!r}: confirm public HF/timm hub id, allowed by this challenge"
            if not pinned:
                msg += "; consider pinning revision=<sha>"
            self.findings.append(Finding(lvl, "pretrained", msg, line, "CLAUDE.md §2.4 / patterns E8"))
        if "argparse" in self.imports:
            self.add("WARN", "argparse", "argparse found: platform passes exactly two positional args; "
                     "use sys.argv directly or parse_known_args", ref="CLAUDE.md §9 (exit code 2)")


def scan(path):
    src = Path(path).read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [Finding("ERROR", "syntax", f"cannot parse: {e}", getattr(e, "lineno", None))]
    return Scanner(src, tree).run()


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 64
    path = argv[1]
    as_json = "--json" in argv
    strict = "--strict" in argv
    findings = scan(path)
    order = {"ERROR": 0, "WARN": 1, "INFO": 2}
    findings.sort(key=lambda f: (order[f.level], f.line or 0))
    if as_json:
        print(json.dumps([f.as_dict() for f in findings], indent=2))
    else:
        print(f"compliance_scan: {path}")
        for f in findings:
            loc = f"line {f.line}" if f.line else "file"
            ref = f"  [{f.ref}]" if f.ref else ""
            print(f"  {f.level:5} {f.rule:22} {loc:9} {f.msg}{ref}")
        n = {k: sum(1 for f in findings if f.level == k) for k in order}
        print(f"summary: {n['ERROR']} error(s), {n['WARN']} warning(s), {n['INFO']} info. "
              "Heuristic scan: verify by hand; a clean scan is not a compliance verdict.")
    if any(f.level == "ERROR" for f in findings):
        return 2
    if strict and any(f.level == "WARN" for f in findings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
