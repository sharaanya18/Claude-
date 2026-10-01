"""Solver for "Weakly Supervised Grounded Integer Program Formulation".

Reads <public_dir>, fine-tunes a public decoder LM (Qwen2.5-0.5B base) inside this script to write a grounded MILP for each word
problem, and writes <submission_out> (case_id, formulation).

Requirements map (challenge rules -> where this script satisfies them)
  * The model must be trained on the supplied data: stage 1 fine-tunes the pretrained weights on the 248 seed formulations
    (several randomly re-numbered views of each). Stages 2-5 are expert iteration: the model samples programs for the other
    training cases, a MILP solver (scipy/HiGHS) keeps only samples whose optimum equals the case's given optimal_value
    (the intended use of weak supervision, training cases only), and the model is retrained on seeds + kept samples.
  * No hand-written rules produce or repair formulations: every submitted string is verbatim model output. Hand-built code
    only (a) annotates the INPUT text with the reference of each number it contains, and (b) parses model outputs to
    check them / choose among the model's own samples for the same case (selection among outputs is allowed).
  * No retrieval or prompting with training cases; no case_id / row-order signal (ids are used only to join files).
  * Test cases are used only as one-case-at-a-time prompts: no fitting, no pseudo-labels, no whole-test statistics.
  * Only public weights (Qwen/Qwen2.5-0.5B from Hugging Face, Apache-2.0, pinned revision); no OR-formulation checkpoints,
    no external data, no API calls.
  * Fixed work plan: epochs, samples, batch sizes, seeds are constants; elapsed time is only logged.
"""
import os
import sys
import json
import math
import random
import re
import time
from pathlib import Path

PUBLIC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./dataset/public")
SUBMISSION_OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./working/submission.csv")
WORK_DIR = SUBMISSION_OUT.parent
WORK_DIR.mkdir(parents=True, exist_ok=True)

os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = str(WORK_DIR / "hf_cache")

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field

# ------------------------------------------------------------------ fixed work plan (no clock/hardware branches)
SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
DEVICE = "cuda"
AMP = "bf16"                 # "bf16" on the A10G; dev runs on older GPUs may set "fp16" or "off"
MAX_LEN = 1280               # prompt + target tokens (asserted: nothing is truncated)
LR = 5e-5
WARMUP_FRAC = 0.03
MICRO_BS = 4
ACCUM = 4
VIEWS_SEED = 4              # random ref-renumbering views per seed example
VIEWS_EI = 2                # views per accepted self-generated target
VIEWS_EI_LARGE = 4          # ... when the target has >= LARGE_VARS variables (test problems are larger)
LARGE_VARS = 6
EPOCHS_S1 = 3
EPOCHS_S3 = 2
EPOCHS_S5 = 2
K_EI = 4                    # samples per unsolved training case per expert-iteration round
T_EI = 0.8
TOP_P = 0.95
MAXNEW_EI = 384
GEN_PROMPTS_PER_BATCH = 64
EI_ROUNDS = 2               # 1 = ship the stage-3 model (fallback plan), 2 = full plan
K_TEST = 8
T_TEST = 0.7
MAXNEW_TEST = 768
TEST_PROMPTS_PER_BATCH = 24
N_PERTURB = 3               # train-side perturbation draws used to filter accepted samples
ALLOWED_LITERALS = (0.0, 1.0, 100.0)   # the only numbers the references may type in instead of referencing
T0 = time.time()            # LOGGING ONLY: never used in a condition


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(4)


# ================================================================== grammar parser + MILP solver (verification tool)
# Re-implementation of the challenge's formulation syntax. It is used (a) to verify training candidates against the given
# optimal values and (b) to filter / compare the model's own candidates for one case. It never rewrites a formulation.

MAX_VARS = 60
MAX_CONS = 150

# optimality gap used for MILP solves. scipy's default (1e-4) is looser than the 1e-6 counterfactual tolerance.
MIP_REL_GAP = 1e-9
# deterministic work cap per MILP (a node count, never a clock) so a degenerate sampled program cannot stall the run
MIP_NODE_LIMIT = 20000


class FormulationError(ValueError):
    """Raised for anything the grader would reject: syntax, nonlinearity, unknown reference, limits."""


# --------------------------------------------------------------------------- tokenizer
_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?%?|\.\d+%?"
_TOKEN_RE = re.compile(
    r"\s*(?:(?P<num>" + _NUM + r")|(?P<id>[A-Za-z_][A-Za-z0-9_]*)|(?P<op><=|>=|=|\+|-|\*|/|\(|\)|,|:))"
)
_REF_RE = re.compile(r"x\d+\Z")
_Q_RE = re.compile(r"q\d+\Z")
_CMP = ("<=", ">=", "=")


def _tokenize(stmt: str):
    toks, pos, n = [], 0, len(stmt)
    while pos < n:
        if stmt[pos:].strip() == "":
            break
        m = _TOKEN_RE.match(stmt, pos)
        if not m:
            raise FormulationError(f"cannot tokenize near {stmt[pos:pos + 20]!r}")
        if m.group("num") is not None:
            toks.append(("num", m.group("num")))
        elif m.group("id") is not None:
            toks.append(("id", m.group("id")))
        else:
            toks.append(("op", m.group("op")))
        pos = m.end()
    return toks


def _num_value(s: str) -> float:
    pct = s.endswith("%")
    if pct:
        s = s[:-1]
    v = float(s.replace(",", ""))
    return v / 100.0 if pct else v


# --------------------------------------------------------------------------- linear expressions
@dataclass
class Lin:
    coefs: dict = field(default_factory=dict)  # variable -> coefficient
    const: float = 0.0

    def is_const(self) -> bool:
        return not self.coefs

    def add(self, other: "Lin", sign: float = 1.0) -> "Lin":
        out = Lin(dict(self.coefs), self.const + sign * other.const)
        for v, c in other.coefs.items():
            out.coefs[v] = out.coefs.get(v, 0.0) + sign * c
        return out

    def scale(self, k: float) -> "Lin":
        return Lin({v: c * k for v, c in self.coefs.items()}, self.const * k)


class _Parser:
    def __init__(self, toks, numbers, var_registry):
        self.t, self.i, self.numbers, self.vars = toks, 0, numbers, var_registry

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def expr(self) -> Lin:
        acc = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()[1]
            acc = acc.add(self.term(), 1.0 if op == "+" else -1.0)
        return acc

    def term(self) -> Lin:
        acc = self.factor()
        while self.peek() in (("op", "*"), ("op", "/")):
            op = self.take()[1]
            rhs = self.factor()
            if op == "*":
                if acc.coefs and rhs.coefs:
                    raise FormulationError("nonlinear product (variables on both sides)")
                acc = rhs.scale(acc.const) if not acc.coefs else acc.scale(rhs.const)
            else:
                if rhs.coefs:
                    raise FormulationError("nonlinear quotient (variable in denominator)")
                if rhs.const == 0:
                    raise FormulationError("division by zero")
                acc = acc.scale(1.0 / rhs.const)
        return acc

    def factor(self) -> Lin:
        kind, val = self.take()
        if kind is None:
            raise FormulationError("unexpected end of expression")
        if kind == "op":
            if val in ("+", "-"):
                f = self.factor()
                return f.scale(1.0 if val == "+" else -1.0)
            if val == "(":
                e = self.expr()
                if self.take() != ("op", ")"):
                    raise FormulationError("missing ')'")
                return e
            raise FormulationError(f"unexpected operator {val!r}")
        if kind == "num":
            return Lin({}, _num_value(val))
        # identifier: reference or variable
        if _REF_RE.match(val):
            if val not in self.numbers:
                raise FormulationError(f"reference {val} does not exist")
            return Lin({}, float(self.numbers[val]))
        _check_var_name(val)
        self.vars.setdefault(val, len(self.vars))
        return Lin({val: 1.0}, 0.0)


def _check_var_name(name: str):
    if _REF_RE.match(name) or _Q_RE.match(name):
        raise FormulationError(f"illegal variable name {name!r}")


# --------------------------------------------------------------------------- model
@dataclass
class Model:
    sense: str                      # 'max' | 'min'
    obj: dict                       # variable -> coefficient
    obj_const: float
    rows: list                      # list of (coefs dict, sense in {'<=','>=','='}, rhs float)
    int_vars: set
    bin_vars: set
    variables: list                 # in order of first appearance


def normalize_numbers(numbers) -> dict:
    """Accept numbers_json as a JSON string, a list of [ref, value] or a dict; return {ref: float}."""
    if isinstance(numbers, str):
        numbers = json.loads(numbers)
    if isinstance(numbers, dict):
        return {str(k): float(v) for k, v in numbers.items()}
    return {str(k): float(v) for k, v in numbers}


def parse(text: str, numbers) -> Model:
    """Parse a formulation against `numbers` (numbers_json). Raises FormulationError if the grader would reject it."""
    if not isinstance(text, str):
        raise FormulationError("formulation is not a string")
    numbers = normalize_numbers(numbers)
    stmts = [s for s in text.split(";") if s.strip() != ""]
    if not stmts:
        raise FormulationError("empty program")
    registry: dict = {}
    sense = obj = None
    obj_const = 0.0
    rows, int_vars, bin_vars = [], set(), set()
    for si, stmt in enumerate(stmts):
        toks = _tokenize(stmt)
        if not toks:
            continue
        head = toks[0][1].lower() if toks[0][0] == "id" else None
        if si == 0:
            if head not in ("max", "maximize", "min", "minimize"):
                raise FormulationError("first statement must be the objective (max/min)")
            sense = "max" if head.startswith("max") else "min"
            rest = toks[1:]
            if rest and rest[0] == ("op", ":"):
                rest = rest[1:]
            if not rest:
                raise FormulationError("empty objective")
            p = _Parser(rest, numbers, registry)
            lin = p.expr()
            if p.i != len(rest):
                raise FormulationError("trailing tokens in objective")
            obj, obj_const = lin.coefs, lin.const
            continue
        if head in ("int", "bin") and not any(t[0] == "op" and t[1] in _CMP for t in toks):
            names = []
            expect_name = True
            for kind, val in toks[1:]:
                if kind == "id" and (expect_name or True):
                    _check_var_name(val)
                    names.append(val)
                    expect_name = False
                elif (kind, val) == ("op", ","):
                    expect_name = True
                else:
                    raise FormulationError("bad token in declaration")
            if not names:
                raise FormulationError("empty declaration")
            for nme in names:
                registry.setdefault(nme, len(registry))
                (int_vars if head == "int" else bin_vars).add(nme)
            continue
        cmp_idx = [k for k, t in enumerate(toks) if t[0] == "op" and t[1] in _CMP]
        if len(cmp_idx) != 1:
            raise FormulationError("constraint needs exactly one comparator")
        k = cmp_idx[0]
        lt, rt = toks[:k], toks[k + 1:]
        if not lt or not rt:
            raise FormulationError("empty side in constraint")
        pl, pr = _Parser(lt, numbers, registry), _Parser(rt, numbers, registry)
        left, right = pl.expr(), pr.expr()
        if pl.i != len(lt) or pr.i != len(rt):
            raise FormulationError("trailing tokens in constraint")
        diff = left.add(right, -1.0)           # left - right  (sense) 0
        rows.append((diff.coefs, toks[k][1], -diff.const))
        if len(rows) > MAX_CONS:
            raise FormulationError("too many constraints")
    if len(registry) > MAX_VARS:
        raise FormulationError("too many variables")
    bin_vars -= set()  # bin overrides int below
    int_vars = int_vars - bin_vars
    for row in rows:
        for v, c in row[0].items():
            if not math.isfinite(c):
                raise FormulationError("non-finite coefficient")
    return Model(sense, obj, obj_const, rows, int_vars, bin_vars, list(registry))


# --------------------------------------------------------------------------- solver
def solve(model: Model, mip_rel_gap: float = MIP_REL_GAP):
    """Return the optimal objective value (float) or None if infeasible / unbounded / solver failure."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    vars_ = model.variables
    n = len(vars_)
    idx = {v: i for i, v in enumerate(vars_)}
    sgn = -1.0 if model.sense == "max" else 1.0
    if n == 0:
        for coefs, s, rhs in model.rows:
            if (s == "<=" and 0 > rhs + 1e-9) or (s == ">=" and 0 < rhs - 1e-9) or (s == "=" and abs(rhs) > 1e-9):
                return None
        return float(model.obj_const)
    c = np.zeros(n)
    for v, k in model.obj.items():
        c[idx[v]] = sgn * k
    lb = np.zeros(n)
    ub = np.full(n, np.inf)
    integrality = np.zeros(n)
    for v in model.int_vars:
        integrality[idx[v]] = 1
    for v in model.bin_vars:
        integrality[idx[v]] = 1
        ub[idx[v]] = 1.0
    cons = []
    if model.rows:
        A = np.zeros((len(model.rows), n))
        lo = np.full(len(model.rows), -np.inf)
        hi = np.full(len(model.rows), np.inf)
        for r, (coefs, s, rhs) in enumerate(model.rows):
            for v, k in coefs.items():
                A[r, idx[v]] += k
            if s in ("<=", "="):
                hi[r] = rhs
            if s in (">=", "="):
                lo[r] = rhs
        cons = [LinearConstraint(A, lo, hi)]
    try:
        res = milp(c, constraints=cons, integrality=integrality, bounds=Bounds(lb, ub),
                   options={"mip_rel_gap": mip_rel_gap, "disp": False, "node_limit": MIP_NODE_LIMIT})
    except Exception:
        return None
    if res.status != 0 or res.x is None or not np.isfinite(res.fun):
        return None
    return float(sgn * res.fun + model.obj_const)


def optimum(text, numbers):
    """Optimal value of the program or None for unparsable / nonlinear / bad reference / infeasible / unbounded."""
    try:
        return solve(parse(text, numbers))
    except FormulationError:
        return None


def analyze(text, numbers) -> dict:
    """Validity report for candidate filtering: {'parsed','error','optimum','n_vars','n_rows'}."""
    try:
        m = parse(text, numbers)
    except FormulationError as e:
        return {"parsed": False, "error": str(e), "optimum": None, "n_vars": 0, "n_rows": 0}
    return {"parsed": True, "error": None, "optimum": solve(m), "n_vars": len(m.variables), "n_rows": len(m.rows)}


def rel_close(a, b, rel_tol: float) -> bool:
    """|a-b| <= rel_tol * |b| (b = reference), with a tiny absolute floor for reference values near zero."""
    return abs(a - b) <= rel_tol * abs(b) + 1e-12


def matches_optimum(text, numbers, optimal_value: float, rel_tol: float = 1e-4) -> bool:
    """Weak-supervision verifier: does the program's optimum on the real numbers equal the label?

    This is exactly the value term of the metric; use it as the reward / filter when searching for
    formulations of training cases that have only an optimal_value label.
    """
    v = optimum(text, numbers)
    return v is not None and rel_close(v, float(optimal_value), rel_tol)


# ================================================================== inputs: annotate each number with its reference
# A number is a run of digits with optional decimals and thousands commas; "25%" has value 0.25; digits glued to letters
# (RM1, P2) are not numbers. numbers_json maps each distinct value to one reference, so the annotation is unambiguous.
_NUM_RE = re.compile(r"(?<![A-Za-z_\d.,])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?|(?<![A-Za-z_\d.,])\.\d+%?")
_REF_IN_TEXT = re.compile(r"\bx(\d+)\b")


def _value_key(v):
    return round(float(v), 9)


def make_prompt(problem, numbers_list, perm=None):
    """Prompt text for one case. perm (optional) renumbers the references consistently (augmentation of a real case)."""
    nums = [(str(r), float(v)) for r, v in numbers_list]
    new_ref = {}
    for r, v in nums:
        k = int(r[1:])
        new_ref[r] = f"x{perm[k] if perm is not None else k}"
    by_value = {_value_key(v): new_ref[r] for r, v in nums}
    text = problem.replace("¶", "\n")

    def annotate(m):
        tok = m.group()
        pct = tok.endswith("%")
        val = float(tok.rstrip("%").replace(",", ""))
        val = val / 100.0 if pct else val
        ref = by_value.get(_value_key(val))
        return f"{tok} [{ref}]" if ref is not None else tok

    text = _NUM_RE.sub(annotate, text)
    listing = "; ".join(f"{new_ref[r]}={format_number(v)}"
                        for r, v in sorted(nums, key=lambda rv: int(new_ref[rv[0]][1:])))
    return f"### Problem:\n{text}\n### Numbers:\n{listing}\n### Formulation:\n"


def format_number(v):
    return str(int(v)) if float(v).is_integer() and abs(v) < 1e15 else repr(float(v))


def permute_refs(text, perm):
    return _REF_IN_TEXT.sub(lambda m: f"x{perm[int(m.group(1))]}", text)


def permuted_numbers(numbers_list, perm):
    return [[f"x{perm[int(str(r)[1:])]}", v] for r, v in numbers_list]


@dataclass
class Case:
    case_id: str
    problem: str
    numbers: list            # [[ref, value], ...] as in numbers_json
    label: float = None      # optimal_value (train only)
    seed_formulation: str = None


def load_cases(public_dir):
    train = pd.read_csv(public_dir / "train.csv", keep_default_na=False)
    labels = pd.read_csv(public_dir / "train_labels.csv")
    seeds = pd.read_csv(public_dir / "train_seed_formulations.csv", keep_default_na=False)
    test = pd.read_csv(public_dir / "test.csv", keep_default_na=False)
    lab = dict(zip(labels.case_id, labels.optimal_value))
    sd = dict(zip(seeds.case_id, seeds.formulation))
    tr = [Case(r.case_id, r.problem, json.loads(r.numbers_json), float(lab[r.case_id]), sd.get(r.case_id))
          for r in train.itertuples()]
    te = [Case(r.case_id, r.problem, json.loads(r.numbers_json)) for r in test.itertuples()]
    return tr, te


# ================================================================== model utilities
def amp_ctx():
    if AMP == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if AMP == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cuda", enabled=False)


def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = model.float().to(DEVICE)          # fp32 master weights; autocast does the low-precision math
    model.config.use_cache = True
    return model


def build_examples(cases, tok, perm_rng):
    """Training examples (prompt ids, target ids) with random reference renumbering views of each (case, formulation)."""
    ex = []
    for case, formulation, n_views in cases:
        n = len(case.numbers)
        for v in range(n_views):
            perm = list(range(n)) if v == 0 else [int(i) for i in perm_rng.permutation(n)]
            prompt = make_prompt(case.problem, case.numbers, perm)
            target = permute_refs(formulation, perm)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            t_ids = tok(target, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            if len(p_ids) + len(t_ids) > MAX_LEN:     # only possible for long self-generated targets; seeds are asserted below
                assert case.seed_formulation is None, "seed example longer than MAX_LEN"
                continue
            ex.append((p_ids, t_ids))
    return ex


def train_from_base(examples, epochs, tok, tag):
    """Fine-tune a FRESH copy of the pretrained weights on the examples (full fine-tuning, causal LM loss on the target)."""
    seed_everything(SEED)
    model = load_model()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(len(examples) / (MICRO_BS * ACCUM))
    total = steps_per_epoch * epochs
    warm = max(1, int(WARMUP_FRAC * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=(AMP == "fp16"))
    rng = np.random.default_rng(SEED)
    w0 = model.lm_head.weight.detach().clone()
    pad = tok.pad_token_id
    step = 0
    for ep in range(epochs):
        order = rng.permutation(len(examples))
        # length-bucketed micro batches: sort inside chunks of 8 micro batches to cut padding, deterministic given the rng
        chunks = [order[i:i + MICRO_BS * 8] for i in range(0, len(order), MICRO_BS * 8)]
        micro = []
        for ch in chunks:
            ch = sorted(ch, key=lambda i: len(examples[i][0]) + len(examples[i][1]))
            micro += [ch[i:i + MICRO_BS] for i in range(0, len(ch), MICRO_BS)]
        micro = [micro[i] for i in rng.permutation(len(micro))]
        tot_loss, n_loss = 0.0, 0
        for mi in range(0, len(micro), ACCUM):
            group = micro[mi:mi + ACCUM]
            for mb in group:
                seqs = [examples[i][0] + examples[i][1] for i in mb]
                L = max(len(s) for s in seqs)
                ids = torch.full((len(mb), L), pad, dtype=torch.long)
                att = torch.zeros((len(mb), L), dtype=torch.long)
                tmask = torch.zeros((len(mb), L), dtype=torch.bool)
                for r, i in enumerate(mb):
                    p, t = examples[i]
                    ids[r, :len(p) + len(t)] = torch.tensor(p + t)
                    att[r, :len(p) + len(t)] = 1
                    tmask[r, len(p):len(p) + len(t)] = True
                ids, att, tmask = ids.to(DEVICE), att.to(DEVICE), tmask.to(DEVICE)
                with amp_ctx():
                    h = model.model(input_ids=ids, attention_mask=att).last_hidden_state
                    sel = tmask[:, 1:]
                    logits = model.lm_head(h[:, :-1][sel]).float()
                loss = torch.nn.functional.cross_entropy(logits, ids[:, 1:][sel])
                assert torch.isfinite(loss), "non-finite loss"
                scaler.scale(loss / len(group)).backward()
                tot_loss += float(loss.detach())
                n_loss += 1
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
        log(f"[{tag}] epoch {ep + 1}/{epochs} loss {tot_loss / max(1, n_loss):.4f} ({len(examples)} examples)")
    delta = float((model.lm_head.weight.detach() - w0).abs().max())
    assert delta > 0, "weights did not change: training is not happening"
    model.eval()
    model.config.use_cache = True
    model.gradient_checkpointing_disable()
    # fp32 master weights are only needed for the optimiser; generation and scoring run on a low-precision copy so the
    # weights are not re-cast at every decoding step (several times faster)
    return model.to({"bf16": torch.bfloat16, "fp16": torch.float16}.get(AMP, torch.float32))


@torch.no_grad()
def generate(model, tok, prompts, k, temperature, top_p, max_new, batch, greedy=False, tag="gen"):
    """k samples (or one greedy output) per prompt. Returns list (per prompt) of (text, finished_with_eos)."""
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [None] * len(prompts)
    for bi, s in enumerate(range(0, len(order), batch)):
        idx = order[s:s + batch]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
        torch.manual_seed(SEED + bi)
        kw = dict(do_sample=False) if greedy else dict(do_sample=True, temperature=temperature, top_p=top_p, top_k=0,
                                                       num_return_sequences=k)
        with amp_ctx():
            gen = model.generate(**enc, max_new_tokens=max_new, pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id, use_cache=True, **kw)
        new = gen[:, enc["input_ids"].shape[1]:]
        per = 1 if greedy else k
        for j, i in enumerate(idx):
            res = []
            for q in range(per):
                row = new[j * per + q]
                finished = bool((row == tok.eos_token_id).any())
                res.append((tok.decode(row, skip_special_tokens=True).strip(), finished))
            out[i] = res
        if bi % 10 == 0:
            log(f"[{tag}] generated batch {bi + 1}/{math.ceil(len(order) / batch)}")
    return out


@torch.no_grad()
def mean_logprobs(model, tok, prompt, texts):
    """Mean token log-probability of each text given the prompt (used only to break ties among a case's own samples)."""
    res = []
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    for s in range(0, len(texts), 8):
        chunk = texts[s:s + 8]
        seqs = [p_ids + tok(t, add_special_tokens=False)["input_ids"] + [tok.eos_token_id] for t in chunk]
        L = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, x in enumerate(seqs):
            ids[r, :len(x)] = torch.tensor(x)
            att[r, :len(x)] = 1
        ids, att = ids.to(DEVICE), att.to(DEVICE)
        with amp_ctx():
            logits = model(input_ids=ids, attention_mask=att).logits.float()
        lp = torch.log_softmax(logits[:, :-1], dim=-1).gather(-1, ids[:, 1:, None])[..., 0]
        for r, x in enumerate(seqs):
            res.append(float(lp[r, len(p_ids) - 1:len(x) - 1].mean()))
    return res


# ================================================================== weak supervision: verify candidates for TRAIN cases
def typed_literals_ok(text):
    """True if the only plain numbers the program types in are 0, 1, 100 (everything else must be a reference)."""
    for stmt in text.split(";"):
        try:
            toks = _tokenize(stmt)
        except FormulationError:
            return False
        for kind, val in toks:
            if kind == "num" and _num_value(val) not in ALLOWED_LITERALS:
                return False
    return True


def perturb_numbers(numbers, rng):
    """One counterfactual-style version of a TRAIN case's numbers: each distinct value times its own factor in
    [0.55,0.95] U [1.05,1.45]; equal values share a factor; year-like whole numbers 1990..2030 stay fixed."""
    factors = {}
    out = {}
    for r, v in numbers:
        v = float(v)
        key = _value_key(v)
        if key not in factors:
            if v.is_integer() and 1990 <= v <= 2030:
                factors[key] = 1.0
            else:
                f = rng.uniform(0.55, 1.45 - 0.1)
                factors[key] = f if f < 0.95 else f + 0.1
        out[str(r)] = v * factors[key]
    return out


def accept_targets(cases, samples, tag):
    """Expert-iteration filter over train cases without a known formulation.
    Keep a sample if it parses, is linear, uses only references (plus 0/1/100), and its optimum equals the label
    (rel 1e-4). Then require a finite optimum on N_PERTURB perturbed copies of the train numbers; among surviving
    distinct programs keep one from the largest cluster of equal perturbed-optimum vectors."""
    accepted, n_valid, n_value = {}, 0, 0
    for ci, (case, cands) in enumerate(zip(cases, samples)):
        rng = np.random.default_rng(SEED * 1000 + ci)
        perturbed = [perturb_numbers(case.numbers, rng) for _ in range(N_PERTURB)]
        good = []
        seen = set()
        for text, finished in cands:
            if not finished or text in seen:
                continue
            seen.add(text)
            try:
                model = parse(text, case.numbers)
            except FormulationError:
                continue
            n_valid += 1
            opt = solve(model)
            if opt is None or not rel_close(opt, case.label, 1e-4) or not typed_literals_ok(text):
                continue
            n_value += 1
            vec = []
            for pn in perturbed:
                try:
                    vec.append(solve(parse(text, pn)))
                except FormulationError:
                    vec.append(None)
            if any(v is None for v in vec):
                continue
            good.append((text, tuple(round(v, 6) for v in vec)))
        if good:
            clusters = {}
            for text, vec in good:
                clusters.setdefault(vec, []).append(text)
            best = max(clusters.values(), key=len)
            accepted[ci] = best[0]
    n_progs = sum(len(c) for c in samples)
    log(f"[{tag}] samples {n_progs}; parseable {n_valid}; optimum matches {n_value}; cases accepted "
        f"{len(accepted)}/{len(cases)}")
    return accepted


def n_variables(text, numbers):
    return len(parse(text, numbers).variables)


# ================================================================== test-time selection among the model's own samples
def select_candidate(cands, numbers, logps):
    """Pick one of a case's own candidates (choosing among the model's outputs for the same case). Candidates that parse
    and have a finite optimum are eligible; among them pick the one whose optimum most of the other eligible candidates
    agree with (a vote among the model's own samples); ties go to the higher mean log-probability. No rule inspects or
    edits the text. If nothing is eligible the greedy output (index 0) is submitted verbatim."""
    infos = []
    for i, text in enumerate(cands):
        a = analyze(text, numbers)
        if a["parsed"] and a["optimum"] is not None:
            infos.append((i, a["optimum"]))
    if not infos:
        return 0, "no_valid"
    best, best_key = None, None
    for i, opt in infos:
        agree = sum(1 for j, o2 in infos if j != i and rel_close(opt, o2, 1e-4))
        key = (agree, logps[i], -i)
        if best_key is None or key > best_key:
            best, best_key = i, key
    return best, "ok"


def decode_cases(model, tok, cases):
    prompts = [make_prompt(c.problem, c.numbers) for c in cases]
    greedy = generate(model, tok, prompts, 1, 0.0, 1.0, MAXNEW_TEST, TEST_PROMPTS_PER_BATCH * 2, greedy=True, tag="test-greedy")
    samples = generate(model, tok, prompts, K_TEST, T_TEST, TOP_P, MAXNEW_TEST, TEST_PROMPTS_PER_BATCH, tag="test-sample")
    outputs, stats = [], {"no_valid": 0, "ok": 0}
    for c, prompt, g, s in zip(cases, prompts, greedy, samples):
        cands = [g[0][0]] + [t for t, _ in s]
        logps = mean_logprobs(model, tok, prompt, cands)
        i, status = select_candidate(cands, c.numbers, logps)
        stats[status] += 1
        outputs.append(cands[i])
    log(f"selection: {stats}")
    return outputs


# ================================================================== pipeline
def run_training(train_cases, tok, ei_rounds=None):
    """Stages 1-5. Returns the final fine-tuned model (fine-tuned on seeds + accepted self-generated targets)."""
    ei_rounds = EI_ROUNDS if ei_rounds is None else ei_rounds
    perm_rng = np.random.default_rng(SEED)
    seeds = [c for c in train_cases if c.seed_formulation]
    rest = [c for c in train_cases if not c.seed_formulation]
    log(f"train cases {len(train_cases)}: seeds {len(seeds)}, unlabelled-formulation {len(rest)}")
    seed_items = [(c, c.seed_formulation, VIEWS_SEED) for c in seeds]
    model = train_from_base(build_examples(seed_items, tok, perm_rng), EPOCHS_S1, tok, "S1 seeds")
    accepted = {}
    epochs_by_round = [EPOCHS_S3, EPOCHS_S5]
    for rnd in range(ei_rounds):
        todo = [i for i in range(len(rest)) if i not in accepted]
        prompts = [make_prompt(rest[i].problem, rest[i].numbers) for i in todo]
        samples = generate(model, tok, prompts, K_EI, T_EI, TOP_P, MAXNEW_EI, GEN_PROMPTS_PER_BATCH, tag=f"EI{rnd + 1}")
        acc = accept_targets([rest[i] for i in todo], samples, f"EI{rnd + 1}")
        for j, text in acc.items():
            accepted[todo[j]] = text
        log(f"[EI{rnd + 1}] accepted so far {len(accepted)}/{len(rest)}")
        del model
        torch.cuda.empty_cache()
        items = list(seed_items)
        for i, text in accepted.items():
            nv = n_variables(text, rest[i].numbers)
            items.append((rest[i], text, VIEWS_EI_LARGE if nv >= LARGE_VARS else VIEWS_EI))
        model = train_from_base(build_examples(items, tok, perm_rng), epochs_by_round[rnd], tok,
                                f"S{3 + 2 * rnd} seeds+accepted")
    return model


def validate_submission(sub, sample_path):
    sample = pd.read_csv(sample_path, keep_default_na=False)
    assert list(sub.columns) == list(sample.columns), f"columns {list(sub.columns)} != {list(sample.columns)}"
    assert len(sub) == len(sample), f"rows {len(sub)} != {len(sample)}"
    assert sub["case_id"].astype(str).tolist() == sample["case_id"].astype(str).tolist(), "id mismatch/order"
    assert sub["case_id"].is_unique, "duplicate ids"
    assert (sub["formulation"].astype(str).str.strip().str.len() > 0).all(), "empty formulation"


def oracle_check_seeds(train_cases):
    """Sanity check of the parser/solver against the given labels: every seed must reproduce its optimal_value."""
    seeds = [c for c in train_cases if c.seed_formulation]
    bad = [c.case_id for c in seeds if not matches_optimum(c.seed_formulation, c.numbers, c.label, 1e-4)]
    log(f"seed oracle: {len(seeds) - len(bad)}/{len(seeds)} seed formulations reproduce their optimal value")
    assert len(bad) <= 0.02 * len(seeds), f"parser/solver disagree with labels on {len(bad)} seeds"


def main():
    seed_everything()
    train_cases, test_cases = load_cases(PUBLIC_DIR)
    sample = pd.read_csv(PUBLIC_DIR / "sample_submission.csv", keep_default_na=False)
    assert set(sample.case_id) == {c.case_id for c in test_cases}, "test ids differ from sample_submission"
    oracle_check_seeds(train_cases)
    tok = load_tokenizer()
    model = run_training(train_cases, tok)
    by_id = dict(zip([c.case_id for c in test_cases], decode_cases(model, tok, test_cases)))
    sub = pd.DataFrame({"case_id": sample.case_id, "formulation": [by_id[i] for i in sample.case_id]})
    validate_submission(sub, PUBLIC_DIR / "sample_submission.csv")
    sub.to_csv(SUBMISSION_OUT, index=False)
    log(f"wrote {SUBMISSION_OUT} shape={sub.shape}")


if __name__ == "__main__":
    main()
