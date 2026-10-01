"""Dev harness (NOT part of the submission): hold out whole scenario groups of the training set, run the shipped training +
decode pipeline on the rest, and score the held-out cases with the local metric.
Usage: python3 dev_run.py EXP KEY=VALUE ...   (KEY: solution.py constants, plus FOLD=0..4, EI=<rounds>, NHOLD=<max held-out rows>)"""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
args = sys.argv[1:]
exp, overrides = args[0], dict(a.split("=", 1) for a in args[1:])
sys.argv = [str(HERE / "solution.py"), str(HERE / "dataset" / "public"), str(HERE / "working" / "dev_submission.csv")]
spec = importlib.util.spec_from_file_location("solution_mod", HERE / "solution.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
sys.modules["formulation"] = S          # metric.py imports the grammar tool under this name; solution.py carries the same code
sys.path.insert(0, str(HERE))
import metric as M

FOLD = int(overrides.pop("FOLD", 0))
EI = int(overrides.pop("EI", S.EI_ROUNDS))
NHOLD = int(overrides.pop("NHOLD", 100000))
NSEED = int(overrides.pop("NSEED", 100000))     # smoke-test subsetting of the training side only
NREST = int(overrides.pop("NREST", 100000))
for k, v in overrides.items():
    cur = getattr(S, k)
    setattr(S, k, type(cur)(v) if not isinstance(cur, str) else v)
S.seed_everything(S.SEED)
train, _ = S.load_cases(S.Path(HERE / "dataset" / "public"))


def groups(cases):
    """Scenario groups: same opening sentence or same number set are one group (union-find)."""
    par = list(range(len(cases)))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    first, vals = {}, {}
    for i, c in enumerate(cases):
        keys = [("s", c.problem.split(".")[0].strip().lower()), ("v", tuple(sorted(round(float(v), 9) for _, v in c.numbers)))]
        for k in keys:
            d = first if k[0] == "s" else vals
            if k in d:
                par[find(i)] = find(d[k])
            else:
                d[k] = i
    return [find(i) for i in range(len(cases))]


g = groups(train)
order = sorted(set(g), key=lambda x: (-g.count(x), x))
fold_of, load = {}, [0] * 5
for gid in order:                                   # greedy balance of seed rows then rows
    sz = g.count(gid)
    f = min(range(5), key=lambda j: load[j])
    fold_of[gid] = f
    load[f] += sz
hold_idx = [i for i, gid in enumerate(g) if fold_of[gid] == FOLD][:NHOLD]
hold = [train[i] for i in hold_idx]
tr = [c for i, c in enumerate(train) if fold_of[g[i]] != FOLD]
_s = [c for c in tr if c.seed_formulation][:NSEED]
_r = [c for c in tr if not c.seed_formulation][:NREST]
tr = _s + _r
S.log(f"fold {FOLD}: train {len(tr)} (seeds {sum(bool(c.seed_formulation) for c in tr)}), held out {len(hold)} "
      f"(seeds {sum(bool(c.seed_formulation) for c in hold)})")
S.oracle_check_seeds(train)
tok = S.load_tokenizer()
model = S.run_training(tr, tok, ei_rounds=EI)
t = time.time()
prompts = [S.make_prompt(c.problem, c.numbers) for c in hold]
greedy = S.generate(model, tok, prompts, 1, 0.0, 1.0, S.MAXNEW_TEST, S.TEST_PROMPTS_PER_BATCH * 2, greedy=True, tag="hold-greedy")
samples = S.generate(model, tok, prompts, S.K_TEST, S.T_TEST, S.TOP_P, S.MAXNEW_TEST, S.TEST_PROMPTS_PER_BATCH, tag="hold-sample")
res = []
for c, p, gr, sm in zip(hold, prompts, greedy, samples):
    cands = [gr[0][0]] + [x for x, _ in sm]
    logps = S.mean_logprobs(model, tok, p, cands)
    i, status = S.select_candidate(cands, c.numbers, logps)
    v = [S.matches_optimum(x, c.numbers, c.label, 1e-4) for x in cands]
    row = dict(case_id=c.case_id, seed=bool(c.seed_formulation), status=status, greedy_v=v[0], sel_v=v[i], pass_k=any(v),
               n_valid=sum(S.analyze(x, c.numbers)["optimum"] is not None for x in cands), chosen=cands[i])
    if c.seed_formulation:
        sc = M.case_score(c.seed_formulation, cands[i], c.numbers)
        sg = M.case_score(c.seed_formulation, cands[0], c.numbers)
        row.update(score=sc["score"], v=sc["value"], cf=sc["cf"], st=sc["structure"], greedy_score=sg["score"],
                   n_vars=len(S.parse(c.seed_formulation, c.numbers).variables))
    res.append(row)
rows = [r for r in res]
seed_rows = [r for r in res if r["seed"]]
big = [r for r in seed_rows if r["n_vars"] >= 6]
summ = dict(exp=exp, fold=FOLD, ei=EI, overrides=overrides, n_hold=len(res),
            V_greedy=float(np.mean([r["greedy_v"] for r in rows])), V_selected=float(np.mean([r["sel_v"] for r in rows])),
            pass_at_K=float(np.mean([r["pass_k"] for r in rows])), no_valid=float(np.mean([r["status"] == "no_valid" for r in rows])),
            seed_n=len(seed_rows), seed_score=float(np.mean([r["score"] for r in seed_rows])) if seed_rows else None,
            seed_greedy_score=float(np.mean([r["greedy_score"] for r in seed_rows])) if seed_rows else None,
            seed_terms={k: float(np.mean([r[k] for r in seed_rows])) for k in ("v", "cf", "st")} if seed_rows else None,
            big_n=len(big), big_score=float(np.mean([r["score"] for r in big])) if big else None, wall_s=round(time.time() - S.T0))
print(json.dumps(summ))
out = HERE / "reports" / "dev"
out.mkdir(parents=True, exist_ok=True)
(out / f"{exp}.json").write_text(json.dumps(dict(summary=summ, rows=res), indent=1))
