"""Dev harness (NOT part of the submission): run the cohort CV of solution.py with patched constants and save OOF logits.
Usage: python3 dev_run.py EXP_NAME KEY=VALUE ...   e.g. MODEL_NAME=google/electra-small-discriminator DEVICE=cpu N_FOLDS=3 MAX_EPOCHS=4 BATCH_SIZE=32
Saves reports/oof/<exp>.npz (OOF logits per epoch + labels + temperature) and prints the exact-metric CV per epoch."""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
args = sys.argv[1:]
exp, overrides = args[0], dict(a.split("=", 1) for a in args[1:])
sys.argv = [str(HERE / "solution.py"), str(HERE / "dataset" / "public"), str(HERE / "working" / "dev_submission.csv")]
spec = importlib.util.spec_from_file_location("solution_mod", HERE / "solution.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for k, v in overrides.items():
    cur = getattr(mod, k)
    setattr(mod, k, type(cur)(v) if not isinstance(cur, str) else v)
mod.seed_everything(mod.SEED)
tr = mod.load_train()
tok = mod.make_tok()
roles, ridx, examples, ex_index = mod.prepare(tr)
t = time.time()
L, cv, temps = mod.cross_validate(tr, roles, ridx, examples, ex_index, tok)
out = HERE / "reports" / "oof"
out.mkdir(parents=True, exist_ok=True)
np.savez(out / f"{exp}.npz", **{f"L{ep}": v for ep, v in L.items()}, Y=np.stack([e["y"] for e in examples]), M=np.stack([e["m"] for e in examples]))
print(json.dumps(dict(exp=exp, overrides=overrides, cv=cv, temps=temps, wall_s=round(time.time() - t))))
