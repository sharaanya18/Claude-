"""Solver for <CHALLENGE NAME>.

Reads <public_dir>, trains real models inside this script, writes <submission_out>.

Compliance notes (keep this block accurate; reviewers read it):
  * every statistic, vocabulary, scaler and threshold is fit on TRAIN rows only, then applied to test;
  * each test row is predicted from its own inputs plus the frozen model (no whole-test statistics);
  * fixed work plan: epochs, folds, trials and counts are constants; wall-clock time is logged, never branched on;
  * all randomness is seeded; pretrained weights (if any) come from the HF/timm hub with a pinned revision.

Challenge requirements map (the Prompt Compliance check reads the code against the challenge text; make each explicit
requirement visible here, one line each, and point to where it is satisfied):
  * hardware / runtime stated by the challenge: TODO (e.g. "CPU only, 10 cores, 90 min -> device fixed to cpu, ~25 min")
  * required methods (e.g. fine-tuning, from-scratch, trained ranker): TODO
  * prohibited methods / models / data (copy each ban): TODO -> how this script avoids it
  * output grammar: TODO

Template rules (delete this paragraph when you specialise the file):
  1. Fill TODO blocks only; do not add time-based, hardware-based or environment-based branching.
  2. Write the submission once, at the end, validated; any failure must raise (no placeholder or fallback output).
  3. Run `python3 .claude/scripts/compliance_scan.py solution.py` after every edit.
"""
import os
import random
import sys
import time
from pathlib import Path

PUBLIC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./dataset/public")
SUBMISSION_OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./working/submission.csv")
WORK_DIR = SUBMISSION_OUT.parent
WORK_DIR.mkdir(parents=True, exist_ok=True)

os.environ["PYTHONHASHSEED"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_HOME"] = str(WORK_DIR / "hf_cache")

import numpy as np
import pandas as pd

# ---- fixed plan: all values are constants chosen offline from CV and profiling --------------------
SEED = 42
N_FOLDS = 5          # grouped folds mirroring the hidden split (see reports/split_audit.md)
N_REPEATS = 1        # split repeats used for reporting only
EPOCHS = 4           # fixed; never derived from the clock
BATCH_SIZE = 32
T0 = time.time()     # LOGGING ONLY: never used in a condition, loop bound or library argument


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    # TODO(torch): import torch; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    #   torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    #   torch.use_deterministic_algorithms(True, warn_only=True)


# ---- exact metric (copy the description's formula; keep it a pure function) -----------------------
def metric(y_true, y_pred):
    """TODO: implement the official metric exactly. Return a float where higher is better."""
    raise NotImplementedError


def metric_self_test():
    """Hard-coded extreme cases. Fill with values worked out by hand from the description."""
    # perfect prediction -> best score; reversed/all-wrong -> worst score; trivial constant -> baseline
    # assert abs(metric(y_perfect, y_perfect) - 1.0) < 1e-9
    pass


# ---- submission writing --------------------------------------------------------------------------
def validate_submission(sub, sample_path):
    sample = pd.read_csv(sample_path, keep_default_na=False)
    assert list(sub.columns) == list(sample.columns), f"columns {list(sub.columns)} != {list(sample.columns)}"
    assert len(sub) == len(sample), f"rows {len(sub)} != {len(sample)}"
    id_col = sample.columns[0]
    assert sub[id_col].astype(str).tolist() == sample[id_col].astype(str).tolist(), "id mismatch/order"
    for c in sample.columns[1:]:
        s = sub[c]
        assert not s.isna().any(), f"NaN in {c}"
        if pd.api.types.is_numeric_dtype(s):
            assert np.isfinite(s.to_numpy(dtype=float)).all(), f"non-finite in {c}"
        else:
            assert (s.astype(str).str.len() > 0).all(), f"empty string in {c}"


def write_submission(sub, sample_path):
    validate_submission(sub, sample_path)
    tmp = SUBMISSION_OUT.with_suffix(".tmp")
    sub.to_csv(tmp, index=False)
    os.replace(tmp, SUBMISSION_OUT)     # atomic: a crash never leaves a half-written file


def main():
    seed_everything()
    metric_self_test()
    train = pd.read_csv(PUBLIC_DIR / "train.csv")
    test = pd.read_csv(PUBLIC_DIR / "test.csv")
    sample_path = PUBLIC_DIR / "sample_submission.csv"
    sample = pd.read_csv(sample_path, keep_default_na=False)
    log(f"train {train.shape} test {test.shape}")

    # 1) TODO: derive groups from the data (union-find over shared keys / near-duplicates), build
    #    group-aware folds that mirror the hidden split, then train models per fold.
    # 2) TODO: out-of-fold predictions -> exact metric -> report mean +- std (and per-slice scores).
    # 3) TODO: refit on 100% of train with fixed counts (e.g. mean best epoch of the CV runs),
    #    predict each test row independently, decode with the metric-aware rule fitted on OOF only.
    sub = sample.copy()
    # sub[<target column>] = final_test_predictions
    write_submission(sub, sample_path)
    log(f"wrote {SUBMISSION_OUT} shape={sub.shape}")


if __name__ == "__main__":
    main()
