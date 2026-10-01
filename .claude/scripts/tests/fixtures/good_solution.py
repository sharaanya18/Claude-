"""Good fixture. Mentions pip install, github.com and time in the docstring only."""
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PUBLIC_DIR = Path(sys.argv[1])
SUBMISSION_OUT = Path(sys.argv[2])
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:.0f}s] {msg}")


def main():
    random.seed(0)
    np.random.seed(0)
    train = pd.read_csv(PUBLIC_DIR / "train.csv")
    test = pd.read_csv(PUBLIC_DIR / "test.csv")
    model = LogisticRegression(max_iter=200, random_state=0)
    model.fit(train[["x"]], train["y"])
    pred = model.predict_proba(test[["x"]])[:, 1]
    out = pd.DataFrame({"id": test["id"], "y": pred})
    SUBMISSION_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SUBMISSION_OUT, index=False)
    log("done")


if __name__ == "__main__":
    main()
