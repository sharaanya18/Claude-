import os, shutil, subprocess, sys
import glob
hits = sorted(glob.glob("/kaggle/input/**/train.csv", recursive=True))
assert hits, "train.csv not found under /kaggle/input"
PUB = os.path.dirname(hits[0])
print("data dir:", PUB)
SRC = os.path.dirname(os.path.abspath(__file__))
RUN = "/kaggle/working/run"
shutil.copytree(SRC, RUN, dirs_exist_ok=True)
os.chdir(RUN)
os.makedirs("dataset", exist_ok=True)
shutil.copytree(PUB, "dataset/public", dirs_exist_ok=True)
subprocess.check_call([sys.executable, "dev_run.py"] + ['cvlarge', 'MODEL_NAME=microsoft/deberta-v3-large', 'DEVICE=cuda', 'N_FOLDS=5', 'MAX_EPOCHS=4', 'BATCH_SIZE=16'])
for d in ('reports', 'working'):
    if os.path.isdir(d):
        shutil.copytree(d, '/kaggle/working/' + d, dirs_exist_ok=True)
