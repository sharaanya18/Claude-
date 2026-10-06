#!/usr/bin/env python3
"""Run a script on a Kaggle GPU notebook and fetch its outputs (needs the `kaggle` CLI >= 1.8 and KAGGLE_API_TOKEN in the environment).

  python3 kaggle_gpu_run.py data   <challenge_dir>                     upload dataset/public as a PRIVATE Kaggle dataset
  python3 kaggle_gpu_run.py run    <challenge_dir> <script> [--dataset=owner/slug] [ARGS...]
                                                                       push <script> (solution.py or dev_run.py) as a GPU kernel and wait.
                                                                       --dataset points at an EXISTING Kaggle dataset (e.g. one you uploaded);
                                                                       the kernel finds train.csv anywhere under /kaggle/input
  (run also takes --name=<kernel slug> and repeated --set=KEY=VALUE overrides of module-level constants of the embedded solution.py)
  python3 kaggle_gpu_run.py fetch  <challenge_dir> [--name=<kernel slug>]                     download the kernel's /kaggle/working files into <challenge_dir>/kaggle_out/
  python3 kaggle_gpu_run.py dry    <challenge_dir> <script>            write the metadata files only (no network)

Notes: (1) Kaggle GPUs (T4/P100) are not the platform's A10G: use them for experiments and CV; re-profile runtime and re-check determinism on the
real runtime. (2) Uploading challenge data to a third-party service may conflict with the platform's data terms: confirm before using `data`.
(3) The token is read by the kaggle CLI from KAGGLE_API_TOKEN; this script never prints it. (4) Kaggle notebooks may need phone verification for
internet access (HF weights); the kernel enables internet.
The kernel runs `python <script> /kaggle/input/<dataset>/public /kaggle/working/submission.csv` for solution.py, or `python dev_run.py ...` with its
dataset path patched through the DATA_ROOT env var that this runner sets in a tiny wrapper (dev harnesses only; never in solution.py).
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def sh(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, **kw)


def username():
    r = sh(["kaggle", "config", "view"])
    for line in r.stdout.splitlines():
        if "username" in line.lower():
            return line.split(":")[-1].strip()
    sys.exit("could not read the Kaggle username (is KAGGLE_API_TOKEN set and the CLI >= 1.8 installed?)\n" + r.stdout + r.stderr)


def slug(ch):
    return "".join(c if c.isalnum() else "-" for c in Path(ch).resolve().name.lower()).strip("-")[:40]


def write_meta(ch, script, args, user, dataset=None, sets=(), name=None):
    """dataset: 'owner/slug' of an EXISTING Kaggle dataset (e.g. one the user uploaded); default is the private one created by `data`."""
    ch = Path(ch).resolve()
    ds_full = dataset or f"{user}/{slug(ch)}-data"
    ds, kn = ds_full.split("/")[-1], (name or f"{slug(ch)}-run")
    kdir = ch / "kaggle_kernel"
    shutil.rmtree(kdir, ignore_errors=True)
    kdir.mkdir()
    # Kaggle script kernels upload only the code file, so the helper files are embedded as string literals and written out at run time.
    wrapper = kdir / "main.py"
    embedded = [script] + [e for e in ("solution.py", "metric.py") if (ch / e).exists() and e != script]
    pre = "import os, shutil, subprocess, sys, glob\n"
    for name in embedded:
        pre += f"open({name!r}, 'w', encoding='utf-8').write({(ch / name).read_text(encoding='utf-8')!r})\n"
    if sets:   # dev-only constant overrides applied to the embedded solution.py (module-level `KEY = value` lines)
        pre += "import re\nsrc = open('solution.py', encoding='utf-8').read()\n"
        for kv in sets:
            k, v = kv.split("=", 1)
            pre += f"src, n = re.subn(r'(?m)^{k} = .*$', {k + ' = ' + v!r}, src); assert n == 1, {k!r}\n"
        pre += "open('solution.py', 'w', encoding='utf-8').write(src)\n"
    find = ('hits = sorted(glob.glob("/kaggle/input/**/train.csv", recursive=True))\n'
            'assert hits, "train.csv not found under /kaggle/input"\nPUB = os.path.dirname(hits[0])\nprint("data dir:", PUB)\n')
    if script == "solution.py":
        cmd = 'subprocess.check_call([sys.executable, "solution.py", PUB, "/kaggle/working/submission.csv"])\n'
    else:
        cmd = ('os.makedirs("dataset", exist_ok=True)\n'
               'shutil.copytree(PUB, "dataset/public", dirs_exist_ok=True)\n'
               f'subprocess.check_call([sys.executable, "{script}"] + {args!r})\n')
    post = ""
    cmd = "try:\n" + "".join("    " + ln + "\n" for ln in cmd.strip().splitlines()) + (
        "finally:\n    shutil.rmtree('working/hf_cache', ignore_errors=True)\n    shutil.rmtree('/kaggle/working/hf_cache', ignore_errors=True)\n")
    wrapper.write_text(pre + find + cmd + post)
    (kdir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{kn}", "title": kn, "code_file": "main.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": os.environ.get("KAGGLE_ENABLE_GPU", "1") == "1", "enable_internet": True,
        "dataset_sources": [ds_full], "competition_sources": [], "kernel_sources": []}, indent=2))
    return kdir, f"{user}/{kn}"


def main(argv):
    if len(argv) < 3 or argv[1] not in {"data", "run", "fetch", "dry"}:
        print(__doc__)
        return 64
    cmd, ch = argv[1], Path(argv[2]).resolve()
    rest = argv[3:]
    dataset = next((a.split("=", 1)[1] for a in rest if a.startswith("--dataset=")), None)
    name = next((a.split("=", 1)[1] for a in rest if a.startswith("--name=")), None)
    sets = [a.split("=", 1)[1] if a.startswith("--set=") else None for a in rest]
    sets = [x for x in sets if x]
    rest = [a for a in rest if not a.startswith(("--dataset=", "--name=", "--set="))]
    if cmd == "dry":
        kdir, kid = write_meta(ch, rest[0], rest[1:], "USER", dataset, sets, name)
        print("wrote", kdir, "kernel id", kid)
        return 0
    user = username()
    if cmd == "data":
        d = ch / "kaggle_data"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
        shutil.copytree(ch / "dataset" / "public", d / "public")
        (d / "dataset-metadata.json").write_text(json.dumps({"title": f"{slug(ch)}-data", "id": f"{user}/{slug(ch)}-data",
                                                              "licenses": [{"name": "other"}], "isPrivate": True}))
        r = sh(["kaggle", "datasets", "create", "-p", str(d), "--dir-mode", "zip"])
        print(r.stdout + r.stderr)
        return r.returncode
    if cmd == "run":
        kdir, kid = write_meta(ch, rest[0], rest[1:], user, dataset, sets, name)
        r = sh(["kaggle", "kernels", "push", "-p", str(kdir)])
        print(r.stdout + r.stderr)
        if r.returncode:
            return r.returncode
        while True:                                   # polling interval is orchestration, not solution logic
            st = sh(["kaggle", "kernels", "status", kid]).stdout.strip()
            print(time.strftime("%H:%M:%S"), st)
            if any(w in st.lower() for w in ("complete", "error", "cancel")):
                break
            time.sleep(60)
        return 0 if "complete" in st.lower() else 1
    if cmd == "fetch":
        kid = f"{user}/{next((a.split('=', 1)[1] for a in argv[3:] if a.startswith('--name=')), slug(ch) + '-run')}"
        out = ch / "kaggle_out"
        out.mkdir(exist_ok=True)
        r = sh(["kaggle", "kernels", "output", kid, "-p", str(out)])
        print(r.stdout + r.stderr)
        return r.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv))
