exec(open("/home/user/Claude-/challenges/anchorperm/reports/v2_scratch/exp_rowadapt.py").read().split("sets = {")[0])
import torch, ap_model as M
from dev_noise import noisy
torch.set_num_threads(4)
model = M.train_model(fit, epochs=40, gain_max=1.6, log=lambda *a: None)
def z(S): return (S - S.mean()) / (S.std() + 1e-9)
N = 400
sets = {"clean": va[:N], "noise0.5": noisy(va[:N], 0.5, 3), "prot0.7": amp(va[:N], 1.0, 8, 4, 0.7), "prot1.0": amp(va[:N], 1.0, 8, 4, 1.0), "coordmix0.5": coordmix(va[:N], 0.5, 5), "amp1.6+prot0.7": amp(va[:N], 1.6, 8, 6, 0.7)}
W = [0.0, 0.25, 0.5, 1.0]
for name, rows in sets.items():
    lg = M.predict_logits(model, rows).numpy(); res = {f"nn+glob{w}": [] for w in W}; res.update({f"nn+adapt{w}": [] for w in W}); res["adapt_only"] = []
    for l, r in zip(lg, rows):
        _, Sn = M.decode_row(l, r)
        _, Cg = acc_of(r, W1); a_, Ca = acc_of(r, adapt(r, 30, "em", em_iters=8)); res["adapt_only"].append(a_)
        for w in W:
            for tag, C in (("glob", Cg), ("adapt", Ca)):
                ri, ci = lsa(-(z(Sn) - w * z(C))); res[f"nn+{tag}{w}"].append(np.mean([r["ha"][c] == t for c, t in zip(ci, r["y"])]))
    print(name, {k: round(float(np.mean(v)), 4) for k, v in res.items()}, flush=True)
