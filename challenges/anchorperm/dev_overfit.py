import sys, torch, numpy as np
sys.path.insert(0, "."); import ap_model as M
from dev_common import va, fit, scale
torch.set_num_threads(2)
sub = fit[:700]
for ep in [10, 20, 40, 80, 160]:
    model = M.train_model(fit, epochs=ep, gain_max=1.6, log=lambda *a: None)
    tr_acc = M.evaluate(model, sub)[0].mean(); va_acc = M.evaluate(model, va)[0].mean()
    st = M.evaluate(model, scale(va, 1.5, 0.3, 2))[0].mean()
    print(f"epochs {ep:3d} train(acc, given anchors) {tr_acc:.4f} val {va_acc:.4f} gap {tr_acc - va_acc:+.4f} val-stressed {st:.4f}", flush=True)
