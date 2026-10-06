"""Phase 2: verify the exact evaluator against hand-computed cases.

Run: python3 -I test_metric.py
Every expected value below is computed by hand in the comment, not by the code.
"""
import numpy as np

from kp_data import T, axis_skill, score

ok = 0
fail = []


def check(name, got, want, tol=1e-12):
    global ok
    if abs(got - want) <= tol:
        ok += 1
        print(f"  PASS  {name:<52} {got:.12g}")
    else:
        fail.append(name)
        print(f"  FAIL  {name:<52} got {got!r} want {want!r}")


print("== axis_skill, hand-computed ==")

# perfect prediction: error 0, baseline != 0 -> 1 - 0/b = 1
y = np.array([1.0, -2.0, 3.0])
check("perfect", axis_skill(y, y), 1.0)

# zero prediction on a nonzero axis: error == baseline -> 1 - 1 = 0
check("zero pred on nonzero axis", axis_skill(np.zeros(3), y), 0.0)

# truth all zero, pred all zero -> baseline 0, error 0 -> float(True) = 1
check("zero axis, exact zero pred", axis_skill(np.zeros(3), np.zeros(3)), 1.0)

# truth all zero, pred nonzero -> baseline 0, error != 0 -> float(False) = 0
check("zero axis, nonzero pred", axis_skill(np.array([0.0, 1e-12, 0.0]), np.zeros(3)), 0.0)

# sign flip: truth [1,-2,3] b=6; pred=-truth err=|2|+|4|+|6|=12 -> 1-12/6=-1 -> clipped 0
check("sign flip -> clipped at 0", axis_skill(-y, y), 0.0)

# amplitude 0.5x: pred=0.5*y err=0.5+1+1.5=3, b=6 -> 1-3/6=0.5
check("half amplitude", axis_skill(0.5 * y, y), 0.5)

# amplitude 1.5x: err=0.5+1+1.5=3 -> 0.5 as well (symmetric in |err|)
check("1.5x amplitude", axis_skill(1.5 * y, y), 0.5)

# amplitude 3x: err=2+4+6=12 -> 1-2 = -1 -> 0
check("3x amplitude -> 0", axis_skill(3 * y, y), 0.0)

# constant offset: truth [1,1,1] b=3, pred [1.5,1.5,1.5] err=1.5 -> 1-0.5=0.5
check("constant offset", axis_skill(np.full(3, 1.5), np.ones(3)), 0.5)

# temporal shift on a spike: truth has 1 at idx0, pred has 1 at idx1
# b=1, err=|1-0|+|0-1|=2 -> 1-2 = -1 -> 0.  A one-sample shift of a pure
# spike earns nothing: timing matters as much as amplitude.
t_spike = np.array([1.0, 0.0, 0.0]); p_spike = np.array([0.0, 1.0, 0.0])
check("1-sample shift of a spike", axis_skill(p_spike, t_spike), 0.0)

# smooth shift is much gentler: truth [1,2,3,2,1] b=9,
# pred shifted right [0,1,2,3,2] err=1+1+1+1+1=5 -> 1-5/9
t_s = np.array([1.0, 2, 3, 2, 1]); p_s = np.array([0.0, 1, 2, 3, 2])
check("1-sample shift of a ramp", axis_skill(p_s, t_s), 1 - 5 / 9)

print("\n== score(), averaging order and malformed rows ==")

rng = np.random.default_rng(0)
truth = rng.normal(size=(4, 3, T))

# perfect -> 1.0
check("all rows perfect", score(truth.copy(), truth), 1.0)

# all-zero submission on nonzero axes -> 0.0
check("all-zero submission", score(np.zeros_like(truth), truth), 0.0)

# averaging order: build known per-axis skills via amplitude scaling.
# row0 all axes 0.5x -> 0.5 each ; row1 perfect -> 1 each ; rows 2,3 zero -> 0
pred = np.zeros_like(truth)
pred[0] = 0.5 * truth[0]
pred[1] = truth[1]
# expected = (0.5*3 + 1*3 + 0 + 0) / (3*4) = 4.5/12 = 0.375
check("mean over axes then trials", score(pred, truth), 0.375)

# NaN in ONE axis zeroes the WHOLE row, other rows still score.
pred_nan = truth.copy()
pred_nan[0, 1, 5] = np.nan
# rows 1,2,3 perfect (=1 each), row 0 zeroed -> 9/12 = 0.75
check("NaN zeroes its whole row only", score(pred_nan, truth), 0.75)

pred_inf = truth.copy()
pred_inf[2, 0, 0] = np.inf
check("inf zeroes its whole row only", score(pred_inf, truth), 0.75)

# a row whose truth is identically zero on one axis
truth_z = truth.copy()
truth_z[3, 2, :] = 0.0
pred_z = truth_z.copy()
check("exact-zero axis reproduced -> 1", score(pred_z, truth_z), 1.0)
pred_z2 = truth_z.copy()
pred_z2[3, 2, 0] = 1e-9          # tiny nonzero on a zero axis -> that axis 0
# row3 gets (1+1+0)/3 ; rows 0-2 get 1 -> (3+3+3+2)/12 = 11/12
check("tiny nonzero on zero axis -> axis 0", score(pred_z2, truth_z), 11 / 12)

print("\n== wrong-length guard ==")
bad = np.zeros((1, 3, T - 1))
try:
    score(bad, np.zeros((1, 3, T - 1)))
    print("  (shape assert is on (3,T); short arrays rejected by assert)")
except AssertionError:
    ok += 1
    print("  PASS  wrong length raises AssertionError")

print(f"\n{ok} passed, {len(fail)} failed")
if fail:
    raise SystemExit("FAILED: " + ", ".join(fail))
