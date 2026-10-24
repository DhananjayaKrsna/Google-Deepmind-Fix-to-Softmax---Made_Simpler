"""
test_softmax_fix.py
Plain-assert tests for softmax_fix.py.  No pytest needed:  python test_softmax_fix.py
Each test prints PASS so beginners can see what is being checked.
"""
import numpy as np
from softmax_fix import (std_softmax, std_log_softmax, std_cross_entropy,
                         fixed_log_softmax, fixed_softmax, fixed_cross_entropy,
                         std_soft_target_grad, fixed_soft_target_grad,
                         std_soft_target_grad_framework_style, logit_margin,
                         to_bf16, lse, softplus, CLIFFS)

results = []


def check(name, cond):
    results.append(cond)
    print(("PASS  " if cond else "FAIL  ") + name)
    assert cond, name


rng = np.random.default_rng(0)

# 1. softmax rows sum to 1 and are shift invariant
z = rng.normal(size=(5, 7)).astype(np.float64)
check("softmax rows sum to 1", np.allclose(std_softmax(z).sum(-1), 1.0))
check("softmax is shift invariant", np.allclose(std_softmax(z), std_softmax(z + 123.0)))

# 2. no overflow for huge logits
big = np.array([1000.0, 0.0, -1000.0], dtype=np.float32)
check("no overflow at z=1000 (standard)", np.all(np.isfinite(std_log_softmax(big))))
check("no overflow at z=1000 (fixed)", np.all(np.isfinite(fixed_log_softmax(big))))

# 3. fixed and standard agree when confidence is moderate (float64)
z = rng.normal(scale=2.0, size=(32, 10))
t = rng.integers(0, 10, size=32)
l_s, g_s = std_cross_entropy(z, t)
l_f, g_f, m = fixed_cross_entropy(z, t)
check("fixed == standard loss at moderate margins", np.allclose(l_s, l_f, rtol=1e-10, atol=1e-12))
check("fixed == standard gradient at moderate margins", np.allclose(g_s, g_f, rtol=1e-8, atol=1e-12))
check("fixed log_softmax == standard (moderate)", np.allclose(std_log_softmax(z), fixed_log_softmax(z), atol=1e-12))
check("fixed softmax == standard (moderate)", np.allclose(std_softmax(z), fixed_softmax(z), atol=1e-12))

# 4. gradient check against finite differences (float64, moderate margins)
def loss_only(zz, tt):
    return fixed_cross_entropy(zz, tt)[0]

z1 = rng.normal(size=(1, 6))
t1 = np.array([2])
_, g1, _ = fixed_cross_entropy(z1, t1)
num = np.zeros_like(z1)
h = 1e-6
for j in range(6):
    zp, zm = z1.copy(), z1.copy()
    zp[0, j] += h
    zm[0, j] -= h
    num[0, j] = (loss_only(zp, t1)[0] - loss_only(zm, t1)[0]) / (2 * h)
check("fixed gradient matches finite differences", np.allclose(g1, num, atol=1e-7))

# 5. Softmax Collapse: z = [9, -9, -9] in fp32
z = np.array([[9.0, -9.0, -9.0]], dtype=np.float32)
t = np.array([0])
l_s, g_s = std_cross_entropy(z, t)
l_f, g_f, m = fixed_cross_entropy(z, t)
check("margin is above the fp32 cliff", float(m[0]) > CLIFFS["fp32"])
check("standard loss is exactly 0 (bug)", float(l_s[0]) == 0.0)
check("standard target gradient is exactly 0 (bug)", float(g_s[0, 0]) == 0.0)
check("standard gradient does not sum to 0 (bug)", float(g_s.sum()) > 1e-9)
check("fixed target gradient is non-zero", float(g_f[0, 0]) < 0.0)
check("fixed gradient sums to exactly 0", float(g_f.sum()) == 0.0)
true_e = np.exp(-np.float64(m[0]))
check("fixed loss matches true e^-m (1%)", abs(float(l_f[0]) - true_e) / true_e < 0.01)

# 6. Softmax Cancellation: tiny perturbation in fp32 against an fp64 reference
zT = np.array([[2.0, 1.0, 0.0]], dtype=np.float32)
zS = zT + np.array([[-1e-7, 1e-7, 0.0]], dtype=np.float32)
ref = std_soft_target_grad(zS.astype(np.float64), zT.astype(np.float64))
err_std = np.max(np.abs(std_soft_target_grad(zS, zT) - ref))
err_fix = np.max(np.abs(fixed_soft_target_grad(zS, zT) - ref))
check("fixed soft-target error is >100x smaller", err_fix * 100 < err_std)
check("fixed soft-target gradient is exactly 0 at zS == zT", np.all(fixed_soft_target_grad(zT, zT) == 0.0))
check("fixed soft-target gradient sums to ~0", abs(float(fixed_soft_target_grad(zS, zT).sum())) < 1e-12)

# 7. forward/backward mismatch exists for standard, absent for fixed (float32, several draws)
mism = 0.0
for _ in range(20):
    zz = (3.0 * rng.standard_normal((1, 10))).astype(np.float32)
    mism = max(mism, float(np.max(np.abs(std_soft_target_grad_framework_style(zz, zz)))))
check("framework-style standard grad is non-zero at zS == zT", mism > 0.0)

# 8. helpers
check("softplus matches log1p(exp(x))", np.allclose(softplus(np.array([-5.0, 0.0, 5.0])), np.log1p(np.exp([-5.0, 0.0, 5.0]))))
check("lse matches naive on small values", np.allclose(lse(np.array([1.0, 2.0, 3.0])), np.log(np.exp([1.0, 2.0, 3.0]).sum())))
check("bf16 keeps 1.0", float(to_bf16(1.0)) == 1.0)
check("bf16 absorbs 2^-9 added to 1", float(to_bf16(np.float32(1.0) + np.float32(2.0 ** -9))) == 1.0)
check("bf16 keeps 2^-7 added to 1", float(to_bf16(np.float32(1.0) + np.float32(2.0 ** -7))) == 1.0078125)
check("logit_margin agrees with fixed_cross_entropy", np.allclose(logit_margin(z.astype(np.float64), t), m, rtol=1e-4))

print(f"\n{sum(results)}/{len(results)} checks passed")
