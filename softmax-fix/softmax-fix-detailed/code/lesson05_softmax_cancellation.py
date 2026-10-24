"""
Lesson 05: Bug 2, Softmax Cancellation (knowledge distillation)
===============================================================
Teacher logits zT and student logits zS = zT + small perturbation.
True gradient of soft-target cross-entropy w.r.t. zS is q - p, which -> 0 as zS -> zT.
We compare what float32 code computes against a float64 reference.

Run all:  python lesson05_softmax_cancellation.py
"""
import numpy as np
from _figs import plt, save, HAVE_MPL
from softmax_fix import (std_softmax, std_soft_target_grad, std_soft_target_grad_framework_style,
                         fixed_soft_target_grad)

np.set_printoptions(precision=4, linewidth=120)
sci = {"float_kind": lambda v: f"{v:+.3e}"}

# %% 1. The paper's three-class example ---------------------------------------------
zT = np.array([[2.0, 1.0, 0.0]], dtype=np.float32)
zS = zT + np.array([[-1e-7, 1e-7, 0.0]], dtype=np.float32)
ref = std_soft_target_grad(zS.astype(np.float64), zT.astype(np.float64))
print("fp64 reference q - p :", np.array2string(ref, formatter=sci))
print("standard fp32        :", np.array2string(std_soft_target_grad(zS, zT), formatter=sci))
print("fixed fp32           :", np.array2string(fixed_soft_target_grad(zS, zT), formatter=sci))
print("Standard: wrong magnitudes AND a zero where the truth is +4.5e-9; fixed matches fp64.")

# %% 2. Why subtracting two rounded probabilities is lossy ----------------------------
p32 = std_softmax(zT)[0]
q32 = std_softmax(zS)[0]
print("\np (fp32):", p32, "\nq (fp32):", q32)
print("Many of these digits are identical.  Their difference keeps only the last, noisy digit.")
print("p == q elementwise?", p32 == q32, "  (so q - p is 0 or one-ulp noise)")

# %% 3. Exact zero requirement: zS == zT ------------------------------------------------
rng = np.random.default_rng(1)
worst_std, worst_fix = 0.0, 0.0
for _ in range(200):
    z = (3.0 * rng.standard_normal((1, 10))).astype(np.float32)
    worst_std = max(worst_std, float(np.max(np.abs(std_soft_target_grad_framework_style(z, z)))))
    worst_fix = max(worst_fix, float(np.max(np.abs(fixed_soft_target_grad(z, z)))))
print(f"\nWorst |gradient| over 200 random teachers when student == teacher (fp32):")
print(f"  standard (framework style: p from softmax, q from exp(log_softmax)): {worst_std:.3e}")
print(f"  fixed                                                              : {worst_fix:.3e}")
print("A correct implementation must return exactly 0 here.  Only the fixed one does.")

# %% 4. Sweep the distance between student and teacher ------------------------------------
K = 10
zT = (2.0 * rng.standard_normal((1, K))).astype(np.float32)
direction = rng.standard_normal((1, K)).astype(np.float32)
# Stop at 1e-8: for logits of size ~2 to 6, fp32 spacing is ~2e-7 to 5e-7, so below
# about 1e-8 the perturbation is rounded away and zS == zT exactly (see cell 3 for that case).
scales = np.logspace(-1, -8, 29)
rows = []
for s in scales:
    zS = (zT + np.float32(s) * direction).astype(np.float32)
    # float64 reference from the exact float32 values actually used
    ref = std_soft_target_grad(zS.astype(np.float64), zT.astype(np.float64))
    g_std = std_soft_target_grad_framework_style(zS, zT).astype(np.float64)
    g_fix = fixed_soft_target_grad(zS, zT).astype(np.float64)
    rows.append((s, np.linalg.norm(ref), np.linalg.norm(g_std), np.linalg.norm(g_fix),
                 np.linalg.norm(g_std - ref) / np.linalg.norm(ref),
                 np.linalg.norm(g_fix - ref) / np.linalg.norm(ref)))
rows = np.array(rows)
print("\n scale    ||true||    ||standard||  ||fixed||   rel.err standard   rel.err fixed")
for r in rows[::4]:
    print(f"{r[0]:.1e}  {r[1]:.3e}   {r[2]:.3e}    {r[3]:.3e}   {r[4]:.3e}        {r[5]:.3e}")
print("\nThe standard gradient stops shrinking near 1e-8 to 1e-7 (a noise floor); the fixed one")
print("keeps following the true gradient, and returns exactly 0 once zS and zT are identical.")
zS_same = (zT + np.float32(1e-9) * direction).astype(np.float32)
print("\nAt scale 1e-9 the perturbation is below fp32 spacing, so zS == zT is", bool(np.all(zS_same == zT)))
print("  standard (framework style) norm:", f"{np.linalg.norm(std_soft_target_grad_framework_style(zS_same, zT)):.3e}",
      " <- should be exactly 0")
print("  fixed norm                     :", f"{np.linalg.norm(fixed_soft_target_grad(zS_same, zT)):.3e}")

# %% 5. Figure -----------------------------------------------------------------------------
if HAVE_MPL:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].loglog(rows[:, 0], rows[:, 1], color="black", lw=3, alpha=0.5, label="true (float64)")
    ax[0].loglog(rows[:, 0], rows[:, 2], color="tab:orange", marker="o", ms=3, label="standard fp32")
    ax[0].loglog(rows[:, 0], rows[:, 3], color="tab:blue", marker="s", ms=3, label="fixed fp32")
    ax[0].invert_xaxis()
    ax[0].set_xlabel("size of perturbation ||zS - zT|| scale (smaller to the right)")
    ax[0].set_ylabel("||gradient||"); ax[0].set_title("Gradient norm as student approaches teacher")
    ax[0].legend()
    ax[1].loglog(rows[:, 0], rows[:, 4], color="tab:orange", marker="o", ms=3, label="standard fp32")
    ax[1].loglog(rows[:, 0], np.maximum(rows[:, 5], 1e-9), color="tab:blue", marker="s", ms=3, label="fixed fp32")
    ax[1].invert_xaxis()
    ax[1].set_xlabel("perturbation scale (smaller to the right)")
    ax[1].set_ylabel("relative error vs fp64"); ax[1].set_title("Relative error of the gradient")
    ax[1].legend()
    save(fig, "lesson05_cancellation.png")

# %% 6. Why Adam makes this matter (preview of lesson 06) -------------------------------------
eps = 1e-8
print("\nAdam turns any gradient g with |g| >> eps into a step of about the learning rate.")
for g in [1e-3, 1e-6, 1e-8, 1e-10]:
    print(f"  |g| = {g:.0e}  ->  normalized step |g|/(|g|+eps) = {g / (g + eps):.3f}")
print("A 'noise' gradient of 6e-8 (fp32 residual) is therefore a real step.  Next: lesson 06.")

# %% Exercises --------------------------------------------------------------------------------------
print("\nEXERCISES")
print("1. Verify by hand that q/p = exp(d_k) / sum_i p_i exp(d_i) with d = delta - sum(p*delta).")
print("2. Why must the argument of log1p be >= 0?  (Hint: Jensen or expm1(x) >= x.)")
print("3. Convert the inputs to float16 and compare the noise floor.")
