"""
Lesson 04: Bug 1, Softmax Collapse
==================================
Setup: two classes, logits z = [m, 0], target class 0.  Then the logit margin is
exactly m, the true loss is log(1 + e^-m) and the true target gradient is
    g_t = -e^-m / (1 + e^-m).
We sweep m and compare what each precision computes.

Run all:  python lesson04_softmax_collapse.py
"""
import numpy as np
from _figs import plt, save, HAVE_MPL
from softmax_fix import (std_cross_entropy, fixed_cross_entropy, to_bf16, CLIFFS)

# %% 1. One example, step by step, in float32 -----------------------------------
z = np.array([9.0, -9.0, -9.0], dtype=np.float32)
shifted = z - z.max()
e = np.exp(shifted)
denom = e.sum()
q = e / denom
sci = {"float_kind": lambda v: f"{v:.3e}"}
print("shifted logits      :", shifted)
print("exponentials        :", np.array2string(e, formatter=sci))
print("denominator         :", repr(denom), "   <- exactly 1.0, the 3e-8 was absorbed")
print("q                   :", np.array2string(q, formatter=sci))
print("gradient q - onehot :", np.array2string(q - np.array([1, 0, 0], np.float32), formatter=sci))
print("sum of gradient     :", float((q - np.array([1, 0, 0], np.float32)).sum()), " <- should be 0")

# %% 2. Reference and four precisions --------------------------------------------
def true_gt(m):
    """Exact target gradient in float64 (accurate for m < ~700)."""
    em = np.exp(-np.float64(m))
    return -em / (1.0 + em)

def std_gt(m, dtype):
    z = np.array([[m, 0.0]], dtype=dtype)
    _, g = std_cross_entropy(z, np.array([0]))
    return float(g[0, 0])

def fix_gt(m, dtype):
    z = np.array([[m, 0.0]], dtype=dtype)
    _, g, _ = fixed_cross_entropy(z, np.array([0]))
    return float(g[0, 0])

def std_gt_bf16(m):
    """Standard algorithm with every operation rounded to bfloat16."""
    r = to_bf16
    e2 = r(np.exp(-np.float32(m)))        # exp(0 - m) for the wrong class
    s = r(np.float32(1.0) + e2)           # denominator 1 + e^-m  <-- absorbed
    q_t = r(np.float32(1.0) / s)
    return float(r(q_t - np.float32(1.0)))

def fix_gt_bf16(m):
    """Fixed algorithm with every operation rounded to bfloat16 (K = 2 case)."""
    r = to_bf16
    margin = r(np.float32(m))             # -logsumexp over the single wrong class
    loss = r(np.logaddexp(np.float32(0.0), -margin))  # softplus(-m)
    g_j = r(np.exp(r(-margin - loss)))    # exp(z_j - z_t - loss)
    return float(-g_j)

ms = np.arange(0.0, 45.0, 0.25)
curves = {
    "true (float64)": np.array([abs(true_gt(m)) for m in ms]),
    "standard bf16": np.array([abs(std_gt_bf16(m)) for m in ms]),
    "standard fp16": np.array([abs(std_gt(m, np.float16)) if m < 11 else 0.0 for m in ms]),
    "standard fp32": np.array([abs(std_gt(m, np.float32)) for m in ms]),
    "standard fp64": np.array([abs(std_gt(m, np.float64)) for m in ms]),
    "fixed bf16": np.array([abs(fix_gt_bf16(m)) for m in ms]),
    "fixed fp32": np.array([abs(fix_gt(m, np.float32)) for m in ms]),
}

# %% 3. Where does each standard implementation hit exactly zero? -----------------
print("\nFirst margin at which the STANDARD target gradient is exactly 0")
print("format   measured    formula (p+1) ln 2   paper")
for name, key, paper in [("bf16", "standard bf16", 5.5), ("fp32", "standard fp32", 16.6), ("fp64", "standard fp64", 36.7)]:
    fine = np.arange(0.0, 45.0, 0.01)
    if name == "bf16":
        vals = np.array([abs(std_gt_bf16(m)) for m in fine])
    elif name == "fp32":
        vals = np.array([abs(std_gt(m, np.float32)) for m in fine])
    else:
        vals = np.array([abs(std_gt(m, np.float64)) for m in fine])
    first_zero = fine[np.argmax(vals == 0.0)]
    print(f"{name}     {first_zero:6.2f}      {CLIFFS[name]:6.2f}             {paper}")
print("(The measured value is within a hundredth of the formula. Differences of order")
print(" 0.1 are possible because the exact tie-to-even point depends on rounding.)")

# %% 4. What the FIXED version returns at the same margins -------------------------
print("\nmargin   true |g_t|     standard fp32   fixed fp32     relative error (fixed)")
for m in [5.0, 10.0, 16.0, 17.0, 20.0, 30.0, 40.0, 60.0]:
    tr = abs(true_gt(m))
    st = abs(std_gt(m, np.float32))
    fx = abs(fix_gt(m, np.float32))
    print(f"{m:5.1f}    {tr:.4e}   {st:.4e}      {fx:.4e}   {abs(fx - tr) / tr:.1e}")

# %% 5. Gradient does not sum to zero in the standard version -----------------------
print("\nsum of gradient entries at m = 20 (should be 0):")
z = np.array([[20.0, 0.0]], dtype=np.float32)
_, gs = std_cross_entropy(z, np.array([0]))
_, gf, _ = fixed_cross_entropy(z, np.array([0]))
print("  standard:", gs, " sum =", float(gs.sum()))
print("  fixed   :", gf, " sum =", float(gf.sum()))

# %% 6. Figure ----------------------------------------------------------------------
if HAVE_MPL:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    floor = 1e-60
    style = {
        "true (float64)": dict(color="black", lw=2.5, alpha=0.5),
        "standard bf16": dict(color="tab:red", ls="--"),
        "standard fp32": dict(color="tab:orange", ls="--"),
        "standard fp64": dict(color="tab:brown", ls="--"),
        "fixed bf16": dict(color="tab:cyan"),
        "fixed fp32": dict(color="tab:blue"),
    }
    for k, st in style.items():
        ax[0].semilogy(ms, np.maximum(curves[k], floor), label=k, **st)
    for name, col in [("bf16", "tab:red"), ("fp32", "tab:orange"), ("fp64", "tab:brown")]:
        ax[0].axvline(CLIFFS[name], color=col, ls=":", alpha=0.7)
    ax[0].set_ylim(1e-22, 2)
    ax[0].set_xlabel("logit margin m"); ax[0].set_ylabel("|target gradient|")
    ax[0].set_title("Target gradient vs margin (drops to 1e-60 means exactly 0)")
    ax[0].legend(fontsize=8)

    sum_std = np.array([float(std_cross_entropy(np.array([[m, 0.0]], np.float32), np.array([0]))[1].sum()) for m in ms])
    sum_fix = np.array([float(fixed_cross_entropy(np.array([[m, 0.0]], np.float32), np.array([0]))[1].sum()) for m in ms])
    ax[1].semilogy(ms, np.maximum(np.abs(sum_std), floor), color="tab:orange", label="standard fp32")
    ax[1].semilogy(ms, np.maximum(np.abs(sum_fix), floor), color="tab:blue", label="fixed fp32")
    ax[1].axvline(CLIFFS["fp32"], color="tab:orange", ls=":")
    ax[1].set_ylim(1e-22, 2)
    ax[1].text(0.5, 0.35, "fixed fp32: sum is exactly 0 everywhere\n(below the plotted range)",
               transform=ax[1].transAxes, color="tab:blue", ha="center")
    ax[1].set_xlabel("logit margin m"); ax[1].set_ylabel("|sum of gradient|")
    ax[1].set_title("Zero-sum check (should be 0; 1e-60 means exactly 0)")
    ax[1].legend()
    save(fig, "lesson04_collapse.png")

# %% Exercises --------------------------------------------------------------------
print("\nEXERCISES")
print("1. Why does the standard WRONG-class gradient stay non-zero beyond the cliff?")
print("2. Re-run with z = [m, 0, 0, 0, 0].  Does the cliff move?  (Hint: margin is defined")
print("   with the log-sum-exp of the wrong classes, so use logit_margin.)")
print("3. Use np.float16 for the fixed version and check it never reaches exactly 0 early.")
