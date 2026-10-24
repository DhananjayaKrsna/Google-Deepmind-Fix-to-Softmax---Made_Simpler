"""
Lesson 01: Floating point, the ground the whole paper stands on
===============================================================
Run all:        python lesson01_floating_point.py
Run in cells:   open in VS Code / Spyder / Jupytext and run each "# %%" cell.

You will see, with your own eyes:
  1. why 0.1 + 0.2 != 0.3
  2. that numbers are spaced RELATIVELY (fine near 0, coarse near 1)
  3. that adding a small number to 1.0 can change nothing ("absorption")
  4. where exactly the absorption threshold is: 2^-(p+1)
  5. overflow and underflow of exp
  6. why log1p and expm1 exist (catastrophic cancellation)
"""
import numpy as np
from _figs import plt, save, HAVE_MPL

# %% 1. The classic surprise ---------------------------------------------------
print("0.1 + 0.2 == 0.3 ?", 0.1 + 0.2 == 0.3)
print("0.1 + 0.2        =", repr(0.1 + 0.2))
print("Reason: 0.1 and 0.2 have no exact binary representation, like 1/3 has no")
print("exact decimal representation.  Every float is a rounded version of the truth.")

# %% 2. Machine epsilon: the spacing of floats just above 1.0 -----------------
print("\nformat   fraction bits p   spacing at 1.0 (2^-p)    half spacing 2^-(p+1)")
for name, dt, p in [("fp16", np.float16, 10), ("fp32", np.float32, 23), ("fp64", np.float64, 52)]:
    eps = np.finfo(dt).eps
    assert eps == 2.0 ** -p
    print(f"{name}     {p:>3d}              {eps:<22.3e}   {2.0 ** -(p + 1):.3e}")
print("bf16        7              7.813e-03                3.906e-03   (NumPy has no bf16; see to_bf16)")

# %% 3. Spacing is RELATIVE: fine near zero, coarse near large numbers --------
print("\nSpacing between neighbouring float32 numbers at different magnitudes:")
for x in [1e-30, 1e-10, 1e-3, 1.0, 1e3, 1e10]:
    sp = float(np.spacing(np.float32(x)))
    print(f"  near {x:8.0e}: spacing = {sp:.3e}   relative = {sp / x:.3e}")
print("The RELATIVE spacing stays near 1e-7, so absolute spacing near 1e-30 is tiny,")
print("while absolute spacing near 1.0 is about 1.2e-7.  This asymmetry causes the bug.")

# %% 4. Absorption: 1.0 + tiny == 1.0 -----------------------------------------
print("\nAbsorption test (does 1 + x differ from 1?)")
for name, dt, p in [("fp16", np.float16, 10), ("fp32", np.float32, 23), ("fp64", np.float64, 52)]:
    one = dt(1.0)
    half = dt(2.0 ** -(p + 1))                       # exactly half the spacing: a tie
    above = dt(2.0 ** -(p + 1) * (1 + 2.0 ** -(p - 2)))  # a hair above half
    print(f"{name}: 1 + 2^-(p+1)      == 1 ? {one + half == one}   (tie, rounds to even = 1.0)")
    print(f"{name}: 1 + (just above)  == 1 ? {one + above == one}  (now rounds up)")
print("\nConclusion: 1 + x == 1 whenever x <= 2^-(p+1).  With x = e^-m this means")
print("m >= (p+1) ln 2.  That is the 'precision cliff' of the paper.")
for name, p in [("bf16", 7), ("fp16", 10), ("fp32", 23), ("fp64", 52)]:
    print(f"  {name}: c = {(p + 1) * np.log(2):6.2f}")

# %% 5. Overflow and underflow ---------------------------------------------------
print("\nOverflow / underflow of exp in float32:")
with np.errstate(over="ignore", under="ignore"):
    for x in [80.0, 88.0, 89.0, -80.0, -100.0, -104.0]:
        print(f"  exp({x:7.1f}) = {np.exp(np.float32(x))}")
print("Largest float32 ~3.4e38 so exp(x) overflows for x > ~88.7 (this is why softmax")
print("subtracts the max).  Below ~1e-38 values become subnormal, then 0.0.")

# %% 6. Catastrophic cancellation and the cure: log1p / expm1 -------------------
print("\nComputing log(1+x) and exp(x)-1 for tiny x in float64:")
print("      x          log(1+x)      log1p(x)      exp(x)-1      expm1(x)")
with np.errstate(divide="ignore"):
    for x in [1e-3, 1e-8, 1e-12, 1e-16, 1e-20]:
        print(f"  {x:8.0e}   {np.log(1 + x):<12.4e}  {np.log1p(x):<12.4e}  "
              f"{np.exp(x) - 1:<12.4e}  {np.expm1(x):<12.4e}")
print("The naive forms lose digits, then return exactly 0, because 1 + x == 1.")
print("log1p and expm1 never form 1 + x, so they stay accurate.")

# %% 7. Figure: relative vs absolute spacing -----------------------------------
if HAVE_MPL:
    xs = np.logspace(-30, 30, 400)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, dt, col in [("fp32", np.float32, "tab:blue"), ("fp64", np.float64, "tab:orange")]:
        with np.errstate(over="ignore", under="ignore"):
            sp = np.array([np.spacing(dt(x)) for x in xs], dtype=np.float64)
        ok = np.isfinite(sp) & (sp > 0)
        ax[0].loglog(xs[ok], sp[ok], color=col, label=name)
        ax[1].loglog(xs[ok], sp[ok] / xs[ok], color=col, label=name)
    ax[0].axvline(1.0, color="gray", ls=":")
    ax[0].set_title("Absolute spacing between neighbouring floats")
    ax[0].set_xlabel("magnitude x"); ax[0].set_ylabel("spacing"); ax[0].legend()
    ax[1].set_title("Relative spacing (spacing / x): roughly constant")
    ax[1].set_xlabel("magnitude x"); ax[1].set_ylabel("spacing / x"); ax[1].legend()
    save(fig, "lesson01_spacing.png")

# %% Exercises ------------------------------------------------------------------
print("\nEXERCISES")
print("1. Predict, then test: np.float32(1) + np.float32(1e-8) == np.float32(1)?")
print("2. Find the smallest x with np.float32(1) + x != 1 by halving a loop variable.")
print("3. Why is (a + b) + c sometimes different from a + (b + c) for floats?")
