"""
Lesson 02: Softmax, log-sum-exp and the max-shift trick, from scratch
=====================================================================
Run all:        python lesson02_softmax_from_scratch.py
Run in cells:   run each "# %%" cell in VS Code / Spyder / Jupytext.
"""
import numpy as np
from _figs import plt, save, HAVE_MPL
from softmax_fix import std_softmax, std_log_softmax, lse

np.set_printoptions(precision=4, suppress=True)

# %% 1. Softmax by the definition ----------------------------------------------
def naive_softmax(z):
    e = np.exp(z)
    return e / e.sum()

z = np.array([2.0, 1.0, 0.0])
q = naive_softmax(z)
print("logits      :", z)
print("exp(logits) :", np.exp(z))
print("softmax     :", q, " sum =", q.sum())
print("Largest logit gets the largest probability; probabilities sum to 1.")

# %% 2. The naive version breaks for big logits --------------------------------
big = np.array([1000.0, 1001.0, 1002.0])
with np.errstate(over="ignore", invalid="ignore"):
    print("\nnaive softmax of [1000, 1001, 1002] ->", naive_softmax(big), "(inf / inf = nan)")
print("stable softmax of [1000, 1001, 1002] ->", std_softmax(big))

# %% 3. Why the shift is legal: softmax ignores constants -----------------------
print("\nShift invariance:")
for c in [-5.0, 0.0, 100.0]:
    print(f"  softmax(z + {c:6.1f}) =", std_softmax(z + c))
print("Proof: exp(z+c) = exp(z) * exp(c) and exp(c) cancels in numerator and denominator.")

# %% 4. log-sum-exp and log-softmax ---------------------------------------------
print("\nlog-sum-exp of z      :", lse(z), " (naive:", np.log(np.exp(z).sum()), ")")
print("log-softmax = z - LSE :", z - lse(z))
print("log(softmax)          :", np.log(std_softmax(z)))

# %% 5. Why log-softmax is a separate function ------------------------------------
z32 = np.array([0.0, -200.0], dtype=np.float32)
with np.errstate(divide="ignore", under="ignore"):
    print("\nfloat32 logits [0, -200]:")
    print("  log(softmax(z))   =", np.log(std_softmax(z32)), "   <- -inf: the probability underflowed to 0")
print("  log_softmax(z)    =", std_log_softmax(z32), "   <- correct, stays in range")

# %% 6. The "1" inside the denominator (set-up for the paper) ---------------------
z = np.array([9.0, -9.0, -9.0])
zs = z - z.max()
sci = {"float_kind": lambda v: f"{v:.3e}"}
print("\nAfter the max shift, logits are", zs, "-> exponentials",
      np.array2string(np.exp(zs), formatter=sci))
print("The denominator is 1 + 1.5e-8 + 1.5e-8.  In float64 that is fine;")
print("in float32 the small terms are below half the spacing near 1, so the sum is exactly 1.0.")
print("float32:", np.float32(1) + np.float32(np.exp(-18)) + np.float32(np.exp(-18)))

# %% 7. Temperature: how softmax sharpness changes -------------------------------
z = np.array([2.0, 1.0, 0.0])
Ts = [0.25, 0.5, 1.0, 2.0, 4.0, 10.0]
print("\nTemperature T divides the logits before softmax:")
for T in Ts:
    print(f"  T = {T:5.2f} ->", std_softmax(z / T))
if HAVE_MPL:
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.13
    for i, T in enumerate(Ts):
        ax.bar(np.arange(3) + (i - 2.5) * w, std_softmax(z / T), width=w, label=f"T={T}")
    ax.set_xticks(range(3)); ax.set_xticklabels(["class 0 (z=2)", "class 1 (z=1)", "class 2 (z=0)"])
    ax.set_ylabel("probability"); ax.set_title("Softmax at different temperatures"); ax.legend(ncol=3)
    save(fig, "lesson02_temperature.png")

# %% Exercises ------------------------------------------------------------------
print("\nEXERCISES")
print("1. Show by hand that softmax([a, b]) = [sigmoid(a-b), sigmoid(b-a)].")
print("2. What is softmax of [0, 0, 0, 0]?  Of [1000, 0, 0]?")
print("3. What happens as T -> 0 and as T -> infinity?")
