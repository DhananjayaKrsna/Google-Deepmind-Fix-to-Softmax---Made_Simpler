"""
Lesson 07: A mini experiment, margins under AdamW (weight decay and epsilon)
============================================================================
A small softmax-regression model on separable blobs is trained in float32 with AdamW.
This is a TOY.  It is NOT a reproduction of the paper's Transformer / MLP grokking runs.
It lets you check, on a model small enough to read, three statements from the paper's
Section 5:

  (a) Without weight decay the margin keeps growing.
  (b) With weight decay it settles at an equilibrium.
  (c) That equilibrium moves with Adam's epsilon like log(1/epsilon):
      every 100x smaller epsilon should raise it by about ln(100) = 4.61.

  (d) With the STANDARD gradient the hardest example's margin ends up pinned near the
      fp32 cliff (16.6), while with the FIXED gradient it ends higher, near log(1/epsilon).

What the toy does NOT show: Slingshot loss spikes, or an accuracy drop.  Here the examples do
not all cross the cliff at the same time, so the batch gradient never vanishes entirely.
In the paper's networks it can.  Treat this as a small, honest check of the margin picture,
not as a replication of the paper's grokking experiments.

Run all:  python lesson07_margin_growth.py      (about 20 seconds)
"""
import numpy as np
from _figs import plt, save, HAVE_MPL
from optim import AdamW
from softmax_fix import std_cross_entropy, fixed_cross_entropy, logit_margin, CLIFFS

# %% 1. Data: three well separated blobs, features (x1, x2, 1) ---------------------------
def make_data(seed=0, n_per=30):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 3.0], [3.0, -2.0], [-3.0, -2.0]])
    X = np.concatenate([c + 0.6 * rng.standard_normal((n_per, 2)) for c in centers])
    y = np.repeat(np.arange(3), n_per)
    X = np.hstack([X, np.ones((len(X), 1))]).astype(np.float32)
    return X, y

X, y = make_data()
N, D = X.shape

# %% 2. Training function ----------------------------------------------------------------------
def train(fixed, wd, eps, steps=12000, lr=1e-2, seed=1, record_every=50, data_seed=0):
    X, y = make_data(data_seed)
    N, D = X.shape
    rng = np.random.default_rng(seed)
    W = (0.1 * rng.standard_normal((D, 3))).astype(np.float32)
    opt = AdamW(lr=lr, beta1=0.9, beta2=0.98, eps=eps, wd=wd, dtype=np.float32)
    h = {"step": [], "mean_m": [], "min_m": [], "acc": []}
    for step in range(steps + 1):
        logits = (X @ W).astype(np.float32)
        if fixed:
            loss, G, m = fixed_cross_entropy(logits, y)
        else:
            loss, G = std_cross_entropy(logits, y)
            m = logit_margin(logits, y)
        if step % record_every == 0:
            h["step"].append(step)
            h["mean_m"].append(float(np.mean(m)))
            h["min_m"].append(float(np.min(m)))
            h["acc"].append(float(np.mean(logits.argmax(1) == y)))
        grad = (X.T @ G / np.float32(N)).astype(np.float32)
        W = opt.step(W, grad)
    h = {k: np.array(v) for k, v in h.items()}
    h["max_abs_W"] = float(np.abs(W).max())
    return h

# %% 3. (a) and (b): weight decay on and off, standard vs fixed gradient ---------------------------
eps = 1e-8
print(f"AdamW, float32, eps={eps}, lr=1e-2, full batch, {N} examples, K=3")
print(f"fp32 precision cliff = {CLIFFS['fp32']:.2f},  log(1/eps) = {np.log(1 / eps):.2f}\n")
configs = [("standard, wd=0", False, 0.0), ("fixed,    wd=0", True, 0.0),
           ("standard, wd=0.01", False, 0.01), ("fixed,    wd=0.01", True, 0.01),
           ("fixed,    wd=1", True, 1.0)]
runs = {}
print("config               mean margin @6000   @12000   min margin @12000   max|W|    accuracy")
for name, fixed, wd in configs:
    h = train(fixed, wd, eps)
    runs[name] = h
    i6000 = int(np.where(h["step"] == 6000)[0][0])
    print(f"{name:19s}  {h['mean_m'][i6000]:12.2f}   {h['mean_m'][-1]:9.2f}   {h['min_m'][-1]:12.2f}   "
          f"{h['max_abs_W']:9.2f}   {h['acc'][-1]:.3f}")
print("\nReading the table (compare the '@6000' and '@12000' columns):")
print(" * wd=0   : the margin is higher at 12000 than at 6000, so it is still growing (statement a).")
print(" * wd=0.01: the two columns are nearly equal, so the margin has settled (statement b).")
print(" * wd=1   : weights are capped near 1/wd = 1 because Adam's normalized step is at most ~1,")
print("            so this LINEAR model cannot reach a large margin.  Deep networks can, because")
print("            small weights multiply across layers.  This is a toy-model limitation.")
print(" * standard vs fixed: compare the 'min margin' column; the next cell repeats this over 5 seeds.")

# %% 4. (c): epsilon sweep, fixed gradient, weight decay 0.01 ----------------------------------------
print("\nEpsilon sweep (fixed CE, wd=0.01, 12000 steps)")
print("  eps      log(1/eps)   min margin   mean margin   shift of min margin per 100x (theory 4.61)")
sweep, prev = {}, None
for e in [1e-6, 1e-8, 1e-10, 1e-12]:
    h = train(True, 0.01, e)
    sweep[e] = h
    shift = "" if prev is None else f"{h['min_m'][-1] - prev:.2f}"
    print(f"  {e:.0e}   {np.log(1 / e):8.2f}   {h['min_m'][-1]:10.2f}   {h['mean_m'][-1]:11.2f}   {shift}")
    prev = h["min_m"][-1]
print("\nThe smallest margin (the example that still carries gradient) follows log(1/eps) within")
print("about 1.0 and shifts by roughly ln(100) per 100x change in eps.  The MEAN margin shifts a bit")
print("more, because easier examples sit above the smallest one.  The paper plots the mean margin")
print("for its networks; this toy shows the same scaling law qualitatively, not the paper's numbers.")

# %% 4b. (d): is the standard run pinned at the cliff?  Five different datasets and initializations --
print("\nStandard vs fixed gradient, wd=0.01, eps=1e-8, five seeds (different data and init each time)")
print(f"fp32 cliff = {CLIFFS['fp32']:.2f},  log(1/eps) = {np.log(1 / 1e-8):.2f}")
print("seed   standard: mean   min    | fixed: mean   min    | fixed - standard (min)")
mins_std, mins_fix = [], []
for s in range(5):
    a = train(False, 0.01, 1e-8, seed=s + 100, data_seed=s)
    b = train(True, 0.01, 1e-8, seed=s + 100, data_seed=s)
    mins_std.append(a["min_m"][-1]); mins_fix.append(b["min_m"][-1])
    print(f"  {s}        {a['mean_m'][-1]:6.2f}   {a['min_m'][-1]:6.2f}  |        {b['mean_m'][-1]:6.2f}   {b['min_m'][-1]:6.2f}  |  {b['min_m'][-1] - a['min_m'][-1]:5.2f}")
print(f"standard min margin: {min(mins_std):.2f} to {max(mins_std):.2f}  (cliff {CLIFFS['fp32']:.2f})")
print(f"fixed    min margin: {min(mins_fix):.2f} to {max(mins_fix):.2f}  (log(1/eps) {np.log(1 / 1e-8):.2f})")
print("The hardest example ends up at the cliff with the standard gradient.  This is consistent with")
print("the paper's explanation (a rounded-to-zero target gradient removes the push beyond the cliff),")
print("but this toy does not isolate the cause.  For eps=1e-6 the equilibrium 13.8 is below the cliff,")
print("which the paper also predicts: for eps >= 1e-7 standard CE settles before reaching the cliff.")
print("It is still a toy: accuracy stays 100% and no loss spike appears.")

# %% 5. Figure ----------------------------------------------------------------------------------------
if HAVE_MPL:
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    cols = {"standard, wd=0": "tab:orange", "fixed,    wd=0": "tab:blue",
            "standard, wd=0.01": "tab:red", "fixed,    wd=0.01": "tab:cyan", "fixed,    wd=1": "tab:green"}
    for name, h in runs.items():
        ax[0].plot(h["step"], h["mean_m"], color=cols[name], label=name)
    ax[0].axhline(CLIFFS["fp32"], color="gray", ls=":", label="fp32 cliff 16.6")
    ax[0].axhline(np.log(1 / eps), color="black", ls="--", lw=0.8, label="log(1/eps) = 18.4")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("mean logit margin")
    ax[0].set_title("Margins under AdamW (float32, eps=1e-8)"); ax[0].legend(fontsize=8)
    for e, h in sweep.items():
        line, = ax[1].plot(h["step"], h["min_m"], label=f"eps={e:.0e}")
        ax[1].axhline(np.log(1 / e), color=line.get_color(), ls=":", lw=1)
    ax[1].set_xlabel("step"); ax[1].set_ylabel("MIN logit margin")
    ax[1].set_title("Fixed CE, wd=0.01: plateau vs log(1/eps) (dotted)"); ax[1].legend(fontsize=8)
    save(fig, "lesson07_margin_growth.png")

# %% Exercises ----------------------------------------------------------------------------------------
print("\nEXERCISES")
print("1. Make the data harder (blob noise 0.6 -> 2.0).  Is the data still separable?  What does the")
print("   margin do?")
print("2. Predict, then test, what a 2-layer linear model W2 @ W1 does with wd=1.")
print("3. Replace AdamW by SGD (optim.SGD).  Is there an epsilon-dependent equilibrium at all?")
