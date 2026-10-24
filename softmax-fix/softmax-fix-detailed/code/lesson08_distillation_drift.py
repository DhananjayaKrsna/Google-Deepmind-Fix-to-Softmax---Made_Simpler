"""
Lesson 08: Distillation drift, the practical face of Softmax Cancellation
=========================================================================
Teacher: a fixed random linear model producing 10-class logits.
Student: a linear model initialised with EXACTLY the teacher's weights, so the true
gradient is 0 and a correct training loop must leave the student unchanged.

We train the student with Adam on the soft-target loss at temperature T = 4 and compare
  * standard gradient (framework style: p = softmax, q = exp(log_softmax)), computed in
    float64, float32 and float16 (float16 is a stand-in for low precision; NumPy has no bf16)
  * the fixed gradient (equation 9 of the paper), in the same dtypes.

This mirrors the logic of the paper's Section 7 on a model small enough to read.  It is NOT
the paper's network, so the numbers will differ; the pattern is what to look for.

Run all:  python lesson08_distillation_drift.py      (about 30 seconds)
"""
import numpy as np
from _figs import plt, save, HAVE_MPL
from optim import AdamW
from softmax_fix import (std_softmax, std_soft_target_grad_framework_style,
                         fixed_soft_target_grad)

# %% 1. Teacher, data and evaluation helpers -----------------------------------------------
rng = np.random.default_rng(0)
N, D, K, T = 256, 20, 10, 4.0
X = rng.standard_normal((N, D)).astype(np.float32)
X_test = rng.standard_normal((512, D)).astype(np.float32)
W_T = (0.5 * rng.standard_normal((D, K))).astype(np.float32)

def kl_teacher_student(W):
    """KL(p_T || q_S) on held-out data, evaluated in float64 at temperature T."""
    zT = (X_test.astype(np.float64) @ W_T.astype(np.float64)) / T
    zS = (X_test.astype(np.float64) @ W.astype(np.float64)) / T
    p, q = std_softmax(zT), std_softmax(zS)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=-1)))

def agreement(W):
    return float(np.mean((X_test @ W).argmax(1) == (X_test @ W_T).argmax(1)))

print(f"teacher logit std = {float((X @ W_T).std()):.2f},  temperature T = {T}")
print("KL at initialisation (student == teacher):", kl_teacher_student(W_T.copy()))

# %% 2. Residual gradient at zS == zT, per dtype -----------------------------------------------
print("\nGradient residual at zS == zT (should be exactly 0), mean |g| per entry:")
for name, dt in [("float64", np.float64), ("float32", np.float32), ("float16", np.float16)]:
    zT = ((X @ W_T) / np.float32(T)).astype(dt)
    g_std = std_soft_target_grad_framework_style(zT, zT)
    g_fix = fixed_soft_target_grad(zT, zT)
    print(f"  {name}: standard {float(np.mean(np.abs(g_std.astype(np.float64)))):.2e}   "
          f"fixed {float(np.mean(np.abs(g_fix.astype(np.float64)))):.2e}")

# %% 2b. What Adam actually sees: the PARAMETER gradient at step 0 --------------------------------
# The logit residual is averaged over the batch and divided by T before it reaches the weights,
# so the parameter gradient is smaller than the logit residual.  Adam then maps it to a step.
print("\nParameter gradient at step 0 (student == teacher) and Adam's first normalized step:")
print("  dtype     mean |param grad|   mean |g|/(|g|+eps)  (1.0 would be a full step of size lr)")
for name, dt in [("float64", np.float64), ("float32", np.float32), ("float16", np.float16)]:
    zT = ((X @ W_T) / np.float32(T)).astype(dt)
    G = std_soft_target_grad_framework_style(zT, zT)
    pg = np.abs((X.T @ (G.astype(np.float32) / np.float32(T)) / np.float32(N)).astype(np.float64))
    print(f"  {name}   {pg.mean():.3e}          {np.mean(pg / (pg + 1e-8)):.3f}")

# %% 3. Train the student ----------------------------------------------------------------------
def train(dtype, fixed, steps=2000, lr=2e-3, eps=1e-8):
    W = W_T.copy()
    opt = AdamW(lr=lr, beta1=0.9, beta2=0.999, eps=eps, wd=0.0, dtype=np.float32)
    hist = {"step": [], "kl": [], "agree": [], "drift": [], "inv": [], "fun": []}
    zT = ((X @ W_T) / np.float32(T)).astype(dtype)
    for step in range(steps + 1):
        zS = ((X @ W) / np.float32(T)).astype(dtype)
        G = fixed_soft_target_grad(zS, zT) if fixed else std_soft_target_grad_framework_style(zS, zT)
        grad = (X.T @ (G.astype(np.float32) / np.float32(T)) / np.float32(N)).astype(np.float32)
        if step % 50 == 0:
            hist["step"].append(step)
            hist["kl"].append(kl_teacher_student(W))
            hist["agree"].append(agreement(W))
            dW = (W - W_T).astype(np.float64)
            common = dW.mean(axis=1, keepdims=True)            # same shift for every class
            hist["drift"].append(float(np.linalg.norm(dW)))
            hist["inv"].append(float(np.linalg.norm(np.broadcast_to(common, dW.shape))))
            hist["fun"].append(float(np.linalg.norm(dW - common)))
        W = opt.step(W, grad)
    return {k: np.array(v) for k, v in hist.items()}

print("\nAdam, lr=2e-3, eps=1e-8, 2000 steps, student initialised at the teacher")
print("'invisible' = part of the drift that shifts all class logits equally (softmax ignores it);")
print("'visible'   = the rest, which can change the predicted distribution.")
print("variant            final KL     max KL      ||dW||     invisible   visible    agreement")
runs = {}
for dname, dt in [("float64", np.float64), ("float32", np.float32), ("float16", np.float16)]:
    for fixed in (False, True):
        label = f"{'fixed   ' if fixed else 'standard'} {dname}"
        h = train(dt, fixed)
        runs[label] = h
        print(f"{label:17s}  {h['kl'][-1]:.2e}   {h['kl'].max():.2e}   {h['drift'][-1]:.2e}   "
              f"{h['inv'][-1]:.2e}   {h['fun'][-1]:.2e}   {h['agree'][-1]:.3f}")

# %% 4. Reading the result -----------------------------------------------------------------------
print("\nHow to read it:")
print(" * float64 standard: the parameter gradient (~1e-19) is far below eps = 1e-8, so Adam's step is")
print("   about 0 and the student does not move.")
print(" * float32 standard: the parameter gradient (~1e-10, after averaging over the batch) is still")
print("   below eps, so Adam takes steps of only ~1% of lr.  Small but non-zero drift, and nothing")
print("   stops it from accumulating (see the 'drift' curve).")
print(" * float16 standard: the parameter gradient (~1e-6) is well above eps, so Adam takes near-full")
print("   steps (0.97 of lr) and the student leaves the teacher although the true gradient is 0.")
print(" * Most of that drift is 'invisible': the residual does not sum to zero across classes, so")
print("   Adam pushes every class the same way, and a common shift of all logits does not change the")
print("   softmax (lesson 02).  That is why KL stays tiny while the weights wander.  This is the")
print("   zero-sum bug of lesson 04 showing up again, in a different place.")
print(" * fixed: gradient is exactly 0, Adam gets 0/(0+eps) = 0, the student stays put in every dtype.")
print("The paper reports test-accuracy loss for bf16 in its network; this toy shows weight drift but")
print("only a tiny change in the predicted distribution.  Do not over-read it.")

# %% 5. Figure -------------------------------------------------------------------------------------
if HAVE_MPL:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    cols = {"float64": "tab:green", "float32": "tab:orange", "float16": "tab:red"}
    for label, h in runs.items():
        dname = label.split()[1]
        ls = "-" if label.startswith("fixed") else "--"
        ax[0].semilogy(h["step"], np.maximum(h["kl"], 1e-30), color=cols[dname], ls=ls, label=label)
        ax[1].semilogy(h["step"], np.maximum(h["drift"], 1e-30), color=cols[dname], ls=ls, label=label)
    ax[0].set_xlabel("step"); ax[0].set_ylabel("held-out KL(teacher || student)")
    ax[0].set_title("Student should stay at KL = 0 (1e-30 = exactly 0)"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("step"); ax[1].set_ylabel("||W_student - W_teacher||")
    ax[1].set_title("Parameter drift"); ax[1].legend(fontsize=7)
    save(fig, "lesson08_distillation.png")

# %% Exercises --------------------------------------------------------------------------------------
print("\nEXERCISES")
print("1. Raise eps to 1e-4.  Which variants stop drifting?  What does that say about eps as a")
print("   crude safety net?")
print("2. Use a larger T (8).  Does the float16 residual change?")
print("3. Perturb the student slightly (W_T + 1e-3 noise).  Do standard and fixed converge to the")
print("   same KL?  Which converges more precisely?")
