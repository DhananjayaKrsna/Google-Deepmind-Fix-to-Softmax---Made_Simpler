"""
Lesson 03: Cross-entropy, the gradient q - p, and a tiny training loop
======================================================================
Run all:        python lesson03_gradients_and_sgd.py
"""
import numpy as np
from _figs import plt, save, HAVE_MPL
from softmax_fix import std_softmax, std_log_softmax, std_cross_entropy

np.set_printoptions(precision=5, suppress=True)
rng = np.random.default_rng(0)

# %% 1. Cross-entropy for a hard target ----------------------------------------
z = np.array([2.0, 1.0, 0.0])
t = 0
q = std_softmax(z)
loss = -np.log(q[t])
print("q =", q, " target class", t)
print("loss = -log q_t =", loss, "  (small when q_t is near 1, large when q_t is near 0)")
for qt in [0.99, 0.9, 0.5, 0.1, 0.01]:
    print(f"  q_t = {qt:5.2f} -> loss {-np.log(qt):.3f}")

# %% 2. Check the gradient formula g = q - p with finite differences ------------
def ce_loss(zz, tt):
    return -std_log_softmax(zz)[tt]

def numeric_grad(zz, tt, h=1e-6):
    g = np.zeros_like(zz)
    for j in range(len(zz)):
        zp, zm = zz.copy(), zz.copy()
        zp[j] += h
        zm[j] -= h
        g[j] = (ce_loss(zp, tt) - ce_loss(zm, tt)) / (2 * h)
    return g

z = rng.normal(size=5)
t = 3
analytic = std_softmax(z) - np.eye(5)[t]
print("\nanalytic q - p :", analytic)
print("finite diff    :", numeric_grad(z, t))
print("max difference :", np.max(np.abs(analytic - numeric_grad(z, t))))

# %% 3. The gradient sums to zero and why ---------------------------------------
print("\nsum of gradient entries:", analytic.sum(), " (should be 0)")
print("Reason: adding the same constant to all logits does not change the loss,")
print("so the slope along the direction (1,1,...,1) is zero.")
q = std_softmax(z)
J = np.diag(q) - np.outer(q, q)              # Jacobian of softmax
print("Jacobian times ones =", J @ np.ones(5), "  <- exact null direction")

# %% 4. A real, tiny training loop: softmax regression on 3 blobs ----------------
def make_blobs(n_per=60):
    centers = np.array([[0.0, 2.5], [2.5, -1.5], [-2.5, -1.5]])
    X = np.concatenate([c + 0.9 * rng.standard_normal((n_per, 2)) for c in centers])
    y = np.repeat(np.arange(3), n_per)
    X = np.hstack([X, np.ones((len(X), 1))])        # bias feature
    return X, y

X, y = make_blobs()
W = np.zeros((3, 3))                                 # (features+bias, classes)
lr, steps = 0.1, 300
losses, accs = [], []
for step in range(steps):
    logits = X @ W
    loss, G = std_cross_entropy(logits, y)           # G = q - onehot, shape (N, 3)
    W -= lr * (X.T @ G) / len(X)                     # chain rule: dL/dW = X^T G / N
    losses.append(loss.mean())
    accs.append((logits.argmax(1) == y).mean())
print(f"\nstep   0: loss {losses[0]:.4f}  accuracy {accs[0]:.3f}")
print(f"step 299: loss {losses[-1]:.4f}  accuracy {accs[-1]:.3f}")

if HAVE_MPL:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(losses); ax[0].set_title("Training loss (SGD)"); ax[0].set_xlabel("step")
    for k, c in enumerate(["tab:blue", "tab:orange", "tab:green"]):
        ax[1].scatter(X[y == k, 0], X[y == k, 1], s=12, c=c, label=f"class {k}")
    gx, gy = np.meshgrid(np.linspace(-6, 6, 200), np.linspace(-5, 6, 200))
    grid = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)], axis=1)
    ax[1].contourf(gx, gy, (grid @ W).argmax(1).reshape(gx.shape), alpha=0.15, levels=[-.5, .5, 1.5, 2.5])
    ax[1].set_title("Learned decision regions"); ax[1].legend()
    save(fig, "lesson03_sgd.png")

# %% 5. Keep training: the margin grows, the gradient shrinks like e^-m ---------
print("\nKeep training on separable data and watch margin m and target gradient:")
from softmax_fix import logit_margin
for extra in [0, 2000, 8000]:
    for _ in range(extra):
        logits = X @ W
        _, G = std_cross_entropy(logits, y)
        W -= lr * (X.T @ G) / len(X)
    ms = logit_margin(X @ W, y)
    _, G = std_cross_entropy(X @ W, y)
    gt = np.abs(G[np.arange(len(y)), y])
    print(f"  +{extra:5d} steps: median margin {np.median(ms):6.2f}   median |g_t| {np.median(gt):.3e}   "
          f"e^-(median margin) {np.exp(-np.median(ms)):.3e}")
print("Confident models have tiny gradients, shrinking like e^-m.")
print("How tiny is tiny, and what floating point does there, is exactly what the paper studies.")

# %% Exercises ------------------------------------------------------------------
print("\nEXERCISES")
print("1. Derive dL/dz_j = q_j - p_j yourself using L = -sum_k p_k (z_k - LSE(z)).")
print("2. Change lr to 1.0.  What happens to the loss curve?  Why?")
print("3. Add L2 weight decay (W -= lr * wd * W) and watch the margin stop growing.")
