"""
Lesson 06: Adam and AdamW from scratch, and why tiny gradients can still move parameters
========================================================================================
Run all:  python lesson06_adam_from_scratch.py
"""
import numpy as np
from _figs import plt, save, HAVE_MPL
from optim import SGD, AdamW

# %% 1. SGD: step is proportional to the gradient -------------------------------------
print("SGD on f(theta) = c * theta^2 / 2, starting at theta = 1, lr = 0.1, one step:")
for c in [1e-6, 1.0, 1e6]:
    g = np.array([c * 1.0])
    new = SGD(0.1).step(np.array([1.0]), g)
    print(f"  curvature c = {c:8.0e}  gradient = {c:8.0e}  ->  step = {1.0 - new[0]:.3e}")

# %% 2. Adam: the first step has size ~lr, whatever the gradient scale ---------------------
print("\nAdam (eps tiny), same problem, one step, lr = 0.1:")
for c in [1e-6, 1.0, 1e6]:
    opt = AdamW(lr=0.1, eps=1e-30, dtype=np.float64)
    new = opt.step(np.array([1.0]), np.array([c * 1.0]))
    print(f"  gradient = {c:8.0e}  ->  step = {1.0 - new[0]:.6f}")
print("Adam divides by the running size of the gradient, so the step is ~lr for ANY nonzero")
print("gradient (as long as |g| is much larger than eps).  Scale-free.")

# %% 3. Where epsilon matters: normalized step |g| / (|g| + eps) ---------------------------
eps = 1e-8
gs = np.logspace(-14, 0, 200)
step = gs / (gs + eps)
print(f"\nWith eps = {eps}: the step is {1 - 1e-8 / (1e-4 + 1e-8):.4f} at |g|=1e-4, 0.5 at |g|=eps,")
print(f"and falls like |g|/eps below that (|g|=1e-10 -> {1e-10 / (1e-10 + eps):.3f}).")

# %% 4. Bias correction: why m_hat and v_hat divide by (1 - beta^t) -----------------------------
print("\nWithout bias correction the first steps would be too small, because m and v start at 0:")
g = 1.0
m = v = 0.0
b1, b2 = 0.9, 0.98
for t in range(1, 6):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    print(f"  t={t}: m={m:.4f}  v={v:.4f}   m/sqrt(v) raw={m / np.sqrt(v):.3f}   "
          f"corrected={(m / (1 - b1 ** t)) / np.sqrt(v / (1 - b2 ** t)):.3f}")

# %% 5. A quiet period: what happens to v and the step when gradients vanish -----------------------
print("\nGradient history: steady g=1e-3 for 300 steps, then g=0 for N steps, then g=1e-3 again.")
print("(beta1=0.9, beta2=0.98, eps=1e-8, float64)")
print("  N     sqrt(v_hat) just before resuming   first normalized step after resuming")
traces = {}
for N_quiet in [0, 50, 200, 500]:
    opt = AdamW(lr=1.0, eps=1e-8, dtype=np.float64)
    th = np.array([0.0])
    sv, ns = [], []
    for t in range(300 + N_quiet + 30):
        g = np.array([1e-3]) if (t < 300 or t >= 300 + N_quiet) else np.array([0.0])
        th = opt.step(th, g)
        v_hat = opt.v / (1 - opt.b2 ** opt.t)
        sv.append(float(np.sqrt(v_hat[0])))
        ns.append(float(opt.last_normalized_step()[0]))
    traces[N_quiet] = (np.array(sv), np.array(ns))
    before = sv[300 + N_quiet - 1]
    after = ns[300 + N_quiet]
    print(f"  {N_quiet:3d}   {before:.3e}                         {after:.3f}")
print("\nVerified here:")
print(" * sqrt(v_hat) decays like 0.99^N while gradients are absent (v itself decays 0.98 per step).")
print(" * m_hat decays faster (0.9 per step), so updates during the quiet period are near zero.")
print(" * When the gradient returns after a long pause, the normalized step tends to")
print(f"   (1-beta1)/sqrt(1-beta2) = {0.1 / np.sqrt(0.02):.3f}: bounded, so one parameter never takes a huge step.")
print("Reported by the paper, NOT verified here: the Slingshot loss spike (Section 4).  The paper")
print("attributes it to a restored gradient divided by a faded denominator.  Per parameter the jump")
print("is from ~0 to ~0.7*lr; whether many parameters doing this together spikes the loss depends on")
print("the network, and is shown by the paper's experiments, not by this script.")

# %% 6. Equilibrium preview: AdamW stationary point -------------------------------------------------------
print("\nStationary point of AdamW: g / (|g| + eps) + wd * theta = 0")
print("If |g| << eps this is g/eps ~ -wd*theta, so |g| ~ eps * wd * |theta|.")
print("Since |g| ~ e^-m, the margin settles near m ~ log(1/eps) (minus log(wd*|theta|)).  See lesson 07.")

# %% 7. Figure -----------------------------------------------------------------------------------------------
if HAVE_MPL:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    ax[0].semilogx(gs, step, color="tab:blue")
    ax[0].axvline(eps, color="gray", ls=":"); ax[0].set_xlabel("|g|"); ax[0].set_ylabel("|g| / (|g| + eps)")
    ax[0].set_title("Adam's normalized step, eps = 1e-8")
    for N_quiet, (sv, ns) in traces.items():
        ax[1].semilogy(np.maximum(sv, 1e-300), label=f"quiet N={N_quiet}")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("sqrt(v_hat)"); ax[1].set_title("Second-moment estimate fades in quiet periods")
    ax[1].legend(fontsize=8)
    for N_quiet, (sv, ns) in traces.items():
        ax[2].plot(ns, label=f"quiet N={N_quiet}")
    ax[2].set_xlabel("step"); ax[2].set_ylabel("|m_hat| / (sqrt(v_hat) + eps)"); ax[2].set_title("Normalized step (bounded)")
    ax[2].legend(fontsize=8)
    save(fig, "lesson06_adam.png")

# %% Exercises -------------------------------------------------------------------------------------------------------
print("\nEXERCISES")
print("1. Minimize f(x) = x^2 with SGD and with Adam (lr=0.1).  Compare trajectories.")
print("2. Set beta2 = 0.999.  How fast does v_hat fade in a quiet period now?")
print("3. Why does eps = 0 make Adam divide by zero when g = 0 for a long time?")
