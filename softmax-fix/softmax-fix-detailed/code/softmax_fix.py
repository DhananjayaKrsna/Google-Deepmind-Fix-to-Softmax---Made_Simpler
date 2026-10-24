"""
softmax_fix.py
==============
A small, readable NumPy library used by every lesson script in this kit.

It contains
  1. STANDARD implementations (what most libraries do): max-shift softmax,
     log-softmax and cross-entropy with gradient q - p.
  2. FIXED implementations that follow arXiv:2610.31642v1 (Barron,
     Gnanapragasam, Tschannen): the log-space rewrite using softplus,
     expm1 and log1p.
  3. A bfloat16 emulator, because NumPy has no native bf16 type.

All functions are batched: the last axis of `z` is the class axis, so `z` may
have shape (K,), (B, K), (B, T, K), and so on.  Output dtypes follow the input
dtype (float16, float32, float64), which is the whole point of the lessons.

Written for teaching.  It is NOT optimized and has no autodiff: the gradients
are written out analytically so you can read every line.
"""
import numpy as np


# --------------------------------------------------------------------------
# Small numerical helpers
# --------------------------------------------------------------------------
def lse(x, axis=-1, keepdims=False):
    """Stable log-sum-exp: log(sum(exp(x))) computed as max + log(sum(exp(x - max)))."""
    mx = np.max(x, axis=axis, keepdims=True)
    finite_mx = np.where(np.isfinite(mx), mx, np.zeros_like(mx))
    out = finite_mx + np.log(np.sum(np.exp(x - finite_mx), axis=axis, keepdims=True))
    out = np.where(np.isfinite(mx), out, mx)  # all -inf stays -inf
    return out if keepdims else np.squeeze(out, axis=axis)


def softplus(x):
    """log(1 + e^x), safe for very large and very negative x."""
    return np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))


def to_bf16(x):
    """Round float32 values to the nearest bfloat16 (kept inside a float32 array).

    bfloat16 is the top 16 bits of a float32: 1 sign, 8 exponent, 7 fraction bits.
    Rounding is round-to-nearest-even, like hardware.  Use it to emulate bf16
    arithmetic by wrapping each operation:  to_bf16(a + b).
    """
    x = np.array(x, dtype=np.float32, order="C")  # copy, keeps the shape (even 0-d)
    u = x.view(np.uint32)
    rounding_bias = ((u >> 16) & 1) + np.uint32(0x7FFF)
    u = (u + rounding_bias) & np.uint32(0xFFFF0000)
    return u.view(np.float32)


# --------------------------------------------------------------------------
# STANDARD implementations (these contain the two bugs)
# --------------------------------------------------------------------------
def std_softmax(z):
    """Standard softmax: shift by the max, exponentiate, divide."""
    zt = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(zt)
    return e / np.sum(e, axis=-1, keepdims=True)


def std_log_softmax(z):
    """Standard log-softmax: z~ - log(sum(exp(z~))).  The sum contains the exact 1."""
    zt = z - np.max(z, axis=-1, keepdims=True)
    return zt - np.log(np.sum(np.exp(zt), axis=-1, keepdims=True))


def _take_target(z, t):
    return np.take_along_axis(z, np.asarray(t)[..., None], axis=-1)[..., 0]


def _one_hot(t, K, dtype):
    return (np.arange(K) == np.asarray(t)[..., None]).astype(dtype)


def std_cross_entropy(z, t):
    """Standard hard-target cross-entropy.  Returns (loss, grad) with grad = q - onehot(t)."""
    logq = std_log_softmax(z)
    loss = -_take_target(logq, t)
    q = std_softmax(z)
    grad = q - _one_hot(t, z.shape[-1], z.dtype)
    return loss, grad


def std_soft_target_grad(zS, zT):
    """Standard soft-target gradient q - p, where q and p are rounded separately."""
    return std_softmax(zS) - std_softmax(zT)


def std_soft_target_grad_framework_style(zS, zT):
    """Same gradient, but q is recovered as exp(log_softmax(zS)) like an autodiff backward pass.

    In many frameworks the forward pass uses softmax for p and the backward pass
    uses exp(logsoftmax) for q.  These two routes are different sequences of
    floating-point operations, so q - p is not exactly 0 even when zS == zT.
    """
    q = np.exp(std_log_softmax(zS))
    p = std_softmax(zT)
    return q - p


# --------------------------------------------------------------------------
# FIXED implementations (paper, Equations 5 to 9)
# --------------------------------------------------------------------------
def fixed_log_softmax(z):
    """Equation 5.  The arg-max slot is the constant 0; the margin comes from the others."""
    K = z.shape[-1]
    i_max = np.argmax(z, axis=-1)
    is_max = np.arange(K) == i_max[..., None]
    z_max = np.max(z, axis=-1, keepdims=True)
    shifted = np.where(is_max, np.zeros_like(z), z - z_max)  # exact 0 at the arg-max
    others = np.where(is_max, -np.inf, shifted)              # leave the arg-max out of the sum
    m = -lse(others, axis=-1, keepdims=True)                 # logit margin
    return shifted - softplus(-m)


def fixed_softmax(z):
    return np.exp(fixed_log_softmax(z))


def fixed_cross_entropy(z, t):
    """Equations 6 and 7.  Returns (loss, grad, margin).

        m      = -logsumexp_{j != t}(z_j - z_t)
        loss   = softplus(-m)
        g_j    = exp(z_j - z_t - loss)   for j != t
        g_t    = -sum_{j != t} g_j       -> the gradient sums to zero
    """
    K = z.shape[-1]
    z_t = _take_target(z, t)[..., None]
    is_t = np.arange(K) == np.asarray(t)[..., None]
    diffs = z - z_t
    others = np.where(is_t, -np.inf, diffs)
    m = -lse(others, axis=-1)
    loss = softplus(-m)
    g = np.exp(others - loss[..., None])          # exp(-inf) = 0 at the target slot
    g_t = -np.sum(g, axis=-1)
    g = np.where(is_t, g_t[..., None], g)
    return loss, g.astype(z.dtype), m


def fixed_soft_target_grad(zS, zT):
    """Equation 9 plus the zero-sum clean-up.  Computes q - p directly from zS - zT."""
    p = std_softmax(zT)
    q = np.exp(fixed_log_softmax(zS))
    delta = zS - zT
    d = delta - np.sum(p * delta, axis=-1, keepdims=True)                  # center under p
    s = d - np.log1p(np.sum(p * np.expm1(d), axis=-1, keepdims=True))      # log(q / p)
    u = p * np.expm1(s)                                                    # q - p
    return u - q * np.sum(u, axis=-1, keepdims=True)                       # remove drift


def fixed_soft_target_loss(zS, zT):
    """Forward value of the soft-target cross-entropy (uses the fixed log-softmax)."""
    p = std_softmax(zT)
    return -np.sum(p * fixed_log_softmax(zS), axis=-1)


# --------------------------------------------------------------------------
# Quantities used in the lessons
# --------------------------------------------------------------------------
def logit_margin(z, t):
    """m = z_t - log(sum_{j != t} exp(z_j))   (paper, Equation 2)."""
    K = z.shape[-1]
    is_t = np.arange(K) == np.asarray(t)[..., None]
    z_t = _take_target(z, t)
    return z_t - lse(np.where(is_t, -np.inf, z), axis=-1)


def precision_cliff(fraction_bits):
    """c = (p + 1) * ln 2 : margin beyond which 1 + exp(-m) rounds to 1.0."""
    return (fraction_bits + 1) * np.log(2.0)


CLIFFS = {"bf16": precision_cliff(7), "fp16": precision_cliff(10),
          "fp32": precision_cliff(23), "fp64": precision_cliff(52)}
