"""
check_my_framework.py
Test whether YOUR cross-entropy implementation shows Softmax Collapse.

The test case is the paper's example: logits z = [9, -9, -9], target class 0, float32.
Its margin is 17.31, above the float32 precision cliff (16.64).  Correct answers:
    loss             = 3.046e-8      (NOT 0)
    gradient (target) = -3.046e-8     (NOT 0)
    sum of gradient   = 0

What runs here:
  * the NumPy "standard" and "fixed" versions from softmax_fix.py (always).
What is OPTIONAL and was NOT executed when this kit was built (PyTorch and JAX were not
installed in the build environment): the PyTorch and JAX blocks below.  They use standard,
documented APIs and are skipped automatically if the library is missing.  Run them yourself
and compare with the correct answers above.

    python check_my_framework.py
"""
import numpy as np
from softmax_fix import std_cross_entropy, fixed_cross_entropy

np.set_printoptions(linewidth=120)
CORRECT_LOSS = float(np.exp(-(18.0 - np.log(2.0))))      # e^-m with m = 18 - ln 2
print(f"correct loss ~ {CORRECT_LOSS:.3e}, correct target gradient ~ {-CORRECT_LOSS:.3e}\n")


def report(name, loss, grad):
    grad = np.asarray(grad, dtype=np.float64).ravel()
    ok = loss != 0.0 and grad[0] != 0.0 and abs(grad.sum()) < 1e-12
    print(f"{name:28s} loss={loss:.3e}  grad={np.array2string(grad, formatter={'float_kind': lambda v: f'{v:+.3e}'})}  "
          f"sum={grad.sum():+.2e}   -> {'OK' if ok else 'COLLAPSED (bug present)'}")


# --- 1. NumPy versions from this kit -------------------------------------------------------
z = np.array([[9.0, -9.0, -9.0]], dtype=np.float32)
t = np.array([0])
l, g = std_cross_entropy(z, t)
report("NumPy standard (this kit)", float(l[0]), g)
l, g, _ = fixed_cross_entropy(z, t)
report("NumPy fixed (this kit)", float(l[0]), g)

# --- 2. Optional: PyTorch (NOT executed when this kit was built) -----------------------------
try:
    import torch
    zt = torch.tensor([[9.0, -9.0, -9.0]], dtype=torch.float32, requires_grad=True)
    loss = torch.nn.functional.cross_entropy(zt, torch.tensor([0]))
    loss.backward()
    report("PyTorch F.cross_entropy", float(loss.item()), zt.grad.numpy())
except ImportError:
    print("PyTorch not installed: skipped (optional).")

# --- 3. Optional: JAX + optax (NOT executed when this kit was built) ---------------------------
try:
    import jax
    import jax.numpy as jnp
    zj = jnp.array([9.0, -9.0, -9.0], dtype=jnp.float32)

    def jax_ce(zz):
        return -jax.nn.log_softmax(zz)[0]

    loss, grad = jax.value_and_grad(jax_ce)(zj)
    report("JAX -log_softmax(z)[0]", float(loss), np.asarray(grad))
except ImportError:
    print("JAX not installed: skipped (optional).")

print("\nHow to read: 'COLLAPSED' means the loss or target gradient is exactly 0 or the gradient does")
print("not sum to 0.  The paper reports this behaviour for standard JAX and optax functions;")
print("whether your PyTorch / TensorFlow version shows it is for you to find out with this script.")
