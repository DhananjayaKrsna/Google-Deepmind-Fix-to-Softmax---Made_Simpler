"""
optim.py
Minimal optimizers written for reading, not speed.

  SGD    : theta <- theta - lr * g
  AdamW  : the update used in the paper
              theta <- theta - lr * ( m_hat / (sqrt(v_hat) + eps) + wd * theta )
           with bias-corrected moving averages m_hat, v_hat.
Setting wd = 0 gives plain Adam.  Parameters and state keep the dtype of `theta`.
"""
import numpy as np


class SGD:
    def __init__(self, lr):
        self.lr = lr

    def step(self, theta, g):
        return theta - self.lr * g


class AdamW:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.98, eps=1e-8, wd=0.0, dtype=np.float32):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, beta1, beta2, eps, wd
        self.dtype = dtype
        self.t = 0
        self.m = None
        self.v = None

    def step(self, theta, g):
        dt = self.dtype
        g = g.astype(dt)
        if self.m is None:
            self.m = np.zeros_like(theta, dtype=dt)
            self.v = np.zeros_like(theta, dtype=dt)
        self.t += 1
        b1, b2 = dt(self.b1), dt(self.b2)
        self.m = b1 * self.m + (dt(1) - b1) * g
        self.v = b2 * self.v + (dt(1) - b2) * g * g
        m_hat = self.m / dt(1 - self.b1 ** self.t)
        v_hat = self.v / dt(1 - self.b2 ** self.t)
        update = m_hat / (np.sqrt(v_hat) + dt(self.eps)) + dt(self.wd) * theta
        return theta - dt(self.lr) * update

    def last_normalized_step(self):
        """|m_hat| / (sqrt(v_hat) + eps) for the most recent step, handy for plots."""
        dt = self.dtype
        m_hat = self.m / dt(1 - self.b1 ** self.t)
        v_hat = self.v / dt(1 - self.b2 ** self.t)
        return np.abs(m_hat) / (np.sqrt(v_hat) + dt(self.eps))
