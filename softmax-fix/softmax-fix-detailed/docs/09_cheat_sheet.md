# Guide 09: One-Page Cheat Sheet

## Definitions

| Symbol | Meaning |
|---|---|
| $z$ | logits (raw scores) |
| $q=\mathrm{softmax}(z)$ | predicted probabilities |
| $p$ | target (one-hot or soft) |
| $\mathrm{LSE}(z)=\log\sum e^{z_j}$ | log-sum-exp |
| $m=z_t-\mathrm{LSE}_{j\ne t}(z_j)$ | logit margin |
| $\sigma(m)=q_t$ | sigmoid of the margin |

## Core formulas

* Stable LSE: $\mathrm{LSE}(z)=\max z+\log\sum e^{z_j-\max z}$
* Loss (hard target): $-\log q_t=\mathrm{softplus}(-m)=\log(1+e^{-m})$
* Gradient wrt logits: $g=q-p$, always $\sum g=0$
* Hard-target gradient: $g_t=-\dfrac{e^{-m}}{1+e^{-m}}\approx-e^{-m}$
* KL: $\mathrm{KL}(p\|q)=H(p,q)-H(p)$
* Weight gradient (linear layer): $\nabla_W=X^\top G/N$

## Floating point

| format | fraction bits $p$ | cliff $(p+1)\ln2$ |
|---|---|---|
| bf16 | 7 | 5.55 |
| fp16 | 10 | 7.62 |
| fp32 | 23 | 16.64 |
| fp64 | 52 | 36.74 |

Absorption: `1 + x == 1` iff $x\le2^{-(p+1)}$.

## The two bugs

| | Softmax Collapse | Softmax Cancellation |
|---|---|---|
| Where | hard-target CE | soft-target CE / distillation |
| What | target gradient rounds to 0 past the cliff | $q-p$ from two rounded softmaxes |
| Symptom | gradient not zero-sum | tiny nonzero gradient at $q=p$, drift |
| Fix | margin from differences, softplus, target gradient $=-\sum$ others | paper Equation 9 (difference form with `expm1`, `log1p`) |

## Adam in 30 seconds

$m\leftarrow\beta_1m+(1-\beta_1)g$, $v\leftarrow\beta_2v+(1-\beta_2)g^2$, bias-correct, step $=\eta\hat m/(\sqrt{\hat v}+\epsilon)$.
Normalised step $\approx|g|/(|g|+\epsilon)$. After a quiet period the step is bounded by $(1-\beta_1)/\sqrt{1-\beta_2}\,\eta$ (0.707 for 0.9 and 0.98).
Equilibrium (heuristic): $m^\star\approx\log(1/\epsilon)$.

## Verified numbers (this kit)

* Cliffs measured equal the formula: 5.55 / 16.64 / 36.74 (bf16 / fp32 / fp64).
* Adam normalised step at $\epsilon=10^{-8}$: $|g|=10^{-3}$: 1.000; $10^{-6}$: 0.990; $10^{-8}$: 0.500; $10^{-10}$: 0.010.
* Margin sweep: min margin 13.72 / 17.99 / 22.34 / 26.74 for $\log(1/\epsilon)$ = 13.82 / 18.42 / 23.03 / 27.63.
* Distillation drift (toy): 0 / 7.5e-3 / 13.8 for fp64 / fp32 / fp16 standard; exactly 0 for fixed.

## Paper-only claims

Slingshot spikes are caused by the chain above; spikes vanish with the fixed loss; Table 1 margins.

## Sanity checks to remember

* Uniform prediction over $K$ classes gives loss $\ln K$.
* Any gradient wrt logits must sum to 0.
* If student equals teacher, the gradient must be exactly 0.
