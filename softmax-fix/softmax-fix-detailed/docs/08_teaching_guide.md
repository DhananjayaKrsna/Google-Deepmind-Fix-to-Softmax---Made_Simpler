# Guide 08: Teaching Guide (for the person explaining this to a beginner)

Use this file to run the material as a short course. Total time: three sessions of about 90 minutes.

---

## 1. Learning goals

After the course a learner should be able to:

1. Explain why `1 + 1e-8 == 1` in float32 and what a "precision cliff" is.
2. Write softmax and log-sum-exp stably and say why the max-shift works.
3. Derive $\partial\mathcal L/\partial z=q-p$ and state why it sums to zero.
4. Describe Softmax Collapse and Softmax Cancellation in one sentence each.
5. Explain why Adam can turn a tiny, wrong gradient into a real parameter move.
6. Separate what this kit verified from what only the paper reports.

---

## 2. Session plan

| Session | Topics | Guides | Lessons to run live |
|---|---|---|---|
| 1. Numbers | floating point, softmax, log-sum-exp | 01, 02 | 01, 02 |
| 2. The bugs | cross-entropy, collapse, cancellation | 03, 04, 05 | 03, 04, 05 |
| 3. Dynamics | Adam, margins, distillation drift, limits | 06, 07 | 06, 07, 08 |

Suggested pacing inside a session: 15 minutes of explanation, 20 minutes running the lesson script cell by cell (`# %%` markers), 15 minutes of exercises, 10 minutes of questions.

---

## 3. Analogies that work

| Idea | Analogy | Where it breaks |
|---|---|---|
| Floating point | A ruler with finer marks near zero and coarser marks far away | The ruler's marks are powers of two, not evenly spread |
| Absorption (`1 + tiny == 1`) | Adding one grain of sand to a truck scale | The scale does not round randomly; it rounds to nearest, ties to even |
| Margin | How many steps ahead the winning class is | It is measured in log-odds, not steps |
| Adam's scale-free step | A cruise control that always drives at the set speed, even on a gentle slope | The speed is set by $\eta$, and the control is blind below $\epsilon$ |
| Cancellation | Weighing a ship with and without the captain, then subtracting | Each weighing has its own rounding error |

---

## 4. Misconceptions to listen for

* "More precision always fixes it." Higher precision moves the cliff (36.74 for fp64) but does not remove it. The fix is a different computation.
* "The model is wrong." The model's maths is fine; the arithmetic that evaluates its gradient is wrong.
* "A tiny gradient does nothing." Under Adam it can move parameters (Guide 06).
* "The toy proved the paper." It checked ingredients only (Guide 07, section 6).

---

## 5. Quiz

1. In float32, below which $x$ does `1 + x == 1` hold, and what is the cliff in nats?
2. Why subtract the maximum before exponentiating?
3. What does the gradient of softmax cross-entropy with respect to logits sum to?
4. Give the formula for the margin $m$ and the loss in terms of $m$.
5. In one sentence: what is Softmax Collapse?
6. In one sentence: what is Softmax Cancellation?
7. For Adam with $\beta_1=0.9$, $\beta_2=0.98$, what bounds the per-parameter step after a long quiet period?
8. What is the heuristic equilibrium margin and what does it depend on?
9. Name one thing the kit does not verify.

<details><summary>Answers</summary>

1. For float32 (23 fraction bits), `1 + x == 1` iff $x\le2^{-24}$ (ties to even). The cliff is $c=(p+1)\ln2=24\ln2=16.64$ nats.
2. It makes the largest exponent 0, so $e^{\cdot}\le1$ and nothing overflows, and the sum is at least 1, so nothing underflows to a zero denominator. The result is unchanged because softmax is shift invariant.
3. Zero.
4. $m=z_t-\mathrm{LSE}_{j\ne t}(z_j)$; loss $=\mathrm{softplus}(-m)=\log(1+e^{-m})$.
5. Past the precision cliff the target gradient is rounded to exactly zero while wrong-class gradients stay positive, so the gradient no longer sums to zero.
6. The soft-target gradient $q-p$ is computed from two separately rounded softmaxes, so for near-equal $p$ and $q$ the difference is dominated by rounding error.
7. About $(1-\beta_1)/\sqrt{1-\beta_2}=0.707$ times $\eta$.
8. $m^\star\approx\log(1/\epsilon)$, depending mainly on Adam's $\epsilon$ and weight decay.
9. Any of: the Slingshot loss spikes, grokking, accuracy loss in the paper's network, bf16 on real hardware (NumPy has no bf16, so it is emulated), PyTorch/JAX behaviour.
</details>

---

## 6. Assignments

1. **Reproduce a cliff.** Pick a dtype and find the margin where `std_cross_entropy` returns a zero target gradient. Compare with the formula.
2. **Break and repair.** Replace `fixed_cross_entropy` by a version without the max-shift and find the smallest logit that overflows.
3. **Framework check.** Run `code/check_my_framework.py` in an environment with PyTorch or JAX and report whether the framework's loss has the collapse. (The PyTorch/JAX blocks in that script are untested by the author.)
4. **Critique.** Write half a page on why a toy run that shows no spike does not refute the paper.

---

## 7. Discussion on limitations (be honest with learners)

* The kit uses NumPy only; bf16 is emulated by rounding float32.
* The toy models are linear and small. They show mechanisms, not the paper's results.
* The equilibrium formula is a heuristic.
* The paper is a preprint, and its numbers here are quoted, not reproduced.

Next: [Guide 09: Cheat sheet](09_cheat_sheet.md).
