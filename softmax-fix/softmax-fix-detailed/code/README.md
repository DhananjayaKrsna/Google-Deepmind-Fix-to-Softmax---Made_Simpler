# Code

Only NumPy and Matplotlib are required (`pip install -r ../requirements.txt`). Run from this folder.

| File | Purpose |
|---|---|
| `softmax_fix.py` | Library: standard and fixed softmax, cross-entropy, soft-target gradients, bf16 emulation, cliff formula |
| `optim.py` | SGD and AdamW from scratch |
| `_figs.py` | Figure helper (saves to `../docs/figures`; set `SHOW=1` to open windows) |
| `test_softmax_fix.py` | 26 assert checks |
| `lesson01_floating_point.py` | Rounding, absorption, cancellation |
| `lesson02_softmax_from_scratch.py` | Softmax, log-sum-exp, max-shift |
| `lesson03_gradients_and_sgd.py` | Cross-entropy gradient, tiny training run |
| `lesson04_softmax_collapse.py` | Bug 1: cliff and zero-sum failure |
| `lesson05_softmax_cancellation.py` | Bug 2: rounding noise in $q-p$ |
| `lesson06_adam_from_scratch.py` | Adam, bias correction, quiet periods |
| `lesson07_margin_growth.py` | Margins, weight decay, $\epsilon$ sweep (about 30 s) |
| `lesson08_distillation_drift.py` | Student drift from a perfect teacher (about 30 s) |
| `check_my_framework.py` | Probe your own framework (NumPy runs; PyTorch/JAX blocks untested) |
| `run_all.py` | Runs tests and all lessons (`--quick` for a faster pass) |

Lesson scripts use `# %%` cell markers, so they run cell by cell in VS Code, Spyder or Jupyter (via jupytext), or top to bottom with `python lessonNN_*.py`.

Note: NumPy has no bfloat16, so bf16 is emulated by rounding float32 (round-to-nearest-even). float16 is used as a stand-in for low precision in lesson 08.
