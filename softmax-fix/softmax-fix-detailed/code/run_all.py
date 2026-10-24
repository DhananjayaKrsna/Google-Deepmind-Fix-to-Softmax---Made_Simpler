"""
run_all.py
Runs the tests and every lesson script, prints PASS/FAIL and the run time of each,
and regenerates all figures in ../docs/figures.

    python run_all.py            # everything (about 1.5 minutes)
    python run_all.py --quick    # skip the two slower training lessons (07, 08)
"""
import subprocess
import sys
import time
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    "test_softmax_fix.py",
    "lesson01_floating_point.py",
    "lesson02_softmax_from_scratch.py",
    "lesson03_gradients_and_sgd.py",
    "lesson04_softmax_collapse.py",
    "lesson05_softmax_cancellation.py",
    "lesson06_adam_from_scratch.py",
    "lesson07_margin_growth.py",
    "lesson08_distillation_drift.py",
    "check_my_framework.py",
]
SLOW = {"lesson07_margin_growth.py", "lesson08_distillation_drift.py"}

quick = "--quick" in sys.argv
failed = []
for name in SCRIPTS:
    if quick and name in SLOW:
        print(f"SKIP  {name} (--quick)")
        continue
    t0 = time.time()
    proc = subprocess.run([sys.executable, name], cwd=HERE, capture_output=True, text=True)
    dt = time.time() - t0
    ok = proc.returncode == 0
    print(f"{'PASS' if ok else 'FAIL'}  {name:38s} {dt:6.1f}s")
    if not ok:
        failed.append(name)
        print(proc.stdout[-1500:])
        print(proc.stderr[-1500:])

print("\nAll good." if not failed else f"\nFailed: {failed}")
sys.exit(1 if failed else 0)
