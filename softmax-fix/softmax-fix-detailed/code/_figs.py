"""
_figs.py
Tiny helper shared by the lesson scripts.

* Works on machines without a display (uses the Agg backend unless SHOW=1).
* save(fig, "name.png") writes into ../docs/figures/ next to the markdown guides.
* If matplotlib is not installed the lessons still run; figures are skipped.
"""
import os

try:
    import matplotlib
    if os.environ.get("SHOW", "0") != "1":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:  # pragma: no cover
    plt = None
    HAVE_MPL = False

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "figures")


def save(fig, name):
    """Save a figure to docs/figures and optionally show it (SHOW=1)."""
    if not HAVE_MPL:
        print(f"[figure skipped, matplotlib missing: {name}]")
        return
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"[figure saved: docs/figures/{name}]")
    if os.environ.get("SHOW", "0") == "1":
        plt.show()
    plt.close(fig)
