import os
import sys

# Make `surrogate_rollout.*` importable when pytest runs from the repo root.
_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
