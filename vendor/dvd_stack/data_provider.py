"""Dataset provider — component 1.

The `providers` package lives beside this module (vendored from longVideoPO),
so it is importable as soon as this directory is on sys.path, which
`surrogate_rollout.config.PROMPT_SENS_ROOT` guarantees.

Common sample schema (see providers/base.py):
    dataset, sample_id, video_path, duration_sec, split, task_format,
    question, options, answer, answer_available, extra
"""

from __future__ import annotations

import os
import sys

_VENDOR_ROOT = os.path.dirname(os.path.abspath(__file__))
if _VENDOR_ROOT not in sys.path:
    sys.path.append(_VENDOR_ROOT)

from providers.registry import load_provider  # noqa: E402
from providers.base import BaseProvider  # noqa: E402

__all__ = ["load_provider", "BaseProvider", "get_provider"]


def get_provider(name: str, split: str | None = None, data_root: str | None = None):
    """Thin alias over providers.registry.load_provider.

    name: "videomme" | "egolife" | "hourvideo"
    """
    return load_provider(name, data_root=data_root, split=split)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a dataset provider")
    parser.add_argument("name", nargs="?", default="videomme")
    parser.add_argument("--split", default="long")
    args = parser.parse_args()

    p = get_provider(args.name, split=args.split)
    print(repr(p))
    s = p[0]
    print("first sample:")
    for k, v in s.items():
        if k == "extra":
            continue
        print(f"  {k}: {v}")
    print(f"  video exists: {os.path.exists(s['video_path'])}")
