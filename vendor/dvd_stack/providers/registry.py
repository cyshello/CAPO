"""Provider 레지스트리.

    from providers.registry import load_provider
    p = load_provider("hourvideo", split="sample")
    p = load_provider("videomme", split="short")
"""

from __future__ import annotations

import importlib
import os

# Vendored for meta-prompt optimization: Video-MME only. The other providers
# stay in the upstream longVideoPO repository.
PROVIDERS = {
    "videomme": "providers.videomme.VideoMMEProvider",
}

# 데이터셋별 기본 data_root.
# $DATA_ROOT 환경변수가 설정돼 있으면 $DATA_ROOT/<name>_data가 우선한다.
# (현재 물리 배치: HourVideo는 /hub_data2, Video-MME는 /hub_data3 — 2026-07 용량 사정)
DEFAULT_DATA_ROOTS = {
    "videomme": os.environ.get(
        "SR_VIDEOMME_DATA_ROOT", "/hub_data3/videomme_data"),
}

# 데이터셋별 기본 split (인터페이스 기본값 "test"가 없는 데이터셋 대비)
DEFAULT_SPLITS = {
    "videomme": "short",
}


def _resolve_data_root(name: str) -> str:
    env_root = os.environ.get("DATA_ROOT")
    if env_root:
        candidate = os.path.join(env_root, f"{name}_data")
        if os.path.isdir(candidate):
            return candidate
    return DEFAULT_DATA_ROOTS[name]


def load_provider(name: str, data_root: str | None = None, split: str | None = None):
    """이름만 넣으면 해당 provider 인스턴스 반환.

    data_root 생략 시 $DATA_ROOT/<name>_data (없으면 데이터셋별 기본 경로).
    split 생략 시 데이터셋별 기본 split.
    """
    if name not in PROVIDERS:
        raise KeyError(f"unknown provider {name!r}; available: {list(PROVIDERS)}")
    module_path, cls_name = PROVIDERS[name].rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)
    if data_root is None:
        data_root = _resolve_data_root(name)
    if split is None:
        split = DEFAULT_SPLITS.get(name, "test")
    return cls(data_root=data_root, split=split)
