"""공통 provider 인터페이스.

모든 provider는 공통 sample 스키마를 따르는 dict를 반환한다:

{
    "dataset": str,            # "hourvideo" | "videomme" | "egolife" (Phase 2에서 추가 예정)
    "sample_id": str,
    "video_path": str,         # 단일 mp4. 단, egolife는 인물 egocentric 스트림 디렉터리
    "duration_sec": float,     # 모를 경우 -1.0
    "split": str,
    "task_format": str,        # "mcq"

    "question": str | None,
    "options": list[str] | None,
    "answer": str | None,      # 정답 라벨 ("A"~"E"), 비공개면 None
    "answer_available": bool,

    "extra": dict,             # 데이터셋 원본 필드 보관용
}
"""

from __future__ import annotations


class BaseProvider:
    """모든 데이터셋 provider의 공통 인터페이스."""

    name: str = "base"

    def __init__(self, data_root: str, split: str = "test"):
        self.data_root = data_root
        self.split = split

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        raise NotImplementedError

    def available_splits(self) -> list[str]:
        raise NotImplementedError

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(split={self.split!r}, n={len(self)}, data_root={self.data_root!r})"
