"""Video-MME provider.

데이터 레이아웃 (data_root = .../videomme_data):
    Video-MME/videomme/test-00000-of-00001.parquet  # QA 2700개 (정답 전부 공개)
    Video-MME/videos/{short,medium,long}/{videoID}.mp4
    Video-MME/subtitles/subtitle/{videoID}.srt   # 자막 없는 영상도 있음 (744/900)

split은 공식 duration 분할("short"/"medium"/"long")을 그대로 사용하고,
해당 폴더의 영상에 속한 QA만 순회한다. "all"을 주면 세 split 전체 순회.
Video-MME는 정답이 전부 공개돼 있으므로 answer_available=True 고정.
"""

from __future__ import annotations

import os

import pandas as pd

from .base import BaseProvider

_SPLITS = ["short", "medium", "long"]

# parquet 원본 그대로 extra에 보관할 컬럼들
_EXTRA_COLS = ["video_id", "domain", "sub_category", "url", "videoID", "task_type"]


class VideoMMEProvider(BaseProvider):
    name = "videomme"

    def __init__(self, data_root: str, split: str = "short"):
        if split in ("all", "test"):  # 편의 alias: 세 split 전체
            splits = _SPLITS
            split = "all"
        elif split in _SPLITS:
            splits = [split]
        else:
            raise ValueError(
                f"unknown split {split!r} for videomme; available: {_SPLITS} (or 'all')"
            )
        super().__init__(data_root, split)

        repo = os.path.join(data_root, "Video-MME")
        parquet_path = os.path.join(repo, "videomme", "test-00000-of-00001.parquet")
        df = pd.read_parquet(parquet_path)
        df = df[df["duration"].isin(splits)].reset_index(drop=True)

        self._videos_dir = os.path.join(repo, "videos")
        self._subtitles_dir = os.path.join(repo, "subtitles", "subtitle")
        self._samples = [self._to_sample(row) for row in df.itertuples(index=False)]

    def _to_sample(self, row) -> dict:
        duration_split = row.duration  # short | medium | long
        video_path = os.path.join(
            self._videos_dir, duration_split, f"{row.videoID}.mp4"
        )
        subtitle_path = os.path.join(self._subtitles_dir, f"{row.videoID}.srt")
        extra = {c: getattr(row, c) for c in _EXTRA_COLS}
        extra["duration_split"] = duration_split
        extra["subtitle_path"] = subtitle_path if os.path.exists(subtitle_path) else None
        return {
            "dataset": self.name,
            "sample_id": row.question_id,
            "video_path": video_path,
            "duration_sec": -1.0,  # parquet에는 초 단위 길이가 없음 (short/medium/long 분류만 제공)
            "split": duration_split,
            "task_format": "mcq",
            "question": row.question,
            "options": list(row.options),
            "answer": row.answer,
            "answer_available": True,
            "extra": extra,
        }

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        return self._samples[idx]

    def available_splits(self) -> list[str]:
        return list(_SPLITS)
