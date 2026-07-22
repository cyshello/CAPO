"""Small captioner interface used by surrogate rollout experiments."""

from __future__ import annotations

import io
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

ImageInput = Union[str, os.PathLike[str], "PILImage"]
ImagesInput = Union[ImageInput, list[ImageInput], tuple[ImageInput, ...]]


class BaseCaptioner(ABC):
    """Code-level captioning API: image(s) plus prompt to text."""

    name = "base"

    def caption(
        self,
        images: ImagesInput,
        prompt: str,
        *,
        max_tokens: int | None = None,
        **kwargs,
    ) -> str:
        return self.caption_batch([images], prompt, max_tokens=max_tokens, **kwargs)[0]

    @abstractmethod
    def caption_batch(
        self,
        images_list: list[ImagesInput],
        prompt: str | list[str],
        *,
        max_tokens: int | None = None,
        **kwargs,
    ) -> list[str]:
        """Caption several image groups.

        Each item in images_list may be one image or a list of images. `prompt`
        may be a single shared prompt or one prompt per item.
        """


def as_image_list(images: ImagesInput) -> list[ImageInput]:
    if images is None:
        return []
    if isinstance(images, (list, tuple)):
        return list(images)
    return [images]


def broadcast_prompt(prompt: str | list[str], n: int) -> list[str]:
    if isinstance(prompt, str):
        return [prompt] * n
    if len(prompt) != n:
        raise ValueError(f"prompt length {len(prompt)} does not match batch size {n}")
    return list(prompt)


def is_url(value: object) -> bool:
    return isinstance(value, str) and (
        value.startswith("http://") or value.startswith("https://")
    )


def load_pil(image: ImageInput) -> "PILImage":
    from PIL import Image

    if hasattr(image, "convert"):
        return image.convert("RGB")
    if is_url(image):
        import urllib.request

        with urllib.request.urlopen(str(image)) as response:
            return Image.open(io.BytesIO(response.read())).convert("RGB")
    return Image.open(Path(image)).convert("RGB")


def resolve_local_model(model_id: str) -> str:
    """Prefer the local Hugging Face cache; fall back to MODEL_PATH or model_id."""

    explicit = os.environ.get("MODEL_PATH")
    if explicit:
        return explicit
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_id, local_files_only=True)
    except Exception:
        return model_id


def sample_video_frames(
    video_path: str | os.PathLike[str],
    *,
    max_frames: int = 8,
) -> list["PILImage"]:
    """Uniformly sample frames from a local video path as PIL images."""

    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")
    try:
        import cv2
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("sample_video_frames requires opencv-python and pillow") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        raise RuntimeError(f"cannot read frame count from video: {video_path}")

    if max_frames == 1:
        indices = [total // 2]
    else:
        indices = [
            round(i * (total - 1) / (max_frames - 1)) for i in range(max_frames)
        ]

    frames: list[PILImage] = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = cap.read()
        if ok:
            frames.append(Image.fromarray(bgr[:, :, ::-1]).convert("RGB"))
    cap.release()

    if not frames:
        raise RuntimeError(f"no frames sampled from video: {video_path}")
    return frames

