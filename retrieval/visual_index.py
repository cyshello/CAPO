"""Cached per-video frame-embedding index (PHASE2_3 §6).

Embeds the same decoded frames the DVD captioner consumes (frames_fps{f}) with
SigLIP once per video, preserving the frame-to-clip mapping used by
dvd_captioning._build_clips (t_i = i * duration / n, contiguous CLIP_SECS
windows). Embeddings are L2-normalized at build time; retrieval is a batched
matrix product against normalized text embeddings. The captioner VLM is never
loaded for retrieval.

Cache key: video_id, vision model, frame-sampling config, frame-source hash,
preprocessing version. The index is deterministic given those inputs; the
embedder is injectable so tests build indexes without model weights.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from surrogate_rollout import config
from surrogate_rollout.schemas import sha256_json


def frames_source_hash(frames_dir: str) -> str:
    """Identity of the exact frame files (sorted name + size)."""
    entries = sorted(
        (f, os.path.getsize(os.path.join(frames_dir, f)))
        for f in os.listdir(frames_dir) if f.endswith(".jpg")
    )
    return sha256_json(entries)


def index_cache_key(
    *,
    video_id: str,
    frames_dir: str,
    model_id: str,
    sample_fps: float,
    clip_secs: int,
    preprocessing_version: str,
) -> dict:
    return {
        "video_id": video_id,
        "vision_model_id": model_id,
        "frame_sampling_config": {"sample_fps": sample_fps, "clip_secs": clip_secs},
        "frame_source_hash": frames_source_hash(frames_dir),
        "preprocessing_version": preprocessing_version,
    }


@dataclass
class VisualIndex:
    video_id: str
    model_id: str
    frame_names: list[str]        # sorted jpg basenames
    clip_ids: list[str]           # per-frame clip key
    timestamps: list[float]       # per-frame seconds
    embeddings: np.ndarray        # (n_frames, dim) float32, L2-normalized
    key: dict

    @property
    def all_clip_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for c in self.clip_ids:
            seen.setdefault(c)
        return list(seen)


def clip_key_for_time(t: float, duration: float, clip_secs: int) -> str:
    start = int(t // clip_secs) * clip_secs
    end = min(start + clip_secs, duration)
    return f"{int(start)}_{int(end)}"


def build_visual_index(
    *,
    video_id: str,
    frames_dir: str,
    duration: float,
    embed_images_fn: Callable[[list[str]], np.ndarray],
    model_id: str,
    sample_fps: float = config.SAMPLE_FPS,
    clip_secs: int = config.CLIP_SECS,
    preprocessing_version: str = config.VISUAL_INDEX_PREPROCESSING_VERSION,
) -> VisualIndex:
    names = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    if not names:
        raise ValueError(f"no frames in {frames_dir}")
    n = len(names)
    ts = [i * duration / n for i in range(n)]
    clip_ids = [clip_key_for_time(t, duration, clip_secs) for t in ts]

    emb = np.asarray(
        embed_images_fn([os.path.join(frames_dir, f) for f in names]),
        dtype=np.float32,
    )
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norms, 1e-12, None)

    key = index_cache_key(
        video_id=video_id, frames_dir=frames_dir, model_id=model_id,
        sample_fps=sample_fps, clip_secs=clip_secs,
        preprocessing_version=preprocessing_version,
    )
    return VisualIndex(video_id=video_id, model_id=model_id, frame_names=names,
                       clip_ids=clip_ids, timestamps=ts, embeddings=emb, key=key)


# ------------------------------ persistence -------------------------------- #
def index_dir(key: dict, root: str | None = None) -> str:
    root = root or config.VISUAL_INDEX_ROOT
    return os.path.join(root, key["video_id"], sha256_json(key)[:16])


def save_visual_index(index: VisualIndex, root: str | None = None) -> str:
    d = index_dir(index.key, root)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, "embeddings.npz"), embeddings=index.embeddings)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({
            "video_id": index.video_id, "model_id": index.model_id,
            "frame_names": index.frame_names, "clip_ids": index.clip_ids,
            "timestamps": index.timestamps, "key": index.key,
        }, f)
    return d


def load_visual_index(key: dict, root: str | None = None) -> VisualIndex | None:
    d = index_dir(key, root)
    meta_p, emb_p = os.path.join(d, "meta.json"), os.path.join(d, "embeddings.npz")
    if not (os.path.exists(meta_p) and os.path.exists(emb_p)):
        return None
    with open(meta_p) as f:
        meta = json.load(f)
    if meta["key"] != key:
        raise ValueError(f"visual index cache key mismatch in {d}")  # abort, never reuse
    emb = np.load(emb_p)["embeddings"]
    return VisualIndex(video_id=meta["video_id"], model_id=meta["model_id"],
                       frame_names=meta["frame_names"], clip_ids=meta["clip_ids"],
                       timestamps=meta["timestamps"], embeddings=emb, key=key)


# ------------------------------ SigLIP embedder ----------------------------- #
class SiglipEmbedder:
    """Lazy SigLIP image/text embedder (HF transformers, local cache)."""

    def __init__(self, model_id: str = config.VISUAL_INDEX_MODEL_ID,
                 device: str | None = None,
                 batch_size: int = config.VISUAL_INDEX_BATCH_SIZE) -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        self._device = device
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoProcessor

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = AutoModel.from_pretrained(
                self.model_id, torch_dtype=torch.float32).to(self._device).eval()
            self._processor = AutoProcessor.from_pretrained(self.model_id)
        return self._model, self._processor

    def embed_images(self, paths: Sequence[str]) -> np.ndarray:
        import torch
        from PIL import Image

        model, processor = self._load()
        out = []
        with torch.no_grad():
            for i in range(0, len(paths), self.batch_size):
                imgs = [Image.open(p).convert("RGB")
                        for p in paths[i:i + self.batch_size]]
                inputs = processor(images=imgs, return_tensors="pt").to(self._device)
                feats = model.get_image_features(**inputs)
                out.append(feats.cpu().numpy())
        emb = np.concatenate(out, axis=0).astype(np.float32)
        return emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        model, processor = self._load()
        with torch.no_grad():
            inputs = processor(text=list(texts), padding="max_length",
                               return_tensors="pt").to(self._device)
            feats = model.get_text_features(**inputs).cpu().numpy().astype(np.float32)
        return feats / np.clip(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12, None)


def load_or_build_visual_index(
    *,
    video_id: str,
    frames_dir: str,
    duration: float,
    embedder: SiglipEmbedder | None = None,
    root: str | None = None,
    model_id: str = config.VISUAL_INDEX_MODEL_ID,
) -> VisualIndex:
    key = index_cache_key(
        video_id=video_id, frames_dir=frames_dir, model_id=model_id,
        sample_fps=config.SAMPLE_FPS, clip_secs=config.CLIP_SECS,
        preprocessing_version=config.VISUAL_INDEX_PREPROCESSING_VERSION,
    )
    cached = load_visual_index(key, root)
    if cached is not None:
        return cached
    embedder = embedder or SiglipEmbedder(model_id)
    index = build_visual_index(
        video_id=video_id, frames_dir=frames_dir, duration=duration,
        embed_images_fn=embedder.embed_images, model_id=model_id,
    )
    save_visual_index(index, root)
    return index
