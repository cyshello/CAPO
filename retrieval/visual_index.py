"""Cached per-video frame-embedding index (PHASE2_3 §6)."""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
import traceback
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from surrogate_rollout import config
from surrogate_rollout.schemas import sha256_json


_PROCESS_ENVIRONMENT_LOCK = threading.Lock()
_ISOLATED_EMBEDDER_REGISTRY_LOCK = threading.Lock()
_LIVE_ISOLATED_EMBEDDERS: set["ProcessIsolatedSiglipEmbedder"] = set()


def frames_source_hash(frames_dir: str) -> str:
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
    frame_names: list[str]
    clip_ids: list[str]
    timestamps: list[float]
    embeddings: np.ndarray
    key: dict

    @property
    def all_clip_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for clip_id in self.clip_ids:
            seen.setdefault(clip_id)
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
    timestamps = [i * duration / n for i in range(n)]
    clip_ids = [clip_key_for_time(t, duration, clip_secs) for t in timestamps]
    embeddings = np.asarray(
        embed_images_fn([os.path.join(frames_dir, f) for f in names]),
        dtype=np.float32,
    )
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)
    key = index_cache_key(
        video_id=video_id,
        frames_dir=frames_dir,
        model_id=model_id,
        sample_fps=sample_fps,
        clip_secs=clip_secs,
        preprocessing_version=preprocessing_version,
    )
    return VisualIndex(
        video_id=video_id, model_id=model_id, frame_names=names,
        clip_ids=clip_ids, timestamps=timestamps, embeddings=embeddings, key=key)


def index_dir(key: dict, root: str | None = None) -> str:
    root = root or config.VISUAL_INDEX_ROOT
    return os.path.join(root, key["video_id"], sha256_json(key)[:16])


def save_visual_index(index: VisualIndex, root: str | None = None) -> str:
    directory = index_dir(index.key, root)
    os.makedirs(directory, exist_ok=True)
    np.savez_compressed(
        os.path.join(directory, "embeddings.npz"), embeddings=index.embeddings)
    with open(os.path.join(directory, "meta.json"), "w") as handle:
        json.dump({
            "video_id": index.video_id,
            "model_id": index.model_id,
            "frame_names": index.frame_names,
            "clip_ids": index.clip_ids,
            "timestamps": index.timestamps,
            "key": index.key,
        }, handle)
    return directory


def load_visual_index(key: dict, root: str | None = None) -> VisualIndex | None:
    directory = index_dir(key, root)
    meta_path = os.path.join(directory, "meta.json")
    embeddings_path = os.path.join(directory, "embeddings.npz")
    if not (os.path.exists(meta_path) and os.path.exists(embeddings_path)):
        return None
    with open(meta_path) as handle:
        meta = json.load(handle)
    if meta["key"] != key:
        raise ValueError(f"visual index cache key mismatch in {directory}")
    embeddings = np.load(embeddings_path)["embeddings"]
    return VisualIndex(
        video_id=meta["video_id"], model_id=meta["model_id"],
        frame_names=meta["frame_names"], clip_ids=meta["clip_ids"],
        timestamps=meta["timestamps"], embeddings=embeddings, key=key)


class SiglipEmbedder:
    """Lazy SigLIP image/text embedder."""

    def __init__(
        self,
        model_id: str = config.VISUAL_INDEX_MODEL_ID,
        device: str | None = None,
        batch_size: int = config.VISUAL_INDEX_BATCH_SIZE,
    ) -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        self._device = device
        self._model = None
        self._processor = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _load(self):
        with self._load_lock:
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

        with self._inference_lock:
            model, processor = self._load()
            outputs = []
            with torch.no_grad():
                for index in range(0, len(paths), self.batch_size):
                    images = [
                        Image.open(path).convert("RGB")
                        for path in paths[index:index + self.batch_size]
                    ]
                    inputs = processor(
                        images=images, return_tensors="pt").to(self._device)
                    outputs.append(model.get_image_features(**inputs).cpu().numpy())
        embeddings = np.concatenate(outputs, axis=0).astype(np.float32)
        return embeddings / np.clip(
            np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        with self._inference_lock:
            model, processor = self._load()
            with torch.no_grad():
                inputs = processor(
                    text=list(texts), padding="max_length",
                    return_tensors="pt").to(self._device)
                embeddings = model.get_text_features(
                    **inputs).cpu().numpy().astype(np.float32)
        return embeddings / np.clip(
            np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)

    def close(self) -> None:
        with self._inference_lock:
            with self._load_lock:
                loaded = self._model is not None or self._processor is not None
                self._model = None
                self._processor = None
            if loaded and str(self._device or "").startswith("cuda"):
                import gc
                import torch

                gc.collect()
                torch.cuda.empty_cache()


class IsolatedSiglipWorkerError(RuntimeError):
    pass


def _isolated_siglip_worker_main(
    connection, physical_gpu: str, model_id: str, batch_size: int,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_gpu
    embedder = SiglipEmbedder(
        model_id=model_id, device="cuda:0", batch_size=batch_size)
    try:
        connection.send({
            "status": "ready", "pid": os.getpid(),
            "physical_gpu": physical_gpu,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "logical_device": "cuda:0",
        })
        while True:
            message = connection.recv()
            operation = message.get("operation")
            if operation == "close":
                connection.send({"status": "closed"})
                return
            try:
                if operation == "embed_images":
                    value = embedder.embed_images(tuple(message["values"]))
                elif operation == "embed_texts":
                    value = embedder.embed_texts(tuple(message["values"]))
                else:
                    raise ValueError(f"unknown SigLIP worker operation: {operation}")
                connection.send({"status": "ok", "value": value})
            except BaseException as exc:
                connection.send({
                    "status": "error", "error_type": type(exc).__name__,
                    "error": str(exc), "traceback": traceback.format_exc(),
                })
                return
    except (EOFError, BrokenPipeError):
        return
    finally:
        embedder.close()
        connection.close()


class ProcessIsolatedSiglipEmbedder:
    """Serialize SigLIP requests through one GPU-owning spawn subprocess."""

    def __init__(
        self, *,
        physical_gpu: str,
        model_id: str = config.VISUAL_INDEX_MODEL_ID,
        batch_size: int = config.VISUAL_INDEX_BATCH_SIZE,
        startup_timeout_seconds: float = 30.0,
        worker_target: Callable = _isolated_siglip_worker_main,
    ) -> None:
        if not physical_gpu or "," in physical_gpu:
            raise ValueError("physical_gpu must be exactly one GPU ID")
        self.model_id = model_id
        self.batch_size = batch_size
        self.physical_gpu = physical_gpu
        self.logical_device = "cuda:0"
        self._lock = threading.Lock()
        self._closed = False
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        self._connection = parent_connection
        self._process = context.Process(
            target=worker_target,
            args=(child_connection, physical_gpu, model_id, batch_size),
            name=f"siglip-gpu-{physical_gpu}", daemon=False)
        with _PROCESS_ENVIRONMENT_LOCK:
            previous_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = physical_gpu
            try:
                self._process.start()
            finally:
                if previous_visible is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = previous_visible
        child_connection.close()
        if not self._connection.poll(startup_timeout_seconds):
            self._process.terminate()
            self._process.join(timeout=5.0)
            self._connection.close()
            self._closed = True
            raise IsolatedSiglipWorkerError(
                "timed out starting isolated SigLIP worker")
        try:
            ready = self._connection.recv()
        except (EOFError, BrokenPipeError, OSError) as exc:
            self._process.join(timeout=5.0)
            self._connection.close()
            self._closed = True
            raise IsolatedSiglipWorkerError(
                "isolated SigLIP worker exited during startup") from exc
        if ready.get("status") != "ready" or \
                ready.get("cuda_visible_devices") != physical_gpu or \
                ready.get("logical_device") != "cuda:0":
            self.close()
            raise IsolatedSiglipWorkerError(
                f"invalid isolated SigLIP worker startup: {ready}")
        self.startup_metadata = dict(ready)
        with _ISOLATED_EMBEDDER_REGISTRY_LOCK:
            _LIVE_ISOLATED_EMBEDDERS.add(self)

    @property
    def worker_pid(self) -> int | None:
        return self._process.pid

    @property
    def is_alive(self) -> bool:
        return self._process.is_alive()

    def _request(self, operation: str, values: Sequence[str]) -> np.ndarray:
        with self._lock:
            if self._closed or not self._process.is_alive():
                raise IsolatedSiglipWorkerError(
                    "isolated SigLIP worker is not running")
            try:
                self._connection.send({
                    "operation": operation, "values": tuple(values)})
                response = self._connection.recv()
            except (EOFError, BrokenPipeError, OSError) as exc:
                raise IsolatedSiglipWorkerError(
                    f"isolated SigLIP worker exited during {operation}") from exc
            if response.get("status") != "ok":
                raise IsolatedSiglipWorkerError(
                    f"isolated SigLIP {operation} failed: "
                    f"{response.get('error_type')}: {response.get('error')}")
            return np.asarray(response["value"], dtype=np.float32)

    def embed_images(self, paths: Sequence[str]) -> np.ndarray:
        return self._request("embed_images", paths)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        return self._request("embed_texts", texts)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._process.is_alive():
                try:
                    self._connection.send({"operation": "close"})
                    if self._connection.poll(10.0):
                        self._connection.recv()
                except (EOFError, BrokenPipeError, OSError):
                    pass
            self._process.join(timeout=10.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5.0)
            self._connection.close()
            with _ISOLATED_EMBEDDER_REGISTRY_LOCK:
                _LIVE_ISOLATED_EMBEDDERS.discard(self)


def close_all_isolated_siglip_embedders() -> tuple[dict[str, object], ...]:
    with _ISOLATED_EMBEDDER_REGISTRY_LOCK:
        embedders = tuple(_LIVE_ISOLATED_EMBEDDERS)
    records = []
    for embedder in embedders:
        pid = embedder.worker_pid
        physical_gpu = embedder.physical_gpu
        embedder.close()
        records.append({
            "pid": pid, "physical_gpu": physical_gpu,
            "released": not embedder.is_alive,
        })
    return tuple(records)


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
