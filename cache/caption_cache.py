"""Harness-level caption-cache identity and manifest.

The vendored DVD cache tag (md5(caption_prompt||merge_prompt)[:8]) is left
untouched — changing it would orphan the existing five videos' caches
(CLAUDE.md correction: preserve as read-only legacy). This module adds the
stronger identity on top:

- CaptionCacheKey: prompt hash (sha256, full), caption-model ID, decoding
  hash, source hash (CLAUDE.md §11).
- cache_manifest.jsonl: one record per registered caption set. Legacy DVD
  caches are registered with `legacy: true, read_only: true`; new candidate
  caches (Phase 2) must be registered here before use and are keyed by the
  strong key.

Nothing here rewrites caption content. `assert_writable` is the guard every
future cache writer must call: it aborts on any attempt to write into a
registered read-only cache directory (cache-contamination protection).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime

from surrogate_rollout import config
from surrogate_rollout.schemas import CaptionCacheKey, sha256_json, sha256_text


def prompt_hash(caption_prompt: str, merge_prompt: str) -> str:
    """Full sha256 over the same string DVD's legacy md5 tag hashes."""
    return sha256_text(f"{caption_prompt}||{merge_prompt}")


def legacy_md5_tag(caption_prompt: str, merge_prompt: str) -> str:
    import hashlib

    return hashlib.md5(f"{caption_prompt}||{merge_prompt}".encode()).hexdigest()[:8]


def source_hash(video_path: str, sample_fps: float, clip_secs: int,
                subtitle_path: str | None) -> str:
    """Identity of the caption inputs other than prompt/model/decoding.

    Uses video file size (not mtime — copies keep identity) plus the sampling
    and clipping parameters and the subtitle file identity.
    """
    ident = {
        "video_basename": os.path.basename(video_path),
        "video_size_bytes": os.path.getsize(video_path) if os.path.exists(video_path) else None,
        "sample_fps": sample_fps,
        "clip_secs": clip_secs,
        "subtitle_basename": os.path.basename(subtitle_path) if subtitle_path else None,
        "subtitle_size_bytes": (os.path.getsize(subtitle_path)
                                if subtitle_path and os.path.exists(subtitle_path) else None),
    }
    return sha256_json(ident)


def build_cache_key(*, video_id: str, video_path: str, caption_prompt: str,
                    merge_prompt: str, subtitle_path: str | None,
                    segment_id: str = "*",
                    caption_model_id: str = config.CAPTION_MODEL_ID,
                    sample_fps: float = config.SAMPLE_FPS,
                    clip_secs: int = config.CLIP_SECS) -> CaptionCacheKey:
    return CaptionCacheKey(
        video_id=video_id,
        segment_id=segment_id,
        prompt_hash=prompt_hash(caption_prompt, merge_prompt),
        caption_model_id=caption_model_id,
        decoding_hash=config.decoding_hash(),
        source_hash=source_hash(video_path, sample_fps, clip_secs, subtitle_path),
    )


def build_history_aware_cache_key(
    *,
    video_id: str,
    video_path: str,
    caption_prompt: str,
    merge_prompt: str,
    subtitle_path: str | None,
    segment_id: str,
    history_hash: str,
    composed_prompt_hash: str,
    bank_version: str,
    router_version: str,
    scaffold_version: str,
    contract_version: str,
    backend_id: str,
    history_config_hash: str,
    intervention_identity_hash: str | None = None,
    caption_model_id: str = config.CAPTION_MODEL_ID,
    sample_fps: float = config.SAMPLE_FPS,
    clip_secs: int = config.CLIP_SECS,
) -> CaptionCacheKey:
    """Strong identity for a history-dependent per-segment caption.

    This is deliberately separate from ``build_cache_key``. A caller cannot
    accidentally omit history/component lineage and therefore cannot resolve
    to a legacy history-free cache entry.
    """
    required_identity = {
        "history_hash": history_hash,
        "composed_prompt_hash": composed_prompt_hash,
        "bank_version": bank_version,
        "router_version": router_version,
        "scaffold_version": scaffold_version,
        "contract_version": contract_version,
        "backend_id": backend_id,
        "history_config_hash": history_config_hash,
    }
    missing = [name for name, value in required_identity.items() if not value]
    if missing:
        raise ValueError(
            f"history-aware cache identity has empty fields: {missing}")
    key = build_cache_key(
        video_id=video_id,
        video_path=video_path,
        caption_prompt=caption_prompt,
        merge_prompt=merge_prompt,
        subtitle_path=subtitle_path,
        segment_id=segment_id,
        caption_model_id=caption_model_id,
        sample_fps=sample_fps,
        clip_secs=clip_secs,
    )
    values = asdict(key)
    values.update(
        history_hash=history_hash,
        composed_prompt_hash=composed_prompt_hash,
        bank_version=bank_version,
        router_version=router_version,
        scaffold_version=scaffold_version,
        contract_version=contract_version,
        backend_id=backend_id,
        history_config_hash=history_config_hash,
        intervention_identity_hash=intervention_identity_hash,
    )
    return CaptionCacheKey(**values)


def captions_content_hash(captions_json_path: str) -> str:
    """Content hash of the exact captions.json a vector DB was built from.
    A DB may only be reused when this hash matches (correction #5)."""
    with open(captions_json_path, "rb") as f:
        import hashlib

        return hashlib.sha256(f.read()).hexdigest()


# --------------------------------------------------------------------------- #
#                                 manifest                                     #
# --------------------------------------------------------------------------- #
def _manifest_path(path: str | None = None) -> str:
    return path or config.CACHE_MANIFEST_PATH


def load_manifest(path: str | None = None) -> list[dict]:
    p = _manifest_path(path)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def register_cache(entry: dict, path: str | None = None) -> dict:
    """Append one cache record. Required fields are validated; duplicate
    (cache_dir) registrations raise unless identical."""
    required = {"video_id", "cache_dir", "key", "read_only", "legacy"}
    missing = required - set(entry)
    if missing:
        raise ValueError(f"cache manifest entry missing fields: {sorted(missing)}")

    existing = {e["cache_dir"]: e for e in load_manifest(path)}
    prev = existing.get(entry["cache_dir"])
    if prev is not None:
        comparable = {k: prev.get(k) for k in entry if k != "registered_at"}
        if comparable != {k: entry.get(k) for k in entry if k != "registered_at"}:
            raise ValueError(
                f"cache_dir already registered with different identity: {entry['cache_dir']}"
            )
        return prev

    entry = dict(entry)
    entry.setdefault("registered_at", datetime.now().astimezone().isoformat())
    p = _manifest_path(path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry


def assert_writable(cache_dir: str, path: str | None = None) -> None:
    """Abort before any write into a read-only (legacy/incumbent) cache."""
    cache_dir = os.path.abspath(cache_dir)
    for e in load_manifest(path):
        if os.path.abspath(e["cache_dir"]) == cache_dir and e.get("read_only"):
            raise PermissionError(
                f"attempt to write into read-only caption cache: {cache_dir}"
            )


def register_legacy_dvd_caches(default_caption_prompt: str,
                               default_merge_prompt: str,
                               workspace: str | None = None,
                               manifest: str | None = None) -> list[dict]:
    """Register every existing DVD workspace caption dir as read-only legacy.

    Caches whose tag matches the current default prompts get the full strong
    key; caches produced by unknown prompt variants are registered with
    prompt_hash "unknown_legacy:<tag>" so they can never collide with a real
    key.
    """
    workspace = workspace or config.DVD_RUN_WORKSPACE
    known_tag = legacy_md5_tag(default_caption_prompt, default_merge_prompt)
    known_hash = prompt_hash(default_caption_prompt, default_merge_prompt)

    registered = []
    for video_id in sorted(os.listdir(workspace)):
        vdir = os.path.join(workspace, video_id)
        if not os.path.isdir(vdir):
            continue
        for name in sorted(os.listdir(vdir)):
            if not name.startswith("captions"):
                continue
            cache_dir = os.path.join(vdir, name)
            if not os.path.isdir(cache_dir):
                continue
            tag = name.rsplit("_", 1)[-1] if "_" in name else ""
            is_default = tag == known_tag
            captions_json = os.path.join(cache_dir, "captions.json")
            entry = {
                "video_id": video_id,
                "cache_dir": cache_dir,
                "legacy_dvd_tag": tag,
                "key": {
                    "video_id": video_id,
                    "segment_id": "*",
                    "prompt_hash": known_hash if is_default else f"unknown_legacy:{tag or name}",
                    "caption_model_id": config.CAPTION_MODEL_ID,
                    "decoding_hash": config.decoding_hash(),
                    "source_hash": None,  # legacy: inputs not re-derivable without video scan
                },
                "captions_content_hash": (captions_content_hash(captions_json)
                                          if os.path.exists(captions_json) else None),
                "n_ckpt_files": len(os.listdir(os.path.join(cache_dir, "ckpt")))
                if os.path.isdir(os.path.join(cache_dir, "ckpt")) else 0,
                "read_only": True,
                "legacy": True,
            }
            registered.append(register_cache(entry, manifest))
    return registered


def new_candidate_cache_dir(key: CaptionCacheKey, root: str | None = None) -> str:
    """Directory for a NEW candidate caption cache under the harness root
    (never inside the legacy DVD workspace). Phase 2 writers must call
    assert_writable + register_cache."""
    root = root or config.CAPTION_CACHE_ROOT
    return os.path.join(
        root, key.video_id,
        f"p{key.prompt_hash[:12]}_d{key.decoding_hash[:8]}_s{(key.source_hash or 'na')[:8]}",
    )


def new_history_aware_cache_dir(key: CaptionCacheKey,
                                root: str | None = None) -> str:
    """Isolated cache directory for one frozen-history segment identity."""
    if not key.history_hash or not key.composed_prompt_hash:
        raise ValueError("history-aware cache key is missing history/prompt identity")
    versions_hash = sha256_json({
        "bank": key.bank_version,
        "router": key.router_version,
        "scaffold": key.scaffold_version,
        "contract": key.contract_version,
    })
    root = root or config.CAPTION_CACHE_ROOT
    safe_segment = key.segment_id.replace("/", "_")
    model_hash = sha256_text(key.caption_model_id)
    backend_hash = sha256_text(key.backend_id or "unknown")
    intervention_suffix = (
        f"_i{key.intervention_identity_hash[:12]}"
        if key.intervention_identity_hash else "")
    return os.path.join(
        root,
        key.video_id,
        "history_v1",
        safe_segment,
        f"c{key.composed_prompt_hash[:12]}_p{key.prompt_hash[:12]}_"
        f"h{key.history_hash[:12]}_v{versions_hash[:12]}_"
        f"m{model_hash[:8]}_d{key.decoding_hash[:8]}_"
        f"b{backend_hash[:8]}_g{(key.history_config_hash or 'na')[:8]}"
        f"{intervention_suffix}_"
        f"s{(key.source_hash or 'na')[:8]}",
    )


def key_as_dict(key: CaptionCacheKey) -> dict:
    value = asdict(key)
    if key.intervention_identity_hash is None:
        value.pop("intervention_identity_hash")
    return value
