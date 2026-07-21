"""Temporary mixed caption views (PHASE2_3 §3).

A mixed view replaces ONLY the selected clips' captions with candidate-prompt
captions; every other clip keeps the frozen baseline snapshot's caption text
verbatim. The subject registry is derived state: it is re-merged over the
mixed per-clip registries (baseline per-clip registries come from the baseline
ckpt dir, candidate ones from the candidate cache). The resulting captions.json
lives only in the Phase 2 work root and its content hash keys the vector DB —
a mixed view can never be committed as a full cache (CLAUDE.md §7/§11).

`merge_fn` is injectable: the default is DVD's merge_subject_registries (codex,
nondeterministic); tests inject a deterministic stub.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable

from surrogate_rollout.cache.caption_cache import assert_writable, captions_content_hash


def default_merge_fn(partial_registries: list) -> dict | list | None:
    from dvd.frame_caption import merge_subject_registries

    return merge_subject_registries(partial_registries)


def caption_entry_from_parsed(parsed: dict) -> dict | None:
    """DVD captions.json entry from one validated caption artifact."""
    if parsed and parsed.get("clip_description"):
        return {"caption": parsed["clip_description"]}
    return None


def load_clip_registries(ckpt_dir: str, clip_keys: list[str]) -> dict[str, dict]:
    """Per-clip subject registries from a ckpt dir (baseline or candidate)."""
    out: dict[str, dict] = {}
    for k in clip_keys:
        p = os.path.join(ckpt_dir, f"{k}.json")
        if os.path.exists(p):
            with open(p) as f:
                parsed = json.load(f)
            if parsed.get("subject_registry"):
                out[k] = parsed["subject_registry"]
    return out


def write_captions_json(captions: dict, out_dir: str) -> tuple[str, str]:
    """Write a captions.json (DVD format, indent=4 like the vendored writer)
    and return (path, content hash). Refuses read-only registered caches."""
    assert_writable(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "captions.json")
    with open(out_path, "w") as f:
        json.dump(captions, f, indent=4)
    return out_path, captions_content_hash(out_path)


@dataclass
class MixedViewArtifact:
    captions_path: str
    captions_hash: str
    database_path: str
    selected_clip_ids: list[str] = field(default_factory=list)
    replaced_clip_ids: list[str] = field(default_factory=list)  # actually swapped


class MixedViewBuilder:
    """Builds one temporary mixed caption view under an isolated work root."""

    def __init__(self, merge_fn: Callable[[list], object] | None = None) -> None:
        self.merge_fn = merge_fn or default_merge_fn

    def build(
        self,
        baseline_captions_path: str,
        candidate_captions: dict[str, dict],
        selected_clip_ids: set[str],
        work_root: str,
        *,
        video_id: str,
        baseline_ckpt_dir: str | None = None,
        baseline_registries: dict[str, object] | None = None,
        view_name: str = "mixed",
    ) -> MixedViewArtifact:
        """`candidate_captions` maps clip key -> parsed captioner JSON for (at
        least) the selected clips. The baseline snapshot is never modified.

        The vector DB is NOT built here — it is keyed by the returned content
        hash and built lazily by the shared reasoning path (dvd_qa), which is
        the only component with the embedding backend installed."""
        with open(baseline_captions_path) as f:
            baseline = json.load(f)
        clip_keys = [k for k in baseline
                     if k not in ("subject_registry", "character_registry")]

        unknown = selected_clip_ids - set(clip_keys)
        if unknown:
            raise ValueError(
                f"selected clips not in baseline caption view: {sorted(unknown)[:5]}")

        mixed: dict = {}
        replaced: list[str] = []
        for k in clip_keys:
            entry = None
            if k in selected_clip_ids:
                entry = caption_entry_from_parsed(candidate_captions.get(k) or {})
                if entry is not None:
                    replaced.append(k)
            if entry is None:  # unselected, or candidate caption parse-failed
                entry = baseline[k]
            mixed[k] = entry

        # registry is derived state: re-merge over the mixed per-clip registries
        registries = (load_clip_registries(baseline_ckpt_dir, clip_keys)
                      if baseline_ckpt_dir else {})
        registries.update(baseline_registries or {})
        for k in replaced:
            reg = (candidate_captions.get(k) or {}).get("subject_registry")
            if reg:
                registries[k] = reg
            else:
                registries.pop(k, None)
        ordered = [registries[k] for k in clip_keys if k in registries]
        mixed["subject_registry"] = self.merge_fn(ordered)

        out_dir = os.path.join(work_root, video_id, f"captions_{view_name}")
        captions_path, chash = write_captions_json(mixed, out_dir)
        workdir = os.path.join(work_root, video_id)
        return MixedViewArtifact(
            captions_path=captions_path,
            captions_hash=chash,
            database_path=os.path.join(workdir, f"database_c{chash[:16]}.json"),
            selected_clip_ids=sorted(selected_clip_ids),
            replaced_clip_ids=sorted(replaced),
        )
