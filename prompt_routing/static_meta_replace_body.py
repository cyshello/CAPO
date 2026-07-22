"""Static meta-prompt (replace-body) policy — parity copy of the baseline repo.

The private baseline repo runs this exact condition standalone: a fixed meta
prompt sees a 0.5 fps, half-resolution subset of a segment's frames plus the
frozen preceding caption history, and writes that segment's captioning task
section. The generated text REPLACES the base template's task text; only the
transcript block and the verbatim JSON output contract survive.

This module is the same policy expressed for this repository so both sides
produce byte-identical generator prompts and byte-identical composed caption
prompts. It is intentionally dependency-light — pure functions over strings and
dicts, no model calls, no imports from the optimization package — and it is
mirrored, not shared: the two repositories stay independently deployable.
`META_PROMPT_SHA256` pins the shared text; `tests/test_static_meta_parity.py`
fails if either side drifts.

QA-leakage guard: the request is built only from video/segment ids, timestamps,
frame paths and previous captions. Questions, options, answers and scores never
enter it.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Mapping, Sequence

from surrogate_rollout.schemas import sha256_text

# --------------------------------------------------------------------------- #
#                          versions (cache identity)                           #
# --------------------------------------------------------------------------- #
META_PROMPT_ID = "static_meta_prompt_full_body_c_v1"
META_PROMPT_VERSION = "static_meta_prompt_full_body_v1_plain_text"
REQUEST_SCHEMA_VERSION = "static_meta_instruction_request_v1"
HISTORY_SCHEMA_VERSION = "frozen_local_caption_history_v1"
PARSER_VERSION = "static_meta_instruction_plain_text_parser_v1"
COMPOSITION_VERSION = "static_meta_replace_body_composition_v1"

DEFAULT_HISTORY_BLOCK_SECONDS = 300.0
DEFAULT_MAX_HISTORY_CAPTIONS = 30
DEFAULT_GENERATOR_MAX_TOKENS = 512

# Frame policy: the generator sees FEWER and SMALLER frames than the captioner —
# every other frame (0.5 fps against the captioner's 1.0 fps) at half
# resolution.
GENERATOR_SAMPLE_FPS = 0.5
GENERATOR_IMAGE_SCALE = 0.5
GENERATOR_JPEG_QUALITY = 90

META_PROMPT_SHA256 = (
    "4e7ca02d27e84339e6e5a7163b983e312d9c7293183ede2d320515951f04c57f")
META_PROMPT_TEXT = """You write the task section of the prompt that a video captioning model will receive for one video segment.

Inspect the sampled frames from the current video segment and the preceding
caption history.

Write the complete instructions telling the captioning model how to describe
THIS segment: what to attend to, what level of detail to give, which entities to
track and how to refer to them, and what to leave out. Replace generic captioning
guidance with guidance specific to what is actually visible here.

Focus only on information supported by the current visual inputs and context.

Do not answer any downstream question.
Do not assume that you know what question will later be asked.
Do not summarize the entire video.
Do not write the caption itself, and do not describe the segment as if you were captioning it.
Do not invent identities, actions, states, relationships, or events.
Do not mention transcripts, JSON, output formats, or field names: the output
contract is appended separately and must not be restated or contradicted.

Return only the instruction text as plain text, at most 12 lines. Do not use JSON or Markdown."""

JSON_CONTRACT_ANCHOR = "Output ONLY valid JSON"
TRANSCRIPT_HEADER = "Transcript of current clip:"
REQUIRED_PLACEHOLDERS = (
    "TRANSCRIPT_PLACEHOLDER", "CLIP_START_TIME", "CLIP_END_TIME")


class StaticMetaPromptError(ValueError):
    """Generator input incomplete, output unusable, or composition invalid."""


def dumps_canonical(value: Any) -> str:
    """Deterministic JSON text (sorted keys, stable separators)."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def segment_times(segment_id: str) -> tuple[float, float]:
    try:
        start, end = (float(value) for value in segment_id.split("_", 1))
    except (ValueError, TypeError) as exc:
        raise StaticMetaPromptError(
            f"segment_id must be '<start>_<end>': {segment_id!r}") from exc
    if start < 0 or end <= start:
        raise StaticMetaPromptError(f"invalid segment interval: {segment_id!r}")
    return start, end


# --------------------------------------------------------------------------- #
#                            generator frame policy                            #
# --------------------------------------------------------------------------- #
def select_generator_frames(
    files: Sequence[str],
    *,
    caption_fps: float,
    generator_fps: float = GENERATOR_SAMPLE_FPS,
) -> list[str]:
    """Deterministic temporal subsample: every `caption_fps/generator_fps`-th
    frame, so the generator's frames are a strict subset of the captioner's."""
    frames = [str(item) for item in files]
    if not frames:
        return []
    if generator_fps <= 0:
        raise StaticMetaPromptError("generator_fps must be positive")
    step = max(1, int(round(caption_fps / generator_fps)))
    return frames[::step]


def downscaled_frames_dir(frames_dir: str,
                          scale: float = GENERATOR_IMAGE_SCALE) -> str:
    tag = f"{int(round(scale * 100)):d}"
    return os.path.join(frames_dir, f"generator_frames_scale{tag}")


def downscale_frames(
    files: Sequence[str],
    out_dir: str,
    *,
    scale: float = GENERATOR_IMAGE_SCALE,
    jpeg_quality: int = GENERATOR_JPEG_QUALITY,
) -> list[str]:
    """Half-resolution copies of `files` in `out_dir` (written once, reused).
    Originals are never modified."""
    import cv2

    if not 0 < scale <= 1:
        raise StaticMetaPromptError("scale must be in (0, 1]")
    os.makedirs(out_dir, exist_ok=True)
    out_paths: list[str] = []
    for path in files:
        out_path = os.path.join(out_dir, os.path.basename(path))
        out_paths.append(out_path)
        if os.path.exists(out_path):
            continue
        image = cv2.imread(path)
        if image is None:
            raise StaticMetaPromptError(f"could not read frame {path}")
        height, width = image.shape[:2]
        target = (max(1, int(width * scale)), max(1, int(height * scale)))
        resized = cv2.resize(image, target, interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode(
            ".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok:
            raise StaticMetaPromptError(
                f"could not encode downscaled frame {out_path}")
        tmp_path = f"{out_path}.{os.getpid()}.tmp"
        with open(tmp_path, "wb") as handle:
            handle.write(buffer.tobytes())
        os.replace(tmp_path, out_path)
    return out_paths


def prepare_generator_frames(files: Sequence[str], *,
                             caption_fps: float) -> list[str]:
    """The frames one generator call receives: 0.5 fps subset, half resolution.
    Downscaled copies live beside the decoded frames they came from."""
    selected = select_generator_frames(files, caption_fps=caption_fps)
    if not selected:
        return []
    frames_dir = os.path.dirname(os.path.abspath(selected[0]))
    return downscale_frames(selected, downscaled_frames_dir(frames_dir))


# --------------------------------------------------------------------------- #
#                            generator request                                 #
# --------------------------------------------------------------------------- #
def build_generator_request(
    *,
    video_id: str,
    segment_id: str,
    frame_references: Sequence[str],
    serialized_history: str,
    generator_max_tokens: int = DEFAULT_GENERATOR_MAX_TOKENS,
    meta_prompt_id: str = META_PROMPT_ID,
    meta_prompt_version: str = META_PROMPT_VERSION,
    meta_prompt_text: str = META_PROMPT_TEXT,
) -> dict[str, Any]:
    """Query-independent request: frames + frozen history, nothing else.

    Takes no question/options/answer argument by construction.
    `frame_references` are the downscaled 0.5 fps generator frames.
    """
    frames = tuple(str(item) for item in frame_references)
    if not frames:
        raise StaticMetaPromptError("current segment frames are required")
    if not serialized_history:
        raise StaticMetaPromptError("serialized caption history is required")
    start, end = segment_times(segment_id)
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "current_segment": {
            "video_id": video_id,
            "segment_id": segment_id,
            "timestamp_start": start,
            "timestamp_end": end,
            "frame_references": list(frames),
        },
        "preceding_caption_history": json.loads(serialized_history),
        "generator_identity": {
            "meta_prompt_id": meta_prompt_id,
            "meta_prompt_version": meta_prompt_version,
            "meta_prompt_hash": sha256_text(meta_prompt_text),
            "sample_fps": GENERATOR_SAMPLE_FPS,
            "image_scale": GENERATOR_IMAGE_SCALE,
            "max_tokens": generator_max_tokens,
        },
    }


def render_generator_prompt(
    request: Mapping[str, Any], *, meta_prompt_text: str = META_PROMPT_TEXT,
) -> str:
    """The exact text handed to the generator alongside its segment frames."""
    return meta_prompt_text + "\n" + dumps_canonical(dict(request))


def parse_generated_instruction(raw: str | None) -> str:
    """Whole non-empty response, stripped. No JSON/Markdown interpretation and
    no repair — an empty reply is a hard failure, not a silent fallback."""
    text = (raw or "").strip()
    if not text:
        raise StaticMetaPromptError("generator returned an empty response")
    return text


# --------------------------------------------------------------------------- #
#                          replace-body composition                            #
# --------------------------------------------------------------------------- #
def split_contract_tail(base_template: str) -> tuple[str, str]:
    """(task body, output-contract tail). The tail is carried over verbatim, so
    subject_registry / clip_description parsing is unchanged."""
    anchor = base_template.find(JSON_CONTRACT_ANCHOR)
    if anchor == -1:
        raise StaticMetaPromptError(
            f"caption template has no {JSON_CONTRACT_ANCHOR!r} contract section")
    line_start = base_template.rfind("\n", 0, anchor) + 1
    return base_template[:line_start], base_template[line_start:]


def compose_caption_prompt(base_template: str,
                           instruction: str) -> tuple[str, str]:
    """Replace the base template's task section with the generated text.

    Returns (composed_prompt, composition_path). Every per-clip placeholder must
    survive exactly once.
    """
    if not base_template:
        raise StaticMetaPromptError("base_template must be non-empty")
    instruction = instruction.strip()
    if not instruction:
        raise StaticMetaPromptError("instruction must be non-empty")
    _, contract = split_contract_tail(base_template)
    composed = (f"{instruction}\n\n"
                f"{TRANSCRIPT_HEADER}\nTRANSCRIPT_PLACEHOLDER\n\n"
                f"{contract}")
    validate_composed(composed, instruction)
    return composed, "replace_body"


def validate_composed(composed: str, instruction: str) -> None:
    for placeholder in REQUIRED_PLACEHOLDERS:
        count = composed.count(placeholder)
        if count != 1:
            raise StaticMetaPromptError(
                f"composed caption prompt has {count} occurrences of "
                f"{placeholder} (expected exactly 1)")
    if instruction not in composed:
        raise StaticMetaPromptError(
            "composed caption prompt does not contain the generated text "
            "verbatim")


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def instruction_is_degenerate(instruction: str) -> bool:
    """True when the generator repeated one sentence — logged, never auto-fixed."""
    chunks = [c.strip().casefold() for c in _SENTENCE_SPLIT.split(instruction)
              if c.strip()]
    return len(chunks) > 1 and len(set(chunks)) == 1
