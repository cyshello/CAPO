#!/usr/bin/env python3
"""Write one immutable plain-text-output parent meta-prompt and pointer."""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from surrogate_rollout.optimization.schemas import MetaPromptVersion
from surrogate_rollout.prompt_routing.free_form_instruction_generator import (
    DEFAULT_TEMPLATE_VERSION,
    FREE_FORM_REQUEST_SCHEMA_VERSION,
    GENERATOR_TEMPLATES,
)
from surrogate_rollout.prompt_routing.free_form_instruction_parser import (
    PARSER_VERSION,
)
from surrogate_rollout.prompt_routing.persistence import _atomic_write_text
from surrogate_rollout.prompt_routing.schemas import dumps_canonical
from surrogate_rollout.schemas import sha256_json, sha256_text


def _write_once(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    _atomic_write_text(str(path), dumps_canonical(value) + "\n")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--previous-meta-prompt-id", required=True)
    parser.add_argument("--created-at")
    args = parser.parse_args(argv)

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "prompt_routing/free_form_instruction_generator.py"
    text = GENERATOR_TEMPLATES[DEFAULT_TEMPLATE_VERSION]
    identity = {
        "schema_version": "plain_text_meta_prompt_bootstrap_identity_v1",
        "parent_meta_prompt_id": args.previous_meta_prompt_id,
        "template_version": DEFAULT_TEMPLATE_VERSION,
        "template_sha256": sha256_text(text),
        "request_contract_version": FREE_FORM_REQUEST_SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
    }
    meta_prompt_id = "meta_prompt_" + sha256_json(identity)[:20]
    created_at = args.created_at or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    artifact = output / "parent_meta_prompt.json"
    version = MetaPromptVersion(
        meta_prompt_id=meta_prompt_id,
        parent_meta_prompt_id=args.previous_meta_prompt_id,
        text=text,
        created_at=created_at,
        status="parent",
    )
    _write_once(artifact, version)
    artifact_hash = _sha256_file(artifact)
    _write_once(output / "current_meta_prompt.json", {
        "schema_version": "current_meta_prompt_pointer_v1",
        "active_meta_prompt_id": meta_prompt_id,
        "artifact_path": str(artifact),
        "artifact_sha256": artifact_hash,
    })
    _write_once(output / "bootstrap_manifest.json", {
        "schema_version": "plain_text_meta_prompt_bootstrap_manifest_v1",
        "identity": identity,
        "meta_prompt_id": meta_prompt_id,
        "artifact_path": str(artifact),
        "artifact_sha256": artifact_hash,
        "source": {
            "path": str(source),
            "sha256": _sha256_file(source),
            "symbol": f"GENERATOR_TEMPLATES[{DEFAULT_TEMPLATE_VERSION!r}]",
        },
        "runtime_pointer_modified": False,
        "write_policy": "atomic_write_once_new_directory",
    })
    print(dumps_canonical({
        "meta_prompt_id": meta_prompt_id,
        "parent_meta_prompt": str(artifact),
        "current_pointer": str(output / "current_meta_prompt.json"),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
