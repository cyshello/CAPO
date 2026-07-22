"""Register all existing DVD workspace caption caches as read-only legacy
entries in the harness cache manifest. Idempotent.

Usage:
    conda run -n local_llm_vllm python -m surrogate_rollout.scripts.register_legacy_caches
"""

from __future__ import annotations

import sys

from surrogate_rollout import config
from surrogate_rollout.cache.caption_cache import register_legacy_dvd_caches

for _p in (config.PROMPT_SENS_ROOT, config.DVD_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> None:
    from dvd.prompts import get_prompts

    p = get_prompts()
    entries = register_legacy_dvd_caches(p.caption_prompt, p.merge_prompt)
    for e in entries:
        print(f"{e['video_id']}: {e['cache_dir'].rsplit('/', 1)[-1]} "
              f"tag={e['legacy_dvd_tag']} ckpt={e['n_ckpt_files']} "
              f"prompt_hash={e['key']['prompt_hash'][:20]}... read_only={e['read_only']}")
    print(f"{len(entries)} caches registered -> {config.CACHE_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
