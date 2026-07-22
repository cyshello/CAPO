Test only the surrogate rollout with Qwen2.5-VL-Instruct.

## Qwen2.5-VL captioner

Code-level API:

```python
from surrogate_rollout.captioning import Qwen25VLCaptioner

captioner = Qwen25VLCaptioner(max_images_per_prompt=8)
caption = captioner.caption(
    ["frame_000.jpg", "frame_001.jpg", "frame_002.jpg", "frame_003.jpg",
     "frame_004.jpg", "frame_005.jpg", "frame_006.jpg", "frame_007.jpg"],
    "Describe the sequence across these frames.",
    max_tokens=128,
)
```

Smoke test with 8 generated images:

```bash
conda run -n local_llm_vllm python -m surrogate_rollout.scripts.smoke_qwen25vl_captioner --gpu 1
```

Smoke test with a frame directory:

```bash
conda run -n local_llm_vllm python -m surrogate_rollout.scripts.smoke_qwen25vl_captioner \
  --gpu 1 \
  --image-dir /path/to/frames \
  --num-images 8
```

TODO
- run/text VideoARM with different prompts
- cherrypick examples
