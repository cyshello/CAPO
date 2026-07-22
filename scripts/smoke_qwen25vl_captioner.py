"""Run a Qwen2.5-VL captioning smoke test with at least eight images."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _make_dummy_images(count: int, directory: Path) -> list[Path]:
    from PIL import Image, ImageDraw

    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx in range(count):
        path = directory / f"dummy_frame_{idx:02d}.jpg"
        image = Image.new("RGB", (384, 256), (30 + idx * 20, 80, 160))
        draw = ImageDraw.Draw(image)
        draw.rectangle((24, 24, 360, 232), outline=(255, 255, 255), width=4)
        draw.text((42, 96), f"Frame {idx + 1}", fill=(255, 255, 255))
        image.save(path, quality=92)
        paths.append(path)
    return paths


def _collect_images(args: argparse.Namespace) -> list:
    if args.images:
        return [Path(p) for p in args.images]
    if args.image_dir:
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        paths = [
            p for p in sorted(Path(args.image_dir).iterdir()) if p.suffix.lower() in exts
        ]
        if not paths:
            raise RuntimeError(f"no images found in {args.image_dir}")
        return paths[: args.num_images]
    if args.video:
        from surrogate_rollout.captioning import sample_video_frames

        return sample_video_frames(args.video, max_frames=args.num_images)
    tmpdir = Path(tempfile.mkdtemp(prefix="qwen25vl_smoke_"))
    return _make_dummy_images(args.num_images, tmpdir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="*", help="Image paths for one prompt")
    parser.add_argument("--image-dir", help="Directory of frames/images")
    parser.add_argument("--video", help="Local video path to sample into frames")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument(
        "--prompt",
        default=(
            "Describe the sequence across these frames. Mention the visible "
            "objects, changes over time, and any text you can read."
        ),
    )
    parser.add_argument("--model-path")
    parser.add_argument("--gpu", default=None, help="Sets CUDA_VISIBLE_DEVICES")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--image-max-pixels", type=int, default=512 * 512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_images < 8:
        raise ValueError("--num-images must be at least 8 for this smoke test")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from surrogate_rollout.captioning import Qwen25VLCaptioner

    images = _collect_images(args)
    if len(images) < 8:
        raise RuntimeError(f"need at least 8 images, got {len(images)}")

    captioner = Qwen25VLCaptioner(
        model_path=args.model_path,
        max_images_per_prompt=max(args.num_images, 8),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        image_max_pixels=args.image_max_pixels,
    )
    caption = captioner.caption(
        images[: args.num_images],
        args.prompt,
        max_tokens=args.max_tokens,
    )

    print(f"images_used={len(images[: args.num_images])}")
    print("caption:")
    print(caption)


if __name__ == "__main__":
    main()
