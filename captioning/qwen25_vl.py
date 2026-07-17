"""Qwen2.5-VL-7B-Instruct captioner backed by vLLM."""

from __future__ import annotations

from typing import Any, Mapping

from .base import (
    BaseCaptioner,
    ImagesInput,
    as_image_list,
    broadcast_prompt,
    load_pil,
    resolve_local_model,
)

_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


class Qwen25VLCaptioner(BaseCaptioner):
    """Caption one or more images with Qwen2.5-VL-7B-Instruct.

    The default `max_images_per_prompt=8` is intentional: the surrogate rollout
    smoke test checks that one request can contain at least eight frames.
    """

    name = "qwen2.5-vl-7b-instruct"

    def __init__(
        self,
        model_path: str | None = None,
        *,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.85,
        tensor_parallel_size: int = 1,
        max_images_per_prompt: int = 8,
        dtype: str = "bfloat16",
        default_max_tokens: int = 256,
        seed: int = 0,
        image_min_pixels: int | None = None,
        image_max_pixels: int | None = None,
    ) -> None:
        from transformers import AutoProcessor
        from vllm import LLM

        self.model_path = model_path or resolve_local_model(_MODEL_ID)
        self.max_images_per_prompt = max_images_per_prompt
        self.default_max_tokens = default_max_tokens
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.llm = LLM(
            model=self.model_path,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt={"image": max_images_per_prompt},
            seed=seed,
            trust_remote_code=True,
        )

    def _image_item(self, image) -> dict:
        item = {"type": "image", "image": load_pil(image)}
        if self.image_min_pixels is not None:
            item["min_pixels"] = self.image_min_pixels
        if self.image_max_pixels is not None:
            item["max_pixels"] = self.image_max_pixels
        return item

    def _messages(self, images: ImagesInput, prompt: str) -> list[dict]:
        image_refs = as_image_list(images)
        if not image_refs:
            raise ValueError("at least one image is required")
        if len(image_refs) > self.max_images_per_prompt:
            raise ValueError(
                f"got {len(image_refs)} images, but max_images_per_prompt is "
                f"{self.max_images_per_prompt}"
            )
        content = [self._image_item(image) for image in image_refs]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _to_vllm_input(self, messages: list[dict]) -> dict:
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        if video_inputs:
            raise ValueError("Qwen25VLCaptioner expects image inputs, not video inputs")

        multi_modal_data = {}
        if image_inputs:
            multi_modal_data["image"] = image_inputs
        return {"prompt": text, "multi_modal_data": multi_modal_data}

    def _sampling(
        self,
        max_tokens: int | None,
        temperature: float,
        top_p: float,
        *,
        json_schema: Mapping[str, Any] | None = None,
    ):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        structured_outputs = None
        if json_schema is not None:
            structured_outputs = StructuredOutputsParams(
                json=dict(json_schema),
                disable_fallback=True,
                disable_additional_properties=True,
            )

        return SamplingParams(
            max_tokens=max_tokens or self.default_max_tokens,
            temperature=temperature,
            top_p=top_p,
            structured_outputs=structured_outputs,
        )

    def caption_batch(
        self,
        images_list: list[ImagesInput],
        prompt: str | list[str],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        json_schema: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> list[str]:
        prompts = broadcast_prompt(prompt, len(images_list))
        inputs = [
            self._to_vllm_input(self._messages(images, item_prompt))
            for images, item_prompt in zip(images_list, prompts)
        ]
        outputs = self.llm.generate(
            inputs,
            self._sampling(
                max_tokens,
                temperature,
                top_p,
                json_schema=json_schema,
            ),
        )
        return [output.outputs[0].text.strip() for output in outputs]
