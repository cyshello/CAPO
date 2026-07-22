"""Local backend for DVD: routes DVD model calls through local or API backends.

DVD normally calls `call_openai_model_with_tools` (orchestrator o3 + caption/
tool VLM gpt-4.1-mini) and Azure `text-embedding-3-large`. This module
monkeypatches DVD so that:

    * image calls (clip captioning, frame_inspect)  -> Qwen2.5-VL (vLLM, local)
    * text calls WITH tools (orchestrator)          -> OpenAI API when enabled,
                                                       otherwise Codex CLI shim
    * text calls WITHOUT tools (global_browse/merge)-> OpenAI API by default,
                                                       or legacy Codex CLI
    * embeddings                                    -> local BGE (SentenceTransformer)

Nothing in DVD's agent architecture changes; only the transport is swapped by
rebinding the names DVD imported.
"""

from __future__ import annotations

import json
import re
import uuid

import os

import dvd.config as config
import dvd.build_database as build_database
import dvd.dvd_core as dvd_core
import dvd.frame_caption as frame_caption
import dvd.utils as utils
from codex_infer import codex_infer

# DVD's real OpenAI/Azure HTTP call, captured before we monkeypatch the name.
# Used for genuine function-calling when an OpenAI key with quota is available.
_ORIG_CALL = utils.call_openai_model_with_tools

# Extra orchestrator retries (with tool_choice="required") when the model
# returns a text-only response instead of a tool call.
_OPENAI_TOOL_RETRIES = 2


def _load_openai_key() -> str | None:
    """OPENAI_API_KEY from the environment or prompt_sensitivity/.env."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.isfile(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


# --------------------------------------------------------------------------- #
#                         Local BGE embedding service                         #
# --------------------------------------------------------------------------- #
_EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
_EMBED_DIM = 384
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(_EMBED_MODEL_ID, device="cpu")
    return _embedder


def _serial_preprocess_captions(caption_json_path):
    """Serial replacement for build_database.preprocess_captions.

    The original forks a multiprocessing.Pool to embed captions, but the parent
    process has already initialised CUDA (vLLM/Qwen); forking after CUDA init
    deadlocks. We embed serially on CPU (BGE) instead. Returns the same
    (timestamp, cap_info, embedding) tuples init_single_video_db expects.
    """
    import json as _json

    with open(caption_json_path) as f:
        captions = _json.load(f)
    captions.pop("subject_registry", None)
    captions.pop("character_registry", None)

    scripts = []
    for timestamp, cap_info in captions.items():
        caption = cap_info.get("caption")
        if not caption:
            continue
        if isinstance(caption, list):
            cap_info["caption"] = caption[0]
        elif not isinstance(caption, str):
            cap_info["caption"] = str(caption)
        ts = list(map(float, timestamp.split("_")))
        scripts.append((ts, cap_info["caption"], cap_info))

    if not scripts:
        return []
    embs = _local_get_embeddings(input_text=[s[1] for s in scripts])
    return [(scripts[i][0], scripts[i][2], embs[i]["embedding"]) for i in range(len(scripts))]


def _local_get_embeddings(endpoints=None, model_name=None, input_text=None, api_key=None):
    """Drop-in for AzureOpenAIEmbeddingService.get_embeddings.

    Returns a list of {"embedding": [...]} matching the Azure response shape.
    """
    if isinstance(input_text, str):
        input_text = [input_text]
    vecs = _get_embedder().encode(input_text, normalize_embeddings=True)
    return [{"embedding": v.tolist()} for v in vecs]


# --------------------------------------------------------------------------- #
#                       Qwen2.5-VL captioner (vision)                         #
# --------------------------------------------------------------------------- #
_captioner = None
_tensor_parallel = 1  # number of GPUs to shard Qwen across (set by install_backend)


def get_captioner():
    """Lazily build the shared Qwen captioner used for all vision calls."""
    global _captioner
    if _captioner is None:
        from captioner import build_captioner

        _captioner = build_captioner(
            max_model_len=12288,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=_tensor_parallel,
            max_images_per_prompt=config.AOAI_TOOL_VLM_MAX_FRAME_NUM,
            image_max_pixels=200704,  # ~256 vision tokens/image
        )
    return _captioner


# --------------------------------------------------------------------------- #
#                     Codex function-calling JSON shim                         #
# --------------------------------------------------------------------------- #
_PROTOCOL = """You are the decision engine of a tool-using agent. Read the conversation and the available tools, then decide the SINGLE next action.

Respond with ONLY a JSON object, no prose, no code fence. Schema:
{"name": "<one tool name>", "arguments": {<arg>: <value>, ...}}

Rules:
- Choose exactly one tool that best advances toward answering the user's question.
- Arguments must match the chosen tool's parameters. Never include a `database` argument.
- To give the final answer to the user, call the `finish` tool with its `answer` argument.
"""


def _strip_fences(text: str) -> str:
    """Remove a leading ```json / ``` fence and trailing ``` if present."""
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _extract_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    start = t.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in codex output:\n{text}")
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start : i + 1])
    raise ValueError(f"unbalanced JSON in codex output:\n{text}")


def _render_tools(tools: list) -> str:
    blocks = []
    for t in tools:
        fn = t["function"]
        params = fn.get("parameters", {}).get("properties", {})
        args = "\n".join(
            f"    - {k}: {v.get('description', '')}"
            for k, v in params.items()
            if k != "database"
        )
        blocks.append(f"* {fn['name']}: {fn['description']}\n{args}")
    return "\n\n".join(blocks)


def _render_messages(messages: list) -> str:
    out = []
    for m in messages:
        role = m.get("role", "?")
        if m.get("tool_calls"):
            calls = "; ".join(
                f"{c['function']['name']}({c['function']['arguments']})"
                for c in m["tool_calls"]
            )
            out.append(f"[{role}] called: {calls}")
        elif m.get("content"):
            out.append(f"[{role}] {m['content']}")
    return "\n".join(out)


def _codex_tool_call(messages: list, tools: list, model: str) -> dict:
    prompt = (
        _PROTOCOL
        + "\n\n### Available tools\n"
        + _render_tools(tools)
        + "\n\n### Conversation\n"
        + _render_messages(messages)
        + "\n\n### Your JSON action:"
    )
    obj = _extract_json(codex_infer(prompt, model=model))
    name = obj["name"]
    args = obj.get("arguments", {})
    # DVD's _exec_tool injects the real DB only if the key already exists
    # (`if "database" in args`). The model never supplies it, so add a
    # placeholder for any tool that declares a `database` parameter.
    schema = next((t["function"] for t in tools if t["function"]["name"] == name), None)
    if schema and "database" in schema.get("parameters", {}).get("properties", {}):
        args.setdefault("database", None)
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args),
                },
            }
        ],
    }


def _codex_text(messages: list, model: str, return_json: bool) -> dict:
    prompt = _render_messages(messages)
    if return_json:
        prompt += "\n\nRespond with ONLY a valid JSON object, no prose, no code fence."
    text = codex_infer(prompt, model=model)
    if return_json:
        try:  # normalise to compact JSON, but never raise (caller retries)
            text = json.dumps(_extract_json(text))
        except Exception:
            text = _strip_fences(text)
    return {"content": text.strip(), "tool_calls": None}


def _openai_text(messages: list, model: str, api_key: str,
                 return_json: bool, max_tokens: int, temperature: float) -> dict:
    """Plain text OpenAI API call used for global_browse/merge reasoning."""
    resp = _ORIG_CALL(
        messages=messages,
        endpoints=None,
        model_name=model,
        api_key=api_key,
        tools=[],
        max_tokens=max_tokens,
        temperature=temperature,
        tool_choice="none",
        return_json=return_json,
    )
    if resp is None:
        raise RuntimeError(
            "OpenAI text call failed; check OPENAI_API_KEY, quota, and model name"
        )
    return resp


# --------------------------------------------------------------------------- #
#                    Unified router (call_openai_* replacement)               #
# --------------------------------------------------------------------------- #
def make_router(inference_model: str, tool_calling_model: str,
                tool_openai_api_key: str | None,
                text_openai_api_key: str | None,
                text_backend: str = "openai"):
    """Return a call_openai_model_with_tools-compatible router.

    Routing:
        image_paths -> Qwen (vision)
        tools       -> real OpenAI function-calling if a key with quota is
                       available, else the Codex JSON shim (fallback)
        plain text  -> OpenAI API by default, or Codex CLI if text_backend=codex
    """
    if text_backend not in {"openai", "codex"}:
        raise ValueError("text_backend must be 'openai' or 'codex'")

    def call(messages, endpoints, model_name, api_key=None, tools=(),
             image_paths=(), max_tokens=4096, temperature=0.0,
             tool_choice="auto", return_json=False):
        # Vision call: clip captioning or frame_inspect -> Qwen
        if image_paths:
            prompt = messages[-1]["content"] if messages else ""
            text = get_captioner().caption(
                list(image_paths), prompt, max_tokens=max_tokens
            )
            if return_json:  # Qwen fences JSON in ```json ... ```; unwrap the
                # fence but let the caller's json.loads validate + retry.
                text = _strip_fences(text)
            return {"content": text.strip(), "tool_calls": None}
        # Orchestrator (function-calling)
        if tools:
            if tool_openai_api_key:
                # Genuine OpenAI tool-calling (per user: key only for this).
                # gpt-4o-mini sometimes replies text-only (tool_calls=None);
                # retry forcing tool_choice="required" to get an actual call.
                last = None
                for attempt in range(_OPENAI_TOOL_RETRIES + 1):
                    tc = "required" if attempt > 0 else tool_choice
                    resp = _ORIG_CALL(
                        messages=messages, endpoints=None,
                        model_name=tool_calling_model, api_key=tool_openai_api_key,
                        tools=list(tools), max_tokens=max_tokens,
                        temperature=temperature, tool_choice=tc,
                    )
                    if resp is None:
                        break  # quota/error -> codex fallback
                    resp.setdefault("role", "assistant")
                    if resp.get("tool_calls"):
                        return resp
                    last = resp  # text-only; retry with required
                if last is not None:
                    return last  # DVD loop tolerates the empty tool_calls
            return _codex_tool_call(messages, list(tools), inference_model)
        # Plain text reasoning (global_browse / merge).
        if text_backend == "openai":
            if not text_openai_api_key:
                raise RuntimeError(
                    "text_backend='openai' requires OPENAI_API_KEY or .env"
                )
            return _openai_text(
                messages, tool_calling_model, text_openai_api_key,
                return_json, max_tokens, temperature,
            )
        return _codex_text(messages, inference_model, return_json)

    return call


# --------------------------------------------------------------------------- #
#                              Install / patch                                #
# --------------------------------------------------------------------------- #
def install_backend(inference_model: str = "gpt-5.5", tool_vlm_max_frames: int = 16,
                    tool_calling_model: str = "gpt-4o-mini", openai_api_key: str | None = None,
                    use_openai_tools: bool = True, text_backend: str = "openai",
                    tensor_parallel_size: int = 1):
    """Monkeypatch DVD to use Qwen vision, BGE embeddings, and configured text.

    Tool-calling (orchestrator) uses the real OpenAI API when a key with quota
    is available (`use_openai_tools` + resolvable OPENAI_API_KEY), otherwise it
    falls back to the Codex JSON shim. Plain text reasoning uses OpenAI API by
    default; set `text_backend="codex"` to keep the previous Codex CLI behavior.
    Call once before running the agent.
    """
    global _tensor_parallel
    _tensor_parallel = tensor_parallel_size
    config.AOAI_EMBEDDING_LARGE_DIM = _EMBED_DIM
    config.AOAI_TOOL_VLM_MAX_FRAME_NUM = tool_vlm_max_frames

    key = (openai_api_key or _load_openai_key()) if use_openai_tools else None
    text_key = openai_api_key or _load_openai_key()
    router = make_router(inference_model, tool_calling_model, key, text_key, text_backend)
    # Rebind in every module that did `from dvd.utils import call_openai_...`.
    utils.call_openai_model_with_tools = router
    dvd_core.call_openai_model_with_tools = router
    build_database.call_openai_model_with_tools = router
    frame_caption.call_openai_model_with_tools = router

    # Local embeddings (class is shared by reference across imports).
    utils.AzureOpenAIEmbeddingService.get_embeddings = staticmethod(_local_get_embeddings)
    # Serial embedding at DB-build time (avoid fork-after-CUDA deadlock).
    build_database.preprocess_captions = _serial_preprocess_captions

    return router
