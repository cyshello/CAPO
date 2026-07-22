"""Text-only inference via the Codex CLI.

Codex CLI is used as a drop-in replacement for the OpenAI API (no API key
available; ChatGPT OAuth is already configured in ~/.codex/auth.json).

Only textual reasoning is used here — no vision. `codex exec` runs
non-interactively; we redirect stdin from /dev/null to avoid the known
"Reading additional input from stdin..." infinite hang, and capture the final
assistant message via `--output-last-message`.
"""

from __future__ import annotations

import os
import subprocess
import tempfile


# Models supported by the ChatGPT-account Codex backend (from
# ~/.codex/models_cache.json): gpt-5.6-sol, gpt-5.6-terra, gpt-5.6-luna,
# gpt-5.5, gpt-5.4, gpt-5.4-mini. Plain "gpt-5" is NOT accepted.
DEFAULT_MODEL = "gpt-5.5"


class CodexError(RuntimeError):
    """Raised when the codex CLI call fails."""


def codex_infer(
    prompt: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 600,
    codex_bin: str = "codex",
) -> str:
    """Run a single text reasoning call through the Codex CLI.

    Args:
        prompt: The full prompt text sent to the model.
        model: Model name passed to `codex exec -m`.
        timeout: Seconds before the subprocess is killed.
        codex_bin: Path/name of the codex executable.

    Returns:
        The final assistant message as a stripped string.

    Raises:
        CodexError: On non-zero exit, timeout, or empty output.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "last_message.txt")
        err_path = os.path.join(tmpdir, "stderr.log")

        # Pass the prompt via stdin ("-"), not as an argv element: long prompts
        # (e.g. global_browse with many captions) exceed the OS ARG_MAX and
        # raise OSError [Errno 7] Argument list too long. Feeding stdin then
        # closing it gives a clean EOF, so codex does not hang waiting for more.
        cmd = [
            codex_bin,
            "exec",
            "-",
            "-m",
            model,
            "-o",
            out_path,
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--color",
            "never",
        ]

        with open(err_path, "wb") as errf:
            try:
                proc = subprocess.run(
                    cmd,
                    input=prompt.encode(),
                    stdout=subprocess.DEVNULL,
                    stderr=errf,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as e:
                raise CodexError(
                    f"codex exec timed out after {timeout}s"
                ) from e

        stderr = _read(err_path)
        if proc.returncode != 0:
            raise CodexError(
                f"codex exec exited with {proc.returncode}.\nstderr:\n{stderr}"
            )

        result = _read(out_path).strip()
        if not result:
            raise CodexError(
                f"codex exec produced empty output.\nstderr:\n{stderr}"
            )
        return result


def _read(path: str) -> str:
    try:
        with open(path, "r", errors="replace") as f:
            return f.read()
    except FileNotFoundError:
        return ""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Codex CLI text inference")
    parser.add_argument("prompt")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL)
    parser.add_argument("-t", "--timeout", type=int, default=600)
    args = parser.parse_args()

    print(codex_infer(args.prompt, model=args.model, timeout=args.timeout))
