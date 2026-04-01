from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path


class HashingEmbedder:
    """Deterministic offline embedder for reproducible demos and tests."""

    def __init__(self, dimensions: int = 64):
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    def embed_text(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_bytes(self, data: bytes, mime_type: str) -> list[float]:
        seed = hashlib.sha256(mime_type.encode("utf-8") + b":" + data).hexdigest()
        return self._embed(seed)

    def embed_image(
        self,
        source: str | Path | bytes,
        description: str | None = None,
        mime_type: str | None = "image/png",
    ) -> list[float]:
        return self._embed_media(source, mime_type or "image/png", description)

    def embed_audio(
        self,
        source: str | Path | bytes,
        description: str | None = None,
        mime_type: str | None = "audio/mpeg",
    ) -> list[float]:
        return self._embed_media(source, mime_type or "audio/mpeg", description)

    def embed_video(
        self,
        source: str | Path | bytes,
        description: str | None = None,
        mime_type: str | None = "video/mp4",
    ) -> list[float]:
        return self._embed_media(source, mime_type or "video/mp4", description)

    def embed_pdf(
        self,
        source: str | Path | bytes,
        description: str | None = None,
        mime_type: str | None = "application/pdf",
    ) -> list[float]:
        return self._embed_media(source, mime_type or "application/pdf", description)

    def _embed_media(self, source: str | Path | bytes, mime_type: str, description: str | None) -> list[float]:
        seed = self._media_seed(source, mime_type)
        if description:
            seed = f"{description} {seed}"
        return self._embed(seed)

    def _media_seed(self, source: str | Path | bytes, mime_type: str) -> str:
        return hashlib.sha256(mime_type.encode("utf-8") + b":" + self._read_source(source)).hexdigest()

    def _read_source(self, source: str | Path | bytes) -> bytes:
        if isinstance(source, bytes):
            return source
        return Path(source).read_bytes()

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            index = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self._dimensions
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class DeterministicSupportLLM:
    """Grounded responder used for deterministic benchmarking.

    It only answers from the current message or from explicit memory/procedure
    context passed in the system prompt.
    """

    def generate(self, *, system_prompt: str, user_message: str) -> str:
        evidence = f"{system_prompt}\n{user_message}"
        question = user_message.lower()

        name = _match(evidence, r"my name is ([A-Za-z][A-Za-z .'-]+)")
        plan = _match(evidence, r"i(?: am|'m) on the ([a-z0-9 -]+) plan")
        timezone = _match(evidence, r"my timezone is ([A-Za-z0-9_+:/\- ]+)")
        theme = _match(evidence, r"i prefer the ([a-z0-9 -]+) theme")
        laptop = _match(evidence, r"i use a ([A-Za-z0-9 .'-]+)")
        tried = _match(evidence, r"i already tried ([^.!\n]+)")
        issue = _match(evidence, r"issue: ([^.!\n]+)")
        summary_issue = _match(evidence, r"my issue is ([^.!\n]+)")

        procedure_steps = _extract_procedure_steps(system_prompt)

        if "what is my name" in question or "what's my name" in question:
            return _or_unknown(name, "Your name is {value}.")
        if "what plan am i on" in question:
            return _or_unknown(plan, "You are on the {value} plan.")
        if "what is my timezone" in question or "what's my timezone" in question:
            return _or_unknown(timezone, "Your timezone is {value}.")
        if "which theme do i prefer" in question:
            return _or_unknown(theme, "You prefer the {value} theme.")
        if "what laptop am i using" in question or "which laptop am i using" in question:
            return _or_unknown(laptop, "You are using a {value}.")
        if "what have we already tried" in question:
            return _or_unknown(tried, "We already tried {value}.")
        if "what issue am i reporting" in question:
            return _or_unknown(summary_issue or issue, "You are reporting {value}.")
        if "what notification channel do i prefer" in question:
            channel = _match(evidence, r"i prefer ([a-z]+) notifications")
            return _or_unknown(channel, "You prefer {value} notifications.")
        if "how should we troubleshoot" in question or "what should we do next" in question:
            if procedure_steps:
                return "Recommended next steps:\n" + "\n".join(
                    f"{index}. {step}" for index, step in enumerate(procedure_steps, start=1)
                )
            return "I don't have enough context yet."

        if any(keyword in question for keyword in ("my name is", "i'm on the", "my timezone is", "i prefer", "i use a", "i already tried")):
            return "Got it, I'll remember that."
        return "I don't have enough context yet."


def _match(text: str, pattern: str) -> str | None:
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if not matches:
        return None
    return matches[-1].strip()


def _extract_procedure_steps(system_prompt: str) -> list[str]:
    steps: list[str] = []
    in_block = False
    for raw_line in system_prompt.splitlines():
        line = raw_line.strip()
        if line == "PROCEDURE CONTEXT:":
            in_block = True
            continue
        if in_block and not line:
            break
        if in_block and line.startswith("- "):
            value = line[2:]
            if value != "none":
                steps.append(value)
    return steps


def _or_unknown(value: str | None, template: str) -> str:
    if value is None:
        return "I don't have enough context yet."
    return template.format(value=value)
