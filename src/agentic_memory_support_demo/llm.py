from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from .deterministic import DeterministicSupportLLM


class ChatLLM(Protocol):
    def generate(self, *, system_prompt: str, user_message: str) -> str:
        ...


@dataclass
class GeminiChatLLM:
    model: str = "gemini-2.5-flash"

    def __post_init__(self) -> None:
        from google import genai

        self._client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def generate(self, *, system_prompt: str, user_message: str) -> str:
        response = self._client.models.generate_content(
            model=self.model,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": f"{system_prompt}\n\nUser: {user_message}"},
                    ],
                }
            ],
        )
        return (response.text or "").strip()


def make_default_llm() -> ChatLLM:
    if os.getenv("GEMINI_API_KEY"):
        return GeminiChatLLM()
    return DeterministicSupportLLM()
