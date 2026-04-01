from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from .deterministic import DeterministicSupportLLM


class ChatLLM(Protocol):
    def generate(self, *, system_prompt: str, user_message: str) -> str:
        ...


@dataclass
class OpenAIChatLLM:
    model: str = "gpt-4.1-mini"

    def __post_init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, *, system_prompt: str, user_message: str) -> str:
        response = self._client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.output_text.strip()


def make_default_llm() -> ChatLLM:
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIChatLLM()
    return DeterministicSupportLLM()
