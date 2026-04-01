from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_memory import Memory

from .deterministic import HashingEmbedder
from .llm import ChatLLM, make_default_llm


@dataclass
class ChatbotResponse:
    text: str
    recalled_memories: list[str]
    recalled_procedures: list[str]


class SupportChatbot:
    def __init__(
        self,
        *,
        llm: ChatLLM | None = None,
        memory: Memory | None = None,
        enable_memory: bool = True,
        session_id: str = "demo-session",
    ) -> None:
        self.llm = llm or make_default_llm()
        self.enable_memory = enable_memory
        self.session_id = session_id
        self.memory = memory or self._make_default_memory()
        self._seed_default_procedures()

    def reply(self, user_message: str) -> ChatbotResponse:
        if self.enable_memory:
            self._capture_user_message(user_message)

        recalled_memories: list[str] = []
        recalled_procedures: list[str] = []
        if self.enable_memory:
            recalled_memories = [result.record.content for result in self.memory.recall(user_message, top_k=4)]
            recalled_procedures = [
                match.record.content for match in self.memory.recall_procedures(user_message, top_k=1)
            ]

        system_prompt = self._build_system_prompt(recalled_memories, recalled_procedures)
        answer = self.llm.generate(system_prompt=system_prompt, user_message=user_message)

        if self.enable_memory:
            self.memory.remember_episode(
                f"Assistant replied: {answer}",
                session=self.session_id,
                participants=["assistant"],
                summary="assistant_response",
                metadata={"role": "assistant"},
            )

        return ChatbotResponse(
            text=answer,
            recalled_memories=recalled_memories,
            recalled_procedures=recalled_procedures,
        )

    def _capture_user_message(self, user_message: str) -> None:
        self.memory.remember_episode(
            f"User said: {user_message}",
            session=self.session_id,
            participants=["user"],
            summary="user_message",
            metadata={"role": "user"},
        )

        fact = _extract_fact(user_message)
        if fact is not None:
            self.memory.remember(
                fact["content"],
                category=fact["category"],
                domain="support_demo",
                metadata=fact["metadata"],
            )

    def _build_system_prompt(self, recalled_memories: list[str], recalled_procedures: list[str]) -> str:
        memory_lines = "\n".join(f"- {item}" for item in recalled_memories) or "- none"
        procedure_lines = "\n".join(
            f"- {step}"
            for step in _procedure_steps_from_contents(recalled_procedures)
        ) or "- none"
        return (
            "You are a concise support copilot.\n"
            "Answer using the available memory and procedure context.\n\n"
            "MEMORY CONTEXT:\n"
            f"{memory_lines}\n\n"
            "PROCEDURE CONTEXT:\n"
            f"{procedure_lines}\n"
        )

    def _seed_default_procedures(self) -> None:
        if not self.enable_memory:
            return
        existing = self.memory.recall_procedures("login loop", top_k=5)
        if existing:
            return

        procedures = [
            (
                "Troubleshoot a login loop after an SSO or cookie mismatch",
                [
                    "Confirm whether the login loop started after an SSO or cookie change",
                    "Clear cookies for the product domain",
                    "Retry sign-in in an incognito window",
                    "Re-authenticate the SSO session",
                ],
            ),
            (
                "Troubleshoot webhook delivery failures after a billing deployment",
                [
                    "Check the deployment timestamp against the first failed webhook",
                    "Inspect signature verification settings",
                    "Replay one failed event in a staging or dry-run environment",
                    "Escalate with request ids if signature replay still fails",
                ],
            ),
        ]

        for content, steps in procedures:
            self.memory.remember_procedure(
                content,
                steps=steps,
                importance=0.9,
                metadata={"seeded": True},
            )

    def _make_default_memory(self) -> Memory:
        chroma_path = Path(tempfile.mkdtemp(prefix="agentic_memory_support_demo_"))
        media_root = chroma_path / "media"
        embedder = HashingEmbedder(dimensions=64)
        return Memory(
            chroma_path=str(chroma_path),
            media_root=str(media_root),
            embedder=embedder,
            embedding_dimensions=64,
        )


def _extract_fact(message: str) -> dict[str, Any] | None:
    lowered = message.lower()
    patterns = [
        ("my name is ", "profile", "name"),
        ("i'm on the ", "profile", "plan"),
        ("i am on the ", "profile", "plan"),
        ("my timezone is ", "profile", "timezone"),
        ("i prefer the ", "preference", "theme"),
        ("i prefer ", "preference", "notifications"),
        ("i use a ", "profile", "device"),
        ("i already tried ", "troubleshooting", "attempted_step"),
        ("my issue is ", "issue", "issue_summary"),
        ("issue: ", "issue", "issue_summary"),
    ]

    for prefix, category, field in patterns:
        index = lowered.find(prefix)
        if index == -1:
            continue
        content = message[index:].strip().rstrip(".!?")
        return {
            "content": content,
            "category": category,
            "metadata": {"field": field},
        }
    return None


def _procedure_steps_from_contents(contents: list[str]) -> list[str]:
    steps: list[str] = []
    for content in contents:
        lowered = content.lower()
        if "login loop" in lowered:
            steps.extend(
                [
                    "Confirm whether the login loop started after an SSO or cookie change",
                    "Clear cookies for the product domain",
                    "Retry sign-in in an incognito window",
                    "Re-authenticate the SSO session",
                ]
            )
        if "webhook delivery failures" in lowered:
            steps.extend(
                [
                    "Check the deployment timestamp against the first failed webhook",
                    "Inspect signature verification settings",
                    "Replay one failed event in a staging or dry-run environment",
                    "Escalate with request ids if signature replay still fails",
                ]
            )
    return steps
