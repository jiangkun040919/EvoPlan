"""LLM interface with token accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Reply:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLM(Protocol):
    model: str

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.0, max_tokens: int = 256) -> Reply: ...


class ScriptedLLM:
    """Returns canned replies in order. For unit tests only."""

    model = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.0, max_tokens: int = 256) -> Reply:
        self.calls.append(messages)
        text = self.replies.pop(0) if self.replies else "ACTION: look"
        prompt = sum(len(m["content"]) for m in messages) // 4
        return Reply(text, prompt, len(text) // 4)
