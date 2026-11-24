"""Client for any OpenAI-compatible /chat/completions endpoint (Zhipu GLM, DeepSeek, Qwen, OpenAI, vLLM...).

Configured from environment variables (see .env.example):
    EVOPLAN_BASE_URL   e.g. https://open.bigmodel.cn/api/paas/v4
    EVOPLAN_API_KEY
    EVOPLAN_MODEL      e.g. glm-4-flash
Standard library only, so the core package has no dependencies.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from evoplan.llm.base import Reply


class OpenAICompatLLM:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None,
                 timeout: float = 60.0, retries: int = 3) -> None:
        self.base_url = (base_url or os.environ.get("EVOPLAN_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("EVOPLAN_API_KEY", "")
        self.model = model or os.environ.get("EVOPLAN_MODEL", "")
        if not (self.base_url and self.api_key and self.model):
            raise ValueError("Set EVOPLAN_BASE_URL, EVOPLAN_API_KEY and EVOPLAN_MODEL (see .env.example)")
        self.timeout = timeout
        self.retries = retries

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.0, max_tokens: int = 256) -> Reply:
        body = json.dumps({"model": self.model, "messages": messages,
                           "temperature": temperature, "max_tokens": max_tokens}).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, method="POST", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                usage = data.get("usage") or {}
                return Reply(data["choices"][0]["message"]["content"] or "",
                             int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))
            except (urllib.error.URLError, TimeoutError, KeyError) as exc:
                last = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM request failed after {self.retries} attempts: {last}")


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines, existing environment wins."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"'))
