"""
Thin LLM client wrapper. Defaults to the Groq API, but the interface
(`complete(system, user) -> str`) is what agents depend on, so swapping in
OpenAI/local models means writing one new class.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import groq


def load_local_env():
    paths = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[1] / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    for p in paths:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip().strip('"').strip("'")
            break


class LLMClient:
    def __init__(self, model: str = "llama-3.3-70b-versatile", api_key: Optional[str] = None):
        load_local_env()
        self.model = model
        self.client = groq.Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))

    def complete(self, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.2) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content


