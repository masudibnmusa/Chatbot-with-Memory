import json
import re

from app import config


def parse_json(text: str):
    """Parse JSON from an LLM reply, tolerating code fences and stray prose."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return None
    return None


class LLMClient:
    """Thin wrapper over the Claude or OpenAI chat APIs."""

    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = provider or config.LLM_PROVIDER
        self.model = model or config.LLM_MODEL
        self._client = None

        if self.provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        elif self.provider == "openai":
            from openai import OpenAI

            self._client = OpenAI(api_key=config.OPENAI_API_KEY)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")

    def chat(self, system: str, messages: list[dict], max_tokens: int = 1024) -> str:
        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model, system=system, messages=messages, max_tokens=max_tokens
            )
            return "".join(b.text for b in response.content if b.type == "text")

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}] + messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def chat_json(self, system: str, user: str, max_tokens: int = 1024):
        text = self.chat(system, [{"role": "user", "content": user}], max_tokens)
        return parse_json(text)