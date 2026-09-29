from app.llm.prompt_templates import CHAT_SYSTEM_PROMPT


class PromptBuilder:
    """Combine system prompt + retrieved memories + recent history."""

    def build(self, memories, history: list[dict]) -> tuple[str, list[dict]]:
        if memories:
            lines = "\n".join(f"- {m.text}" for m in memories)
            block = f"Things you know about this user:\n{lines}"
        else:
            block = "You don't have any stored memories about this user yet."

        system = CHAT_SYSTEM_PROMPT.format(memories_block=block)

        messages = [dict(m) for m in history]
        # Chat APIs require the first message to come from the user.
        while messages and messages[0]["role"] != "user":
            messages.pop(0)
        return system, messages