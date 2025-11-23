"""Stub implementation of ChatOllama for local unit tests."""


class ChatOllama:
    def __init__(self, model: str, base_url: str | None = None):
        self.model = model
        self.base_url = base_url

    async def ainvoke(self, prompt: str):
        # Minimal async stub that returns a deterministic response
        return "__ok__"
