"""Test doubles shared by the pipeline tests (no network)."""


class FakeLLM:
    """Stands in for ChatOpenAI: returns a fixed structured result and keeps the prompt it was sent."""

    def __init__(self, result):
        self.result = result
        self.messages = None
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, messages):
        self.messages = messages
        return self.result
