"""A scripted stand-in for the LLM so agent tests need no key and no network.

Each step is either say("text") for a final reply, or call(name, **args) for a
tool request (several call() objects in a list = several tool calls at once).
"""
import json
from types import SimpleNamespace


def say(text):
    return SimpleNamespace(content=text, tool_calls=None)


def call(name, **args):
    return SimpleNamespace(
        id=f"call_{name}", function=SimpleNamespace(name=name, arguments=json.dumps(args))
    )


def calls(*tool_calls):
    return SimpleNamespace(content=None, tool_calls=list(tool_calls))


class FakeLLM:
    def __init__(self, *steps):
        self.steps = list(steps)
        self.requests = []  # what the agent sent each time

    def __call__(self, settings, messages, tools=None, timeout=None):
        self.requests.append({"messages": list(messages), "tools": tools, "timeout": timeout})
        if not self.steps:
            raise AssertionError("The agent asked the LLM more times than the test scripted")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step
