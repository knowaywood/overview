"""for prompt of agents."""

from typing import TypedDict


class BaseState(TypedDict):
    """base state of agent."""

    state: str


MAIN_AGENT_PROMPT: str = (
    "you are a helpful assistant that helps users to accomplish tasks with tools like."
)
