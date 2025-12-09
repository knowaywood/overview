"""save and resume with hard drive."""

import json

from langchain_core.messages import convert_to_messages

import overview.config as cfg


def save(path: str, history: cfg.BaseState) -> None:
    state_json = history.model_dump_json()
    with open(path, "w", encoding="utf-8") as f:
        f.write(state_json)
    print(f"✅ Save history to {path}")


def resume(path: str) -> cfg.BaseState:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "messages" in data:
                data["messages"] = convert_to_messages(data["messages"])
        return cfg.BaseState(**data)
    except FileNotFoundError as e:
        raise e
