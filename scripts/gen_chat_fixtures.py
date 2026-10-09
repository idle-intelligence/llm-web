#!/usr/bin/env python3
"""Writes lean-chat's exactness fixtures: for each model below, its chat
template and, for a set of conversations, the prompt text transformers
renders with

    tokenizer.apply_chat_template(messages, tools=tools,
                                  add_generation_prompt=..., tokenize=False)

crates/lean-chat/tests/hf_exactness.rs renders the same conversations with
lean-chat and requires byte-equal text.

Needs transformers in a venv (tokenizer files only, no weights, no torch):

    python3 -m venv .venv && .venv/bin/pip install transformers jinja2
    .venv/bin/python scripts/gen_chat_fixtures.py
"""
import json
import os

import transformers
from transformers import AutoTokenizer

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "crates", "lean-chat", "tests", "fixtures")

# name -> (Hugging Face repo, whether its template renders tools and tool messages)
MODELS = {
    "qwen2_5": ("Qwen/Qwen2.5-0.5B-Instruct", True),
    "qwen3": ("Qwen/Qwen3-0.6B", True),
    "smollm2": ("HuggingFaceTB/SmolLM2-360M-Instruct", False),
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a city. Pass <city> names & 'quotes' as given.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. Zürich"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_volume",
            "description": "Set the speaker volume in a room.",
            "parameters": {
                "type": "object",
                "properties": {
                    "room": {"type": "string"},
                    "level": {"type": "integer", "minimum": 0, "maximum": 100},
                    "fade": {"type": "number"},
                    "mute": {"type": "boolean"},
                },
                "required": ["level"],
            },
        },
    },
]


def sys_msg(text):
    return {"role": "system", "content": text}


def user(text):
    return {"role": "user", "content": text}


def assistant(text):
    return {"role": "assistant", "content": text}


def tool_calls(*calls):
    return {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in calls]}


def tool(text):
    return {"role": "tool", "content": text}


# (name, messages, use tools, add_generation_prompt, needs a tool-capable template)
CASES = [
    ("user_only", [user("What is the capital of France?")], False, True, False),
    ("system_user", [sys_msg("Answer briefly, in one or two sentences."), user("What is the capital of France?")], False, True, False),
    (
        "multi_turn",
        [
            sys_msg("Answer briefly, in one or two sentences."),
            user("What is the capital of France?"),
            assistant("The capital of France is Paris."),
            user("And of Germany?"),
        ],
        False,
        True,
        False,
    ),
    ("multi_turn_closed", [user("Hello!"), assistant("Hi! How can I help?")], False, False, False),
    ("unicode", [user("Écris une phrase sur la Seine 🌊, avec <balises> & \"guillemets\".\nNouvelle ligne.\tTab.")], False, True, False),
    ("tools_user", [user("What's the weather in Zürich?")], True, True, False),
    ("tools_system", [sys_msg("You control the house."), user("Turn the kitchen up to 30.")], True, True, True),
    (
        "tool_call_roundtrip",
        [
            sys_msg("You control the house."),
            user("What's the weather in Zürich?"),
            tool_calls(("get_weather", {"unit": "celsius", "city": "Zürich"})),
            tool('{"temperature": 21, "condition": "sunny"}'),
            assistant("It is 21 °C and sunny in Zürich."),
            user("Set the kitchen volume to 30."),
        ],
        True,
        True,
        True,
    ),
    (
        "parallel_tool_calls",
        [
            user("Mute the kitchen and tell me the weather in Paris."),
            tool_calls(("set_volume", {"room": "kitchen", "level": 0, "fade": 0.5, "mute": True}), ("get_weather", {"city": "Paris"})),
            tool('{"ok": true}'),
            tool('{"temperature": 18, "condition": "rain", "alerts": null}'),
        ],
        True,
        True,
        True,
    ),
    (
        "string_arguments",
        [
            user("Weather in Paris?"),
            tool_calls(("get_weather", '{"city": "Paris"}')),
            tool("18 °C"),
            assistant("18 °C in Paris."),
            user("Thanks!"),
        ],
        True,
        True,
        True,
    ),
    (
        "think_in_history",
        [
            user("What is 2 + 2?"),
            assistant("<think>\nTwo plus two is four.\n</think>\n\n4"),
            user("And 3 + 3?"),
        ],
        False,
        True,
        False,
    ),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, (repo, tool_capable) in MODELS.items():
        tok = AutoTokenizer.from_pretrained(repo)
        cases = []
        for case_name, messages, use_tools, add_generation_prompt, needs_tools in CASES:
            if needs_tools and not tool_capable:
                continue
            tools = TOOLS if use_tools else None
            rendered = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=add_generation_prompt, tokenize=False)
            cases.append({"name": case_name, "messages": messages, "tools": tools, "add_generation_prompt": add_generation_prompt, "rendered": rendered})
        out = {"model": repo, "transformers": transformers.__version__, "template": tok.chat_template, "cases": cases}
        path = os.path.join(OUT_DIR, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"{path}: {len(cases)} cases")


if __name__ == "__main__":
    main()
