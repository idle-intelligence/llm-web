"""
Build fixtures/reference/inputs/{02,03,04}*.json from the real Sonos MCP
tool schemas (fixtures/sonos/tools-12.json, fixtures/sonos/tools.json),
replacing the hand-written PLACEHOLDER tool list that used to live inline
in those files. 01_no_tools.json has no tools and is left untouched.

venv mon ami: run with scripts/.venv/bin/python (see scripts/README.md).

MCP -> transformers/OpenAI tool shape conversion:
  MCP tools/list entry:  {"name", "description", "inputSchema"}
  -> function-calling entry: {"type": "function", "function": {
        "name": <name>, "description": <description>, "parameters": <inputSchema>
     }}
Copied verbatim field-for-field (no edits to descriptions/schemas) -- this
matches the shape already used in 02_tools_single.json/03_tools_multiturn.json
before this script existed, and is the shape export_reference.py's
tokenizer.apply_chat_template(..., tools=tools) call expects as input (see
docs/MODELS.md: the chat template itself just tojson-dumps whatever object
sits in each tools[] entry -- it doesn't unwrap the type/function wrapper --
so this wrapped shape ends up serialized as-is inside the rendered prompt's
system block, same as it did with the placeholders).

Messages (system prompt, user text, the fake multi-turn assistant tool-call
+ tool-result exchange in 03) are NOT touched by this script -- they're
copied from the existing 02/03 input files' "messages" key as-is. Only the
"tools" key is regenerated.
"""
import json
import os

REPO_ROOT = os.environ.get(
    "LLM_REPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
INPUTS_DIR = os.path.join(REPO_ROOT, "fixtures/reference/inputs")
SONOS_DIR = os.path.join(REPO_ROOT, "fixtures/sonos")


def mcp_to_function_tools(mcp_tools):
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["inputSchema"],
            },
        }
        for t in mcp_tools
    ]


def load_messages(name):
    with open(os.path.join(INPUTS_DIR, f"{name}.json")) as f:
        return json.load(f)["messages"]


def write_input(name, messages, tools):
    path = os.path.join(INPUTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump({"messages": messages, "tools": tools}, f, indent=2)
        f.write("\n")
    print(f"wrote {path} ({len(tools)} tools)")


def main():
    with open(os.path.join(SONOS_DIR, "tools-12.json")) as f:
        tools_12 = json.load(f)
    with open(os.path.join(SONOS_DIR, "tools.json")) as f:
        tools_34 = json.load(f)

    messages_02 = load_messages("02_tools_single")
    messages_03 = load_messages("03_tools_multiturn")

    write_input("02_tools_single", messages_02, mcp_to_function_tools(tools_12))
    write_input("03_tools_multiturn", messages_03, mcp_to_function_tools(tools_12))
    # 04: same messages/shape as 02 but all 34 tools -- prefill-size probe,
    # no logits (see export_reference.py --render-only).
    write_input("04_tools_all", messages_02, mcp_to_function_tools(tools_34))


if __name__ == "__main__":
    main()
