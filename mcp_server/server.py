import asyncio
import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

try:
    from .common import handle_search, handle_store
except ImportError:  # pragma: no cover
    from mcp_server.common import handle_search, handle_store

app = Server("dag-memory")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search",
            description=(
                "Search for information in DAG memory. "
                "Pass the last N conversation turns: "
                "the server builds the query and returns the most relevant nodes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "turns": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "enum": ["user", "assistant"]},
                                "content": {"type": "string"},
                            },
                            "required": ["role", "content"],
                        },
                    },
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["turns"],
            },
        ),
        types.Tool(
            name="store_conversation",
            description=(
                "Process a conversation and update DAG memory. "
                "Call this at the end of a conversation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "turns": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "enum": ["user", "assistant"]},
                                "content": {"type": "string"},
                            },
                            "required": ["role", "content"],
                        },
                    }
                },
                "required": ["turns"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "search":
        return [types.TextContent(type="text", text=json.dumps(handle_search(arguments), ensure_ascii=False, indent=2))]
    if name == "store_conversation":
        return [types.TextContent(type="text", text=handle_store(arguments))]
    return [types.TextContent(type="text", text=f"Tool '{name}' not recognized.")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
