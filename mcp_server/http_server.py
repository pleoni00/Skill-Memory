import asyncio
import json

from mcp.server.fastmcp.server import FastMCP

from .common import handle_search, handle_store

mcp = FastMCP(
    name="dag-memory",
    streamable_http_path="/mcp",
    host="0.0.0.0",
    port=8000,
    stateless_http=False,
)


@mcp.tool(
    name="search",
    description=(
        "Search for information in DAG memory. "
        "Pass the last N conversation turns: "
        "the server builds the query and returns the most relevant nodes."
    ),
)
async def search(turns: list[dict], top_k: int = 5) -> dict:
    return handle_search({"turns": turns, "top_k": top_k})


@mcp.tool(
    name="store_conversation",
    description=(
        "Process a conversation and update DAG memory. "
        "Call this at the end of a conversation."
    ),
)
async def store_conversation(turns: list[dict]) -> str:
    return handle_store({"turns": turns})


async def main():
    await mcp.run_streamable_http_async()


if __name__ == "__main__":
    asyncio.run(main())
