import os
import sys
import asyncio
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from core.entities import Turn, Conversation, Chunk
from storage import SqliteGraphStore, SqliteVectorStore
from agents import (
    OpenAICompatibleLLMClient,
    OpenAICompatibleEmbeddingService,
    LLMExtractor,
    LLMMerger,
    HybridRetriever,
    LLMQueryBuilder,
    LLMSummaryUpdater,
    LLMMergeDecisionAgent,
)

# ── Config ────────────────────────────────────────────────────────────────────

LLM_BASE_URL   = os.environ.get("LLM_BASE_URL",   "https://api.x.ai/v1")
LLM_MODEL      = os.environ.get("LLM_MODEL",      "grok-3-mini")
LLM_API_KEY    = os.environ.get("LLM_API_KEY",    "dummy")
EMBED_BASE_URL = os.environ.get("EMBED_BASE_URL",  LLM_BASE_URL)
EMBED_MODEL    = os.environ.get("EMBED_MODEL",     "text-embedding-3-small")
EMBED_API_KEY  = os.environ.get("EMBED_API_KEY",   LLM_API_KEY)
DAG_DB_PATH    = os.environ.get("DAG_DB_PATH",    "./data/vec.db")
VEC_DB_PATH    = os.environ.get("VEC_DB_PATH",    "./data/vec.db")
EMBED_DIM      = int(os.environ.get("EMBED_DIM",  "1536"))

Path("./data").mkdir(exist_ok=True)

# ── Storage (SqliteGraphStore come in A2A) ────────────────────────────────────

graph  = SqliteGraphStore(DAG_DB_PATH)
vector = SqliteVectorStore(VEC_DB_PATH, embedding_dim=EMBED_DIM)

# ── LLM client ────────────────────────────────────────────────────────────────

llm = OpenAICompatibleLLMClient(
    model    = LLM_MODEL,
    base_url = LLM_BASE_URL,
    api_key  = LLM_API_KEY,
)
embed_service = OpenAICompatibleEmbeddingService(
    model    = EMBED_MODEL,
    base_url = EMBED_BASE_URL,
    api_key  = EMBED_API_KEY,
)

# ── Agenti (stessa istanza di A2A) ────────────────────────────────────────────

extractor       = LLMExtractor(embed_service, llm)
decision_agent  = LLMMergeDecisionAgent(llm, graph)
retriever       = HybridRetriever(llm, graph, vector)
merger          = LLMMerger(graph, vector, embed_service)
query_builder   = LLMQueryBuilder(llm)
summary_updater = LLMSummaryUpdater(graph, llm)

# ── MCP Server ────────────────────────────────────────────────────────────────

app = Server("dag-memory")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name        = "search",
            description = (
                "Cerca informazioni nella memoria DAG. "
                "Passa gli ultimi N turni della conversazione: "
                "il server costruisce la query e restituisce i nodi più rilevanti."
            ),
            inputSchema = {
                "type": "object",
                "properties": {
                    "turns": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role":    {"type": "string", "enum": ["user", "assistant"]},
                                "content": {"type": "string"}
                            },
                            "required": ["role", "content"]
                        }
                    },
                    "top_k": {"type": "integer", "default": 5}
                },
                "required": ["turns"]
            }
        ),
        types.Tool(
            name        = "store_conversation",
            description = (
                "Processa una conversazione e aggiorna la memoria DAG. "
                "Da chiamare a fine conversazione."
            ),
            inputSchema = {
                "type": "object",
                "properties": {
                    "turns": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role":    {"type": "string", "enum": ["user", "assistant"]},
                                "content": {"type": "string"}
                            },
                            "required": ["role", "content"]
                        }
                    }
                },
                "required": ["turns"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:

    if name == "search":
        return await _handle_search(arguments)

    if name == "store_conversation":
        return await _handle_store(arguments)

    return [types.TextContent(type="text", text=f"Tool '{name}' non riconosciuto.")]


async def _handle_search(args: dict) -> list[types.TextContent]:
    turns = [
        Turn(role=t["role"], content=t["content"])
        for t in args.get("turns", [])
    ]

    if not query_builder._needs_retrieval(turns):
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "nodes": [],
                        "query_used": ""
                    }
                )
            )
        ]

    query = query_builder.build(turns)
    if not query:
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "nodes": [],
                        "query_used": ""
                    }
                )
            )
        ]

    embedding = embed_service.embed_one(query)
    synthetic_chunk = Chunk(
        text=query,
        embedding=embedding,
        topic="search",
        source_conversation_id="search",
    )
    retrieved = retriever.retrieve(synthetic_chunk)
    output = {
        "query_used": query,
        "nodes": [
            n.to_dict()
            for n in retrieved
        ]
    }

    return [
        types.TextContent(
            type="text",
            text=json.dumps(
                output,
                ensure_ascii=False,
                indent=2,
            ),
        )
    ]


async def _handle_store(args: dict) -> list[types.TextContent]:
    turns = [
        Turn(role=t["role"], content=t["content"])
        for t in args.get("turns", [])
    ]

    convo = Conversation(turns=turns)
    log = []
    chunks = extractor.extract(convo)

    log.append(f"Estratti {len(chunks)} chunk.")
    if not chunks:
        return [types.TextContent(type="text", text="\n".join(log))]
    nodes = retriever.retrieve_and_decide(chunks)
    batch = decision_agent.decide(chunks, nodes)

    for decision in batch.decisions:
        target_id = (
            decision.target_node.id
            if decision.target_node
            else None
        )
        parent_id = (
            decision.parent_node.id
            if decision.parent_node
            else None
        )
        line = (
            f"Chunk '{decision.chunk.topic}': "
            f"{decision.action.value}"
        )

        if target_id:
            line += f" → target={target_id}"
        if parent_id:
            line += f" parent={parent_id}"
        line += f" ({decision.rationale})"
        log.append(line)

    affected_nodes = merger.apply(batch)
    nodes_modified = {n.id for n in affected_nodes}
    log.append(f"{len(affected_nodes)} nodi creati o modificati.")
    for node_id in nodes_modified:
        summary_updater.update_ancestors(node_id)
    log.append(f"Summary aggiornati per {len(nodes_modified)} nodi.")

    return [types.TextContent(type="text", text="\n".join(log))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())