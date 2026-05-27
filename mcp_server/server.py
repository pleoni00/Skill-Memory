"""
DAG Memory MCP Server
---------------------
Variabili d'ambiente:
  LLM_BASE_URL      — base URL del server LLM  (default: https://api.x.ai/v1)
  LLM_MODEL         — modello da usare          (default: grok-3-mini)
  LLM_API_KEY       — api key del server LLM    (default: dummy)
  EMBED_BASE_URL    — base URL embedding        (default: stesso di LLM_BASE_URL)
  EMBED_MODEL       — modello embedding         (default: text-embedding-3-small)
  DAG_DB_PATH       — path DB Kuzu             (default: ./data/dag)
  VEC_DB_PATH       — path DB sqlite-vec       (default: ./data/vec.db)
  EMBED_DIM         — dimensione embedding     (default: 1536)

Avvio:
  LLM_API_KEY=xai-... LLM_MODEL=grok-3-mini python mcp/server.py
"""

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
from storage import KuzuGraphStore, SqliteVectorStore
from agents import (
    OpenAICompatibleLLMClient,
    OpenAICompatibleEmbeddingService,
    LocalEmbeddingService,
    LLMExtractor,
    HybridRetriever,
    LLMMerger,
    LLMQueryBuilder,
    LLMSummaryUpdater,
)

# ── Config ────────────────────────────────────────────────────────────────────

LLM_BASE_URL   = os.environ.get("LLM_BASE_URL",   "https://api.x.ai/v1")
LLM_MODEL      = os.environ.get("LLM_MODEL",      "grok-3-mini")
LLM_API_KEY    = os.environ.get("LLM_API_KEY",    "dummy")
EMBED_BASE_URL = os.environ.get("EMBED_BASE_URL",  LLM_BASE_URL)
EMBED_MODEL    = os.environ.get("EMBED_MODEL",     "text-embedding-3-small")
EMBED_API_KEY  = os.environ.get("EMBED_API_KEY",   LLM_API_KEY)
DAG_DB_PATH    = os.environ.get("DAG_DB_PATH",    "./data/dag")
VEC_DB_PATH    = os.environ.get("VEC_DB_PATH",    "./data/vec.db")
EMBED_DIM      = int(os.environ.get("EMBED_DIM",  "1536"))

Path("./data").mkdir(exist_ok=True)

# ── Storage ───────────────────────────────────────────────────────────────────

graph  = KuzuGraphStore(DAG_DB_PATH)
vector = SqliteVectorStore(VEC_DB_PATH, embedding_dim=EMBED_DIM)

# ── LLM client ────────────────────────────────────────────────────────────────

llm = OpenAICompatibleLLMClient(
    model    = LLM_MODEL,
    base_url = LLM_BASE_URL,
    api_key  = LLM_API_KEY,
)

# ── Embedding service ─────────────────────────────────────────────────────────

try:
    embed_service = OpenAICompatibleEmbeddingService(
        model    = EMBED_MODEL,
        base_url = EMBED_BASE_URL,
        api_key  = EMBED_API_KEY,
    )
except Exception:
    print("[warning] Embedding remoto non disponibile, uso locale (384d)", file=sys.stderr)
    embed_service = LocalEmbeddingService()

# ── Agenti ────────────────────────────────────────────────────────────────────

extractor       = LLMExtractor(embed_service, llm)
retriever       = HybridRetriever(llm, graph, vector)
merger          = LLMMerger(graph, vector, embed_service, llm)
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
        ),
        types.Tool(
            name        = "configure",
            description = (
                "Configura il contesto dell'agente. "
                "Imposta l'obiettivo, il ruolo e le istruzioni permanenti "
                "che guidano come la memoria viene costruita e interrogata."
            ),
            inputSchema = {
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "Descrizione dell'agente, obiettivo, contesto permanente."
                    }
                },
                "required": ["context"]
            }
        ),
        types.Tool(
            name        = "get_node",
            description = "Legge un nodo specifico del DAG per ID.",
            inputSchema = {
                "type": "object",
                "properties": {"node_id": {"type": "string"}},
                "required": ["node_id"]
            }
        ),
        types.Tool(
            name        = "get_children",
            description = "Restituisce i figli di un nodo per navigazione manuale del DAG.",
            inputSchema = {
                "type": "object",
                "properties": {"node_id": {"type": "string"}},
                "required": ["node_id"]
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:

    if name == "search":
        return await _handle_search(arguments)

    if name == "store_conversation":
        return await _handle_store(arguments)

    if name == "get_node":
        node = graph.get_node(arguments["node_id"])
        if not node:
            return [types.TextContent(type="text", text="Nodo non trovato.")]
        return [types.TextContent(type="text", text=json.dumps(node.to_dict(), ensure_ascii=False))]

    if name == "get_children":
        children = graph.get_children(arguments["node_id"])
        return [types.TextContent(type="text", text=json.dumps(
            [c.to_dict() for c in children], ensure_ascii=False
        ))]

    if name == "configure":
        return await _handle_configure(arguments)

    return [types.TextContent(type="text", text=f"Tool '{name}' non riconosciuto.")]


async def _handle_search(args: dict) -> list[types.TextContent]:
    turns  = [Turn(role=t["role"], content=t["content"]) for t in args.get("turns", [])]
    top_k  = args.get("top_k", 5)

    query = query_builder.build(turns)
    if not query:
        return [types.TextContent(type="text", text=json.dumps({"nodes": [], "query_used": ""}))]

    query_embedding  = embed_service.embed_one(query)
    synthetic_chunk  = Chunk(
        text                   = query,
        embedding              = query_embedding,
        topic                  = "search",
        source_conversation_id = "search",
    )

    results = retriever.retrieve(synthetic_chunk, top_k=top_k)

    output = {
        "query_used": query,
        "nodes": [
            {
                **r.node.to_dict(),
                "vector_score":   round(r.vector_score, 3),
                "dag_score":      round(r.dag_score, 3),
                "combined_score": round(r.combined_score, 3),
            }
            for r in results
        ]
    }

    return [types.TextContent(type="text", text=json.dumps(output, ensure_ascii=False, indent=2))]


async def _handle_store(args: dict) -> list[types.TextContent]:
    turns = [Turn(role=t["role"], content=t["content"]) for t in args.get("turns", [])]
    convo = Conversation(turns=turns)
    log   = []

    chunks = extractor.extract(convo)
    log.append(f"Estratti {len(chunks)} chunk.")

    if not chunks:
        return [types.TextContent(type="text", text="\n".join(log))]

    nodes_modified = set()
    for chunk in chunks:
        candidates = retriever.retrieve(chunk, top_k=3)
        decision   = merger.decide(chunk, candidates)
        result     = merger.apply(decision)

        log.append(
            f"Chunk '{chunk.topic}': {decision.action.value}"
            + (f" → {result.id}" if result else "")
            + f" ({decision.rationale})"
        )

        if result:
            nodes_modified.add(result.id)

    for node_id in nodes_modified:
        summary_updater.update_ancestors(node_id)

    log.append(f"Summary aggiornati per {len(nodes_modified)} nodi.")

    return [types.TextContent(type="text", text="\n".join(log))]


async def _handle_configure(args: dict) -> list[types.TextContent]:
    context = args.get("context", "").strip()
    turns = [Turn(role="system", content=context)]
    convo = Conversation(turns=turns)
    log   = []

    chunks = extractor.extract(convo)
    log.append(f"Estratti {len(chunks)} chunk.")

    if not chunks:
        return [types.TextContent(type="text", text="\n".join(log))]

    for chunk in chunks:
        candidates = retriever.retrieve(chunk, top_k=3)
        decision   = merger.decide(chunk, candidates)
        result     = merger.apply(decision)

        log.append(
            f"Chunk '{chunk.topic}': {decision.action.value}"
            + (f" → {result.id}" if result else "")
            + f" ({decision.rationale})"
        )
    
    return [types.TextContent(type="text", text="\n".join(log))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())