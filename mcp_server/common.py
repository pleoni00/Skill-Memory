import os
from pathlib import Path

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

graph  = SqliteGraphStore(DAG_DB_PATH)
vector = SqliteVectorStore(VEC_DB_PATH, embedding_dim=EMBED_DIM)

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

extractor       = LLMExtractor(embed_service, llm)
decision_agent  = LLMMergeDecisionAgent(llm, graph)
retriever       = HybridRetriever(llm, graph, vector)
merger          = LLMMerger(graph, vector, embed_service)
query_builder   = LLMQueryBuilder(llm)
summary_updater = LLMSummaryUpdater(graph, llm)


def handle_search(args: dict) -> dict:
    turns = [Turn(role=t["role"], content=t["content"]) for t in args.get("turns", [])]

    if not query_builder._needs_retrieval(turns):
        return {"nodes": [], "query_used": ""}

    query = query_builder.build(turns)
    if not query:
        return {"nodes": [], "query_used": ""}

    embedding = embed_service.embed_one(query)
    synthetic_chunk = Chunk(
        text=query,
        embedding=embedding,
        topic="search",
        source_conversation_id="search",
    )
    retrieved = retriever.retrieve(synthetic_chunk)
    return {
        "query_used": query,
        "nodes": [n.to_dict() for n in retrieved],
    }


def handle_store(args: dict) -> str:
    turns = [Turn(role=t["role"], content=t["content"]) for t in args.get("turns", [])]
    convo = Conversation(turns=turns)
    log = []

    chunks = extractor.extract(convo)
    log.append(f"Extracted {len(chunks)} chunks.")
    if not chunks:
        return "\n".join(log)

    nodes = retriever.retrieve_and_decide(chunks)
    batch = decision_agent.decide(chunks, nodes)

    for decision in batch.decisions:
        target_id = decision.target_node.id if decision.target_node else None
        parent_id = decision.parent_node.id if decision.parent_node else None
        line = f"Chunk '{decision.chunk.topic}': {decision.action.value}"
        if target_id:
            line += f" -> target={target_id}"
        if parent_id:
            line += f" parent={parent_id}"
        line += f" ({decision.rationale})"
        log.append(line)

    affected_nodes = merger.apply(batch)
    nodes_modified = {n.id for n in affected_nodes}
    log.append(f"{len(affected_nodes)} nodes created or modified.")

    for node_id in nodes_modified:
        summary_updater.update_ancestors(node_id)

    log.append(f"Summaries updated for {len(nodes_modified)} nodes.")
    return "\n".join(log)
