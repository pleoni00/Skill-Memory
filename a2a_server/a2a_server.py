import os
from pathlib import Path
from starlette.applications import Starlette
import uvicorn

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.types import AgentCard, AgentSkill, AgentCapabilities, AgentInterface

from agents import OpenAICompatibleLLMClient, OpenAICompatibleEmbeddingService, LLMExtractor, LLMMerger, HybridRetriever, LLMQueryBuilder, LLMSummaryUpdater, LLMMergeDecisionAgent
from storage import SqliteGraphStore, SqliteVectorStore

from a2a_server.agent_executor import ConvMemoryExecutor

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

# ── Storage ───────────────────────────────────────────────────────────────────

graph  = SqliteGraphStore(DAG_DB_PATH)
vector = SqliteVectorStore(VEC_DB_PATH, embedding_dim=EMBED_DIM)

# ── LLM client ────────────────────────────────────────────────────────────────

llm = OpenAICompatibleLLMClient(
    model    = LLM_MODEL,
    base_url = LLM_BASE_URL,
    api_key  = LLM_API_KEY,
)
embed_service = OpenAICompatibleEmbeddingService(model = EMBED_MODEL, base_url = EMBED_BASE_URL, api_key = EMBED_API_KEY)
extractor = LLMExtractor(embed_service, llm)
decision_agent = LLMMergeDecisionAgent(llm, graph)
retriever = HybridRetriever(llm, graph, vector)
merger = LLMMerger(graph, vector, embed_service)
query_builder = LLMQueryBuilder(llm)
summary_updater =  LLMSummaryUpdater(graph, llm)

agent_card = AgentCard(
    name='Conv Memory Agent',
    description='Agent for managing behavior memory through a hybrid DAG-based memory store.',
    version='1.0.0',
    supported_interfaces=[
        AgentInterface(protocol_binding='JSONRPC', url='http://localhost:8000/')
    ],
    capabilities=AgentCapabilities(streaming=True),
    skills=[
        AgentSkill(
            id='retrieval',
            name='Knowledge graph search',
            description=(
                'Given a set of conversation turns, retrieve the most relevant '
                'nodes from the hybrid behavior memory graph (DAG + vector store). '
                'Returns a list of nodes with title and content to inject as '
                'context into the LLM system prompt.'
            ),
            tags=['retrieval', 'search', 'memory', 'graph', 'vector'],
            examples=[
                'Retrieve relevant context for a conversation about DAG memory systems',
                'Find nodes related to LLM orchestration and agent memory design',
            ],
            input_modes=['application/json'],
            output_modes=['application/json'],
        ),
        AgentSkill(
            id='ingestion',
            name='Behavior memory storage',
            description=(
                'Receives a list of conversation turns (role + content), '
                'processes them by extracting entities and relationships, and persists '
                'them in the hybrid behavior memory graph while updating both the DAG and vector store.'
            ),
            tags=['ingestion', 'memory', 'store', 'graph', 'embedding'],
            examples=[
                'Store a behavior feedback conversation in memory',
                'Persist feedback about tone, verbosity, and structure',
            ],
            input_modes=['application/json'],
            output_modes=['text/plain'],
        ),
    ]
)

request_handler = DefaultRequestHandler(
    agent_executor=ConvMemoryExecutor(
        llm, 
        graph, 
        vector,
        extractor,
        retriever,
        decision_agent,
        merger,
        query_builder,
        summary_updater,
        embed_service
    ),
    task_store=InMemoryTaskStore(),
    agent_card=agent_card,
)

routes = []
routes.extend(create_agent_card_routes(agent_card))
routes.extend(create_jsonrpc_routes(request_handler, rpc_url='/'))

app = Starlette(routes=routes)

if __name__ == "__main__":
    for route in routes:
        print(route.path)
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level="info")