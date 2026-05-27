import json

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.helpers import new_task_from_user_message, new_text_message
from a2a.types import Part, TaskState

from core.entities import Turn, Conversation, Chunk
from core.interfaces import LLMClient, GraphStore, VectorStore, Extractor, Retriever, Merger, QueryBuilder, SummaryUpdater, EmbeddingService

class ConvMemoryExecutor(AgentExecutor):
    def __init__(
        self,
        llm: LLMClient,
        graph: GraphStore,
        vector: VectorStore,
        extractor: Extractor,
        retriever: Retriever,
        merger: Merger,
        query_builder: QueryBuilder,
        summary_updater: SummaryUpdater,
        embed_service: EmbeddingService,
    ):
        self._llm    = llm
        self._graph  = graph
        self._vector = vector
        self._extractor = extractor,
        self._retriever = retriever,
        self._merger = merger,
        self._query_builder = query_builder,
        self._summary_updater = summary_updater,
        self._embed_service   = embed_service

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # ── setup task ────────────────────────────────────────────────────────
        task    = context.current_task or new_task_from_user_message(context.message)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await event_queue.enqueue_event(task)

        try:
            payload = json.loads(context.get_user_input())
        except (json.JSONDecodeError, TypeError):
            await updater.update_status(
                TaskState.TASK_STATE_FAILED,
                new_text_message("Payload non valido: atteso JSON.", task_id=task.id, context_id=task.context_id),
                final=True,
            )
            return

        skill = payload.get("skill")

        if skill == "retrieval":
            await self._handle_retrieval(payload, updater, task)
        elif skill == "ingestion":
            await self._handle_ingestion(payload, updater, task)
        else:
            await updater.update_status(
                TaskState.TASK_STATE_FAILED,
                new_text_message(f"Skill '{skill}' non riconosciuta.", task_id=task.id, context_id=task.context_id),
                final=True,
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Cancellazione non supportata.")

    # ── skill: retrieval ──────────────────────────────────────────────────────

    async def _handle_retrieval(self, payload: dict, updater: TaskUpdater, task) -> None:
        await updater.update_status(
            TaskState.TASK_STATE_WORKING,
            new_text_message("Ricerca nel knowledge graph...", task_id=task.id, context_id=task.context_id),
        )

        turns = [
            Turn(role=t["role"], content=t["content"])
            for t in payload.get("turns", [])
        ]
        top_k = payload.get("top_k", 5)

        query = self._query_builder.build(turns)
        if not query:
            await updater.add_artifact(
                [Part(text=json.dumps({"nodes": [], "query_used": ""}))],
                name="retrieval_result",
            )
            await updater.complete()
            return

        query_embedding = self._embed_service.embed_one(query)
        synthetic_chunk = Chunk(
            text                   = query,
            embedding              = query_embedding,
            topic                  = "search",
            source_conversation_id = "search",
        )

        results = self._retriever.retrieve(synthetic_chunk, top_k=top_k)

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
            ],
        }

        await updater.add_artifact(
            [Part(text=json.dumps(output, ensure_ascii=False, indent=2))],
            name="retrieval_result",
        )
        await updater.complete()

    # ── skill: ingestion ──────────────────────────────────────────────────────

    async def _handle_ingestion(self, payload: dict, updater: TaskUpdater, task) -> None:
        await updater.update_status(
            TaskState.TASK_STATE_WORKING,
            new_text_message("Processamento conversazione...", task_id=task.id, context_id=task.context_id),
        )

        turns = [
            Turn(role=t["role"], content=t["content"])
            for t in payload.get("turns", [])
        ]
        convo  = Conversation(turns=turns)
        log    = []

        chunks = self._extractor.extract(convo)
        log.append(f"Estratti {len(chunks)} chunk.")

        if not chunks:
            await updater.add_artifact(
                [Part(text="\n".join(log))],
                name="ingestion_log",
            )
            await updater.complete()
            return

        nodes_modified = set()
        for chunk in chunks:
            candidates = self._retriever.retrieve(chunk, top_k=3)
            decision   = self._merger.decide(chunk, candidates)
            result     = self._merger.apply(decision)

            log.append(
                f"Chunk '{chunk.topic}': {decision.action.value}"
                + (f" → {result.id}" if result else "")
                + f" ({decision.rationale})"
            )

            if result:
                nodes_modified.add(result.id)

        for node_id in nodes_modified:
            self._summary_updater.update_ancestors(node_id)

        log.append(f"Summary aggiornati per {len(nodes_modified)} nodi.")

        await updater.add_artifact(
            [Part(text="\n".join(log))],
            name="ingestion_log",
        )
        await updater.complete()