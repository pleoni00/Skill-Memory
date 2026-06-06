import json

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.helpers import new_task_from_user_message, new_text_message
from a2a.types import Part, TaskState

from core.entities import Turn, Conversation, Chunk
from core.interfaces import LLMClient, GraphStore, VectorStore, Extractor, Retriever, Merger, QueryBuilder, SummaryUpdater, EmbeddingService, MergeDecisionAgent

class ConvMemoryExecutor(AgentExecutor):
    def __init__(
        self,
        llm: LLMClient,
        graph: GraphStore,
        vector: VectorStore,
        extractor: Extractor,
        retriever: Retriever,
        decision_agent: MergeDecisionAgent,
        merger: Merger,
        query_builder: QueryBuilder,
        summary_updater: SummaryUpdater,
        embed_service: EmbeddingService,
    ):
        self._llm    = llm
        self._graph  = graph
        self._vector = vector
        self._extractor = extractor
        self._retriever = retriever
        self._decision_agent = decision_agent
        self._merger = merger
        self._query_builder = query_builder
        self._summary_updater = summary_updater
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
                new_text_message("Invalid payload: JSON expected.", task_id=task.id, context_id=task.context_id),
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
                new_text_message(f"Skill '{skill}' not recognized.", task_id=task.id, context_id=task.context_id),
                final=True,
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Cancellation not supported.")

    # ── skill: retrieval ──────────────────────────────────────────────────────

    async def _handle_retrieval(self, payload: dict, updater: TaskUpdater, task) -> None:
        await updater.update_status(
            TaskState.TASK_STATE_WORKING,
            new_text_message("Searching knowledge graph.....", task_id=task.id, context_id=task.context_id),
        )

        turns = [
            Turn(role=t["role"], content=t["content"])
            for t in payload.get("turns", [])
        ]

        to_retrieve = self._query_builder._needs_retrieval(turns)

        if not to_retrieve:
            await updater.add_artifact(
                [Part(text=json.dumps({"nodes": [], "query_used": ""}))],
                name="retrieval_result",
            )
            await updater.complete()
            return            

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

        retrieved_nodes = self._retriever.retrieve(synthetic_chunk)

        output = {
            "query_used": query,
            "nodes": [n.to_dict() for n in retrieved_nodes],
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
            new_text_message("Processing conversation.....", task_id=task.id, context_id=task.context_id),
        )

        turns = [
            Turn(role=t["role"], content=t["content"])
            for t in payload.get("turns", [])
        ]
        convo = Conversation(turns=turns)
        log   = []

        chunks = self._extractor.extract(convo)
        log.append(f"Extracted {len(chunks)} chunks.")

        if not chunks:
            await updater.add_artifact(
                [Part(text="\n".join(log))],
                name="ingestion_log",
            )
            await updater.complete()
            return

        # Retrieve + decide in batch
        nodes = self._retriever.retrieve_and_decide(chunks)
        batch = self._decision_agent.decide(chunks, nodes)

        for decision in batch.decisions:
            target_id = decision.target_node.id if decision.target_node else None
            parent_id = decision.parent_node.id if decision.parent_node else None

            line = f"Chunk '{decision.chunk.topic}': {decision.action.value}"
            if target_id:
                line += f" → target={target_id}"
            if parent_id:
                line += f" parent={parent_id}"
            line += f" ({decision.rationale})"
            log.append(line)

        # Apply batch
        affected = self._merger.apply(batch)
        nodes_modified = {n.id for n in affected}

        log.append(f"{len(affected)} nodes created or modified.")

        for node_id in nodes_modified:
            self._summary_updater.update_ancestors(node_id)

        log.append(f"Summaries updated for {len(nodes_modified)} nodes.")

        await updater.add_artifact(
            [Part(text="\n".join(log))],
            name="ingestion_log",
        )
        await updater.complete()        
        