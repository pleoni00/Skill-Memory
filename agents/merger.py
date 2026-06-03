# agents/llm_merger.py

from datetime import datetime
from typing import Optional

from core.entities import (
    Chunk, Node, ChunkDecision, BatchMergeDecision, MergeAction, NodeStatus
)
from core.interfaces import GraphStore, VectorStore, EmbeddingService


class LLMMerger:
    """
    Executes merge decisions. No decision logic — only writes.
    Receives a BatchMergeDecision and applies each ChunkDecision to the graph.
    """

    def __init__(
        self,
        graph: GraphStore,
        vector: VectorStore,
        embedding_service: EmbeddingService,
    ):
        self._graph = graph
        self._vector = vector
        self._embed  = embedding_service

    # ─────────────────────────────────────────────────────────────
    # Action handlers
    # ─────────────────────────────────────────────────────────────

    def _apply_add_root(self, decision: ChunkDecision) -> Node:
        chunk = decision.chunk
        embedding = self._embed.embed_one(chunk.text)
        node = Node(
            title     = chunk.topic,
            summary   = decision.new_summary or chunk.text[:200],
            content   = decision.new_content or chunk.text,
            source    = chunk.text,
            embedding = embedding,
        )
        self._graph.add_node(node)
        self._vector.upsert(node.id, embedding, is_root=True)
        return node

    def _apply_add_child(self, decision: ChunkDecision) -> Optional[Node]:
        if not decision.parent_node:
            # Demote to ADD_ROOT if parent is missing
            return self._apply_add_root(decision)

        chunk = decision.chunk
        embedding = self._embed.embed_one(chunk.text)
        node = Node(
            title     = chunk.topic,
            summary   = decision.new_summary or chunk.text[:200],
            content   = decision.new_content or chunk.text,
            source    = chunk.text,
            embedding = embedding,
        )
        self._graph.add_node(node)
        self._graph.add_edge(decision.parent_node.id, node.id)
        self._vector.upsert(node.id, embedding)
        self._graph.mark_stale(decision.parent_node.id)
        return node

    def _apply_update(self, decision: ChunkDecision) -> Optional[Node]:
        target = decision.target_node
        if not target:
            return None

        target.content    = decision.new_content or f"{target.content}\n\n---\n{decision.chunk.text}"
        target.summary    = decision.new_summary or target.summary
        target.updated_at = datetime.utcnow()
        target.status     = NodeStatus.ACTIVE

        embedding        = self._embed.embed_one(target.content)
        target.embedding = embedding

        self._graph.update_node(target)
        self._vector.upsert(target.id, embedding)
        self._graph.mark_stale(target.id)
        return target

    def _apply_merge(self, decision: ChunkDecision) -> Optional[Node]:
        # Same write path as UPDATE — distinction is semantic, made upstream
        return self._apply_update(decision)

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def apply(self, batch: BatchMergeDecision) -> list[Node]:
        """
        Applies all decisions in the batch.
        Returns the list of nodes that were created or modified.
        """
        affected: list[Node] = []

        for decision in batch.decisions:
            result = None

            if decision.action == MergeAction.SKIP:
                continue
            elif decision.action == MergeAction.ADD_ROOT:
                result = self._apply_add_root(decision)
            elif decision.action == MergeAction.ADD_CHILD:
                result = self._apply_add_child(decision)
            elif decision.action == MergeAction.UPDATE:
                result = self._apply_update(decision)
            elif decision.action == MergeAction.MERGE:
                result = self._apply_merge(decision)

            if result:
                affected.append(result)

        return affected