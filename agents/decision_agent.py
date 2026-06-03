# agents/merge_decision_agent.py

import json
from core.entities import (
    Chunk, Node, ChunkDecision, BatchMergeDecision, MergeAction 
)
from core.interfaces import GraphStore, MergeDecisionAgent
from agents.llm_client import LLMClient
from agents.prompts import MERGE_DECISION_PROMPT


class LLMMergeDecisionAgent(MergeDecisionAgent):
    """
    Receives a batch of chunks + relevant existing nodes.
    Returns a BatchMergeDecision: one or more ChunkDecision per chunk.
    Does NOT execute any write — that is LLMMerger's responsibility.
    """

    def __init__(self, llm: LLMClient, graph: GraphStore):
        self._llm   = llm
        self._graph = graph

    # ─────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────

    def _format_chunks(self, chunks: list[Chunk]) -> str:
        return "\n\n".join(
            f"CHUNK {c.id}:\nTopic: {c.topic}\nText: {c.text}"
            for c in chunks
        )

    def _format_nodes(self, nodes: list[Node]) -> str:
        return "\n\n".join(
            f"NODE {n.id}:\nTitle: {n.title}\nSummary: {n.summary or ''}\nContent: {n.content[:400]}"
            for n in nodes
        )

    def _parse_response(
        self,
        raw: str,
        chunks: list[Chunk],
        nodes: list[Node],
    ) -> BatchMergeDecision:
        chunk_index = {c.id: c for c in chunks}
        node_index  = {n.id: n for n in nodes}

        try:
            data = json.loads(raw.strip())
        except json.JSONDecodeError:
            # Fallback: add all chunks as roots
            return BatchMergeDecision(decisions=[
                ChunkDecision(
                    action      = MergeAction.ADD_ROOT,
                    chunk       = c,
                    target_node = None,
                    parent_node = None,
                    new_content = c.text,
                    new_summary = None,
                    rationale   = "Parse error — fallback to ADD_ROOT",
                )
                for c in chunks
            ])

        decisions: list[ChunkDecision] = []

        for item in data.get("decisions", []):
            chunk = chunk_index.get(item.get("chunk_id"))
            if not chunk:
                continue

            action      = MergeAction(item["action"])
            target_node = node_index.get(item.get("target_node_id"))
            parent_node = node_index.get(item.get("parent_node_id"))

            # Fetch from graph if not in local index
            if not target_node and item.get("target_node_id"):
                target_node = self._graph.get_node(item["target_node_id"])
            if not parent_node and item.get("parent_node_id"):
                parent_node = self._graph.get_node(item["parent_node_id"])

            decisions.append(ChunkDecision(
                action      = action,
                chunk       = chunk,
                target_node = target_node,
                parent_node = parent_node,
                new_content = item.get("new_content"),
                new_summary = item.get("new_summary"),
                rationale   = item.get("rationale", ""),
            ))

        return BatchMergeDecision(decisions=decisions)

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def decide(self, chunks: list[Chunk], nodes: list[Node]) -> BatchMergeDecision:
        """
        Decide how to integrate a batch of chunks into the existing graph.
        Returns a BatchMergeDecision with one or more decisions per chunk.
        """
        if len(chunks) == 0:
            return BatchMergeDecision()

        user_msg = (
            f"NEW CHUNKS:\n{self._format_chunks(chunks)}\n\n"
            f"EXISTING NODES:\n{self._format_nodes(nodes) if nodes else 'None.'}"
        )

        raw = self._llm.complete(
            system     = MERGE_DECISION_PROMPT,
            user       = user_msg,
            max_tokens = 1000,
        )

        return self._parse_response(raw, chunks, nodes)