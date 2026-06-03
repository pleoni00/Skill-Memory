import json

from core.entities import Chunk, Node, BatchMergeDecision
from core.interfaces import Retriever, GraphStore, VectorStore
from agents.llm_client import LLMClient
from agents.prompts import RELEVANCE_FILTER


class HybridRetriever(Retriever):
    """
    Retrieval ibrido:
    1. Vector seeding (entry points)
    2. Graph traversal top-down dai capostipiti
    3. LLM-based relevance filtering
    4. Early stopping su saturazione o over-specialization
    """

    MAX_DEPTH = 5
    OVER_SPECIALIZATION_LIMIT = 50

    def __init__(
        self,
        llm: LLMClient,
        graph: GraphStore,
        vector: VectorStore,
        top_k: int = 3,
    ):
        self._llm    = llm
        self._graph  = graph
        self._vector = vector
        self._top_k  = top_k

    # ─────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────

    def _filter_relevant(self, query: Chunk, nodes: list[Node]) -> list[Node]:
        """Calls LLM relevance filter, returns only relevant nodes."""
        nodes_text = "\n".join(
            f"- id: {n.id}, name: {n.title}, summary: {n.summary or ''}"
            for n in nodes
        )
        prompt = RELEVANCE_FILTER.format(
            query_text=query.text,
            nodes_text=nodes_text,
        )
        response = self._llm.complete(
            system="You are a strict information retrieval filter.",
            user=prompt,
            max_tokens=500,
        )
        try:
            relevant_ids = set(json.loads(response))
        except json.JSONDecodeError:
            return []

        return [n for n in nodes if n.id in relevant_ids]

    def _build_seed_frontier(self, chunk: Chunk) -> set[str]:
        """Vector search + root nodes as starting frontier."""
        seed_hits = self._vector.search(chunk.embedding, top_k=self._top_k * 2)
        frontier = {node_id for node_id, _ in seed_hits}
        frontier.update(node.id for node in self._graph.get_roots())
        return frontier

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def retrieve(self, chunk: Chunk) -> list[Node]:
        """
        Pure retrieval. Returns relevant nodes with no scoring.
        Traverses the DAG top-down with LLM filtering at each level.
        """
        frontier = self._build_seed_frontier(chunk)
        visited:  set[str]  = set()
        results:  list[Node] = []

        for _ in range(self.MAX_DEPTH):
            if not frontier:
                break

            current_level = [
                node
                for node_id in frontier
                if node_id not in visited
                and (node := self._graph.get_node(node_id)) is not None
            ]

            for node_id in frontier:
                visited.add(node_id)

            if not current_level:
                break

            relevant = self._filter_relevant(chunk, current_level)

            if not relevant:
                break  # stop condition 1: nothing relevant at this level

            results.extend(relevant)

            next_frontier: set[str] = set()
            for node in relevant:
                children = self._graph.get_children(node.id)
                if 0 < len(children) <= self.OVER_SPECIALIZATION_LIMIT:
                    next_frontier.update(
                        c.id for c in children if c.id not in visited
                    )

            frontier = next_frontier

        return results
    
    def retrieve_and_decide(self, chunks: list[Chunk]) -> list[Node]:
            """
            Retrieves relevant nodes for the batch of chunks,
            then delegates all decisions to MergeDecisionAgent.
            """
            all_nodes: dict[str, Node] = {}
            for chunk in chunks:
                for node in self.retrieve(chunk):
                    all_nodes[node.id] = node

            return list(all_nodes.values())