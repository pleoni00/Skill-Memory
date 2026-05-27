import json

from core.entities import Chunk, Node, RetrievalResult
from core.interfaces import Retriever, GraphStore, VectorStore
from agents import LLMClient

class HybridRetriever(Retriever):
    """
    Retrieval ibrido:
    1. Vector seeding (entry points)
    2. Graph traversal top-down dai capostipiti
    3. LLM-based relevance filtering
    4. Early stopping su saturazione o over-specialization
    """

    def __init__(self, llm: LLMClient, graph: GraphStore, vector: VectorStore):
        self._llm    = llm
        self._graph  = graph
        self._vector = vector

    def _is_relevant(self, query: Chunk, nodes: list[Node]) -> list[str]:
        """
        Returns list of relevant node IDs according to LLM.
        """

        prompt = f"""
You are a relevance filter for a retrieval system.

Query:
{query.text}

Candidate nodes:
"""

        for n in nodes:
            prompt += f"- id: {n.id}, name: {n.name}, summary: {getattr(n, 'summary', '')}\n"

        prompt += """

Return ONLY a JSON list of node IDs that are relevant.
If none are relevant, return [].
"""

        response = self._llm.complete(
            system="You are a strict information retrieval filter.",
            user=prompt,
            max_tokens=500
        )

        try:
            return json.loads(response)
        except:
            return []

    # ─────────────────────────────────────────────────────────────
    # Main retrieval
    # ─────────────────────────────────────────────────────────────
    def retrieve(self, chunk: Chunk, top_k: int = 3) -> list[RetrievalResult]:
        seed_hits = self._vector.search(chunk.embedding, top_k=top_k * 2)
        frontier: set[str] = set(node_id for node_id, _ in seed_hits)
        visited: set[str] = set()
        results: dict[str, RetrievalResult] = {}

        root_nodes = self._graph.get_roots()
        frontier.update(root_nodes)
        depth = 0
        MAX_DEPTH = 5
        while frontier and depth < MAX_DEPTH:
            current_level_nodes = []
            for node_id in frontier:
                if node_id in visited:
                    continue

                node = self._graph.get_node(node_id)
                if node:
                    current_level_nodes.append(node)

                visited.add(node_id)

            if not current_level_nodes:
                break

            relevant_ids = set(self._is_relevant(chunk, current_level_nodes))

            if not relevant_ids:
                break  # STOP CONDITION 1: no relevant nodes

            next_frontier = set()

            for node in current_level_nodes:
                if node.id not in relevant_ids:
                    continue

                if node.id not in results:
                    results[node.id] = RetrievalResult(
                        node=node,
                        vector_score=0.0,
                        dag_score=1.0 / (depth + 1)
                    )

                children = self._graph.get_children(node.id)

                # STOP CONDITION 2: over-specialization
                if len(children) == 0:
                    continue
                if len(children) > 50:  # too broad, avoid explosion
                    continue
                for c in children:
                    if c.id not in visited:
                        next_frontier.add(c.id)

            frontier = next_frontier
            depth += 1

        for node_id, score in seed_hits:
            node = self._graph.get_node(node_id)
            if not node:
                continue

            if node_id not in results:
                results[node_id] = RetrievalResult(
                    node=node,
                    vector_score=score,
                    dag_score=0.2
                )
            else:
                results[node_id].vector_score = max(
                    results[node_id].vector_score,
                    score
                )

        sorted_results = sorted(
            results.values(),
            key=lambda r: r.combined_score,
            reverse=True
        )

        return sorted_results[:top_k]