import json
from datetime import datetime
from typing import Optional

from core.entities import Chunk, Node, RetrievalResult, MergeDecision, MergeAction, NodeStatus
from core.interfaces import Merger, GraphStore, VectorStore, EmbeddingService
from agents.llm_client import LLMClient

MERGER_PROMPT = """Sei un agente che gestisce una base di conoscenza strutturata a DAG.

Ti viene fornito:
- UN CHUNK
- NODI CANDIDATI

Ogni nodo ha un livello implicito nel grafo:
- ROOT = macro-tema
- MID = concetto
- LEAF = dettaglio

Devi decidere UNA azione:

UPDATE / MERGE / ADD / SKIP

------------------------------------------------------------

IMPORTANT RULE:
Do NOT collapse across levels unless similarity is extremely high.

- Macro nodes must remain stable
- Leaf nodes can evolve more freely
- Avoid flattening hierarchy

------------------------------------------------------------

Output JSON:
{
  "action": "...",
  "target_node_id": "...",
  "new_content": "...",
  "new_summary": "...",
  "rationale": "..."
}
"""

class LLMMerger(Merger):

    SKIP_THRESHOLD = 0.95
    ADD_THRESHOLD  = 0.35

    def __init__(
        self,
        graph: GraphStore,
        vector: VectorStore,
        embedding_service: EmbeddingService,
        llm: LLMClient,
    ):
        self._graph  = graph
        self._vector = vector
        self._embed  = embedding_service
        self._llm    = llm

    def decide(self, chunk: Chunk, candidates: list[RetrievalResult]) -> MergeDecision:
        if not candidates:
            return MergeDecision(
                action      = MergeAction.ADD,
                chunk       = chunk,
                target_node = None,
                rationale   = "Nessun nodo candidato trovato."
            )

        best = candidates[0]

        # ── Soglie automatiche (no LLM) ───────────────────────────────────────
        if best.vector_score >= self.SKIP_THRESHOLD:
            return MergeDecision(
                action      = MergeAction.SKIP,
                chunk       = chunk,
                target_node = best.node,
                rationale   = f"Similarity {best.vector_score:.2f} > {self.SKIP_THRESHOLD}"
            )

        if best.vector_score <= self.ADD_THRESHOLD:
            return MergeDecision(
                action      = MergeAction.ADD,
                chunk       = chunk,
                target_node = best.node,
                rationale   = f"Similarity {best.vector_score:.2f} < {self.ADD_THRESHOLD}"
            )

        # ── LLM decision ──────────────────────────────────────────────────────
        candidates_text = "\n\n".join(
            f"NODE {r.node.id} (score={r.combined_score:.2f}):\n"
            f"Title: {r.node.title}\n"
            f"Summary: {r.node.summary}\n"
            f"Content: {r.node.content[:400]}"
            for r in candidates
        )

        user_msg = (
            f"CHUNK:\nTopic: {chunk.topic}\nText: {chunk.text}\n\n"
            f"NODI CANDIDATI:\n{candidates_text}"
        )

        raw = self._llm.complete(
            system     = MERGER_PROMPT,
            user       = user_msg,
            max_tokens = 500,
        )

        try:
            data = json.loads(raw.strip())
        except json.JSONDecodeError:
            return MergeDecision(
                action      = MergeAction.ADD,
                chunk       = chunk,
                target_node = best.node,
                rationale   = "Parse error, fallback ad ADD"
            )

        action      = MergeAction(data["action"].lower())
        target_id   = data.get("target_node_id")
        target_node = self._graph.get_node(target_id) if target_id else best.node

        # Portiamo new_content e new_summary fuori dal dataclass
        # salvandoli come attributi extra sulla decisione
        decision = MergeDecision(
            action      = action,
            chunk       = chunk,
            target_node = target_node,
            rationale   = data.get("rationale", ""),
        )
        decision._new_content = data.get("new_content")
        decision._new_summary = data.get("new_summary")
        return decision

    def apply(self, decision: MergeDecision) -> Optional[Node]:
        action = decision.action
        chunk  = decision.chunk

        if action == MergeAction.SKIP:
            return None

        if action == MergeAction.ADD:
            embedding = self._embed.embed_one(chunk.text)
            node = Node(
                title     = chunk.topic,
                summary   = chunk.text[:200],
                content   = chunk.text,
                source    = chunk.text,
                embedding = embedding,
            )
            self._graph.add_node(node)
            self._vector.upsert(node.id, embedding)

            if decision.target_node:
                self._graph.add_edge(decision.target_node.id, node.id)
                self._graph.mark_stale(decision.target_node.id)

            return node

        if action in (MergeAction.UPDATE, MergeAction.MERGE):
            target = decision.target_node
            if not target:
                return None

            new_content = getattr(decision, "_new_content", None) or \
                          f"{target.content}\n\n---\n{chunk.text}"
            new_summary = getattr(decision, "_new_summary", None) or target.summary

            target.content    = new_content
            target.summary    = new_summary
            target.source     = f"{target.source}\n[updated]"
            target.updated_at = datetime.utcnow()
            target.status     = NodeStatus.ACTIVE

            new_embedding    = self._embed.embed_one(target.content)
            target.embedding = new_embedding

            self._graph.update_node(target)
            self._vector.upsert(target.id, new_embedding)
            self._graph.mark_stale(target.id)

            return target

        return None