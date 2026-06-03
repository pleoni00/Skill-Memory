from core.entities import Turn, Node, NodeStatus
from core.interfaces import QueryBuilder, SummaryUpdater, GraphStore
from agents.llm_client import LLMClient
from agents.prompts import QUERY_BUILDER_PROMPT, SUMMARY_UPDATER_PROMPT, RETRIEVAL_GATE_PROMPT

# ── QueryBuilder ──────────────────────────────────────────────────────────────

class LLMQueryBuilder(QueryBuilder):

    def __init__(self, llm: LLMClient):
        self._llm = llm

    def _needs_retrieval(self, turns: list[Turn]) -> bool:
        last_user = next(
            (t.content for t in reversed(turns) if t.role == "user"),
            None,
        )
        if not last_user:
            return False

        response = self._llm.complete(
            system     = RETRIEVAL_GATE_PROMPT,
            user       = last_user,
            max_tokens = 100,
        ).strip().upper()

        return response.startswith("YES")

    def build(self, turns: list[Turn]) -> str:
        if not turns:
            return ""

        convo_text = "\n".join(
            f"{t.role.upper()}: {t.content}"
            for t in turns[-6:]
        )

        return self._llm.complete(
            system     = QUERY_BUILDER_PROMPT,
            user       = convo_text,
            max_tokens = 1000,
        ).strip()

# ── SummaryUpdater ────────────────────────────────────────────────────────────

class LLMSummaryUpdater(SummaryUpdater):

    def __init__(self, graph: GraphStore, llm: LLMClient):
        self._graph = graph
        self._llm   = llm

    def update_ancestors(self, node_id: str) -> None:
        stale_ancestors = [
            n for n in self._graph.get_ancestors(node_id)
            if n.status == NodeStatus.STALE
        ]

        for ancestor in reversed(stale_ancestors):
            self._update_summary(ancestor)

    def _update_summary(self, node: Node) -> None:
        children = self._graph.get_children(node.id)
        children_summaries = "\n".join(
            f"- {c.title}: {c.summary}" for c in children
        )

        user_msg = (
            f"NODO: {node.title}\n"
            f"CONTENUTO: {node.content[:300]}\n\n"
            f"FIGLI:\n{children_summaries or 'Nessuno'}"
        )

        node.summary = self._llm.complete(
            system     = SUMMARY_UPDATER_PROMPT,
            user       = user_msg,
            max_tokens = 200,
        ).strip()

        node.status = NodeStatus.ACTIVE
        self._graph.update_node(node)