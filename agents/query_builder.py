from core.entities import Turn, Node, NodeStatus
from core.interfaces import QueryBuilder, SummaryUpdater, GraphStore
from agents.llm_client import LLMClient

# ── QueryBuilder ──────────────────────────────────────────────────────────────

QUERY_BUILDER_PROMPT = """Sei un assistente che trasforma una conversazione in una query di ricerca.

Ricevi gli ultimi turni di una conversazione.
Produci UNA singola query semantica in italiano che cattura:
- L'intento principale dell'utente
- Il contesto rilevante degli ultimi messaggi
- Eventuali entità o concetti chiave

Rispondi SOLO con la query, nessun altro testo.
"""


class LLMQueryBuilder(QueryBuilder):

    def __init__(self, llm: LLMClient):
        self._llm = llm

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
            max_tokens = 15000,
        ).strip()


# ── SummaryUpdater ────────────────────────────────────────────────────────────

SUMMARY_UPDATER_PROMPT = """
You maintain a NAVIGATIONAL SUMMARY of a node in a knowledge DAG.

The summary must answer:

"If I search for a concept, should I go into this subtree?"

Rules:
- focus on what exists below
- not on what the node is
- emphasize discrimination (what is here vs not here)
- max 3 sentences

Output ONLY the summary.
"""


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