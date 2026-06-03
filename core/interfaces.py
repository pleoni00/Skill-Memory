from abc import ABC, abstractmethod
from typing import Optional
from .entities import (
    Node, Chunk, Conversation, BatchMergeDecision, Turn
)


class LLMClient(ABC):

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        """Returns the text of the LLM response."""
        ...


class GraphStore(ABC):

    @abstractmethod
    def add_node(self, node: Node) -> None: ...

    @abstractmethod
    def update_node(self, node: Node) -> None: ...

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Node]: ...

    @abstractmethod
    def get_children(self, node_id: str) -> list[Node]: ...

    @abstractmethod
    def get_ancestors(self, node_id: str) -> list[Node]: ...

    @abstractmethod
    def get_roots(self) -> list[Node]: ...

    @abstractmethod
    def add_edge(self, parent_id: str, child_id: str) -> None: ...

    @abstractmethod
    def mark_stale(self, node_id: str) -> None:
        """Marks the node and all its ancestors as stale."""
        ...

    @abstractmethod
    def get_stale_nodes(self) -> list[Node]: ...

    @abstractmethod
    def get_all_nodes(self) -> list[Node]: ...

    @abstractmethod
    def search(self, node: Node) -> list[Node]: ...


class VectorStore(ABC):

    @abstractmethod
    def upsert(self, node_id: str, embedding: list[float], is_root: bool) -> None: ...

    @abstractmethod
    def search(self, embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        """Returns a list of (node_id, score)."""
        ...

    @abstractmethod
    def delete(self, node_id: str) -> None: ...


class EmbeddingService(ABC):

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class Extractor(ABC):

    @abstractmethod
    def extract(self, conversation: Conversation) -> list[Chunk]:
        """
        1. Isolates user turns
        2. Segments by topic change
        3. Produces chunks with embeddings
        """
        ...


class Retriever(ABC):

    @abstractmethod
    def retrieve(self, turn) -> list[Node]:
        """
        Pure retrieval. Returns relevant nodes with no scoring.
        Traverses the DAG top-down with LLM filtering at each level.
        """
        ...

    @abstractmethod
    def retrieve_and_decide(self, chunks: list[Chunk]) -> BatchMergeDecision:
        """
        Retrieves relevant nodes for a batch of chunks and delegates
        all integration decisions to MergeDecisionAgent.
        """
        ...


class MergeDecisionAgent(ABC):

    @abstractmethod
    def decide(self, chunks: list[Chunk], nodes: list[Node]) -> BatchMergeDecision:
        """
        Receives a batch of chunks and the relevant existing nodes.
        Returns one or more ChunkDecision per chunk (split allowed).
        Does NOT execute any write.
        """
        ...


class Merger(ABC):

    @abstractmethod
    def apply(self, batch: BatchMergeDecision) -> list[Node]:
        """
        Executes all decisions in the batch.
        Returns the list of nodes that were created or modified.
        """
        ...


class QueryBuilder(ABC):

    @abstractmethod
    def build(self, turns: list[Turn]) -> str:
        """Transforms the last N turns into an enriched semantic query."""
        ...

    @abstractmethod
    def _needs_retrieval(self, turns: list[Turn]) -> bool:
        """Determines if retrieval is needed based on the last N turns."""
        ...


class SummaryUpdater(ABC):

    @abstractmethod
    def update_ancestors(self, node_id: str) -> None:
        """Updates only STALE nodes walking up the tree. Lazy evaluation."""
        ...