from abc import ABC, abstractmethod
from typing import Optional
from .entities import Node, Chunk, Conversation, RetrievalResult, MergeDecision, SearchResult, Turn

class LLMClient(ABC):

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        """Unico metodo — restituisce il testo della risposta."""
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
        """Marca il nodo e tutti gli antenati come stale."""
        ...

    @abstractmethod
    def get_stale_nodes(self) -> list[Node]: ...

    @abstractmethod
    def get_all_nodes(self) -> list[Node]: ...

    @abstractmethod
    def search(self, node: Node) -> list[Node]: ...

class VectorStore(ABC):

    @abstractmethod
    def upsert(self, node_id: str, embedding: list[float]) -> None: ...

    @abstractmethod
    def search(self, embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        """Restituisce lista di (node_id, score)."""
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
        1. Isola i turni utente
        2. Segmenta per cambio di argomento
        3. Produce chunk con embedding
        """
        ...


class Retriever(ABC):

    @abstractmethod
    def retrieve(self, chunk: Chunk, top_k: int = 3) -> list[RetrievalResult]: ...


class Merger(ABC):

    @abstractmethod
    def decide(self, chunk: Chunk, candidates: list[RetrievalResult]) -> MergeDecision:
        """
        similarity > 0.95 → SKIP  (no LLM)
        similarity < 0.40 → ADD   (no LLM)
        altrimenti        → chiede all'LLM
        """
        ...

    @abstractmethod
    def apply(self, decision: MergeDecision) -> Optional[Node]: ...


class QueryBuilder(ABC):

    @abstractmethod
    def build(self, turns: list[Turn]) -> str:
        """Trasforma gli ultimi N turni in una query semantica arricchita."""
        ...


class SummaryUpdater(ABC):

    @abstractmethod
    def update_ancestors(self, node_id: str) -> None:
        """Aggiorna solo i nodi STALE risalendo l'albero. Lazy."""
        ...
