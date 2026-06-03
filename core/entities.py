from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime
import uuid

class NodeStatus(Enum):
    ACTIVE = "active"
    STALE  = "stale"
    MERGED = "merged"

@dataclass
class Node:
    title:     str
    summary:   str          
    content:   str
    source:    str
    embedding: list[float]
    id:        str          = field(default_factory=lambda: str(uuid.uuid4()))
    parents:   list[str]   = field(default_factory=list)
    children:  list[str]   = field(default_factory=list)
    tags:      list[str]   = field(default_factory=list)
    confidence: float      = 1.0
    status:    NodeStatus  = NodeStatus.ACTIVE
    created_at: datetime   = field(default_factory=datetime.utcnow)
    updated_at: datetime   = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id":         self.id,
            "title":      self.title,
            "summary":    self.summary,
            "content":    self.content,
            "source":     self.source,
            "parents":    self.parents,
            "children":   self.children,
            "tags":       self.tags,
            "confidence": self.confidence,
            "status":     self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class Turn:
    role:    str
    content: str


@dataclass
class Conversation:
    turns: list[Turn]
    id:    str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Chunk:
    text:      str
    embedding: list[float]
    topic:     str
    source_conversation_id: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class RetrievalResult:
    node:           Node
    vector_score:   float
    dag_score:      float = 0.0

    @property
    def combined_score(self) -> float:
        return 0.7 * self.vector_score + 0.3 * self.dag_score


@dataclass
class MergeDecision:
    action:      MergeAction
    chunk:       Chunk
    target_node: Optional[Node]
    rationale:   str


@dataclass
class SearchResult:
    nodes:      list[RetrievalResult]
    query_used: str

# core/entities.py (aggiunte)

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

class MergeAction(str, Enum):
    ADD_ROOT   = "add_root"    # nuovo nodo radice
    ADD_CHILD  = "add_child"   # nuovo nodo figlio di un esistente
    UPDATE     = "update"      # aggiorna contenuto nodo esistente
    MERGE      = "merge"       # fondi chunk in nodo esistente
    SKIP       = "skip"        # nessuna azione


@dataclass
class ChunkDecision:
    """Decisione atomica: un chunk → un target (o nessuno)."""
    action:      MergeAction
    chunk:       Chunk
    target_node: Optional[Node]   # None solo per ADD_ROOT
    parent_node: Optional[Node]   # solo per ADD_CHILD
    new_content: Optional[str]    # per UPDATE / MERGE
    new_summary: Optional[str]    # per UPDATE / MERGE
    rationale:   str


@dataclass
class BatchMergeDecision:
    """Output del MergeDecisionAgent: una decisione per chunk (o frammento)."""
    decisions: list[ChunkDecision] = field(default_factory=list)