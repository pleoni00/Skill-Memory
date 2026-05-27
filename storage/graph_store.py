import json
import kuzu
from datetime import datetime
from typing import Optional

from core.entities import Node, NodeStatus
from core.interfaces import GraphStore


class KuzuGraphStore(GraphStore):

    def __init__(self, db_path: str):
        self._db   = kuzu.Database(db_path)
        self._conn = kuzu.Connection(self._db)
        self._init_schema()

    def _init_schema(self):
        self._conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Node(
                id         STRING,
                title      STRING,
                summary    STRING,
                content    STRING,
                source     STRING,
                tags       STRING,
                confidence DOUBLE,
                status     STRING,
                created_at STRING,
                updated_at STRING,
                PRIMARY KEY(id)
            )
        """)
        self._conn.execute("""
            CREATE REL TABLE IF NOT EXISTS HAS_CHILD(
                FROM Node TO Node
            )
        """)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _parse_result(self, res, prefix: str) -> list[dict]:
        """Converte un QueryResult in lista di dict, strippando il prefisso dalle colonne."""
        cols = [c.replace(f"{prefix}.", "") for c in res.get_column_names()]
        return [dict(zip(cols, row)) for row in res.get_all()]

    def _row_to_node(self, row: dict) -> Node:
        tags = json.loads(row["tags"]) if row["tags"] else []
        return Node(
            id         = row["id"],
            title      = row["title"],
            summary    = row["summary"],
            content    = row["content"],
            source     = row["source"],
            embedding  = [],    # non in Kuzu, sta in VectorStore
            tags       = tags,
            confidence = row["confidence"],
            status     = NodeStatus(row["status"]),
            created_at = datetime.fromisoformat(row["created_at"]),
            updated_at = datetime.fromisoformat(row["updated_at"]),
        )

    def _load_relations(self, node: Node) -> Node:
        res = self._conn.execute(
            "MATCH (p:Node)-[:HAS_CHILD]->(n:Node {id: $id}) RETURN p.id",
            {"id": node.id}
        )
        node.parents = [row[0] for row in res.get_all()]

        res = self._conn.execute(
            "MATCH (n:Node {id: $id})-[:HAS_CHILD]->(c:Node) RETURN c.id",
            {"id": node.id}
        )
        node.children = [row[0] for row in res.get_all()]
        return node

    # ── GraphStore ────────────────────────────────────────────────────────────

    def add_node(self, node: Node) -> None:
        self._conn.execute("""
            CREATE (n:Node {
                id:         $id,
                title:      $title,
                summary:    $summary,
                content:    $content,
                source:     $source,
                tags:       $tags,
                confidence: $confidence,
                status:     $status,
                created_at: $created_at,
                updated_at: $updated_at
            })
        """, {
            "id":         node.id,
            "title":      node.title,
            "summary":    node.summary,
            "content":    node.content,
            "source":     node.source,
            "tags":       json.dumps(node.tags),
            "confidence": node.confidence,
            "status":     node.status.value,
            "created_at": node.created_at.isoformat(),
            "updated_at": node.updated_at.isoformat(),
        })

    def update_node(self, node: Node) -> None:
        node.updated_at = datetime.utcnow()
        self._conn.execute("""
            MATCH (n:Node {id: $id})
            SET n.title      = $title,
                n.summary    = $summary,
                n.content    = $content,
                n.source     = $source,
                n.tags       = $tags,
                n.confidence = $confidence,
                n.status     = $status,
                n.updated_at = $updated_at
        """, {
            "id":         node.id,
            "title":      node.title,
            "summary":    node.summary,
            "content":    node.content,
            "source":     node.source,
            "tags":       json.dumps(node.tags),
            "confidence": node.confidence,
            "status":     node.status.value,
            "updated_at": node.updated_at.isoformat(),
        })

    def get_node(self, node_id: str) -> Optional[Node]:
        res  = self._conn.execute(
            "MATCH (n:Node {id: $id}) RETURN n.*",
            {"id": node_id}
        )
        rows = self._parse_result(res, "n")
        if not rows:
            return None
        return self._load_relations(self._row_to_node(rows[0]))

    def get_children(self, node_id: str) -> list[Node]:
        res   = self._conn.execute(
            "MATCH (p:Node {id: $id})-[:HAS_CHILD]->(c:Node) RETURN c.*",
            {"id": node_id}
        )
        nodes = [self._row_to_node(r) for r in self._parse_result(res, "c")]
        return [self._load_relations(n) for n in nodes]

    def get_ancestors(self, node_id: str) -> list[Node]:
        """BFS verso l'alto."""
        visited, queue, result = set(), [node_id], []
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            res  = self._conn.execute(
                "MATCH (p:Node)-[:HAS_CHILD]->(n:Node {id: $id}) RETURN p.*",
                {"id": current}
            )
            for row in self._parse_result(res, "p"):
                node = self._row_to_node(row)
                result.append(node)
                queue.append(node.id)
        return result

    def get_roots(self) -> list[Node]:
        res = self._conn.execute("""
            MATCH (n:Node)
            WHERE NOT EXISTS { MATCH (p:Node)-[:HAS_CHILD]->(n) }
            RETURN n.*
        """)
        return [self._row_to_node(r) for r in self._parse_result(res, "n")]

    def add_edge(self, parent_id: str, child_id: str) -> None:
        self._conn.execute("""
            MATCH (p:Node {id: $parent_id}), (c:Node {id: $child_id})
            CREATE (p)-[:HAS_CHILD]->(c)
        """, {"parent_id": parent_id, "child_id": child_id})

    def mark_stale(self, node_id: str) -> None:
        ancestors   = self.get_ancestors(node_id)
        ids_to_mark = [node_id] + [a.id for a in ancestors]
        for nid in ids_to_mark:
            self._conn.execute(
                "MATCH (n:Node {id: $id}) SET n.status = $status",
                {"id": nid, "status": NodeStatus.STALE.value}
            )

    def get_stale_nodes(self) -> list[Node]:
        res = self._conn.execute(
            "MATCH (n:Node {status: $status}) RETURN n.*",
            {"status": NodeStatus.STALE.value}
        )
        return [self._row_to_node(r) for r in self._parse_result(res, "n")]

    def get_all_nodes(self) -> list[Node]:
        res = self._conn.execute("MATCH (n:Node) RETURN n.*")
        return [self._row_to_node(r) for r in self._parse_result(res, "n")]
    
    def search(self, node):
        return super().search(node)