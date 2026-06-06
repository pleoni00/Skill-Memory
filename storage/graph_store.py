import json
from datetime import datetime
from typing import Optional

import sqlite3

from core.entities import Node, NodeStatus
from core.interfaces import GraphStore


class SqliteGraphStore(GraphStore):

    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id         TEXT PRIMARY KEY,
                title      TEXT NOT NULL,
                summary    TEXT,
                content    TEXT,
                source     TEXT,
                tags       TEXT,       -- JSON array
                status     TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS edges (
                parent_id  TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                child_id   TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                PRIMARY KEY (parent_id, child_id)
            );

            CREATE INDEX IF NOT EXISTS idx_edges_parent ON edges(parent_id);
            CREATE INDEX IF NOT EXISTS idx_edges_child  ON edges(child_id);
        """)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.commit()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _row_to_node(self, row: sqlite3.Row) -> Node:
        tags = json.loads(row["tags"]) if row["tags"] else []
        return Node(
            id         = row["id"],
            title      = row["title"],
            summary    = row["summary"],
            content    = row["content"],
            source     = row["source"],
            embedding  = [],    # sta in SqliteVectorStore
            tags       = tags,
            status     = NodeStatus(row["status"]),
            created_at = datetime.fromisoformat(row["created_at"]),
            updated_at = datetime.fromisoformat(row["updated_at"]),
        )

    def _load_relations(self, node: Node) -> Node:
        node.parents = [
            r["parent_id"] for r in self._conn.execute(
                "SELECT parent_id FROM edges WHERE child_id = ?", (node.id,)
            ).fetchall()
        ]
        node.children = [
            r["child_id"] for r in self._conn.execute(
                "SELECT child_id FROM edges WHERE parent_id = ?", (node.id,)
            ).fetchall()
        ]
        return node

    def _fetch_nodes(self, query: str, params: tuple = ()) -> list[Node]:
        self._conn.row_factory = sqlite3.Row
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_node(r) for r in rows]

    # ── GraphStore ────────────────────────────────────────────────────────────

    def add_node(self, node: Node) -> None:
        self._conn.execute("""
            INSERT INTO nodes(id, title, summary, content, source,
                              tags, status, created_at, updated_at)
            VALUES (:id, :title, :summary, :content, :source,
                    :tags, :status, :created_at, :updated_at)
        """, {
            "id":         node.id,
            "title":      node.title,
            "summary":    node.summary,
            "content":    node.content,
            "source":     node.source,
            "tags":       json.dumps(node.tags),
            "status":     node.status.value,
            "created_at": node.created_at.isoformat(),
            "updated_at": node.updated_at.isoformat(),
        })
        self._conn.commit()

    def update_node(self, node: Node) -> None:
        node.updated_at = datetime.utcnow()
        self._conn.execute("""
            UPDATE nodes
            SET title      = :title,
                summary    = :summary,
                content    = :content,
                source     = :source,
                tags       = :tags,
                status     = :status,
                updated_at = :updated_at
            WHERE id = :id
        """, {
            "id":         node.id,
            "title":      node.title,
            "summary":    node.summary,
            "content":    node.content,
            "source":     node.source,
            "tags":       json.dumps(node.tags),
            "status":     node.status.value,
            "updated_at": node.updated_at.isoformat(),
        })
        self._conn.commit()

    def get_node(self, node_id: str) -> Optional[Node]:
        self._conn.row_factory = sqlite3.Row
        row = self._conn.execute(
            "SELECT * FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if not row:
            return None
        return self._load_relations(self._row_to_node(row))

    def get_children(self, node_id: str) -> list[Node]:
        nodes = self._fetch_nodes("""
            SELECT n.* FROM nodes n
            JOIN edges e ON e.child_id = n.id
            WHERE e.parent_id = ?
        """, (node_id,))
        return [self._load_relations(n) for n in nodes]

    def get_ancestors(self, node_id: str) -> list[Node]:
        """BFS verso l'alto tramite CTE ricorsiva."""
        self._conn.row_factory = sqlite3.Row
        rows = self._conn.execute("""
            WITH RECURSIVE ancestors AS (
                SELECT parent_id AS id FROM edges WHERE child_id = :id
                UNION
                SELECT e.parent_id FROM edges e
                JOIN ancestors a ON e.child_id = a.id
            )
            SELECT n.* FROM nodes n
            JOIN ancestors a ON n.id = a.id
        """, {"id": node_id}).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_roots(self) -> list[Node]:
        """Nodi senza parent."""
        return self._fetch_nodes("""
            SELECT n.* FROM nodes n
            LEFT JOIN edges e ON e.child_id = n.id
            WHERE e.parent_id IS NULL
        """)

    def add_edge(self, parent_id: str, child_id: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO edges(parent_id, child_id) VALUES (?, ?)",
            (parent_id, child_id)
        )
        self._conn.commit()

    def mark_stale(self, node_id: str) -> None:
        """Marca come STALE il nodo e tutti i suoi antenati."""
        self._conn.execute("""
            WITH RECURSIVE ancestors AS (
                SELECT :id AS id
                UNION
                SELECT e.parent_id FROM edges e
                JOIN ancestors a ON e.child_id = a.id
            )
            UPDATE nodes SET status = :status
            WHERE id IN (SELECT id FROM ancestors)
        """, {"id": node_id, "status": NodeStatus.STALE.value})
        self._conn.commit()

    def get_stale_nodes(self) -> list[Node]:
        return self._fetch_nodes(
            "SELECT * FROM nodes WHERE status = ?", (NodeStatus.STALE.value,)
        )

    def get_all_nodes(self) -> list[Node]:
        return self._fetch_nodes("SELECT * FROM nodes")

    def search(self, node: Node) -> list[Node]:
        return super().search(node)