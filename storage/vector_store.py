import sqlite3
import sqlite_vec
import struct
import numpy as np
from typing import Optional

from core.interfaces import VectorStore

def _serialize(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)

class SqliteVectorStore(VectorStore):

    def __init__(self, db_path: str, embedding_dim: int):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self.embedding_dim = embedding_dim
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._init_schema()

    def _init_schema(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS node_embeddings (
                node_id   TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                is_root   INTEGER NOT NULL DEFAULT 0
            )
        """)

        self._conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_index
            USING vec0(
                node_id TEXT,
                embedding float[{self.embedding_dim}]
            )
        """)

        self._conn.commit()

    def upsert(
        self,
        node_id: str,
        embedding: list[float],
        is_root: bool = False
    ) -> None:
        blob = _serialize(embedding)
        root_flag = 1 if is_root else 0

        self._conn.execute("""
            INSERT INTO node_embeddings(node_id, embedding, is_root)
            VALUES (?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                embedding = excluded.embedding,
                is_root = excluded.is_root
        """, (node_id, blob, root_flag))

        # vec index sync
        self._conn.execute(
            "DELETE FROM vec_index WHERE node_id = ?",
            (node_id,)
        )
        self._conn.execute(
            "INSERT INTO vec_index(node_id, embedding) VALUES (?, ?)",
            (node_id, blob)
        )

        self._conn.commit()

    def search(self, embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        blob = _serialize(embedding)
        rows = self._conn.execute(f"""
            SELECT node_id, distance
            FROM vec_index
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
        """, (blob, top_k)).fetchall()

        # vec0 returns L2 distance; convert it to similarity [0,1]
        results = []
        for node_id, dist in rows:
            score = 1.0 / (1.0 + dist)
            results.append((node_id, score))
        return results

    def delete(self, node_id: str) -> None:
        self._conn.execute(
            "DELETE FROM node_embeddings WHERE node_id = ?", (node_id,)
        )
        self._conn.execute(
            "DELETE FROM vec_index WHERE node_id = ?", (node_id,)
        )
        self._conn.commit()

    def get_embedding(self, node_id: str) -> Optional[list[float]]:
        row = self._conn.execute(
            "SELECT embedding FROM node_embeddings WHERE node_id = ?", (node_id,)
        ).fetchone()
        if not row:
            return None
        blob = row[0]
        n = len(blob) // 4
        return list(struct.unpack(f"{n}f", blob))
    
    def get_roots(self) -> list[str]:
        rows = self._conn.execute("""
            SELECT node_id
            FROM node_embeddings
            WHERE is_root = 1
        """).fetchall()

        return [r[0] for r in rows]
