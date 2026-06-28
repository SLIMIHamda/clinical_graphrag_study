"""Neo4j-backed GraphStore for the full build (runtime only).

Implements the same protocol as InMemoryGraphStore against a real database. The
``neo4j`` driver is imported lazily so the rest of the package (and the test
suite) never requires it. Used on the pod where Neo4j is served from /vol.
"""

from __future__ import annotations

from typing import Any


class Neo4jStore:
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "neo4j"):
        from neo4j import GraphDatabase  # lazy: only needed at runtime

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._ensure_constraints()

    def _run(self, cypher: str, **params: Any):
        with self._driver.session() as s:
            return s.run(cypher, **params).data()

    def _ensure_constraints(self) -> None:
        self._run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
        self._run("CREATE CONSTRAINT concept_cui IF NOT EXISTS FOR (k:Concept) REQUIRE k.cui IS UNIQUE")

    def add_chunk(self, chunk_id: str, text: str, **provenance) -> None:
        self._run(
            "MERGE (c:Chunk {id:$id}) SET c.text=$text, c += $prov",
            id=chunk_id, text=text, prov=provenance,
        )

    def add_concept(self, cui: str) -> None:
        self._run("MERGE (:Concept {cui:$cui})", cui=cui)

    def link_chunk_concept(self, chunk_id: str, cui: str) -> None:
        self._run(
            "MATCH (c:Chunk {id:$id}) MERGE (k:Concept {cui:$cui}) MERGE (c)-[:MENTIONS]->(k)",
            id=chunk_id, cui=cui,
        )

    def link_concept_concept(self, a: str, b: str, rel: str = "related") -> None:
        self._run(
            "MERGE (x:Concept {cui:$a}) MERGE (y:Concept {cui:$b}) MERGE (x)-[:RELATED]->(y)",
            a=a, b=b,
        )

    def chunks_for_concepts(self, cuis: set[str], *, hops: int = 1, limit: int = 10) -> list[tuple[str, float]]:
        rows = self._run(
            f"""
            UNWIND $cuis AS q
            MATCH (k:Concept {{cui:q}})
            OPTIONAL MATCH (k)-[:RELATED*0..{max(0, hops)}]-(n:Concept)
            MATCH (c:Chunk)-[:MENTIONS]->(n)
            WITH c, sum(CASE WHEN n.cui IN $cuis THEN 1.0 ELSE 0.5 END) AS score
            RETURN c.id AS id, score ORDER BY score DESC, id ASC LIMIT $limit
            """,
            cuis=list(cuis), limit=limit,
        )
        return [(r["id"], float(r["score"])) for r in rows]

    def text(self, chunk_id: str) -> str:
        rows = self._run("MATCH (c:Chunk {id:$id}) RETURN c.text AS t", id=chunk_id)
        return rows[0]["t"] if rows else ""

    def concepts_of(self, chunk_id: str) -> set[str]:
        rows = self._run("MATCH (c:Chunk {id:$id})-[:MENTIONS]->(k:Concept) RETURN k.cui AS cui", id=chunk_id)
        return {r["cui"] for r in rows}

    def signature(self) -> str:
        """Stable hash over chunk→concept links and concept edges (graph_hash)."""
        import hashlib

        rows = self._run(
            "MATCH (c:Chunk)-[:MENTIONS]->(k:Concept) "
            "RETURN c.id AS cid, collect(k.cui) AS cuis ORDER BY cid"
        )
        edges = self._run(
            "MATCH (a:Concept)-[:RELATED]->(b:Concept) "
            "RETURN a.cui AS a, b.cui AS b ORDER BY a, b"
        )
        parts = [f"C|{r['cid']}|{','.join(sorted(r['cuis']))}" for r in rows]
        parts += [f"E|{e['a']}|{e['b']}" for e in edges]
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()

    def close(self) -> None:
        self._driver.close()
