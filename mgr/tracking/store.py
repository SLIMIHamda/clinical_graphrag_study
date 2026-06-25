"""DuckDB query layer over the Parquet records (Doc 00 section 4).

Zero-server: DuckDB reads results/run_records.parquet directly. These are
*views* (derived), never the record of truth. The canonical query objects:

  v_status  -- status x phase x priority (mirrors Budget_Summary progress)
  v_cost    -- tokens & $ actual vs cost_est_usd, per phase/condition/backbone

(v_seed_spread and v_rgd arrive once the metrics layer emits per-seed numbers.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _connect(parquet: str | Path):
    import duckdb

    con = duckdb.connect()
    # read_parquet inside CREATE VIEW can't take a bound parameter, so inline the
    # path with single-quote escaping (path is runner-controlled, not user input).
    safe = str(parquet).replace("'", "''")
    con.execute(f"CREATE VIEW runs AS SELECT * FROM read_parquet('{safe}')")
    return con


def query(sql: str, parquet: str | Path = "results/run_records.parquet") -> list[dict[str, Any]]:
    """Run an arbitrary SQL query against the ``runs`` view."""
    con = _connect(parquet)
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        con.close()


def v_status(parquet: str | Path = "results/run_records.parquet") -> list[dict[str, Any]]:
    return query(
        "SELECT status, COUNT(*) AS n FROM runs GROUP BY status ORDER BY status",
        parquet,
    )


def v_cost(parquet: str | Path = "results/run_records.parquet") -> list[dict[str, Any]]:
    return query(
        """
        SELECT condition, backbone,
               SUM(cost_est_usd)    AS est_usd,
               SUM(cost_actual_usd) AS actual_usd
        FROM runs
        GROUP BY condition, backbone
        ORDER BY est_usd DESC
        """,
        parquet,
    )
