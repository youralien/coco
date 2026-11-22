"""One-off: remove degenerate observer output from Coco's memory database.

Uses the same compression-ratio check as the live observer guard
(sensing.observer_guard) to find observations that are repetition loops, then
removes them together with any memory derived *only* from them:

* propositions whose every supporting observation is degenerate
* proposition updates whose every evidence observation is degenerate

Propositions/updates that also rest on healthy observations are kept; their
links to the removed observations disappear via ON DELETE CASCADE, and the FTS
indexes are maintained by the schema's triggers.

Dry run by default. Quit the Coco app before --apply so the memory engine is
not mid-batch on an observation this deletes.

    .venv/bin/python scripts/cleanup_degenerate_memory.py            # report
    .venv/bin/python scripts/cleanup_degenerate_memory.py --apply    # backup + delete
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "sensing"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "memory"))

from memory import default_memory_db_path  # noqa: E402
from sensing.observer_guard import (  # noqa: E402
    compression_ratio,
    degenerate_output_reason,
)


def _only_from(support: dict, bad: set[str]) -> list:
    return sorted(k for k, ids in support.items() if ids and ids <= bad)


def plan(conn: sqlite3.Connection) -> tuple[list[str], list[int], list[int]]:
    bad = {
        oid
        for oid, content in conn.execute("SELECT id, content FROM observations")
        # Token counts are not stored in memory; only the ratio check applies.
        if degenerate_output_reason(content, output_tokens=0, max_tokens=0)
    }
    prop_support: dict[int, set[str]] = defaultdict(set)
    for oid, pid in conn.execute(
        "SELECT observation_id, proposition_id FROM observation_proposition"
    ):
        prop_support[pid].add(oid)
    update_support: dict[int, set[str]] = defaultdict(set)
    for uid, oid in conn.execute(
        "SELECT update_id, observation_id FROM proposition_update_observation"
    ):
        update_support[uid].add(oid)
    return sorted(bad), _only_from(prop_support, bad), _only_from(update_support, bad)


def report(conn, bad, props, updates) -> None:
    total = conn.execute("SELECT count(*) FROM observations").fetchone()[0]
    print(f"Degenerate observations: {len(bad)} of {total}")
    for oid in bad:
        content, created, processed = conn.execute(
            "SELECT content, created_at, processed_at FROM observations WHERE id=?",
            (oid,),
        ).fetchone()
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(created))
        state = "processed" if processed else "pending"
        print(
            f"  {oid}  {when}  {state:9}  ratio={compression_ratio(content):5.1f}  "
            f"{content[:60]!r}"
        )
    print(f"\nPropositions derived only from them: {len(props)}")
    for pid in props:
        (text,) = conn.execute(
            "SELECT text FROM propositions WHERE id=?", (pid,)
        ).fetchone()
        print(f"  #{pid}  {text[:100]!r}")
    print(f"\nProposition updates derived only from them: {len(updates)}")
    for uid in updates:
        (summary,) = conn.execute(
            "SELECT summary FROM proposition_updates WHERE id=?", (uid,)
        ).fetchone()
        print(f"  #{uid}  {summary[:100]!r}")


def backup(db: Path) -> Path:
    dest = db.with_name(f"{db.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    src = sqlite3.connect(db)
    out = sqlite3.connect(dest)
    try:
        src.backup(out)  # consistent snapshot, includes un-checkpointed WAL
    finally:
        out.close()
        src.close()
    return dest


def apply(db: Path, bad, props, updates) -> None:
    conn = sqlite3.connect(db, timeout=30)
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        with conn:  # one transaction: all or nothing
            conn.executemany(
                "DELETE FROM propositions WHERE id=?", [(p,) for p in props]
            )
            conn.executemany(
                "DELETE FROM proposition_updates WHERE id=?", [(u,) for u in updates]
            )
            conn.executemany("DELETE FROM observations WHERE id=?", [(o,) for o in bad])
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", type=Path, default=default_memory_db_path())
    parser.add_argument(
        "--apply", action="store_true", help="back up, then delete (default: dry run)"
    )
    args = parser.parse_args()
    db = args.db.expanduser()
    if not db.exists():
        sys.exit(f"No memory database at {db}")

    ro = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)
    try:
        bad, props, updates = plan(ro)
        print(f"Memory database: {db}\n")
        report(ro, bad, props, updates)
    finally:
        ro.close()

    if not bad:
        print("\nNothing to clean.")
        return
    if not args.apply:
        print("\nDry run: nothing changed. Quit Coco, then re-run with --apply.")
        return
    print(f"\nBackup written to {backup(db)}")
    apply(db, bad, props, updates)
    print(
        f"Deleted {len(bad)} observations, {len(props)} propositions, "
        f"{len(updates)} proposition updates."
    )


if __name__ == "__main__":
    main()
