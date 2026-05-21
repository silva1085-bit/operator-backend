#!/usr/bin/env python3
"""
One-shot migration: local MongoDB  ->  MongoDB Atlas.

Usage (run on the machine that can reach BOTH the local Mongo and Atlas):

    export SOURCE_URI='mongodb://localhost:27017'
    export TARGET_URI='mongodb+srv://<user>:<pw>@<cluster>.mongodb.net/?retryWrites=true&w=majority'
    export SOURCE_DB='ember_breath'
    export TARGET_DB='ember_breath'
    python scripts/migrate_to_atlas.py

What it does:
  1. Connects to source + target.
  2. For every collection in the source DB, copies all documents
     into the same-named collection in the target DB.
  3. Replaces (upserts by _id) so the script is idempotent — safe to re-run.
  4. Prints a per-collection count summary at the end.

No external dependencies beyond `pymongo`.
"""
import os
import sys
import time
from typing import Any, Dict, List

try:
    from pymongo import MongoClient, ReplaceOne
    from pymongo.errors import BulkWriteError, ServerSelectionTimeoutError
except ImportError:
    sys.stderr.write("\n[migrate] pymongo is not installed. Run: pip install pymongo dnspython\n\n")
    sys.exit(1)

BATCH = 500


def die(msg: str, code: int = 1) -> None:
    sys.stderr.write(f"\n[migrate]  ERROR: {msg}\n\n")
    sys.exit(code)


def main() -> None:
    src_uri = os.environ.get("SOURCE_URI", "mongodb://localhost:27017")
    tgt_uri = os.environ.get("TARGET_URI", "")
    src_db_name = os.environ.get("SOURCE_DB", "ember_breath")
    tgt_db_name = os.environ.get("TARGET_DB", src_db_name)

    if not tgt_uri:
        die("TARGET_URI is required (your Atlas SRV connection string).")

    print(f"[migrate] Source: {src_uri}  DB={src_db_name}")
    print(f"[migrate] Target: {_redact(tgt_uri)}  DB={tgt_db_name}")
    print("[migrate] Connecting...")

    try:
        src = MongoClient(src_uri, serverSelectionTimeoutMS=8000)
        src.admin.command("ping")
    except ServerSelectionTimeoutError as e:
        die(f"Cannot reach SOURCE Mongo: {e}")

    try:
        tgt = MongoClient(tgt_uri, serverSelectionTimeoutMS=15000)
        tgt.admin.command("ping")
    except ServerSelectionTimeoutError as e:
        die(f"Cannot reach TARGET Atlas: {e}\n"
            "Common fixes: (a) confirm the SRV string copied correctly, "
            "(b) make sure Atlas Network Access allows your IP / 0.0.0.0/0, "
            "(c) confirm DB user has readWriteAnyDatabase role.")

    src_db = src[src_db_name]
    tgt_db = tgt[tgt_db_name]

    collections: List[str] = src_db.list_collection_names()
    if not collections:
        die(f"No collections found in source DB '{src_db_name}'.")

    print(f"[migrate] Found {len(collections)} collection(s): {', '.join(collections)}\n")

    summary: Dict[str, Dict[str, int]] = {}
    total_started = time.time()

    for col_name in collections:
        t0 = time.time()
        src_col = src_db[col_name]
        tgt_col = tgt_db[col_name]

        total_docs = src_col.count_documents({})
        if total_docs == 0:
            print(f"  - {col_name}: empty, skipping.")
            summary[col_name] = {"source": 0, "copied": 0}
            continue

        print(f"  - {col_name}: copying {total_docs} docs ...", end="", flush=True)
        ops: List[ReplaceOne] = []
        copied = 0

        cursor = src_col.find({})
        for doc in cursor:
            ops.append(ReplaceOne({"_id": doc["_id"]}, doc, upsert=True))
            if len(ops) >= BATCH:
                copied += _flush(tgt_col, ops)
                ops = []
        if ops:
            copied += _flush(tgt_col, ops)

        dt = time.time() - t0
        print(f" done ({copied} in {dt:.1f}s)")
        summary[col_name] = {"source": total_docs, "copied": copied}

    # ---- summary ----
    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)
    print(f"{'Collection':<28} {'Source':>10} {'Copied':>10}")
    print("-" * 60)
    ok = True
    for col, c in summary.items():
        flag = "" if c["source"] == c["copied"] else "   MISMATCH"
        if c["source"] != c["copied"]:
            ok = False
        print(f"{col:<28} {c['source']:>10} {c['copied']:>10}{flag}")
    print("-" * 60)
    print(f"Total time: {time.time() - total_started:.1f}s")
    print("=" * 60)

    if ok:
        print("\n[migrate]  SUCCESS  Atlas now matches the source DB.\n")
        sys.exit(0)
    else:
        print("\n[migrate]  WARN  Some collections did not fully copy. Re-run safe.\n")
        sys.exit(2)


def _flush(col: Any, ops: List[Any]) -> int:
    try:
        result = col.bulk_write(ops, ordered=False)
        return result.upserted_count + result.modified_count + (
            result.matched_count - result.modified_count
        )
    except BulkWriteError as e:
        # Even on partial failure, count what got written.
        return e.details.get("nUpserted", 0) + e.details.get("nModified", 0)


def _redact(uri: str) -> str:
    """Hide username/password when printing a Mongo URI."""
    try:
        if "@" in uri and "://" in uri:
            scheme, rest = uri.split("://", 1)
            _creds, host = rest.split("@", 1)
            return f"{scheme}://***:***@{host}"
    except Exception:
        pass
    return uri


if __name__ == "__main__":
    main()
