"""
run_benchmark.py

The actual experiment: does writing a Qdrant collection the way an agent
memory system does (continuous small batches, interleaved with reads, plus
periodic re-writes of already-stored memories) cost anything, compared to
loading the same data the way a normal RAG corpus gets loaded (one upsert
call), against a real, frozen held-out query set?

Two real Qdrant collections, same 5,882 real LoCoMo memory records, same
pre-computed FastEmbed vectors (embed_corpus.py already ran once, so
embedding cost is identical and excluded from both legs; only the write
pattern differs):

  agent_memory_batch     one upsert() call, the traditional corpus-load path
  agent_memory_streamed  the same records, same order, upserted in batches
                          of 1-5 (agent-memory write granularity), with a
                          real query run and a footprint/segment snapshot at
                          fixed checkpoints, plus a real "touch" (re-upsert,
                          unchanged) on 5% of writes to simulate an agent
                          re-confirming a memory it already wrote

Whatever the checkpoints show is what gets reported. No expected direction
is assumed going in.
"""

import argparse
import json
import random
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models

CONTAINER_NAME = "qdrant-ternlight-eval"
QDRANT_URL = "http://localhost:6333"

DATA_DIR = Path(__file__).parent / "data"
EMBED_DIR = Path(__file__).parent / "embeddings"
RESULTS_PATH = Path(__file__).parent / "results.json"

VECTOR_SIZE = 384
STREAM_BATCH_MIN, STREAM_BATCH_MAX = 1, 5
TOUCH_RATE = 0.05          # fraction of streamed writes that are re-upserts of an existing record
CHECKPOINT_EVERY = 500     # measure every N records written in the streamed leg
SETTLE_SECONDS = 30        # matches payload-audit's method: let WAL/segments settle before measuring
SEED = 42
UUID_NAMESPACE = uuid.UUID("a3f1c2d4-0000-4000-8000-000000000000")


def point_id(record_id: str) -> str:
    return str(uuid.uuid5(UUID_NAMESPACE, record_id))


def load_data():
    records = json.loads((DATA_DIR / "memory_records.json").read_text())
    frozen_queries = json.loads((DATA_DIR / "frozen_query_sample.json").read_text())
    record_vecs = json.loads((EMBED_DIR / "record_vectors.json").read_text())
    query_vecs = json.loads((EMBED_DIR / "query_vectors.json").read_text())
    assert len(records) == len(record_vecs)
    assert len(frozen_queries) == len(query_vecs)
    return records, record_vecs, frozen_queries, query_vecs


def ensure_collection(client: QdrantClient, name: str) -> None:
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        optimizers_config=models.OptimizersConfigDiff(default_segment_number=1),
    )


def measure_recall_and_latency(client: QdrantClient, collection: str,
                                frozen_queries: list[dict], query_vecs: list[list[float]],
                                ks=(1, 5, 10)) -> dict:
    hits = {k: 0 for k in ks}
    latencies = []
    for q, qvec in zip(frozen_queries, query_vecs):
        t0 = time.perf_counter()
        results = client.query_points(collection_name=collection, query=qvec, limit=max(ks)).points
        latencies.append((time.perf_counter() - t0) * 1000)
        evidence_point_ids = {point_id(e) for e in q["evidence_ids"]}
        for k in ks:
            top_k_ids = {r.id for r in results[:k]}
            if evidence_point_ids & top_k_ids:
                hits[k] += 1
    n = len(frozen_queries)
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[int(len(latencies) * 0.99)]
    return {
        **{f"recall@{k}": round(hits[k] / n, 4) for k in ks},
        "query_p50_ms": round(p50, 3),
        "query_p99_ms": round(p99, 3),
        "n_queries": n,
    }


def fetch_process_metrics() -> dict:
    """Real jemalloc stats Qdrant exposes on its own /metrics endpoint,
    same method as payload-audit. Process-wide, not per-collection,
    which is why the container is restarted between the batch and streamed
    legs below: with only one collection resident at a time, a process-wide
    reading is an isolated reading."""
    with urllib.request.urlopen(f"{QDRANT_URL}/metrics", timeout=10) as resp:
        text = resp.read().decode()
    values = {}
    for line in text.splitlines():
        if line.startswith("memory_") and " " in line:
            name, val = line.rsplit(" ", 1)
            try:
                values[name] = int(val)
            except ValueError:
                pass
    return values


def measure_footprint(client: QdrantClient, collection: str) -> dict:
    info = client.get_collection(collection)
    disk_bytes = None
    try:
        out = subprocess.run(
            ["docker", "exec", CONTAINER_NAME, "du", "-sb", f"/qdrant/storage/collections/{collection}"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0:
            disk_bytes = int(out.stdout.split()[0])
    except Exception:
        pass

    mem = fetch_process_metrics()

    return {
        "points_count": info.points_count,
        "segments_count": info.segments_count,
        "disk_bytes": disk_bytes,
        "memory_allocated_bytes": mem.get("memory_allocated_bytes"),
        "memory_resident_bytes": mem.get("memory_resident_bytes"),
    }


def restart_qdrant_for_clean_baseline() -> None:
    """Restarts the Qdrant container so process-wide memory metrics start
    from a fresh baseline before each leg, isolating one collection's real
    memory cost from whatever ran before it in the same process."""
    print(f"  restarting {CONTAINER_NAME} for a clean memory baseline ...")
    subprocess.run(["docker", "restart", CONTAINER_NAME], capture_output=True, text=True, timeout=60)
    for _ in range(30):
        try:
            urllib.request.urlopen(f"{QDRANT_URL}/readyz", timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("Qdrant did not come back up after restart")


def run_batch_leg(client: QdrantClient, records, record_vecs, frozen_queries, query_vecs) -> dict:
    name = "agent_memory_batch"
    print(f"\n=== batch: {name} ===")
    ensure_collection(client, name)

    t0 = time.perf_counter()
    client.upsert(
        collection_name=name,
        points=models.Batch(
            ids=[point_id(r["id"]) for r in records],
            vectors=record_vecs,
            payloads=[{"dia_id": r["id"], "conversation_id": r["conversation_id"]} for r in records],
        ),
    )
    upsert_seconds = time.perf_counter() - t0
    print(f"  upserted {len(records)} points in one call, {upsert_seconds:.2f}s")

    print(f"  settling {SETTLE_SECONDS}s ...")
    time.sleep(SETTLE_SECONDS)

    metrics = measure_recall_and_latency(client, name, frozen_queries, query_vecs)
    footprint = measure_footprint(client, name)
    print(f"  {metrics}")
    print(f"  {footprint}")

    return {"upsert_seconds": round(upsert_seconds, 2), **metrics, **footprint}


def run_streamed_leg(client: QdrantClient, records, record_vecs, frozen_queries, query_vecs) -> dict:
    name = "agent_memory_streamed"
    print(f"\n=== streamed: {name} ===")
    ensure_collection(client, name)

    rng = random.Random(SEED)
    written_ids: list[int] = []  # indices into records already upserted, for touch-update sampling
    checkpoints = []
    i = 0
    t_start = time.perf_counter()

    while i < len(records):
        batch_size = rng.randint(STREAM_BATCH_MIN, STREAM_BATCH_MAX)
        batch_idx = list(range(i, min(i + batch_size, len(records))))

        # Real re-upsert of an already-written record's real payload/vector,
        # unchanged, at TOUCH_RATE probability, simulating an agent
        # re-confirming a memory it already wrote (not fabricated content).
        if written_ids and rng.random() < TOUCH_RATE:
            touch_idx = rng.choice(written_ids)
            client.upsert(
                collection_name=name,
                points=models.Batch(
                    ids=[point_id(records[touch_idx]["id"])],
                    vectors=[record_vecs[touch_idx]],
                    payloads=[{"dia_id": records[touch_idx]["id"],
                               "conversation_id": records[touch_idx]["conversation_id"]}],
                ),
            )

        client.upsert(
            collection_name=name,
            points=models.Batch(
                ids=[point_id(records[j]["id"]) for j in batch_idx],
                vectors=[record_vecs[j] for j in batch_idx],
                payloads=[{"dia_id": records[j]["id"], "conversation_id": records[j]["conversation_id"]} for j in batch_idx],
            ),
        )
        written_ids.extend(batch_idx)
        i += batch_size

        crossed_checkpoint = (len(written_ids) // CHECKPOINT_EVERY) > (len(checkpoints))
        if crossed_checkpoint or i >= len(records):
            elapsed = time.perf_counter() - t_start
            print(f"  checkpoint @ {len(written_ids)}/{len(records)} records ({elapsed:.1f}s elapsed) ...")
            metrics = measure_recall_and_latency(client, name, frozen_queries, query_vecs)
            footprint = measure_footprint(client, name)
            checkpoints.append({
                "records_written": len(written_ids),
                "elapsed_seconds": round(elapsed, 2),
                **metrics,
                **footprint,
            })
            print(f"    {metrics}")
            print(f"    {footprint}")

    total_seconds = time.perf_counter() - t_start
    print(f"  streamed write complete: {len(records)} records, {total_seconds:.2f}s total, "
          f"{TOUCH_RATE:.0%} touch-update rate, batches of {STREAM_BATCH_MIN}-{STREAM_BATCH_MAX}")

    return {"total_write_seconds": round(total_seconds, 2), "checkpoints": checkpoints, "final": checkpoints[-1]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    args = parser.parse_args()

    records, record_vecs, frozen_queries, query_vecs = load_data()
    print(f"Real LoCoMo data: {len(records)} memory records, {len(frozen_queries)} frozen held-out queries")

    client = QdrantClient(url=args.qdrant_url)

    # Clear out unrelated leftover collections from other repos sharing this
    # same local Qdrant container, so they can't contribute to the
    # process-wide memory readings below.
    for existing in client.get_collections().collections:
        if existing.name not in ("agent_memory_batch", "agent_memory_streamed"):
            print(f"Removing unrelated leftover collection: {existing.name}")
            client.delete_collection(existing.name)

    restart_qdrant_for_clean_baseline()
    client = QdrantClient(url=args.qdrant_url)

    results = {
        "n_records": len(records),
        "n_queries": len(frozen_queries),
        "touch_rate": TOUCH_RATE,
        "stream_batch_range": [STREAM_BATCH_MIN, STREAM_BATCH_MAX],
        "checkpoint_every": CHECKPOINT_EVERY,
        "settle_seconds": SETTLE_SECONDS,
        "seed": SEED,
    }
    results["batch"] = run_batch_leg(client, records, record_vecs, frozen_queries, query_vecs)

    client.delete_collection("agent_memory_batch")
    restart_qdrant_for_clean_baseline()
    client = QdrantClient(url=args.qdrant_url)

    results["streamed"] = run_streamed_leg(client, records, record_vecs, frozen_queries, query_vecs)

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
