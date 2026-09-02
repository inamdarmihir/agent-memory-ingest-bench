"""
run_benchmark.py

Compares writing a Qdrant collection the way an agent-memory system does
(continuous small batches, interleaved reads, plus periodic re-writes of
already-stored memories) against loading the same data the way a normal RAG
corpus gets loaded (one bulk upsert call), against a real, frozen held-out
query set. Two collections (`agent_memory_batch`, `agent_memory_streamed`)
are built from the same LoCoMo memory records and pre-computed FastEmbed
vectors so embedding cost is identical and excluded from both legs; only the
write pattern differs. All modeling parameters below are CLI flags, not
hidden constants, so this can be re-run against different data or a
differently named Qdrant container without editing the script.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

VECTOR_SIZE = 384
SEED = 42
UUID_NAMESPACE = uuid.UUID("a3f1c2d4-0000-4000-8000-000000000000")

BATCH_COLLECTION = "agent_memory_batch"
STREAMED_COLLECTION = "agent_memory_streamed"


def point_id(record_id: str) -> str:
    return str(uuid.uuid5(UUID_NAMESPACE, record_id))


def load_data(data_dir: Path, embed_dir: Path) -> tuple[list[dict], list[list[float]], list[dict], list[list[float]]]:
    records = json.loads((data_dir / "memory_records.json").read_text())
    frozen_queries = json.loads((data_dir / "frozen_query_sample.json").read_text())
    record_vecs = json.loads((embed_dir / "record_vectors.json").read_text())
    query_vecs = json.loads((embed_dir / "query_vectors.json").read_text())
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
                                ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, Any]:
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


def fetch_process_metrics(qdrant_url: str) -> dict[str, int]:
    """Real jemalloc stats Qdrant exposes on its own /metrics endpoint.
    Process-wide, not per-collection, which is why the container is
    restarted between the batch and streamed legs below: with only one
    collection resident at a time, a process-wide reading is an isolated
    reading."""
    with urllib.request.urlopen(f"{qdrant_url}/metrics", timeout=10) as resp:
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


def measure_footprint(client: QdrantClient, collection: str, container: str, qdrant_url: str) -> dict[str, Any]:
    """Collection footprint plus process-wide memory. Disk usage is read via
    `docker exec <container> du`; if that fails, this raises loudly instead
    of silently reporting `disk_bytes: None`, since a silent None is
    indistinguishable from "measured and it's zero"."""
    info = client.get_collection(collection)

    try:
        out = subprocess.run(
            ["docker", "exec", container, "du", "-sb", f"/qdrant/storage/collections/{collection}"],
            capture_output=True, text=True, timeout=30,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Could not run `docker`; is Docker installed and on PATH? "
            "Disk usage for the Qdrant collection can't be measured without it."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"`docker exec {container} du ...` timed out. If your Qdrant container isn't named "
            f"'{container}', pass --container <name>."
        ) from exc

    if out.returncode != 0:
        raise RuntimeError(
            f"`docker exec {container} du ...` failed (exit {out.returncode}): {out.stderr.strip()}. "
            f"This usually means no container named '{container}' exists. Pass --container <name> "
            f"to point this script at your actual Qdrant container."
        )

    disk_bytes = int(out.stdout.split()[0])
    mem = fetch_process_metrics(qdrant_url)

    return {
        "points_count": info.points_count,
        "segments_count": info.segments_count,
        "disk_bytes": disk_bytes,
        "memory_allocated_bytes": mem.get("memory_allocated_bytes"),
        "memory_resident_bytes": mem.get("memory_resident_bytes"),
    }


def restart_qdrant_for_clean_baseline(container: str, qdrant_url: str) -> None:
    """Restarts the Qdrant container so process-wide memory metrics start
    from a fresh baseline before each leg, isolating one collection's real
    memory cost from whatever ran before it in the same process."""
    print(f"  restarting {container} for a clean memory baseline ...")
    result = subprocess.run(["docker", "restart", container], capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(
            f"`docker restart {container}` failed: {result.stderr.strip()}. If your Qdrant container "
            f"isn't named '{container}', pass --container <name>."
        )
    for _ in range(30):
        try:
            urllib.request.urlopen(f"{qdrant_url}/readyz", timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("Qdrant did not come back up after restart")


def run_batch_leg(client: QdrantClient, records: list[dict], record_vecs: list[list[float]],
                   frozen_queries: list[dict], query_vecs: list[list[float]],
                   settle_seconds: int, container: str, qdrant_url: str) -> dict[str, Any]:
    name = BATCH_COLLECTION
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

    print(f"  settling {settle_seconds}s ...")
    time.sleep(settle_seconds)

    metrics = measure_recall_and_latency(client, name, frozen_queries, query_vecs)
    footprint = measure_footprint(client, name, container, qdrant_url)
    print(f"  {metrics}")
    print(f"  {footprint}")

    return {"upsert_seconds": round(upsert_seconds, 2), **metrics, **footprint}


def run_streamed_leg(client: QdrantClient, records: list[dict], record_vecs: list[list[float]],
                      frozen_queries: list[dict], query_vecs: list[list[float]],
                      stream_batch_min: int, stream_batch_max: int, touch_rate: float,
                      checkpoint_every: int, container: str, qdrant_url: str) -> dict[str, Any]:
    name = STREAMED_COLLECTION
    print(f"\n=== streamed: {name} ===")
    ensure_collection(client, name)

    rng = random.Random(SEED)
    written_ids: list[int] = []  # indices into records already upserted, for touch-update sampling
    checkpoints = []
    i = 0
    t_start = time.perf_counter()

    while i < len(records):
        batch_size = rng.randint(stream_batch_min, stream_batch_max)
        batch_idx = list(range(i, min(i + batch_size, len(records))))

        # Real re-upsert of an already-written record's real payload/vector,
        # unchanged, at touch_rate probability, simulating an agent
        # re-confirming a memory it already wrote (not fabricated content).
        if written_ids and rng.random() < touch_rate:
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

        crossed_checkpoint = (len(written_ids) // checkpoint_every) > (len(checkpoints))
        if crossed_checkpoint or i >= len(records):
            elapsed = time.perf_counter() - t_start
            print(f"  checkpoint @ {len(written_ids)}/{len(records)} records ({elapsed:.1f}s elapsed) ...")
            metrics = measure_recall_and_latency(client, name, frozen_queries, query_vecs)
            footprint = measure_footprint(client, name, container, qdrant_url)
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
          f"{touch_rate:.0%} touch-update rate, batches of {stream_batch_min}-{stream_batch_max}")

    return {"total_write_seconds": round(total_seconds, 2), "checkpoints": checkpoints, "final": checkpoints[-1]}


def build_arg_parser() -> argparse.ArgumentParser:
    default_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description="Benchmark bulk upsert vs agent-style streamed writes against a real Qdrant collection.",
    )
    parser.add_argument("--qdrant-url", default="http://localhost:6333",
                         help="Qdrant HTTP URL (default: %(default)s)")
    parser.add_argument("--container", default="qdrant-agent-memory-bench",
                         help="Name of the Docker container running Qdrant, used for `docker exec/restart` "
                              "(default: %(default)s). Must match the container you actually started, "
                              "e.g. via docker-compose.yml.")
    parser.add_argument("--data-dir", type=Path, default=default_dir / "data",
                         help="Directory containing memory_records.json and frozen_query_sample.json "
                              "(default: %(default)s)")
    parser.add_argument("--embed-dir", type=Path, default=default_dir / "embeddings",
                         help="Directory containing record_vectors.json and query_vectors.json "
                              "(default: %(default)s)")
    parser.add_argument("--out", type=Path, default=default_dir / "results.json",
                         help="Path to write results JSON to (default: %(default)s)")
    parser.add_argument("--stream-batch-min", type=int, default=1,
                         help="Minimum streamed write batch size (default: %(default)s)")
    parser.add_argument("--stream-batch-max", type=int, default=5,
                         help="Maximum streamed write batch size (default: %(default)s)")
    parser.add_argument("--touch-rate", type=float, default=0.05,
                         help="Fraction of streamed writes that are re-upserts of an existing record "
                              "(default: %(default)s)")
    parser.add_argument("--checkpoint-every", type=int, default=500,
                         help="Measure recall/latency/footprint every N records written in the streamed leg "
                              "(default: %(default)s)")
    parser.add_argument("--settle-seconds", type=int, default=30,
                         help="Seconds to let WAL/segments settle before measuring the batch leg "
                              "(default: %(default)s)")
    parser.add_argument("--wipe-others", action="store_true", default=False,
                         help="Delete every collection on the connected Qdrant instance other than "
                              f"'{BATCH_COLLECTION}'/'{STREAMED_COLLECTION}' before running. Off by default; "
                              "only pass this against a Qdrant instance you're sure is safe to wipe.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    records, record_vecs, frozen_queries, query_vecs = load_data(args.data_dir, args.embed_dir)
    print(f"Real LoCoMo data: {len(records)} memory records, {len(frozen_queries)} frozen held-out queries")

    client = QdrantClient(url=args.qdrant_url)

    existing = [c.name for c in client.get_collections().collections]
    others = [name for name in existing if name not in (BATCH_COLLECTION, STREAMED_COLLECTION)]
    if others:
        if args.wipe_others:
            for name in others:
                print(f"--wipe-others set: deleting unrelated collection: {name}")
                client.delete_collection(name)
        else:
            print(f"Warning: {len(others)} other collection(s) exist on this Qdrant instance ({others}); "
                  f"memory_allocated_bytes/memory_resident_bytes below are process-wide and will include "
                  f"them too. Pass --wipe-others to remove them first, only if this Qdrant instance is safe to wipe.")

    restart_qdrant_for_clean_baseline(args.container, args.qdrant_url)
    client = QdrantClient(url=args.qdrant_url)

    results: dict[str, Any] = {
        "n_records": len(records),
        "n_queries": len(frozen_queries),
        "touch_rate": args.touch_rate,
        "stream_batch_range": [args.stream_batch_min, args.stream_batch_max],
        "checkpoint_every": args.checkpoint_every,
        "settle_seconds": args.settle_seconds,
        "seed": SEED,
    }
    results["batch"] = run_batch_leg(client, records, record_vecs, frozen_queries, query_vecs,
                                      args.settle_seconds, args.container, args.qdrant_url)

    client.delete_collection(BATCH_COLLECTION)
    restart_qdrant_for_clean_baseline(args.container, args.qdrant_url)
    client = QdrantClient(url=args.qdrant_url)

    results["streamed"] = run_streamed_leg(client, records, record_vecs, frozen_queries, query_vecs,
                                            args.stream_batch_min, args.stream_batch_max, args.touch_rate,
                                            args.checkpoint_every, args.container, args.qdrant_url)

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
