"""
embed_corpus.py

Embeds the real LoCoMo memory records and a frozen, stratified sample of the
real query set exactly once, with FastEmbed's bge-small-en-v1.5 (the same
real embedder used as the fp32 baseline in qdrant-ternlight-techdocs, kept
consistent so results are comparable across this content program's repos).

Embedding happens once and is cached to embeddings/, so the actual
batch-vs-streamed write-pattern experiment (run_benchmark.py) measures only
what the write pattern itself does to a Qdrant collection, not embedding
cost. This mirrors the real method description: same data, same embedder,
only the write pattern differs.
"""

import json
import random
from pathlib import Path

from fastembed import TextEmbedding

DATA_DIR = Path(__file__).parent / "data"
EMBED_DIR = Path(__file__).parent / "embeddings"
QUERY_SAMPLE_SIZE = 300
SEED = 42


def stratified_query_sample(queries: list[dict], n: int, seed: int) -> list[dict]:
    """Samples proportionally across LoCoMo's 5 real question categories, so
    the frozen eval set isn't accidentally dominated by one category."""
    rng = random.Random(seed)
    by_cat: dict[int, list[dict]] = {}
    for q in queries:
        by_cat.setdefault(q["category"], []).append(q)

    total = len(queries)
    sample = []
    for cat, items in sorted(by_cat.items()):
        rng.shuffle(items)
        take = max(1, round(n * len(items) / total))
        sample.extend(items[:take])
    rng.shuffle(sample)
    return sample[:n]


def main() -> None:
    EMBED_DIR.mkdir(exist_ok=True)
    records = json.loads((DATA_DIR / "memory_records.json").read_text())
    queries = json.loads((DATA_DIR / "query_set.json").read_text())

    frozen_queries = stratified_query_sample(queries, QUERY_SAMPLE_SIZE, SEED)
    (DATA_DIR / "frozen_query_sample.json").write_text(json.dumps(frozen_queries, indent=2))
    print(f"Frozen held-out query sample: {len(frozen_queries)} of {len(queries)} real queries "
          f"(stratified by category, seed={SEED})")

    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    print(f"Embedding {len(records)} real memory records ...")
    record_vecs = [v.tolist() for v in model.embed([r["text"] for r in records])]
    (EMBED_DIR / "record_vectors.json").write_text(json.dumps(record_vecs))
    print(f"Wrote {len(record_vecs)} vectors to {EMBED_DIR / 'record_vectors.json'}")

    print(f"Embedding {len(frozen_queries)} real held-out queries ...")
    query_vecs = [v.tolist() for v in model.embed([q["question"] for q in frozen_queries])]
    (EMBED_DIR / "query_vectors.json").write_text(json.dumps(query_vecs))
    print(f"Wrote {len(query_vecs)} vectors to {EMBED_DIR / 'query_vectors.json'}")


if __name__ == "__main__":
    main()
