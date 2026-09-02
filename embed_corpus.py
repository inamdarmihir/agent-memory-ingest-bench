"""
embed_corpus.py

Embeds the real LoCoMo memory records and a frozen, stratified sample of the
real query set exactly once, with FastEmbed's bge-small-en-v1.5. Embedding
happens once and is cached to embeddings/, so the actual batch-vs-streamed
write-pattern experiment (run_benchmark.py) measures only what the write
pattern itself does to a Qdrant collection, not embedding cost: same data,
same embedder, only the write pattern differs.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from fastembed import TextEmbedding

DEFAULT_QUERY_SAMPLE_SIZE = 300
DEFAULT_SEED = 42
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


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


def build_arg_parser() -> argparse.ArgumentParser:
    default_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Embed LoCoMo memory records and a frozen query sample.")
    parser.add_argument("--data-dir", type=Path, default=default_dir / "data",
                         help="Directory containing memory_records.json/query_set.json, and where "
                              "frozen_query_sample.json is written (default: %(default)s)")
    parser.add_argument("--embed-dir", type=Path, default=default_dir / "embeddings",
                         help="Directory to write record_vectors.json/query_vectors.json to "
                              "(default: %(default)s)")
    parser.add_argument("--query-sample-size", type=int, default=DEFAULT_QUERY_SAMPLE_SIZE,
                         help="Size of the frozen, stratified held-out query sample (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                         help="Random seed for the stratified query sample (default: %(default)s)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                         help="FastEmbed model name to embed records and queries with "
                              "(default: %(default)s). Changing this from the published run's "
                              "model makes vectors incomparable with the committed results.json.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.embed_dir.mkdir(exist_ok=True, parents=True)
    records = json.loads((args.data_dir / "memory_records.json").read_text())
    queries = json.loads((args.data_dir / "query_set.json").read_text())

    frozen_queries = stratified_query_sample(queries, args.query_sample_size, args.seed)
    (args.data_dir / "frozen_query_sample.json").write_text(json.dumps(frozen_queries, indent=2))
    print(f"Frozen held-out query sample: {len(frozen_queries)} of {len(queries)} real queries "
          f"(stratified by category, seed={args.seed})")

    model = TextEmbedding(model_name=args.model)

    print(f"Embedding {len(records)} real memory records ...")
    record_vecs = [v.tolist() for v in model.embed([r["text"] for r in records])]
    (args.embed_dir / "record_vectors.json").write_text(json.dumps(record_vecs))
    print(f"Wrote {len(record_vecs)} vectors to {args.embed_dir / 'record_vectors.json'}")

    print(f"Embedding {len(frozen_queries)} real held-out queries ...")
    query_vecs = [v.tolist() for v in model.embed([q["question"] for q in frozen_queries])]
    (args.embed_dir / "query_vectors.json").write_text(json.dumps(query_vecs))
    print(f"Wrote {len(query_vecs)} vectors to {args.embed_dir / 'query_vectors.json'}")


if __name__ == "__main__":
    main()
