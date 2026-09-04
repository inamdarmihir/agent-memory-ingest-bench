"""
fetch_dataset.py

Downloads the real LoCoMo dataset (snap-research/locomo, arXiv:2402.17753),
CC BY-NC 4.0, non-commercial research use, and flattens its 10 real
long-term conversations (5,882 real dialogue turns across 19 sessions per
conversation on average) into `memory_records.json` (every dialogue turn,
in original session/turn order, the unit an agent-memory system would
actually write) and `query_set.json` (every real QA pair with grounding
evidence, the frozen held-out set recall@k is measured against). Nothing
here is synthetic: both files are a direct reshaping of LoCoMo's own real
annotations. The dataset itself is not vendored into this repo; it's fetched
fresh under its own license.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"


def fetch_raw(url: str) -> list[dict]:
    print(f"Downloading real LoCoMo data from {url} ...")
    with urllib.request.urlopen(url) as resp:
        raw = json.loads(resp.read())
    print(f"Loaded {len(raw)} real conversations "
          f"(snap-research/locomo, CC BY-NC 4.0, arXiv:2402.17753)")
    return raw


def flatten_memory_records(raw: list[dict]) -> list[dict]:
    records = []
    for sample in raw:
        conv_id = sample["sample_id"]
        conv = sample["conversation"]
        session_keys = sorted(
            (k for k in conv if k.startswith("session_") and not k.endswith("_date_time")),
            key=lambda k: int(k.split("_")[1]),
        )
        for session_key in session_keys:
            session_idx = int(session_key.split("_")[1])
            date_time = conv.get(f"{session_key}_date_time", "")
            for turn in conv[session_key]:
                records.append({
                    "id": f"{conv_id}:{turn['dia_id']}",
                    "conversation_id": conv_id,
                    "session": session_idx,
                    "date_time": date_time,
                    "speaker": turn["speaker"],
                    "text": f"{turn['speaker']}: {turn['text']}",
                    "dia_id": turn["dia_id"],
                })
    return records


def build_query_set(raw: list[dict]) -> list[dict]:
    """Every real QA pair with grounding evidence, across all five of
    LoCoMo's own question categories (single-hop, temporal, multi-hop,
    open-domain, adversarial). Evidence dia_ids are what recall@k checks
    against; the (sometimes intentionally tricky) answer text itself is
    kept for reference but not used by this repo's retrieval metric."""
    queries = []
    for sample in raw:
        conv_id = sample["sample_id"]
        for qa in sample["qa"]:
            evidence = qa.get("evidence") or []
            if not evidence:
                continue
            queries.append({
                "conversation_id": conv_id,
                "question": qa["question"],
                "category": qa.get("category"),
                "evidence_ids": [f"{conv_id}:{d}" for d in evidence],
                "answer": qa.get("answer", qa.get("adversarial_answer")),
            })
    return queries


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch and flatten the real LoCoMo dataset.")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "data",
                         help="Directory to write memory_records.json and query_set.json to "
                              "(default: %(default)s)")
    parser.add_argument("--url", default=LOCOMO_URL,
                         help="URL to fetch the raw LoCoMo JSON from (default: %(default)s)")
    parser.add_argument("--max-conversations", type=int, default=None,
                         help="Only keep the first N of LoCoMo's 10 real conversations (default: all). "
                              "Still real LoCoMo data, just fewer of it — use this to build a small, "
                              "fast slice for a smoke test instead of the full 5,882-record corpus "
                              "(see 'Smoke test' in README.md).")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.out_dir.mkdir(exist_ok=True, parents=True)
    raw = fetch_raw(args.url)
    if args.max_conversations is not None:
        raw = raw[: args.max_conversations]
        print(f"--max-conversations set: keeping the first {len(raw)} of the fetched conversations")

    records = flatten_memory_records(raw)
    (args.out_dir / "memory_records.json").write_text(json.dumps(records, indent=2))
    print(f"Wrote {len(records)} real memory records to {args.out_dir / 'memory_records.json'}")

    queries = build_query_set(raw)
    (args.out_dir / "query_set.json").write_text(json.dumps(queries, indent=2))
    print(f"Wrote {len(queries)} real grounded queries to {args.out_dir / 'query_set.json'}")


if __name__ == "__main__":
    main()
