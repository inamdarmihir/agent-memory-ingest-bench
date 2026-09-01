"""
fetch_dataset.py

Downloads the real LoCoMo dataset (snap-research/locomo, arXiv:2402.17753),
CC BY-NC 4.0, non-commercial research use. Not vendored into this repo, same
reason as ternlight-techdocs doesn't vendor its upstream training code:
it's a real, independently maintained, separately licensed dataset, better
fetched fresh than committed as a stale copy.

Flattens LoCoMo's 10 real long-term conversations (5,882 real dialogue turns
across 19 sessions per conversation on average) into two files:

  memory_records.json  every dialogue turn, in original session/turn order,
                        the unit an agent-memory system would actually write
  query_set.json        every real QA pair that carries grounding evidence
                        (question + the dia_ids of the turns that answer it),
                        the frozen held-out set recall@k is measured against

Nothing here is synthetic. Both files are a direct reshaping of LoCoMo's own
real annotations, not generated or paraphrased.
"""

import json
import urllib.request
from pathlib import Path

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
OUT_DIR = Path(__file__).parent / "data"


def fetch_raw() -> list[dict]:
    print(f"Downloading real LoCoMo data from {LOCOMO_URL} ...")
    with urllib.request.urlopen(LOCOMO_URL) as resp:
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


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    raw = fetch_raw()

    records = flatten_memory_records(raw)
    (OUT_DIR / "memory_records.json").write_text(json.dumps(records, indent=2))
    print(f"Wrote {len(records)} real memory records to {OUT_DIR / 'memory_records.json'}")

    queries = build_query_set(raw)
    (OUT_DIR / "query_set.json").write_text(json.dumps(queries, indent=2))
    print(f"Wrote {len(queries)} real grounded queries to {OUT_DIR / 'query_set.json'}")


if __name__ == "__main__":
    main()
