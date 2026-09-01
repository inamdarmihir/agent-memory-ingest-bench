# agent-memory-ingest-bench

Does writing a Qdrant collection the way an AI agent's memory system actually
writes it — continuous small batches, interleaved reads, occasional
re-writes of things already stored — cost anything, compared to loading the
same data the way a normal RAG corpus gets loaded (one bulk upsert)?

Real answer, on this benchmark, at this scale: no. The two collections
converge to effectively identical recall, latency, disk, and memory. That's
the actual finding below, not a hedge.

## Why this exists

Agent-memory tools (Mem0, Letta, Qdrant's own local-first `engram`) write
continuously: an agent reads its memory far less often than it writes to it,
in small increments, as it works. Every public RAG/vector-DB benchmark
loads its corpus once, in bulk, before measuring anything. Nobody had
published what the agent-style write pattern itself does to a real index's
recall and latency, as distinct from the embedding or retrieval quality
question those other benchmarks already cover. This repo measures that one
variable directly, at the Qdrant layer, isolated from any specific memory
framework's design choices.

## Method

**Data**: [LoCoMo](https://github.com/snap-research/locomo)
([arXiv:2402.17753](https://arxiv.org/abs/2402.17753)), CC BY-NC 4.0,
non-commercial research use. Ten real long-term dialogues (19 sessions
each on average, human-LLM generated but with real, annotated grounding),
flattened into 5,882 real dialogue turns as memory records, and 1,982 real
QA pairs with grounding evidence, of which a frozen, category-stratified
sample of 300 is the held-out query set (seed 42). Not vendored into this
repo; `fetch_dataset.py` downloads it fresh under its own license.

**Embedder**: FastEmbed `bge-small-en-v1.5`, the same model used as the
fp32 baseline in this program's other repo,
[ternlight-techdocs](https://github.com/inamdarmihir/ternlight-techdocs).
Every record and every query is embedded exactly once (`embed_corpus.py`),
before either collection is built, so embedding cost is identical and
excluded from both legs — the only variable that differs is the write
pattern.

**Two real Qdrant collections, same data, same vectors, only the write
pattern differs:**

- **`agent_memory_batch`** — one `upsert()` call, all 5,882 points.
- **`agent_memory_streamed`** — the same 5,882 points, same chronological
  session order, upserted in batches of 1-5 (agent-memory write
  granularity), with a real held-out query run plus a segment/footprint
  snapshot every 500 records, and a real re-upsert (same real vector and
  payload, unchanged) on 5% of writes, simulating an agent re-confirming a
  memory it already stored.

Both legs use `default_segment_number=1` (deliberately, see Limitations)
and run against a container restarted between legs so process-wide memory
readings aren't contaminated by whichever collection ran first.

## Results

| | Batch (final) | Streamed (final, 5,882/5,882) |
|---|---:|---:|
| recall@1 | 0.2433 | 0.2433 |
| recall@5 | 0.4800 | 0.4800 |
| recall@10 | 0.5667 | 0.5667 |
| query p50 | 3.575 ms | 3.680 ms |
| query p99 | 8.125 ms | 5.981 ms |
| segments | 1 | 1 |
| disk | 136,519,258 B | 136,493,733 B |
| memory allocated | 55,584,872 B | 51,224,616 B |
| memory resident | 185,532,416 B | 192,544,768 B |
| write time | 0.52s (one call) | 18.83s (1,700+ calls) |

Recall and disk are effectively identical (the ~25 KB disk difference and
the memory deltas are within normal run-to-run noise, not a pattern).
Streamed p99 is actually lower here, which is noise at this query count,
not a real streamed-is-faster effect. Full machine-readable results,
including every checkpoint: [`results.json`](results.json). Raw run log:
[`run_benchmark.log`](run_benchmark.log).

### The streamed leg's checkpoint trajectory

| Records written | recall@1 | recall@5 | recall@10 | query p50 |
|---:|---:|---:|---:|---:|
| 502 | 0.0233 | 0.0467 | 0.0533 | 2.951 ms |
| 1,002 | 0.0367 | 0.0867 | 0.1133 | 3.017 ms |
| 1,500 | 0.0400 | 0.1067 | 0.1267 | 3.153 ms |
| 2,004 | 0.0600 | 0.1433 | 0.1767 | 3.049 ms |
| 2,500 | 0.0867 | 0.1967 | 0.2467 | 3.343 ms |
| 3,001 | 0.1133 | 0.2233 | 0.2667 | 3.219 ms |
| 3,502 | 0.1133 | 0.2367 | 0.2900 | 3.156 ms |
| 4,001 | 0.1333 | 0.2700 | 0.3333 | 3.471 ms |
| 4,501 | 0.1700 | 0.3300 | 0.4033 | 3.570 ms |
| 5,001 | 0.1967 | 0.3800 | 0.4500 | 3.565 ms |
| 5,501 | 0.2200 | 0.4300 | 0.5100 | 3.759 ms |
| 5,882 | 0.2433 | 0.4800 | 0.5667 | 3.680 ms |

Recall climbs steadily as more records get written. This is not
degradation, it's the correct behavior: a query's grounding evidence
literally doesn't exist in the collection yet at an early checkpoint, so it
can't be retrieved. Query latency stays flat (3.0-3.8 ms p50) across the
entire write process, from 502 points to 5,882. That flat line, not the
recall climb, is the real signal: nothing about writing in small batches,
interleaved with reads and re-upserts, degraded query latency as the
collection grew.

## What this actually shows

**No measurable cost to agent-style write patterns, at this scale.**
Qdrant's segment architecture appears to absorb continuous small-batch
writes, interleaved reads, and periodic re-upserts cleanly: the streamed
collection lands on the same recall, essentially the same latency, and
the same disk/memory footprint as the same data loaded in one call. This
is a real, useful negative result, not a failed experiment: the honest
hypothesis going in (that streamed writes would measurably degrade recall
or latency past some point) did not hold at 5,882 records.

**Absolute recall is genuinely low (0.24-0.57), and that's a property of
the task, not the write pattern.** Unlike this program's other repo, which
retrieves against a 500-candidate closed set, this is retrieval against
5,882 real conversation turns pulled from ten different, topically
unrelated dialogues, where a question's real phrasing and its grounding
utterance's real phrasing can diverge a lot (LoCoMo's temporal and
multi-hop question categories are deliberately hard this way). Both
collections see the same low numbers, which is the actual point: the write
pattern isn't what's limiting recall here, embedding-level retrieval
difficulty is.

## Limitations

- **Scale.** 5,882 records is a real number but a modest one for testing
  segment fragmentation specifically; the effect measured here (recall,
  latency, footprint parity) might not hold at 100k+ records, where more
  segments would actually accumulate.
- **`default_segment_number=1` was forced on both collections**,
  deliberately, to isolate the write-pattern variable from a
  segment-count variable (the same choice `payload-audit` made for
  the same reason). This means the specific question "does streaming
  create segment fragmentation under Qdrant's default multi-segment
  optimizer" is untested here; this repo tests recall/latency/footprint
  parity under controlled segment count, not fragmentation itself.
- **Checkpoint reads are not truly concurrent with writes.** They run
  between write batches, not on a separate thread mid-write. This
  measures "does streamed loading order and re-upserts change the end
  state," not "does a concurrent reader see degraded results while a
  write is literally in flight." A true concurrency test would need an
  async or multi-threaded client and is a natural follow-up, not
  something claimed here.
- **The 5% touch-rate and 1-5 batch-size range are reasonable, chosen
  parameters, not measured from a real agent-memory system's actual
  traffic.** No public numbers on real Mem0/Letta/`engram` write
  granularity were available to calibrate against; these values are
  labeled as a modeling choice, not a measured fact.

## Reproducing this

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python3 fetch_dataset.py      # downloads real LoCoMo data (CC BY-NC 4.0)
python3 embed_corpus.py       # embeds once, ~20-60 min on CPU depending on hardware
python3 run_benchmark.py      # runs both legs, writes results.json
```

Requires Docker (a local Qdrant container is created/restarted by the
script) and enough free request-size headroom for a ~48 MB single-call
upsert — this repo's own run needed
`QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=128` set on the container; Qdrant's
default limit (32 MB) is too small for a single-call bulk upsert at this
record count and rejects it with a plain `400` rather than a silent
partial write, which is how this was caught during development.

## Adapting this to your own data

The whole method is generic once `data/` is in the right shape; nothing
in `run_benchmark.py` is LoCoMo-specific past `fetch_dataset.py`. To point
this at your own records:

1. Write your own `data/memory_records.json`: a list of objects, each with
   at least `id` (a string, any format, `run_benchmark.py`'s `point_id()`
   UUID5-hashes it since Qdrant needs int/UUID point IDs) and `text` (the
   string that gets embedded).
2. Write your own `data/query_set.json`: a list of objects with `question`
   (the query text), `category` (any label, used only for the stratified
   sample, a single category is fine) and `evidence_ids` (a list of the
   `memory_records.json` `id` values that should count as a correct
   retrieval for that query, the ground truth `measure_recall_and_latency()`
   scores against).
3. Run `python3 embed_corpus.py` to embed both and freeze a held-out query
   sample, then `python3 run_benchmark.py` to run both the batch and
   streamed legs and write `results.json`.

`STREAM_BATCH_MIN`/`MAX`, `TOUCH_RATE`, and `CHECKPOINT_EVERY` are plain
constants near the top of `run_benchmark.py`, edit them to model a
different write granularity or re-upsert rate than the 1-5 batch size and
5% touch rate this repo's own run used. `CONTAINER_NAME` and `QDRANT_URL`
there also assume a local Docker Qdrant; point `QDRANT_URL` at a remote
instance instead if you don't want the script managing a local container
(the `restart_qdrant_for_clean_baseline()` calls, used to get isolated
memory readings between legs, only work against a container it can `docker
restart` by name).

## What's not included, and why

- **The LoCoMo dataset itself.** CC BY-NC 4.0, fetched fresh by
  `fetch_dataset.py`, not redistributed here.
- **The FastEmbed vectors (`embeddings/`).** Regenerable in one command;
  keeping them out of git keeps the repo small.

## Citation

If you use this repo's method or results, a link back is appreciated.
Cite LoCoMo separately if you use the dataset:
[arXiv:2402.17753](https://arxiv.org/abs/2402.17753).

## License

MIT for the code in this repository. See [LICENSE](LICENSE). The LoCoMo
dataset it benchmarks against is separately licensed (CC BY-NC 4.0,
non-commercial) and is not redistributed here.
