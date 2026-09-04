# agent-memory-ingest-bench

Measures whether writing a Qdrant collection the way an AI agent's memory
system actually writes it — continuous small batches, interleaved reads,
occasional re-writes of things already stored — costs anything in recall,
query latency, or disk/memory footprint, compared to loading the same data
the way a normal RAG corpus gets loaded (one bulk upsert). Real LoCoMo
dialogue data, a real Qdrant container, two collections built from identical
vectors where only the write pattern differs. Companion to AI Hive's "What
agent memory actually does to a vector index."

Real answer, on this benchmark, at this scale: no. The two collections
converge to effectively identical recall, latency, disk, and memory. That's
the actual finding below, not a hedge.

## Results at a glance

| | Batch (final) | Streamed (final, 5,882/5,882) |
|---|---:|---:|
| recall@1 | 0.2433 | 0.2433 |
| recall@5 | 0.4800 | 0.4800 |
| recall@10 | 0.5667 | 0.5667 |
| query p50 | 3.575 ms | 3.680 ms |
| query p99 | 8.125 ms | 5.981 ms |
| disk | 136,519,258 B | 136,493,733 B |
| write time | 0.52s (one call) | 18.83s (1,700+ calls) |

Full table, the streamed leg's checkpoint-by-checkpoint trajectory, and how
to plot it: [RESULTS.md](RESULTS.md). Machine-readable, every checkpoint:
[`results.json`](results.json). Raw run output: [`run_benchmark.log`](run_benchmark.log).

## Quickstart

**Three ways to use this repo, depending on how much time you have:**

| Path | Time | What you get |
|---|---|---|
| [(a) Inspect published results](#a-inspect-the-published-results-0-setup) | 0 min | Read the numbers above, no setup |
| [(b) Smoke test](#b-smoke-test-1-2-min) | 1-2 min | Confirm the pipeline runs end-to-end, on your machine |
| [(c) Full reproduction](#c-full-reproduction-25-65-min) | 25-65 min | Regenerate the real 5,882-record benchmark yourself |

The full run's slow step is embedding, not Qdrant: **`embed_corpus.py` takes
~20-60 minutes on CPU** to embed 5,882 records + 300 queries with FastEmbed,
depending on your hardware. Everything else (fetch, benchmark) is seconds to
low minutes. If you just want to know the pipeline works, use (b) instead.

### (a) Inspect the published results (0 setup)

Nothing needs to run. The numbers are already committed:
[`results.json`](results.json) (every checkpoint, machine-readable) and
[`run_benchmark.log`](run_benchmark.log) (the raw run output), summarized in
[Results at a glance](#results-at-a-glance) above and in full in
[RESULTS.md](RESULTS.md).

### (b) Smoke test (1-2 min)

Validates the real pipeline — fetch, embed, both benchmark legs, against a
real (throwaway) Qdrant container — on a small real slice of LoCoMo (1
conversation, ~400-600 records) instead of the full corpus. Doesn't touch
`data/`, `embeddings/`, or the committed `results.json`.

```bash
pip install -r requirements.txt
./smoke_test.sh
```

Requires Docker. Runs in a throwaway container
(`qdrant-agent-memory-bench-smoke` by default) on port 6344, and cleans up
after itself. This is also what runs in CI on every push
(`.github/workflows/smoke.yml`).

### (c) Full reproduction (25-65 min)

```bash
docker compose up -d                 # starts Qdrant on localhost:6333
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python3 fetch_dataset.py             # downloads real LoCoMo data (CC BY-NC 4.0), seconds
python3 embed_corpus.py              # embeds once, ~20-60 min on CPU depending on hardware
python3 run_benchmark.py             # runs both legs, writes results.json, ~1-2 min
```

`docker-compose.yml`'s service is named `qdrant-agent-memory-bench`, which
matches `run_benchmark.py --container`'s default, so if you use the compose
file as-is you can drop that flag. Run `python3 run_benchmark.py --help` for
the full list of flags (batch size, touch rate, checkpoint interval,
data/embedding/output paths, `--wipe-others`, etc.) — none of the
benchmark's modeling parameters are hidden constants. See
[Reproducing this](#reproducing-this) below for the full detail (request
size limits, container-name requirements, what `--wipe-others` does).

**Want to point this at your own data instead of LoCoMo?** See
[Adapting this to your own data](#adapting-this-to-your-own-data).

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

**Embedder**: FastEmbed `bge-small-en-v1.5`. Every record and every query is
embedded exactly once (`embed_corpus.py`), before either collection is
built, so embedding cost is identical and excluded from both legs — the only
variable that differs is the write pattern.

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
the task, not the write pattern.** This is retrieval against 5,882 real
conversation turns pulled from ten different, topically unrelated
dialogues, where a question's real phrasing and its grounding utterance's
real phrasing can diverge a lot (LoCoMo's temporal and multi-hop question
categories are deliberately hard this way). Both collections see the same
low numbers, which is the actual point: the write pattern isn't what's
limiting recall here, embedding-level retrieval difficulty is.

## Limitations

- **Scale.** 5,882 records is a real number but a modest one for testing
  segment fragmentation specifically; the effect measured here (recall,
  latency, footprint parity) might not hold at 100k+ records, where more
  segments would actually accumulate.
- **`default_segment_number=1` was forced on both collections**,
  deliberately, to isolate the write-pattern variable from a
  segment-count variable. This means the specific question "does streaming
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
docker compose up -d          # starts Qdrant on localhost:6333, named qdrant-agent-memory-bench
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python3 fetch_dataset.py      # downloads real LoCoMo data (CC BY-NC 4.0)
python3 embed_corpus.py       # embeds once, ~20-60 min on CPU depending on hardware
python3 run_benchmark.py      # runs both legs, writes results.json
```

Requires Docker (the script `docker restart`s and `docker exec`s the
container between legs to get isolated memory/disk readings, by name via
`--container`) and enough free request-size headroom for a ~48 MB
single-call upsert. `docker-compose.yml` at the repo root already sets
`QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=128` and pins the image to
`qdrant/qdrant:v1.19.0` (matching the `qdrant-client>=1.19` pin in
`requirements.txt`); Qdrant's default request-size limit (32 MB) is too
small for a single-call bulk upsert at this record count and rejects it
with a plain `400` rather than a silent partial write, which is how this
was caught during development.

If you're not using `docker-compose.yml` as-is (different container name,
remote Qdrant, non-default ports), pass the matching flags — run
`python3 run_benchmark.py --help` for the full list, including
`--qdrant-url`, `--container`, `--data-dir`, `--embed-dir`, `--out`,
`--stream-batch-min`/`--stream-batch-max`, `--touch-rate`,
`--checkpoint-every`, `--settle-seconds`, and `--wipe-others`.
`fetch_dataset.py --help` and `embed_corpus.py --help` list the fetch/embed
side's own flags (`--out-dir`, `--url`, `--max-conversations`;
`--data-dir`, `--embed-dir`, `--query-sample-size`, `--seed`, `--model`).

By default `run_benchmark.py` only ever touches its own two collections
(`agent_memory_batch`, `agent_memory_streamed`) and leaves everything else
on the connected Qdrant instance alone, printing a warning if other
collections exist (since the memory metrics it measures are process-wide,
not per-collection). Pass `--wipe-others` to delete every other collection
first — only do this against a Qdrant instance you're sure is safe to wipe;
it is off by default specifically so pointing `--qdrant-url` at a shared or
production instance can't silently delete unrelated data.

## Smoke test

`smoke_test.sh` runs the same three scripts (`fetch_dataset.py`,
`embed_corpus.py`, `run_benchmark.py`) against a small, real slice of
LoCoMo — one conversation instead of ten, ~400-600 records instead of
5,882, a 20-query sample instead of 300 — against a disposable Qdrant
container it starts and tears down itself. It finishes in one to two
minutes and never touches `data/`, `embeddings/`, or the committed
`results.json`; everything runs against its own temp directory and its own
container (`qdrant-agent-memory-bench-smoke` on port 6344 by default,
override with `SMOKE_PORT` or by passing a container name as `$1`).

```bash
pip install -r requirements.txt
./smoke_test.sh
```

This is the fast way to confirm the pipeline itself (fetch → embed → both
benchmark legs → checkpoints → recall/latency/footprint measurement) works
on your machine — a clone health check, not a re-run of the published
benchmark. It's also what `.github/workflows/smoke.yml` runs in CI on every
push and pull request; the underlying flag it relies on is
`fetch_dataset.py --max-conversations N`, which keeps the pipeline entirely
real (real download, real embedder, real Qdrant) while keeping it small.

## Adapting this to your own data

The whole method is generic once your data is in the right shape; nothing
in `run_benchmark.py` is LoCoMo-specific past `fetch_dataset.py`. To point
this at your own records:

1. Write your own `memory_records.json`: a list of objects, each with at
   least `id` (a string, any format, `run_benchmark.py`'s `point_id()`
   UUID5-hashes it since Qdrant needs int/UUID point IDs) and `text` (the
   string that gets embedded).
2. Write your own `query_set.json`: a list of objects with `question` (the
   query text), `category` (any label, used only for the stratified sample,
   a single category is fine) and `evidence_ids` (a list of the
   `memory_records.json` `id` values that should count as a correct
   retrieval for that query, the ground truth `measure_recall_and_latency()`
   scores against).
3. Run `python3 embed_corpus.py` to embed both and freeze a held-out query
   sample, then `python3 run_benchmark.py` to run both the batch and
   streamed legs and write your own results file.

By default both scripts look in `./data` and `./embeddings`; pass
`--data-dir`/`--embed-dir` to either script to use different locations (for
example, to keep your own dataset entirely separate from LoCoMo's). Write
granularity and re-upsert rate are also flags rather than constants:
`--stream-batch-min`/`--stream-batch-max` (default 1-5) and `--touch-rate`
(default 5%) on `run_benchmark.py`, in place of this repo's own LoCoMo run's
values.

## What's not included, and why

- **The LoCoMo dataset itself.** CC BY-NC 4.0, fetched fresh by
  `fetch_dataset.py`, not redistributed here.
- **The FastEmbed vectors (`embeddings/`).** Regenerable in one command;
  keeping them out of git keeps the repo small.
- **Plotting code.** [RESULTS.md](RESULTS.md) has a short snippet to plot
  the checkpoint trajectory straight out of `results.json` if you want a
  chart; it's not required to read or reproduce the numbers.

## Citation

If you use this repo's method or results, a link back is appreciated.
Cite LoCoMo separately if you use the dataset:
[arXiv:2402.17753](https://arxiv.org/abs/2402.17753).

## License

MIT for the code in this repository. See [LICENSE](LICENSE). The LoCoMo
dataset it benchmarks against is separately licensed (CC BY-NC 4.0,
non-commercial) and is not redistributed here.
