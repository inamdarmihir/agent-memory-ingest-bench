# Results

These are the committed, published numbers for this benchmark: real LoCoMo
data (5,882 memory records, 300 frozen held-out queries), one Qdrant
container, two collections built from identical vectors, only the write
pattern differs. Nothing on this page is regenerated or re-derived — it's a
direct read of [`results.json`](results.json) (machine-readable, every
checkpoint) and [`run_benchmark.log`](run_benchmark.log) (the raw run
output). See [README.md](README.md) for method, limitations, and how to
reproduce or adapt this.

## Final numbers

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
Streamed p99 is actually lower here, which is noise at this query count, not
a real streamed-is-faster effect.

## Streamed leg's checkpoint trajectory

The streamed collection is measured every 500 records as it's built, so this
is the same final row above, plus every point along the way:

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

Recall climbs steadily as more records get written — that's correct
behavior, not degradation: a query's grounding evidence literally doesn't
exist in the collection yet at an early checkpoint. Query latency stays flat
(3.0-3.8 ms p50) across the entire write process, from 502 points to 5,882.
That flat line, not the recall climb, is the real signal: nothing about
writing in small batches, interleaved with reads and re-upserts, degraded
query latency as the collection grew. Full field-by-field detail (disk,
memory, segments per checkpoint) is in [`results.json`](results.json).

## Plotting this yourself

No plotting code is included in this repo (keeping it to the benchmark
itself), but every number above comes straight out of `results.json`'s
`streamed.checkpoints` list, so a quick local plot doesn't require re-running
anything:

```python
import json
import matplotlib.pyplot as plt

data = json.loads(open("results.json").read())
checkpoints = data["streamed"]["checkpoints"]

x = [c["records_written"] for c in checkpoints]
plt.plot(x, [c["recall@10"] for c in checkpoints], label="recall@10")
plt.plot(x, [c["query_p50_ms"] for c in checkpoints], label="query p50 (ms)")
plt.xlabel("records written (streamed leg)")
plt.legend()
plt.savefig("checkpoint_trajectory.png")
```

Swap in whichever fields you care about (`disk_bytes`,
`memory_resident_bytes`, `segments_count`, ...); they're all in every
checkpoint dict, plus the top-level `batch` dict for the single-shot
comparison point.
