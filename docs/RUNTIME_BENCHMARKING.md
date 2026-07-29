# Runtime Benchmarking

Runtime measurements are evidence, not training authorization or checkpoint
selection. VASU profiling tools remain non-mutating: use their explicit output
option to write a raw JSON report, then freeze selected numeric measurements
into a versioned benchmark contract.

## Comparison contract

Every `vasu_runtime_benchmark_v1` artifact binds its raw report SHA-256 to:

- a benchmark kind;
- workload identity (for example model, batch size, sequence length, cache
  implementation, data-source identity, and iteration count);
- environment identity (for example device, PyTorch version, CUDA version, and
  operating-system/runtime setting); and
- a measured variant (the intentionally changed setting, such as worker count,
  pinned memory, cache implementation, or optimizer backend); and
- explicitly named numeric metrics.

The comparison command fails closed when benchmark kind, workload identity,
environment identity, metric names, or metric paths differ. This prevents a
CPU timing, different batch shape, or different generation workload from being
reported as a performance regression or improvement.

The variant is reported for each side but is not required to match. This is
what permits a controlled comparison of one explicitly declared performance
setting while preserving the workload and environment contract.

## Workflow

First collect a raw report with an existing read-only profiler, for example:

```powershell
python scripts/profiling/profile_data_pipeline.py --output tmp/profile.json
```

Then create a new benchmark artifact. Never overwrite an existing benchmark;
use a versioned result filename.

```powershell
python scripts/profiling/create_runtime_benchmark.py `
  --label <implementation-or-commit> `
  --kind data_pipeline `
  --raw-report tmp/profile.json `
  --metric loader_tokens_per_second=loader_tokens_per_second `
  --workload dataset=<dataset-id> --workload batch_size=<batch-size> `
  --workload sequence_length=<sequence-length> `
  --environment device=<device> --environment torch=<torch-version> `
  --variant workers=<worker-count> --variant pin_memory=<true-or-false> `
  --output evaluation/results/<new-runtime-benchmark>.json

python scripts/profiling/compare_runtime_benchmarks.py `
  --baseline <baseline-runtime-benchmark>.json `
  --candidate <candidate-runtime-benchmark>.json `
  --output evaluation/results/<new-runtime-comparison>.json
```

The comparison output is descriptive only. A performance change must still be
checked for generation parity, checkpoint compatibility, and training safety
before any implementation is adopted.
