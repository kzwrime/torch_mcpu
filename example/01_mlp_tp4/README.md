# MLP TP4 all_reduce profiler example

This example extracts the MLP shape from `tests/test_compile_mlp.py` and scales it
to a TP=4 inference-style flow on `mcpu`.

Each rank owns one local shard:

```text
(16 x 1024) @ (1024 x 1024) -> native activation -> (16 x 1024) @ (1024 x 1024) -> all_reduce
```

Run from the repository root:

```bash
python example/01_mlp_tp4/run.py
```

The script validates that all four ranks finish on `mcpu`, all ranks use the
same input tensor, the all-reduced TP output matches a serial full-MLP reference,
the main loop stays inside an explicit non-default stream, no fallback CPU copy
operators appear in the profiled path, and every rank writes a readable gzipped
Chrome trace:

```text
example/01_mlp_tp4/traces/rank*/*.pt.trace.json.gz
```

Useful knobs:

```bash
python example/01_mlp_tp4/run.py --profile-iters 8
python example/01_mlp_tp4/run.py --tokens 32 --model-dim 2048 --local-hidden-dim 2048
python example/01_mlp_tp4/run.py --compile
python example/01_mlp_tp4/run.py --skip-serial-check
```
