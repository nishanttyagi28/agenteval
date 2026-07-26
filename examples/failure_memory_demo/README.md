# Failure Memory demo

Zero-network end-to-end demo of Agent Failure Memory:

1. A broken refund agent cancels instead of refunding.
2. Production-style traces are recorded (content capture on).
3. Failures are classified and clustered.
4. A human approval creates a golden regression case.
5. The broken agent fails the suite; the fixed agent passes.

```bash
# from repo root with the package importable
python examples/failure_memory_demo/run_demo.py

# keep artifacts
python examples/failure_memory_demo/run_demo.py --workdir /tmp/fm-demo --keep
```

No API keys, providers, or network access required.
