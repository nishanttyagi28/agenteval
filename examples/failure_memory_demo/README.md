# Failure Memory demo

Zero-network end-to-end demo of Agent Failure Memory:

1. A broken refund agent cancels instead of refunding.
2. Production-style traces are recorded (content capture on) with synthetic secrets.
3. Secrets are redacted before SQLite/JSONL persistence.
4. Failures are classified and clustered.
5. A human approval creates a golden regression case.
6. The broken agent fails the suite; the fixed agent passes.
7. A leakage scan confirms raw secrets never appear under the work directory.

```bash
# from repo root with the package importable (uses a fresh temp dir)
python examples/failure_memory_demo/run_demo.py

# keep artifacts for inspection
python examples/failure_memory_demo/run_demo.py --workdir %TEMP%\fm-demo --keep
```

No API keys, providers, or network access required.
The demo never reuses files from a previous run when `--workdir` is omitted.
