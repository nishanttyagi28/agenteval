#!/usr/bin/env bash
# Build the AgentEval Docker image and verify:
#   1. `docker run <image> --help` works (the CLI entrypoint resolves)
#   2. a fully self-contained evaluation (examples/action_demo -- zero API
#      key, zero external repo) succeeds *inside* the container
#
# Usage: scripts/docker_smoke_test.sh [image_tag]

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${1:-agenteval:smoke-test}"

echo "== Building $IMAGE_TAG from $ROOT_DIR =="
docker build -t "$IMAGE_TAG" "$ROOT_DIR"

echo "== Checking 'docker run $IMAGE_TAG --help' =="
HELP_OUTPUT="$(docker run --rm "$IMAGE_TAG" --help)"
if ! grep -q "usage: agenteval" <<< "$HELP_OUTPUT"; then
  echo "FAIL: --help output did not contain expected usage text" >&2
  echo "$HELP_OUTPUT" >&2
  exit 1
fi
echo "OK: --help works"

echo "== Running a self-contained sample evaluation (examples/action_demo) =="
RUN_OUTPUT="$(docker run --rm "$IMAGE_TAG" run \
  --agent action_demo \
  --registry examples/action_demo/agents.yaml \
  --runs-dir /tmp/agenteval-smoke-runs \
  --quiet)"
if ! grep -q "cases=1" <<< "$RUN_OUTPUT"; then
  echo "FAIL: sample evaluation did not report cases=1" >&2
  echo "$RUN_OUTPUT" >&2
  exit 1
fi
echo "OK: sample evaluation ran successfully inside the container"

echo "== Docker smoke test passed =="
