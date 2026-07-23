# Minimal container image for the AgentEval CLI.
#
# python:3.12-slim (glibc, matching the Python version CI pins) rather than
# an alpine base: AgentEval's runtime dependencies include pandas, which
# ships prebuilt manylinux wheels for glibc but would need a full compiler
# toolchain to build from source on musl/alpine -- slim keeps both the
# image small and the build reliable without one.
FROM python:3.12-slim

WORKDIR /app

# Copy the full repository (not just the installable package) so
# examples/action_demo -- not part of the installed package's own file
# list, only ever resolved from a checkout -- is still available for the
# smoke test below, exactly like pytest already resolves it from the repo
# root today. PYTHONPATH exposes /app itself for that same reason.
COPY . /app
ENV PYTHONPATH=/app

RUN pip install --no-cache-dir . \
    && useradd --create-home --shell /usr/sbin/nologin agenteval \
    && chown -R agenteval:agenteval /app

USER agenteval

ENTRYPOINT ["agenteval"]
CMD ["--help"]
