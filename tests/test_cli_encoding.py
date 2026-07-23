"""Regression coverage for the Windows-console UnicodeEncodeError crash.

`core.metrics` embeds `≈` ("~=") in judge notes and `core.compare`
embeds `→` ("->") in case-transition summaries. Both get `print()`ed by
the CLI. On a plain Windows console (cp1252 or similar legacy codepage),
`print()`ing either character used to raise UnicodeEncodeError and abort the
command -- for `run`, after the report JSON was already written but before
the history ledger got a chance to record it. `agenteval.cli.main` now calls
`_harden_console_encoding()` before doing anything else so this can't happen.
"""

import io
import sys

from agenteval.cli import _harden_console_encoding


def test_reconfigures_streams_that_support_it(monkeypatch):
    stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    stderr = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    _harden_console_encoding()

    # Would raise UnicodeEncodeError on an un-hardened cp1252 stream.
    print("found number ≈ 30.0", file=stdout)
    print("`known`: passed → failed", file=stderr)
    stdout.flush()
    stderr.flush()

    stdout.buffer.seek(0)
    written = stdout.buffer.read().decode("cp1252")
    assert "\\u2248" in written
    assert "≈" not in written  # the raw character was never written


def test_tolerates_streams_without_reconfigure(monkeypatch):
    class NoReconfigureStream:
        def write(self, s):
            return len(s)

        def flush(self):
            pass

    monkeypatch.setattr(sys, "stdout", NoReconfigureStream())
    monkeypatch.setattr(sys, "stderr", NoReconfigureStream())

    _harden_console_encoding()  # must not raise


def test_tolerates_reconfigure_that_raises(monkeypatch):
    class HostileStream:
        def reconfigure(self, **kwargs):
            raise ValueError("already in use")

        def write(self, s):
            return len(s)

        def flush(self):
            pass

    monkeypatch.setattr(sys, "stdout", HostileStream())
    monkeypatch.setattr(sys, "stderr", HostileStream())

    _harden_console_encoding()  # must not raise
