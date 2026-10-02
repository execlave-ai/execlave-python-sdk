"""Wire contract between the Python bypass reporter and the backend.

The server's schema (``backend/src/schema/sdkBypass.ts``) is ``.strict()``: one
field over its bound or one key it does not know rejects the WHOLE report, and
the SDK treats 400 as "never acceptable" and discards it. Drift between the two is
therefore not a loud failure -- it is evidence quietly thrown away. The Python and
TypeScript sides cannot import each other, so they meet at a golden file:

* this test builds bodies with the real reporter and requires them to equal
  ``fixtures/bypass_report_bodies.json``;
* ``backend/tests/unit/sdkBypassParity.test.ts`` runs that same file through the
  server's Zod schema.

Change the reporter's output and this test fails until the fixture is
regenerated (``UPDATE_BYPASS_FIXTURE=1 pytest tests/test_bypass_wire_contract.py``),
at which point the backend test says whether the server still accepts it.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from execlave.bypass_reporter import BypassWindowReporter

FIXTURE = Path(__file__).parent / "fixtures" / "bypass_report_bodies.json"
START = datetime(2026, 9, 21, 12, 0, 0, tzinfo=timezone.utc).timestamp()
SDK = {"name": "execlave-sdk", "language": "python", "version": "1.8.0"}


class _Clock:
    def __init__(self):
        self.t = START

    def __call__(self):
        return self.t


def _capture(build, **opts):
    clock = _Clock()
    counter = iter(range(1, 10_000))
    bodies = []

    def send(body):
        bodies.append(body)
        return 200

    reporter = BypassWindowReporter(
        send,
        SDK,
        now=clock,
        new_id=lambda: f"00000000-0000-4000-8000-{next(counter):012d}",
        **opts,
    )
    build(reporter, clock)
    reporter.drain(5)
    return bodies


def _scenarios():
    def full(r, _c):
        r.record(
            reason="server_error",
            source="fail_open_server_error",
            agent_id="agent-1",
            message="server error 503",
            status=503,
            consecutive_failures=4,
        )

    def minimal(r, _c):
        r.record(reason="network_error", source="fail_open_network_error", agent_id="a")

    def oversized(r, _c):
        r.record(
            reason="network_error",
            source="s" * 500,
            agent_id="a" * 1_000,
            message="\U0001f600" * 600,
        )

    def unstorable_text(r, _c):
        # NUL and a lone surrogate would make the server's jsonb insert throw.
        r.record(
            reason="network_error",
            source="fail_open_network_error",
            agent_id="agent" + chr(0),
            message="a" + chr(0xD83D) + "b" + chr(0x1F600) + chr(0),
        )

    def loss_only(r, _c):
        r.record(reason="network_error", source="fail_open_network_error", agent_id="a")
        r.recover("a")

    def coalesced(r, c):
        for _ in range(3):
            r.record(
                reason="circuit_breaker_open",
                source="fail_open_circuit_breaker",
                agent_id="agent-1",
                message="down",
                consecutive_failures=3,
            )
            c.t += 7.5

    return {
        "full_window": _capture(full),
        "minimal_window": _capture(minimal),
        "oversized_fields_truncated": _capture(oversized),
        "unstorable_text_cleaned": _capture(unstorable_text),
        "loss_only": _capture(loss_only, max_buffered_windows=0),
        "coalesced_window": _capture(coalesced),
    }


def test_the_reporter_still_produces_the_bodies_the_backend_validates():
    actual = _scenarios()
    if os.environ.get("UPDATE_BYPASS_FIXTURE") == "1":
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert json.loads(json.dumps(actual)) == expected


def test_every_scenario_produced_a_request():
    # A scenario that silently sends nothing would leave the fixture (and the
    # backend test that reads it) vacuous.
    for name, bodies in _scenarios().items():
        assert len(bodies) == 1, name
