"""Offline verification for the acceptance send gate and phase ledger.

Every scenario runs through the public host HTTP boundary, the public
Runtime/Agent surface and real temporary files. No test sends a real request,
needs a credential or consumes phase budget.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from acceptance_budget import (
    AcceptanceLedger,
    ModelPrice,
    PhaseLimits,
)
from omh.llm.types import FetchResponse
from support import (
    RecordingFetch,
    SequencedFetch,
    json_error_response,
    sse_response,
    text_stream,
    tool_call_stream,
)

from coding_agent import AgentSessionRuntime, CodingAgentHost
from coding_agent.print_runner import compose_tasks, run_print_text

EXIT_OK = 0
EXIT_FAILURE = 1

#: Conservative peak rates from the local 2026-10-05 snapshot, keyed like the
#: gate expects. Tests use them with controlled transports only.
PRICES = {
    ("opencode-go", "deepseek-v4.1-flash"): ModelPrice(0.30, 1.20),
    ("opencode-go", "deepseek-v4-pro"): ModelPrice(1.32, 3.96),
    ("opencode-go", "kimi-k3"): ModelPrice(3.00, 15.00),
    ("opencode-go", "kimi-k2.7-code"): ModelPrice(0.95, 4.00),
    ("deepseek", "deepseek-flash"): ModelPrice(0.30, 1.20),
    ("deepseek", "deepseek-v4-pro"): ModelPrice(1.32, 3.96),
}


def make_ledger(tmp_path: Path, **overrides: object) -> AcceptanceLedger:
    options: dict[str, object] = {"prices": PRICES, "limits": PhaseLimits()}
    options.update(overrides)
    return AcceptanceLedger(tmp_path / "phase-ledger.jsonl", **options)  # type: ignore[arg-type]


def make_host(tmp_path: Path, fetch: object, **kwargs: object) -> CodingAgentHost:
    agent_dir = kwargs.pop("agent_dir", tmp_path / "agent")
    return CodingAgentHost(
        startup_dir=tmp_path, agent_dir=agent_dir, api_key="offline-key", fetch=fetch, **kwargs,  # type: ignore[arg-type]
    )


async def run_chain(host: CodingAgentHost, *prompts: str, **selection: object) -> tuple[int, str, str]:
    resolved = host.select_new(**selection)  # type: ignore[arg-type]
    stdout, stderr = StringIO(), StringIO()
    tasks = compose_tasks(prompts=prompts)
    code = await run_print_text(host, resolved, tasks, stdout=stdout, stderr=stderr)
    return code, stdout.getvalue(), stderr.getvalue()


def records(ledger: AcceptanceLedger) -> list[dict[str, object]]:
    return [json.loads(line) for line in ledger.path.read_text(encoding="utf-8").splitlines() if line.strip()]


def kinds(ledger: AcceptanceLedger) -> list[str]:
    return [str(record["kind"]) for record in records(ledger)]


def usage_stream(text: str = "done") -> object:
    """An SSE response that also reports usage, as the real providers do."""
    return sse_response({
        "choices": [{"delta": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
    })


def line_stream_response(*lines: str) -> FetchResponse:
    """A response shaped like the real transport: an async line iterator."""

    async def generate() -> object:
        for line in lines:
            yield line

    return FetchResponse(
        status=200, headers={"content-type": "text/event-stream"}, text="", lines=generate(),  # type: ignore[arg-type]
    )


def write_settings(agent_dir: Path, document: dict[str, object]) -> None:
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "settings.json").write_text(json.dumps(document))


# --------------------------------------------------------------------------- #
# Reservation before every send
# --------------------------------------------------------------------------- #


async def test_reserves_the_send_and_injects_the_explicit_output_cap(tmp_path: Path) -> None:
    inner = RecordingFetch(text_stream("done"))
    ledger = make_ledger(tmp_path)
    code, stdout, stderr = await run_chain(
        make_host(tmp_path, ledger.guarded_fetch(inner, task="L01 goes")), "go",
    )

    assert code == EXIT_OK and stdout == "done" and stderr == ""
    assert len(inner.requests) == 1
    assert inner.bodies[0]["max_tokens"] == 4096
    sends = [record for record in records(ledger) if record["kind"] == "send"]
    assert len(sends) == 1
    send = sends[0]
    assert send["task"] == "L01 goes"
    assert send["provider"] == "opencode-go" and send["model"] == "deepseek-v4.1-flash"
    assert send["input_estimate"] > 0 and send["output_cap"] == 4096
    assert send["cost_estimate"] == round(ledger.cost_usd, 6) > 0
    assert send["cumulative"] == {
        "sends": 1, "input_tokens": send["input_estimate"], "output_cap_tokens": 4096,
        "cost_usd": send["cost_estimate"], "direct_cost_usd": 0.0,
    }
    assert ledger.snapshot()["remaining_sends"] == 59


@pytest.mark.parametrize("usage, known", [
    (None, False), ({"prompt_tokens": 12}, False),
    ({"prompt_tokens": 0, "completion_tokens": 0}, True),
])
async def test_usage_known_requires_complete_numeric_counts(tmp_path: Path, usage, known) -> None:
    inner = RecordingFetch(sse_response({
        "choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}],
        "usage": usage,
    }))
    ledger = make_ledger(tmp_path)
    code, _, _ = await run_chain(make_host(tmp_path, ledger.guarded_fetch(inner)), "go")
    assert code == EXIT_OK
    result = next(record for record in records(ledger) if record["kind"] == "result")
    assert result["usage_known"] is known
    assert ledger.sends == 1 and ledger.output_cap_tokens == 4096


async def test_account_words_in_successful_content_do_not_stop_the_phase(tmp_path: Path) -> None:
    inner = SequencedFetch(text_stream("The quota is sufficient"), text_stream("continue"))
    ledger = make_ledger(tmp_path)
    code, stdout, _ = await run_chain(make_host(tmp_path, ledger.guarded_fetch(inner)), "go", "next")
    assert code == EXIT_OK and stdout == "continue"
    assert ledger.sends == 2 and ledger.stop_reason is None


async def test_late_stream_account_error_stops_before_a_new_send(tmp_path: Path) -> None:
    inner = RecordingFetch(line_stream_response(
        "data: " + json.dumps({"choices": [{"delta": {"content": "a" * 20_000}}]}),
        'data: {"error": {"message": "subscription quota exceeded"}}',
        "data: [DONE]",
    ))
    ledger = make_ledger(tmp_path)
    await run_chain(make_host(tmp_path, ledger.guarded_fetch(inner)), "go")
    assert ledger.stop_reason == "quota"
    result = next(record for record in records(ledger) if record["kind"] == "result")
    assert result["outcome"] == "provider_error" and result["usage_known"] is False
    await run_chain(make_host(tmp_path, ledger.guarded_fetch(inner)), "next")
    assert len(inner.requests) == 1


async def test_vision_payload_contributes_to_the_input_estimate(tmp_path: Path) -> None:
    from support import png_bytes

    from coding_agent.attachments import read_file_attachment

    (tmp_path / "dot.png").write_bytes(png_bytes())
    plain = RecordingFetch(text_stream("done"))
    with_image = RecordingFetch(text_stream("done"))
    plain_ledger = make_ledger(tmp_path / "plain")
    image_ledger = make_ledger(tmp_path / "image")

    await run_chain(make_host(tmp_path, plain_ledger.guarded_fetch(plain)), "describe")
    tasks = compose_tasks(
        attachments=[read_file_attachment("dot.png", cwd=tmp_path)], prompts=("describe",),
    )
    host = make_host(tmp_path, image_ledger.guarded_fetch(with_image))
    stdout, stderr = StringIO(), StringIO()
    code = await run_print_text(host, host.select_new(), tasks, stdout=stdout, stderr=stderr)

    assert code == EXIT_OK and stderr.getvalue() == ""
    assert '"image_url"' in json.dumps(with_image.bodies[0])
    assert image_ledger.input_tokens > plain_ledger.input_tokens


# --------------------------------------------------------------------------- #
# Every actual send is counted
# --------------------------------------------------------------------------- #


async def test_tool_continuation_counts_each_send(tmp_path: Path) -> None:
    inner = SequencedFetch(
        tool_call_stream("call-1", "read", {"path": "missing.txt"}), text_stream("missing"),
    )
    ledger = make_ledger(tmp_path)
    code, stdout, _ = await run_chain(
        make_host(tmp_path, ledger.guarded_fetch(inner, task="L01 tools")), "read missing.txt",
    )

    assert code == EXIT_OK and stdout == "missing"
    assert len(inner.requests) == 2
    assert ledger.sends == 2
    assert [record["output_cap"] for record in records(ledger) if record["kind"] == "send"] == [4096, 4096]
    session_ids = [record["parameters"]["session_id"]
                   for record in records(ledger) if record["kind"] == "send"]
    assert session_ids[0] and session_ids[0] == session_ids[1]
    assert session_ids == [request.headers["x-opencode-session"] for request in inner.requests]


async def test_retry_counts_each_actual_send(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    write_settings(agent_dir, {"retry": {"baseDelayMs": 0}})
    inner = SequencedFetch(
        json_error_response(503, {"error": {"message": "service unavailable"}}), text_stream("recovered"),
    )
    ledger = make_ledger(tmp_path)
    code, stdout, _ = await run_chain(
        make_host(tmp_path, ledger.guarded_fetch(inner, task="L01 retry"), agent_dir=agent_dir), "go",
    )

    assert code == EXIT_OK and stdout == "recovered"
    assert len(inner.requests) == 2
    assert ledger.sends == 2
    outcomes = [record["outcome"] for record in records(ledger) if record["kind"] == "result"]
    assert outcomes == ["http_error", "ok"]


async def test_summary_send_is_counted_and_capped(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    write_settings(agent_dir, {"compaction": {
        "enabled": False, "reserveTokens": 100, "keepRecentTokens": 0,
    }})
    inner = SequencedFetch(text_stream("first answer"), text_stream("summary done"))
    ledger = make_ledger(tmp_path)
    host = make_host(tmp_path, ledger.guarded_fetch(inner, task="L01 summary"), agent_dir=agent_dir)
    resolved = host.select_new()
    runtime = AgentSessionRuntime(host.build_options(resolved))
    await runtime.new_session()
    await runtime.prompt("first")
    result = await runtime.compact()

    assert "summary done" in result.summary
    assert len(inner.requests) == 2
    assert "summarization assistant" in json.dumps(inner.bodies[1])
    assert ledger.sends == 2
    # The summary carries its own smaller cap; the gate must clamp, not raise it.
    assert inner.bodies[0]["max_tokens"] == 4096
    assert 0 < inner.bodies[1]["max_tokens"] < 4096


async def test_fixed_thinking_go_models_use_8192_without_an_override(tmp_path: Path) -> None:
    inner = RecordingFetch(text_stream("done"))
    ledger = make_ledger(tmp_path)
    code, _, _ = await run_chain(
        make_host(tmp_path, ledger.guarded_fetch(inner)), "go", model="kimi-k2.7-code",
    )

    assert code == EXIT_OK
    assert inner.bodies[0]["model"] == "kimi-k2.7-code"
    assert inner.bodies[0]["max_tokens"] == 8192
    assert ledger.output_cap_tokens == 8192


# --------------------------------------------------------------------------- #
# Refusal and stopping
# --------------------------------------------------------------------------- #


async def test_a_reservation_over_the_limit_refuses_before_sending(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    write_settings(agent_dir, {"retry": {"baseDelayMs": 0}})
    inner = SequencedFetch(
        tool_call_stream("call-1", "read", {"path": "missing.txt"}), text_stream("never sent"),
    )
    ledger = make_ledger(tmp_path, limits=PhaseLimits(max_sends=1))
    host = make_host(tmp_path, ledger.guarded_fetch(inner, task="over limit"), agent_dir=agent_dir)

    code, stdout, stderr = await run_chain(host, "read missing.txt")

    assert code == EXIT_FAILURE and stdout == ""
    assert len(inner.requests) == 1
    assert ledger.sends == 1
    assert "out of budget" in stderr
    assert "refused" in kinds(ledger)
    assert ledger.stop_reason == "budget"


async def test_a_stopped_phase_refuses_the_next_send_without_calling_the_transport(tmp_path: Path) -> None:
    inner = SequencedFetch(json_error_response(401, {"error": {"message": "Authentication Fails"}}))
    ledger = make_ledger(tmp_path)
    code, _, _ = await run_chain(make_host(tmp_path, ledger.guarded_fetch(inner)), "first")

    assert code == EXIT_FAILURE
    assert ledger.stop_reason == "auth"
    later = RecordingFetch(text_stream("never sent"))
    code, _, _ = await run_chain(make_host(tmp_path, ledger.guarded_fetch(later)), "again")
    assert code == EXIT_FAILURE
    assert later.requests == []
    assert ledger.sends == 1


# --------------------------------------------------------------------------- #
# Usage, failure and interruption still occupy the reservation
# --------------------------------------------------------------------------- #


async def test_reported_usage_is_recorded_without_replacing_the_reservation(tmp_path: Path) -> None:
    inner = RecordingFetch(usage_stream())
    ledger = make_ledger(tmp_path)
    code, _, _ = await run_chain(make_host(tmp_path, ledger.guarded_fetch(inner)), "go")

    assert code == EXIT_OK
    result = next(record for record in records(ledger) if record["kind"] == "result")
    assert result["usage_known"] is True and result["outcome"] == "ok"
    assert ledger.sends == 1 and ledger.cost_usd > 0


async def test_a_streamed_response_records_usage_and_completion(tmp_path: Path) -> None:
    inner = RecordingFetch(line_stream_response(
        'data: {"choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}]}',
        'data: {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}}',
        "data: [DONE]",
    ))
    ledger = make_ledger(tmp_path)
    code, stdout, _ = await run_chain(make_host(tmp_path, ledger.guarded_fetch(inner)), "go")

    assert code == EXIT_OK and stdout == "done"
    result = next(record for record in records(ledger) if record["kind"] == "result")
    assert result["usage_known"] is True and result["outcome"] == "ok"


async def test_missing_usage_keeps_the_reservation_and_is_flagged_unknown(tmp_path: Path) -> None:
    inner = RecordingFetch(text_stream("done"))
    ledger = make_ledger(tmp_path)
    code, _, _ = await run_chain(make_host(tmp_path, ledger.guarded_fetch(inner)), "go")

    assert code == EXIT_OK
    result = next(record for record in records(ledger) if record["kind"] == "result")
    assert result["usage_known"] is False
    assert ledger.sends == 1 and ledger.input_tokens > 0 and ledger.output_cap_tokens == 4096


async def test_a_failed_send_keeps_its_reservation(tmp_path: Path) -> None:
    class Broken:
        def __init__(self) -> None:
            self.requests: list[object] = []

        async def __call__(self, request: object) -> object:
            self.requests.append(request)
            raise RuntimeError("boom")

    inner = Broken()
    ledger = make_ledger(tmp_path)
    code, stdout, stderr = await run_chain(make_host(tmp_path, ledger.guarded_fetch(inner)), "go")

    assert code == EXIT_FAILURE and stdout == "" and "boom" in stderr
    assert len(inner.requests) == 1
    assert ledger.sends == 1 and ledger.cost_usd > 0
    result = next(record for record in records(ledger) if record["kind"] == "result")
    assert result["outcome"] == "interrupted" and result["usage_known"] is False


async def test_a_quota_error_stops_the_phase(tmp_path: Path) -> None:
    inner = SequencedFetch(json_error_response(402, {"error": {"message": "Insufficient Balance"}}))
    ledger = make_ledger(tmp_path)
    code, _, _ = await run_chain(make_host(tmp_path, ledger.guarded_fetch(inner)), "go")

    assert code == EXIT_FAILURE
    assert ledger.stop_reason == "balance"


# --------------------------------------------------------------------------- #
# Direct subtotal, resumption and no credentials
# --------------------------------------------------------------------------- #


async def test_deepseek_direct_estimate_uses_its_own_subtotal(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path)
    direct = RecordingFetch(text_stream("done"))
    code, _, _ = await run_chain(
        make_host(tmp_path, ledger.guarded_fetch(direct, task="direct")),
        "go", provider="deepseek", model="deepseek-flash",
    )

    assert code == EXIT_OK
    assert direct.bodies[0]["model"] == "deepseek-flash"
    assert ledger.direct_cost_usd == ledger.cost_usd > 0


async def test_ledger_resumes_from_its_file_and_keeps_the_remaining_limits(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path, limits=PhaseLimits(max_sends=1))
    await run_chain(make_host(tmp_path, ledger.guarded_fetch(RecordingFetch(text_stream("done")))), "go")

    resumed = AcceptanceLedger(ledger.path, PRICES, limits=PhaseLimits(max_sends=1))
    assert resumed.sends == 1 and resumed.stop_reason is None
    later = RecordingFetch(text_stream("never sent"))
    code, _, _ = await run_chain(
        make_host(tmp_path, resumed.guarded_fetch(later, task="resumed over limit")), "again",
    )
    assert code == EXIT_FAILURE and later.requests == []


async def test_the_ledger_records_headroom_for_the_next_ticket(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path)
    await run_chain(make_host(tmp_path, ledger.guarded_fetch(RecordingFetch(text_stream("done")))), "go")
    ledger.record_snapshot()

    snapshots = [record for record in records(ledger) if record["kind"] == "snapshot"]
    assert len(snapshots) == 1
    assert snapshots[0]["sends"] == 1 and snapshots[0]["remaining_sends"] == 59
    resumed = AcceptanceLedger(ledger.path, PRICES)
    assert resumed.snapshot()["remaining_sends"] == 59


async def test_ledger_never_records_credentials(tmp_path: Path) -> None:
    inner = RecordingFetch(text_stream("done"))
    ledger = make_ledger(tmp_path)
    host = CodingAgentHost(
        startup_dir=tmp_path, agent_dir=tmp_path / "agent",
        api_key="super-secret-key", fetch=ledger.guarded_fetch(inner),
    )
    code, _, _ = await run_chain(host, "go")

    assert code == EXIT_OK
    text = ledger.path.read_text(encoding="utf-8")
    assert "super-secret-key" not in text
    assert "bearer" not in text.lower() and "authorization" not in text.lower()
    send = next(record for record in records(ledger) if record["kind"] == "send")
    assert send["attempt"] == 1 and send["parameters"]["max_tokens"] == 4096
    assert "api_key" not in json.dumps(send).lower()


# --------------------------------------------------------------------------- #
# The live-harness injection works through the installed command
# --------------------------------------------------------------------------- #

_INSTALLED_SITE = '''\
import json
import sys

sys.path.insert(0, {tests_dir!r})
from acceptance_budget import AcceptanceLedger, ModelPrice
from omh.llm.types import FetchResponse
import coding_agent.cli as cli

async def _fetch(request):
    with open({caps!r}, "a", encoding="utf-8") as handle:
        handle.write(str(request.json_body.get("max_tokens")) + chr(10))
    chunk = {{"choices": [{{"delta": {{"content": "answer"}}, "finish_reason": "stop"}}],
              "usage": {{"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8}}}}
    body = "data: " + json.dumps(chunk) + chr(10) + chr(10) + "data: [DONE]" + chr(10)
    return FetchResponse(status=200, headers={{"content-type": "text/event-stream"}}, text=body)

_ledger = AcceptanceLedger({ledger!r}, {{("opencode-go", "deepseek-v4.1-flash"): ModelPrice(0.30, 1.20)}})
_real = cli.CodingAgentHost

def _host(*args, **kwargs):
    kwargs["fetch"] = _ledger.guarded_fetch(_fetch, task="installed")
    return _real(*args, **kwargs)

cli.CodingAgentHost = _host
'''


def test_the_live_injection_records_through_the_installed_command(tmp_path: Path) -> None:
    import subprocess

    from test_cli import clean_env, cli_command

    ledger_path = tmp_path / "phase-ledger.jsonl"
    caps = tmp_path / "caps.txt"
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(_INSTALLED_SITE.format(
        tests_dir=str(Path(__file__).resolve().parent), caps=str(caps), ledger=str(ledger_path),
    ))
    home = tmp_path / "home"
    home.mkdir()
    env = clean_env(home)
    env["PYTHONPATH"] = str(site)

    result = subprocess.run(
        [*cli_command(), "--print", "--api-key", "offline-key", "go"],
        cwd=tmp_path, capture_output=True, text=True, env=env,
    )

    assert result.returncode == EXIT_OK, result.stderr
    assert result.stdout == "answer"
    assert caps.read_text().split() == ["4096"]
    send = next(record for record in records(AcceptanceLedger(ledger_path, PRICES)) if record["kind"] == "send")
    assert send["task"] == "installed" and send["output_cap"] == 4096


# --------------------------------------------------------------------------- #
# The acceptance transport mirrors the SDK default shape
# --------------------------------------------------------------------------- #


async def test_the_acceptance_transport_streams_lines_and_reads_error_bodies(tmp_path: Path) -> None:
    import http.server
    import threading

    from acceptance_budget import create_httpx_transport
    from omh.llm.types import FetchRequest

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("content-length") or 0)
            self.rfile.read(length)
            if self.path == "/error":
                payload = b'{"error": {"message": "bad key"}}'
                self.send_response(401)
                self.send_header("content-type", "application/json")
            else:
                payload = b'data: {"choices": []}\n\ndata: [DONE]\n'
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: object) -> None:
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    transport = create_httpx_transport()
    try:
        ok = await transport(FetchRequest(method="POST", url=f"{base}/ok", headers={}, json_body={}))
        assert ok.status == 200
        assert [line async for line in ok.aiter_lines()][-1].strip() == "data: [DONE]"
        error = await transport(
            FetchRequest(method="POST", url=f"{base}/error", headers={}, json_body={}),
        )
        assert error.status == 401 and "bad key" in error.text
    finally:
        server.shutdown()
        server.server_close()
