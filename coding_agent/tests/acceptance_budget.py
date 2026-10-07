"""Acceptance-only HTTP send gate and resumable phase ledger.

This module is local verification tooling, not part of the product contract. It
exists so the limited live provider smoke tasks can reserve every actual HTTP
send before it happens and keep one phase ledger that tickets 26 and 27 share.
It composes at the product's public injectable transport boundary
(:data:`omh.llm.types.FetchFunction`), so it needs no product wiring, no new SDK
seam and no general runtime budget API.

The implemented rules come from the parent specification T03-T04:

- reserve the attempt, estimated input, output cap and conservative cost before
  each send, and refuse a send whose reservation would exceed a phase limit;
- count every actual HTTP send, because the initial request, a tool
  continuation, a retry, a summary (including a split summary) and a vision
  request all pass through one fetch call;
- keep the reservation and the sent output cap when usage is unknown, the
  response fails or the run is interrupted; a failed send is never free;
- stop the phase on an authentication, balance or quota error instead of
  retrying or falling back to another provider;
- record task, model, time, sent parameters, attempt, input estimate, cap,
  whether usage was known, the conservative estimate, the running totals and the
  stop reason, and never record credentials.

Call :meth:`AcceptanceLedger.guarded_fetch` to install the gate around the real
transport, or around a controlled offline transport in tests.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from omh.llm.types import FetchFunction, FetchRequest, FetchResponse

#: Provider identifiers used by the fixed two-provider route and its subtotal.
DEEPSEEK_PROVIDER = "deepseek"
OPENCODE_GO_PROVIDER = "opencode-go"

#: Host suffix to provider. A request whose host does not match keeps the host
#: name, so an unknown route is refused for a missing price instead of guessed.
DEFAULT_ROUTES: Mapping[str, str] = {
    "api.deepseek.com": DEEPSEEK_PROVIDER,
    "opencode.ai": OPENCODE_GO_PROVIDER,
}

#: Both target providers select their output cap with ``max_tokens``.
MAX_TOKENS_FIELD = "max_tokens"

#: Characters kept for stop-reason classification; enough for an error body.
_MAX_BODY = 16_384

_AUTH_STATUS = frozenset({401, 403})
_BALANCE_STATUS = frozenset({402})
_AUTH_TEXT = re.compile(r"invalid api key|unauthori[sz]ed|authentication failed", re.IGNORECASE)
_BALANCE_TEXT = re.compile(
    r"insufficient (?:account )?(?:balance|credits|funds)|"
    r"out of budget|balance (?:is )?(?:exhaust|deplet)|"
    r"(?:account|credit) (?:balance|credits|funds) (?:is |has been )?(?:exhaust|deplet)",
    re.IGNORECASE,
)
_QUOTA_TEXT = re.compile(
    r"quota|usage limit|monthly limit|weekly limit|subscription limit|"
    r"exhausted your|limit reached|too many requests for this (?:plan|account)",
    re.IGNORECASE,
)


class BudgetExceeded(RuntimeError):
    """A send was refused because its reservation would exceed a phase limit."""


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """Conservative peak input/output USD per million tokens."""

    input: float
    output: float


@dataclass(frozen=True, slots=True)
class PhaseLimits:
    """The shared limits for the whole real-service acceptance phase."""

    max_sends: int = 60
    max_input_tokens: int = 400_000
    max_output_cap_tokens: int = 192_000
    max_cost_usd: float = 5.0
    max_direct_cost_usd: float = 2.0
    default_output_cap: int = 4096


@dataclass(frozen=True, slots=True)
class PreparedSend:
    """One send's reserved facts, computed before the request leaves."""

    task: str
    provider: str
    model: str
    thinking: str
    output_cap: int
    input_estimate: int
    cost_estimate: float


def estimate_input_tokens(body: Mapping[str, object]) -> int:
    """Estimate input tokens from the serialized request payload.

    This is a documented conservative estimate (one token per four characters of
    the serialized body, so system, tools, history, tool output and base64 image
    data all contribute), not a provider tokenizer count.
    """
    return math.ceil(len(json.dumps(body, ensure_ascii=False, separators=(",", ":"))) / 4)


def describe_sent_thinking(body: Mapping[str, object]) -> str:
    """Report the thinking parameter actually present in the payload."""
    effort = body.get("reasoning_effort")
    if isinstance(effort, str) and effort:
        return effort
    thinking = body.get("thinking")
    if isinstance(thinking, dict) and isinstance(thinking.get("type"), str):
        return f"thinking:{thinking['type']}"
    if "thinking" in body:
        return "thinking"
    return "none"


def _stop_reason(status: int | None, text: str) -> str | None:
    """Classify a permanent account failure; everything else may continue."""
    if status in _AUTH_STATUS or _AUTH_TEXT.search(text):
        return "auth"
    if status in _BALANCE_STATUS or _BALANCE_TEXT.search(text):
        return "balance"
    if _QUOTA_TEXT.search(text):
        return "quota"
    return None


def create_httpx_transport() -> FetchFunction:
    """Return the acceptance HTTP transport, mirroring the SDK's default fetch.

    The gate must observe the real send, and the SDK's default SSE transport is
    not a public object, so the acceptance harness composes the same behaviour
    at the public :data:`omh.llm.types.FetchFunction` boundary. Provider
    credentials, payloads and stream parsing stay in the SDK adapter.
    """

    async def fetch(request: FetchRequest) -> FetchResponse:
        import httpx

        timeout = None if request.timeout_ms is None else request.timeout_ms / 1000
        client = httpx.AsyncClient(timeout=timeout)
        try:
            response = await client.send(
                client.build_request(
                    request.method, request.url, headers=request.headers, json=request.json_body,
                ),
                stream=True,
            )
        except BaseException:
            await client.aclose()
            raise
        if response.status_code >= 400:
            try:
                await response.aread()
                return FetchResponse(
                    status=response.status_code, headers=dict(response.headers), text=response.text,
                )
            finally:
                await response.aclose()
                await client.aclose()

        async def lines() -> AsyncIterator[str]:
            try:
                async for line in response.aiter_lines():
                    yield line
            finally:
                await response.aclose()
                await client.aclose()

        return FetchResponse(
            status=response.status_code, headers=dict(response.headers), text="", lines=lines(),
        )

    return fetch


class AcceptanceLedger:
    """Append-only phase ledger that reserves before every guarded send.

    ``prices`` is the conservative peak price table the caller re-verified
    before execution, keyed by ``(provider, model)``. A missing entry refuses
    the send rather than guessing a cost. ``output_caps`` overrides the default
    per-request cap for a ``provider/model`` combination.
    """

    def __init__(
        self, path: str | Path, prices: Mapping[tuple[str, str], ModelPrice], *,
        limits: PhaseLimits | None = None,
        output_caps: Mapping[str, int] | None = None,
        routes: Mapping[str, str] | None = None,
        clock: Any = None,
    ) -> None:
        self.path = Path(path)
        self._prices = dict(prices)
        self._limits = limits if limits is not None else PhaseLimits()
        self._output_caps = dict(output_caps or {})
        self._routes = dict(DEFAULT_ROUTES if routes is None else routes)
        self._clock = clock if clock is not None else self._now
        self.sends = 0
        self.input_tokens = 0
        self.output_cap_tokens = 0
        self.cost_usd = 0.0
        self.direct_cost_usd = 0.0
        self.stop_reason: str | None = None
        self._index = 0
        self._load()

    # ------------------------------------------------------------------ #
    # Observable state
    # ------------------------------------------------------------------ #

    @property
    def limits(self) -> PhaseLimits:
        return self._limits

    def snapshot(self) -> dict[str, object]:
        """Return the running totals and remaining headroom for resumption."""
        return {
            "sends": self.sends,
            "input_tokens": self.input_tokens,
            "output_cap_tokens": self.output_cap_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "direct_cost_usd": round(self.direct_cost_usd, 6),
            "remaining_sends": self._limits.max_sends - self.sends,
            "remaining_input_tokens": self._limits.max_input_tokens - self.input_tokens,
            "remaining_output_cap_tokens": self._limits.max_output_cap_tokens - self.output_cap_tokens,
            "remaining_cost_usd": round(self._limits.max_cost_usd - self.cost_usd, 6),
            "remaining_direct_cost_usd": round(self._limits.max_direct_cost_usd - self.direct_cost_usd, 6),
            "stop_reason": self.stop_reason,
        }

    # ------------------------------------------------------------------ #
    # Gate
    # ------------------------------------------------------------------ #

    def guarded_fetch(self, inner: FetchFunction, *, task: str = "unspecified") -> FetchFunction:
        """Wrap a public transport so every send is reserved and recorded."""

        async def send(request: FetchRequest) -> FetchResponse:
            prepared = self._reserve(task, request)
            index = self._index
            try:
                response = await inner(prepared)
            except BaseException as error:
                self._record_result(index, status=None, outcome="interrupted",
                                    usage_known=False, error=type(error).__name__)
                raise
            self._watch(index, response)
            return response

        return send

    # ------------------------------------------------------------------ #
    # Reservation
    # ------------------------------------------------------------------ #

    def _reserve(self, task: str, request: FetchRequest) -> FetchRequest:
        if self.stop_reason is not None:
            self._refuse(task, f"phase already stopped: {self.stop_reason}")
        prepared, send = self._prepare(task, request)
        projected = {
            "sends": self.sends + 1,
            "input_tokens": self.input_tokens + send.input_estimate,
            "output_cap_tokens": self.output_cap_tokens + send.output_cap,
            "cost_usd": self.cost_usd + send.cost_estimate,
            "direct_cost_usd": self.direct_cost_usd + (
                send.cost_estimate if send.provider == DEEPSEEK_PROVIDER else 0.0
            ),
        }
        exceeded = self._exceeded(projected)
        if exceeded is not None:
            self._refuse(task, exceeded, projected)
        self.sends = projected["sends"]
        self.input_tokens = projected["input_tokens"]
        self.output_cap_tokens = projected["output_cap_tokens"]
        self.cost_usd = projected["cost_usd"]
        self.direct_cost_usd = projected["direct_cost_usd"]
        self._index += 1
        self._append({
            "kind": "send",
            "index": self._index,
            "task": task,
            "time": self._clock(),
            "provider": send.provider,
            "model": send.model,
            "thinking": send.thinking,
            "input_estimate": send.input_estimate,
            "output_cap": send.output_cap,
            "cost_estimate": round(send.cost_estimate, 6),
            "usage_known": False,
            "cumulative": self._rounded(projected),
        })
        return prepared

    def _prepare(self, task: str, request: FetchRequest) -> tuple[FetchRequest, PreparedSend]:
        body = request.json_body
        model = body.get("model")
        if not isinstance(model, str) or not model:
            self._refuse(task, "request has no model id")
        provider = self._provider(request.url)
        price = self._prices.get((provider, model))
        if price is None:
            self._refuse(task, f"no conservative price for {provider}/{model}")
        cap = self._output_cap(provider, model, body)
        estimate = estimate_input_tokens(body)
        cost = price.input * estimate / 1_000_000 + price.output * cap / 1_000_000
        prepared = replace(request, json_body={**body, MAX_TOKENS_FIELD: cap})
        return prepared, PreparedSend(
            task=task, provider=provider, model=model,
            thinking=describe_sent_thinking(body), output_cap=cap,
            input_estimate=estimate, cost_estimate=cost,
        )

    def _output_cap(self, provider: str, model: str, body: Mapping[str, object]) -> int:
        limit = self._output_caps.get(f"{provider}/{model}", self._limits.default_output_cap)
        requested = body.get(MAX_TOKENS_FIELD)
        if isinstance(requested, int) and not isinstance(requested, bool) and requested > 0:
            return min(requested, limit)
        return limit

    def _provider(self, url: str) -> str:
        host = urlparse(url).hostname or ""
        if not host:
            return self._routes.get("", "")
        for suffix, provider in self._routes.items():
            if suffix and (host == suffix or host.endswith(f".{suffix}")):
                return provider
        return host

    def _exceeded(self, projected: Mapping[str, float | int]) -> str | None:
        limits = self._limits
        checks = (
            (projected["sends"] > limits.max_sends,
             f"send attempt {projected['sends']} exceeds {limits.max_sends}"),
            (projected["input_tokens"] > limits.max_input_tokens,
             f"estimated input {projected['input_tokens']} exceeds {limits.max_input_tokens} tokens"),
            (projected["output_cap_tokens"] > limits.max_output_cap_tokens,
             f"sent output cap {projected['output_cap_tokens']} exceeds {limits.max_output_cap_tokens} tokens"),
            (projected["cost_usd"] > limits.max_cost_usd,
             f"conservative estimate {projected['cost_usd']:.4f} exceeds USD {limits.max_cost_usd}"),
            (projected["direct_cost_usd"] > limits.max_direct_cost_usd,
             f"DeepSeek direct estimate {projected['direct_cost_usd']:.4f} "
             f"exceeds USD {limits.max_direct_cost_usd}"),
        )
        reasons = [reason for hit, reason in checks if hit]
        return "; ".join(reasons) if reasons else None

    def _refuse(self, task: str, reason: str, projected: Mapping[str, Any] | None = None) -> None:
        record: dict[str, Any] = {"kind": "refused", "task": task, "time": self._clock(), "reason": reason}
        if projected is not None:
            record["projected"] = self._rounded(projected)
        self._append(record)
        self._stop("budget")
        raise BudgetExceeded(f"acceptance send refused for {task}: {reason} (out of budget)")

    # ------------------------------------------------------------------ #
    # Response observation
    # ------------------------------------------------------------------ #

    def _watch(self, index: int, response: FetchResponse) -> None:
        if response.status >= 400 or response.lines is None:
            self._finish(index, response.status, complete=True,
                         text=response.text, usage='"usage"' in response.text)
            return
        lines = response.lines
        state = {"text": "", "usage": False}
        finished = [False]

        def finish(complete: bool) -> None:
            if finished[0]:
                return
            finished[0] = True
            self._finish(index, response.status, complete=complete,
                         text=state["text"], usage=state["usage"])

        async def watched() -> AsyncIterator[str]:
            try:
                async for line in lines:
                    if '"usage"' in line:
                        state["usage"] = True
                    remaining = _MAX_BODY - len(state["text"])
                    if remaining > 0:
                        state["text"] += line[:remaining]
                    # The adapter stops reading at the terminal marker, so the
                    # completion has to be recorded before it is yielded; a
                    # response that ends without the marker is also complete.
                    if line.strip().endswith("[DONE]"):
                        finish(True)
                    yield line
                finish(True)
            finally:
                finish(False)

        response.lines = watched()

    def _finish(self, index: int, status: int, *, complete: bool, text: str, usage: bool) -> None:
        if status >= 400:
            outcome = "http_error"
        elif complete:
            outcome = "ok"
        else:
            outcome = "interrupted"
        self._record_result(index, status=status, outcome=outcome, usage_known=usage,
                            stop=_stop_reason(status, text))

    def _record_result(
        self, index: int, *, status: int | None, outcome: str, usage_known: bool,
        stop: str | None = None, error: str | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "kind": "result", "index": index, "time": self._clock(),
            "status": status, "outcome": outcome, "usage_known": usage_known,
        }
        if error is not None:
            record["error"] = error
        self._append(record)
        if stop is not None:
            self._stop(stop)

    def _stop(self, reason: str) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason
            self._append({"kind": "stop", "time": self._clock(), "reason": reason})

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            kind = record.get("kind")
            if kind == "send":
                cumulative = record["cumulative"]
                self.sends = cumulative["sends"]
                self.input_tokens = cumulative["input_tokens"]
                self.output_cap_tokens = cumulative["output_cap_tokens"]
                self.cost_usd = cumulative["cost_usd"]
                self.direct_cost_usd = cumulative["direct_cost_usd"]
                self._index = record["index"]
            elif kind == "stop":
                self.stop_reason = record["reason"]

    def _append(self, record: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    @staticmethod
    def _rounded(values: Mapping[str, float | int]) -> dict[str, float | int]:
        return {key: (round(value, 6) if isinstance(value, float) else value)
                for key, value in values.items()}

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
