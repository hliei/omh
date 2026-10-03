"""Local bash behavior through the public AgentTool and Agent boundaries."""

import asyncio
import os
import signal
import subprocess
from pathlib import Path

import pytest

from omh.agent import AgentTool, BashToolOptions, create_bash_tool
from omh.llm.types import AbortController, TextContent


async def test_bash_runs_in_cwd_with_prefix_and_combined_progress(tmp_path: Path) -> None:
    tool = create_bash_tool(tmp_path, BashToolOptions(command_prefix="export GREETING=hello"))
    assert isinstance(tool, AgentTool)
    updates = []
    result = await tool.execute(
        "bash", {"command": "printf '%s\\n' \"$PWD\" \"$GREETING\"; printf 'stderr\\n' >&2"},
        None, updates.append,
    )
    assert result.content == [TextContent(text=f"{tmp_path}\nhello\nstderr\n")]
    assert result.details is None
    assert updates[-1].content == result.content


async def test_bash_bounds_tail_and_spills_complete_output(tmp_path: Path) -> None:
    data = "".join(f"line {i}\n" for i in range(2500))
    (tmp_path / "input").write_text(data)
    updates = []
    result = await create_bash_tool(tmp_path).execute("bash", {"command": "cat input"}, None, updates.append)
    assert isinstance(result.details, dict)
    path = Path(result.details["fullOutputPath"])
    try:
        assert path.read_text() == data
        assert result.details["truncation"]["totalLines"] == 2500
        assert result.details["truncation"]["outputLines"] == 2000
        assert result.details["truncation"]["truncatedBy"] == "lines"
        assert result.content[0].text.startswith("line 500\n")
        assert "Full output:" in result.content[0].text
        assert updates[-1].details == result.details
    finally:
        path.unlink()


async def test_bash_byte_tail_keeps_utf8_boundaries(tmp_path: Path) -> None:
    data = "你" * 20000
    (tmp_path / "input").write_text(data)
    result = await create_bash_tool(tmp_path).execute("bash", {"command": "cat input"}, None, lambda _: None)
    assert isinstance(result.details, dict)
    path = Path(result.details["fullOutputPath"])
    try:
        assert path.read_text() == data
        assert result.content[0].text.split("\n\n")[0] == "你" * 17066
        assert result.details["truncation"]["outputBytes"] == 51198
        assert result.details["truncation"]["lastLinePartial"] is True
    finally:
        path.unlink()


@pytest.mark.parametrize("method", ["abort", "timeout", "cancel"])
async def test_bash_stops_process_group_before_settling(tmp_path: Path, method: str) -> None:
    ready = asyncio.Event()
    pids = []
    updates = []
    controller = AbortController()

    def update(result):
        updates.append(result)
        if result.content and not pids:
            pids.extend(int(value) for value in result.content[0].text.split())
            ready.set()

    args = {"command": "sleep 30 & child=$!; printf '%s %s\\n' \"$$\" \"$child\"; wait"}
    if method == "timeout":
        args["timeout"] = 0.3
    task = asyncio.ensure_future(create_bash_tool(tmp_path).execute("bash", args, controller.signal, update))
    try:
        await asyncio.wait_for(ready.wait(), 2)
        if method == "abort":
            controller.abort()
        elif method == "cancel":
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
        error = asyncio.CancelledError if method == "cancel" else ValueError
        with pytest.raises(error, match=None if method == "cancel" else "aborted|timed out"):
            await asyncio.wait_for(task, 2)
        for pid in pids:
            status = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
            assert not status or status.startswith("Z"), f"process {pid} remains active: {status}"
        count = len(updates)
        await asyncio.sleep(0.05)
        assert len(updates) == count
    finally:
        if pids:
            try:
                os.killpg(pids[0], signal.SIGKILL)
            except ProcessLookupError:
                pass
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("ending, status", [
    ("exit 7", "code 7"),
    ("kill -TERM $$", "signal 15"),
    ("sleep 30", "timed out"),
])
async def test_agent_preserves_spill_details_for_failed_commands(
    tmp_path: Path, ending: str, status: str,
) -> None:
    from omh.agent import Agent, AgentInitialState, AgentOptions
    from omh.llm.types import ToolResultMessage
    from tests.agent.test_agent_tools import (
        ScriptedStreamFn,
        make_model,
        text_message,
        tool_call_message,
    )

    data = "x" * 60000
    (tmp_path / "input").write_text(data)
    stream = ScriptedStreamFn([
        lambda: tool_call_message("bash", {"command": f"cat input; {ending}", "timeout": 0.3}),
        lambda: text_message("done"),
    ])
    agent = Agent(AgentOptions(
        initial_state=AgentInitialState(model=make_model(), tools=[create_bash_tool(tmp_path)]),
        stream_fn=stream,
    ))
    await agent.prompt("run")
    result = next(message for message in agent.state.messages if isinstance(message, ToolResultMessage))
    assert result.is_error
    assert status in result.content[0].text
    assert isinstance(result.details, dict) and "fullOutputPath" in result.details
    path = Path(result.details["fullOutputPath"])
    try:
        assert path.read_text() == data
        assert result.details["truncation"]["totalBytes"] == 60000
    finally:
        path.unlink()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), float("-inf"), True, "1", None, 2147483.648, 10 ** 1000])
async def test_invalid_timeout_does_not_execute(tmp_path: Path, timeout: object) -> None:
    with pytest.raises(ValueError, match="Invalid timeout"):
        await create_bash_tool(tmp_path).execute(
            "bash", {"command": "touch executed", "timeout": timeout}, None, lambda _: None,
        )
    assert not (tmp_path / "executed").exists()


async def test_bash_flushes_incomplete_utf8_and_spills_at_final_boundary(tmp_path: Path) -> None:
    data = b"x" * 51200 + b"\xe4"
    (tmp_path / "input").write_bytes(data)
    result = await create_bash_tool(tmp_path).execute("bash", {"command": "cat input"}, None, lambda _: None)
    assert isinstance(result.details, dict)
    path = Path(result.details["fullOutputPath"])
    try:
        assert path.read_bytes() == data
        assert result.content[0].text.split("\n\n")[0] == "x" * 51197 + "�"
    finally:
        path.unlink()


async def test_shell_exit_cleans_background_group_with_inherited_pipe(tmp_path: Path) -> None:
    result = await asyncio.wait_for(
        create_bash_tool(tmp_path).execute("bash", {"command": "sleep 30 & printf '%s\\n' $!"}, None, lambda _: None),
        2,
    )
    pid = int(result.content[0].text)
    status = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    assert not status or status.startswith("Z")


async def test_pre_aborted_tool_does_not_spawn(tmp_path: Path) -> None:
    controller = AbortController()
    controller.abort()
    with pytest.raises(Exception, match="aborted"):
        await create_bash_tool(tmp_path).execute("bash", {"command": "touch executed"}, controller.signal, lambda _: None)
    assert not (tmp_path / "executed").exists()


async def test_shell_exit_closes_pipe_held_by_detached_descendant(tmp_path: Path) -> None:
    import shlex
    import sys

    pids = []

    def update(result):
        if result.content and not pids:
            pids.append(int(result.content[0].text))

    script = "import subprocess; p = subprocess.Popen(['sleep', '30'], start_new_session=True); print(p.pid, flush=True)"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"
    task = asyncio.ensure_future(create_bash_tool(tmp_path).execute("bash", {"command": command}, None, update))
    try:
        done, _ = await asyncio.wait({task}, timeout=2)
        assert task in done, "inherited pipe must not keep a finished shell invocation alive"
        assert int((await task).content[0].text) == pids[0]
    finally:
        for pid in pids:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await asyncio.gather(task, return_exceptions=True)


async def test_long_final_line_reports_size_even_with_trailing_newline(tmp_path: Path) -> None:
    (tmp_path / "input").write_text("x" * 60000 + "\n")
    result = await create_bash_tool(tmp_path).execute("bash", {"command": "cat input"}, None, lambda _: None)
    path = Path(result.details["fullOutputPath"])
    try:
        assert "line is 58.6KB" in result.content[0].text
    finally:
        path.unlink()


@pytest.mark.parametrize("mode", ["parallel", "sequential"])
async def test_agent_bash_batch_keeps_source_result_order_and_error_results(tmp_path: Path, mode: str) -> None:
    from omh.agent import Agent, AgentInitialState, AgentOptions, ToolExecutionEndEvent
    from omh.llm.types import ToolResultMessage
    from tests.agent.test_agent_tool_batches import (
        ScriptedStreamFn,
        make_model,
        text_message,
        tool_call_message,
    )

    stream = ScriptedStreamFn([
        lambda: tool_call_message([
            ("slow", "bash", {"command": "while [ ! -f release ]; do sleep 0.01; done; printf slow" if mode == "parallel" else "printf slow"}),
            ("failure", "bash", {"command": "printf failure >&2; exit 3"}),
            ("invalid", "bash", {"command": "touch forbidden", "timeout": 0}),
        ]),
        lambda: text_message("done"),
    ])
    agent = Agent(AgentOptions(
        initial_state=AgentInitialState(model=make_model(), tools=[create_bash_tool(tmp_path)]),
        stream_fn=stream, tool_execution=mode,
    ))
    finished = []
    def on_end(event, signal):
        if isinstance(event, ToolExecutionEndEvent):
            finished.append(event.tool_call_id)
            if event.tool_call_id == "failure":
                (tmp_path / "release").touch()

    agent.subscribe(on_end)
    await agent.prompt("run batch")
    results = [message for message in agent.state.messages if isinstance(message, ToolResultMessage)]
    assert [result.tool_call_id for result in results] == ["slow", "failure", "invalid"]
    assert [result.is_error for result in results] == [False, True, True]
    assert "failure" in results[1].content[0].text
    assert not (tmp_path / "forbidden").exists()
    assert finished == (["invalid", "failure", "slow"] if mode == "parallel" else ["slow", "failure", "invalid"])


async def test_cancelled_agent_waiter_keeps_bash_running_until_abort(tmp_path: Path) -> None:
    from omh.agent import (
        Agent,
        AgentInitialState,
        AgentOptions,
        ToolExecutionUpdateEvent,
    )
    from omh.llm.types import ToolResultMessage
    from tests.agent.test_agent_tools import (
        ScriptedStreamFn,
        make_model,
        tool_call_message,
    )

    ready = asyncio.Event()
    stream = ScriptedStreamFn([lambda: tool_call_message("bash", {"command": "printf ready; sleep 30"})])
    agent = Agent(AgentOptions(
        initial_state=AgentInitialState(model=make_model(), tools=[create_bash_tool(tmp_path)]),
        stream_fn=stream,
    ))
    events = []

    def observe(event, signal):
        events.append(event)
        if isinstance(event, ToolExecutionUpdateEvent) and event.partial_result.content:
            ready.set()

    agent.subscribe(observe)
    waiter = asyncio.create_task(agent.prompt("run"))
    try:
        await asyncio.wait_for(ready.wait(), 2)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert agent.state.is_busy
        assert not any(isinstance(message, ToolResultMessage) for message in agent.state.messages)
        agent.abort()
        await asyncio.wait_for(agent.wait_for_idle(), 2)
        result = next(message for message in agent.state.messages if isinstance(message, ToolResultMessage))
        assert result.is_error and "aborted" in result.content[0].text
        count = len(events)
        await asyncio.sleep(0.05)
        assert len(events) == count
    finally:
        await agent.close()


@pytest.mark.parametrize("failure", ["cwd", "shell", "command"])
async def test_execution_failures_become_agent_error_results(tmp_path: Path, failure: str) -> None:
    from omh.agent import Agent, AgentInitialState, AgentOptions
    from omh.llm.types import ToolResultMessage
    from tests.agent.test_agent_tools import (
        ScriptedStreamFn,
        make_model,
        text_message,
        tool_call_message,
    )

    cwd = tmp_path / "missing" if failure == "cwd" else tmp_path
    options = BashToolOptions(shell_path=str(tmp_path / "missing")) if failure == "shell" else None
    command = "unknown_command_omh_test" if failure == "command" else "printf ok"
    stream = ScriptedStreamFn([
        lambda: tool_call_message("bash", {"command": command}), lambda: text_message("done"),
    ])
    agent = Agent(AgentOptions(
        initial_state=AgentInitialState(model=make_model(), tools=[create_bash_tool(cwd, options)]),
        stream_fn=stream,
    ))
    await agent.prompt("run")
    result = next(message for message in agent.state.messages if isinstance(message, ToolResultMessage))
    assert result.is_error
    assert "127" in result.content[0].text if failure == "command" else "No such file or directory" in result.content[0].text


async def test_empty_output_and_explicit_shell(tmp_path: Path) -> None:
    result = await create_bash_tool(tmp_path, BashToolOptions(shell_path="/bin/bash")).execute(
        "bash", {"command": "true"}, None, lambda _: None,
    )
    assert result.content == [TextContent(text="(no output)")]
    assert result.details is None


async def test_utf8_split_between_emissions_is_decoded_incrementally(tmp_path: Path) -> None:
    result = await create_bash_tool(tmp_path).execute(
        "bash", {"command": "printf '\\344'; sleep 0.02; printf '\\275\\240'"}, None, lambda _: None,
    )
    assert result.content == [TextContent(text="你")]


async def test_large_output_keeps_tail_and_original_bytes_in_spill(tmp_path: Path) -> None:
    data = b"old\n" * 100000 + b"final\n" * 2000
    (tmp_path / "input").write_bytes(data)
    updates = []
    result = await create_bash_tool(tmp_path).execute("bash", {"command": "cat input"}, None, updates.append)
    path = Path(result.details["fullOutputPath"])
    try:
        assert path.read_bytes() == data
        assert result.content[0].text.split("\n\n")[0] == ("final\n" * 2000).rstrip("\n")
        assert result.details["truncation"]["totalLines"] == 102000
        assert result.details["truncation"]["totalBytes"] == 412000
        assert all(len(update.content[0].text.encode("utf-8")) <= 51200 for update in updates)
    finally:
        path.unlink()


async def test_cancellation_during_startup_settles_invocation_before_return(tmp_path: Path) -> None:
    task = asyncio.ensure_future(create_bash_tool(tmp_path).execute(
        "bash", {"command": "sleep 0.1; touch escaped"}, None, lambda _: None,
    ))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.15)
    assert not (tmp_path / "escaped").exists()


async def test_progress_callback_failure_still_ends_process_group(tmp_path: Path) -> None:
    pids = []

    def fail_update(result):
        if result.content:
            pids.extend(int(value) for value in result.content[0].text.split())
            raise ValueError("host progress failure")

    with pytest.raises(ValueError, match="host progress failure"):
        await asyncio.wait_for(create_bash_tool(tmp_path).execute(
            "bash", {"command": "sleep 30 & printf '%s %s\\n' \"$$\" $!; wait"}, None, fail_update,
        ), 2)
    for pid in pids:
        status = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        assert not status or status.startswith("Z")
