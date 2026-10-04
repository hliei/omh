"""Local bash execution through the in-process AgentTool contract."""

import asyncio
import math
import os
import signal as process_signal
from dataclasses import dataclass
from pathlib import Path

from omh.agent.execution.tools import (
    AgentTool,
    AgentToolResult,
    AgentToolUpdateCallback,
    _AgentToolResultError,
)
from omh.agent.tools.arguments import required_string
from omh.agent.tools.bash_output import BashOutput
from omh.llm.types import AbortSignal, TextContent


@dataclass(frozen=True, slots=True)
class BashToolOptions:
    """Optional command setup and shell executable; environment is inherited."""

    command_prefix: str | None = None
    shell_path: str = "/bin/bash"


def create_bash_tool(cwd: str | Path, options: BashToolOptions | None = None) -> AgentTool:
    """Create a local bash tool with an explicit working directory."""
    directory = Path(cwd).expanduser().absolute()
    settings = options or BashToolOptions()

    async def execute(
        tool_call_id: str,
        arguments: dict[str, object],
        signal: AbortSignal | None,
        on_update: AgentToolUpdateCallback,
    ) -> AgentToolResult:
        del tool_call_id
        if signal is not None:
            signal.throw_if_aborted()
        command = required_string(arguments, "command")
        if settings.command_prefix:
            command = f"{settings.command_prefix}\n{command}"
        timeout = _timeout_seconds(arguments)
        stop = asyncio.Event()
        if signal is not None:
            signal.add_callback(stop.set)
        task = asyncio.create_task(_execute_command(directory, settings, command, timeout, stop, on_update))
        cancellation: asyncio.CancelledError | None = None
        try:
            while True:
                try:
                    result = await asyncio.shield(task)
                    break
                except asyncio.CancelledError as error:
                    if task.cancelled():
                        raise
                    cancellation = error
                    stop.set()
                except Exception:
                    if cancellation is not None:
                        raise cancellation
                    raise
            if cancellation is not None:
                raise cancellation
            return result
        finally:
            if signal is not None:
                signal.remove_callback(stop.set)

    return AgentTool(
        name="bash", label="bash",
        description=(
            "Execute a bash command in the working directory with combined stdout and stderr. "
            "Returns the last 2000 lines or 50 KiB; truncated output is saved in full to a temp file. "
            "An optional timeout is in seconds; there is no default timeout."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"},
            },
            "required": ["command"],
        },
        execute=execute,
    )


def _timeout_seconds(arguments: dict[str, object]) -> float | None:
    if "timeout" not in arguments:
        return None
    value = arguments["timeout"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Invalid timeout: must be a finite positive number of seconds")
    maximum = 2_147_483_647 / 1000
    if value > maximum:
        raise ValueError(f"Invalid timeout: maximum is {maximum} seconds")
    if value <= 0 or not math.isfinite(value):
        raise ValueError("Invalid timeout: must be a finite positive number of seconds")
    return float(value)


def _kill_group(process: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(process.pid, process_signal.SIGKILL)
    except ProcessLookupError:
        pass


async def _execute_command(
    directory: Path,
    settings: BashToolOptions,
    command: str,
    timeout: float | None,
    stop: asyncio.Event,
    on_update: AgentToolUpdateCallback,
) -> AgentToolResult:
    if stop.is_set():
        raise ValueError("Command aborted")
    read_descriptor, write_descriptor = os.pipe()
    pipe = os.fdopen(read_descriptor, "rb", buffering=0)
    stream = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(stream)
    transport: asyncio.ReadTransport | None = None
    try:
        transport, _ = await asyncio.get_running_loop().connect_read_pipe(lambda: protocol, pipe)
        process = await asyncio.create_subprocess_exec(
            settings.shell_path, "-c", command, cwd=directory,
            stdin=asyncio.subprocess.DEVNULL, stdout=write_descriptor,
            stderr=asyncio.subprocess.STDOUT, start_new_session=True,
        )
    except BaseException:
        if transport is not None:
            transport.close()
        else:
            pipe.close()
        raise
    finally:
        os.close(write_descriptor)
    output = BashOutput()

    async def read_output() -> None:
        while chunk := await stream.read(8192):
            await output.append(chunk)
            on_update(output.snapshot())

    async def wait_for_exit() -> None:
        await process.wait()

    async def wait_for_stop() -> None:
        await stop.wait()

    reader = asyncio.create_task(read_output())
    exited = asyncio.create_task(wait_for_exit())
    stopped = asyncio.create_task(wait_for_stop())
    tasks = {reader, exited, stopped}
    deadline = None if timeout is None else asyncio.get_running_loop().time() + timeout
    status: str | None = None
    try:
        while True:
            remaining = None if deadline is None else max(0, deadline - asyncio.get_running_loop().time())
            done, _ = await asyncio.wait(tasks, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
            if stopped in done:
                status = "Command aborted"
                break
            if not done:
                status = f"Command timed out after {timeout} seconds"
                break
            if reader in done:
                await reader
                tasks.remove(reader)
            if exited in done:
                break
    finally:
        _kill_group(process)
        await process.wait()
        # A descendant may have left the group while retaining stdout. Own the
        # pipe transport so it can close independently of that descendant.
        try:
            await asyncio.wait({reader}, timeout=1)
        finally:
            transport.close()
        stopped.cancel()
        await asyncio.gather(reader, exited, stopped, return_exceptions=True)
        try:
            await output.finish()
        finally:
            await output.close()
    # Surface capture failures discovered while draining after shell exit.
    await reader
    on_update(output.snapshot())
    result = output.snapshot(final=True)
    if status is None and process.returncode != 0:
        status = (
            f"Command terminated by signal {-process.returncode}"
            if process.returncode is not None and process.returncode < 0
            else f"Command exited with code {process.returncode}"
        )
    if status is not None:
        content = result.content[0]
        assert isinstance(content, TextContent)
        message = f"{content.text}\n\n{status}"
        result.content = [TextContent(text=message)]
        raise _AgentToolResultError(message, result)
    return result
