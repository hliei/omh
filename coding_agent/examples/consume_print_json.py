"""Consume the installed print command, checking final message and process exit."""

from __future__ import annotations

import json
import subprocess
import sys


def main() -> int:
    process = subprocess.Popen(
        ["omh", "--mode=json", *sys.argv[1:]], stdout=subprocess.PIPE,
        text=True,  # stderr stays visible; it is independent of the JSON stream.
    )
    assert process.stdout is not None
    final = None
    with process.stdout:
        for line in process.stdout:
            event = json.loads(line)
            if event["type"] == "message_end" and event["message"]["role"] == "assistant":
                final = event["message"]
    code = process.wait()
    if code != 0:
        # A permanent stdout write failure exits 1 without a final message;
        # there is no synthetic result or rescue to consume.
        return code
    if final is None:
        print("No authoritative final assistant message; output is incomplete.", file=sys.stderr)
        return 1
    if final["stopReason"] in {"error", "aborted"}:
        print(final.get("errorMessage") or final["stopReason"], file=sys.stderr)
        return 1
    for block in final["content"]:
        if block["type"] == "text":
            print(block["text"], end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
