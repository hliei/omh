from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from omh.llm.auth.types import (
    AuthOperationOptions,
    Credential,
    CredentialInfo,
)
from omh.llm.utils.abort import operation_signal, race_with_abort_signal


class InMemoryCredentialStore:
    def __init__(self) -> None:
        self._credentials: dict[str, Credential] = {}
        self._chains: dict[str, Awaitable[object]] = {}

    def _enqueue[T](
        self,
        provider_id: str,
        task: Callable[[], Awaitable[T]],
        options: AuthOperationOptions | None,
    ) -> Awaitable[T]:
        signal = operation_signal(options.signal if options else None)

        async def queued() -> T:
            previous = self._chains.get(provider_id)
            if previous is not None:
                try:
                    await previous
                except Exception:
                    pass
            signal.throw_if_aborted()
            return await task()

        operation = queued()
        self._chains[provider_id] = operation
        return race_with_abort_signal(operation, signal)

    async def read(self, provider_id: str, options: AuthOperationOptions | None = None) -> Credential | None:
        if options and options.signal:
            options.signal.throw_if_aborted()
        return self._credentials.get(provider_id)

    async def list(self, options: AuthOperationOptions | None = None) -> Sequence[CredentialInfo]:
        if options and options.signal:
            options.signal.throw_if_aborted()
        return tuple(
            CredentialInfo(provider_id=provider_id, type=credential.type)
            for provider_id, credential in self._credentials.items()
        )

    def modify(
        self,
        provider_id: str,
        fn: Callable[[Credential | None], Awaitable[Credential | None]],
        options: AuthOperationOptions | None = None,
    ) -> Awaitable[Credential | None]:
        async def task() -> Credential | None:
            current = self._credentials.get(provider_id)
            next_credential = await fn(current)
            if options and options.signal:
                options.signal.throw_if_aborted()
            if next_credential is not None:
                self._credentials[provider_id] = next_credential
            return next_credential if next_credential is not None else current

        return self._enqueue(provider_id, task, options)

    def delete(self, provider_id: str, options: AuthOperationOptions | None = None) -> Awaitable[None]:
        async def task() -> None:
            self._credentials.pop(provider_id, None)

        return self._enqueue(provider_id, task, options)
