from __future__ import annotations

from omh.agent.agent_harness import AgentHarnessResources


class ResourceRegistry:
    """Process-local holder for harness-global skills and prompt templates.

    Resources are not durable lane configuration; an application registers them
    through :class:`AgentHarness` options or ``set_resources`` and the explicit
    invocation methods resolve names against the current value.
    """

    def __init__(self, resources: AgentHarnessResources) -> None:
        self._resources = resources

    def get(self) -> AgentHarnessResources:
        return self._resources

    def replace(self, resources: AgentHarnessResources) -> None:
        self._resources = resources
