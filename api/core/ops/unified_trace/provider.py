"""Unified trace runtime and provider adapter contract."""

from collections.abc import Callable
from typing import Protocol, override

from core.ops.base_trace_instance import BaseTraceInstance
from core.ops.entities.config_entity import BaseTracingConfig
from core.ops.entities.trace_entity import BaseTraceInfo
from core.ops.unified_trace.entities import CanonicalTrace
from core.ops.unified_trace.parent_context import (
    ParentContextCoordinator,
    ParentResolution,
    ProviderParentContext,
)
from core.ops.unified_trace.trace_builder import CanonicalTraceBuilder

ParentContextPublisher = Callable[[str, ProviderParentContext], None]


class UnifiedTraceAdapter(Protocol):
    @property
    def provider_name(self) -> str:
        raise NotImplementedError

    @property
    def scope(self) -> str:
        raise NotImplementedError

    def emit(
        self,
        trace: CanonicalTrace,
        parent: ParentResolution | None,
        publish_parent_context: ParentContextPublisher,
    ) -> None:
        """Emit one Core-resolved, parent-first canonical fragment.

        Return only after the provider-specific synchronous acceptance step succeeds.
        Raise RetryableTraceDispatchError when acceptance is unconfirmed because of a
        recoverable provider or transport failure. Raise another exception for a
        terminal failure. Publish parent context only after the corresponding provider
        parent span has been accepted.
        """
        raise NotImplementedError


class UnifiedTraceInstance(BaseTraceInstance):
    """Build, coordinate, and emit a trace without a legacy fallback path."""

    def __init__(
        self,
        trace_config: BaseTracingConfig,
        *,
        builder: CanonicalTraceBuilder,
        adapter: UnifiedTraceAdapter,
        coordinator: ParentContextCoordinator,
    ) -> None:
        super().__init__(trace_config)
        self._builder = builder
        self._adapter = adapter
        self._coordinator = coordinator

    def parent_publisher(self, trace_info: BaseTraceInfo) -> ParentContextPublisher:
        return self._coordinator.publish

    def resolve_required_parent(self, trace_info: BaseTraceInfo, parent_id: str) -> ParentResolution:
        return self._coordinator.resolve_required(
            parent_id, expected_provider=self._adapter.provider_name, expected_scope=self._adapter.scope
        )

    @override
    def trace(self, trace_info: BaseTraceInfo) -> None:
        canonical_trace = self._builder.build(trace_info)
        if canonical_trace is None:
            return

        parent_resolution = None
        if canonical_trace.external_parent is not None:
            parent_resolution = self._coordinator.resolve(
                canonical_trace.external_parent,
                expected_provider=self._adapter.provider_name,
                expected_scope=self._adapter.scope,
            )
        elif canonical_trace.required_parent_context_id is not None:
            parent_resolution = self.resolve_required_parent(trace_info, canonical_trace.required_parent_context_id)

        self._adapter.emit(canonical_trace, parent_resolution, self.parent_publisher(trace_info))
