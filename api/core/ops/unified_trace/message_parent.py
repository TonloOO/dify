"""Store the collector-accepted answer context alongside its owning message."""

from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError

from core.db.session_factory import create_session
from core.ops.exceptions import (
    InvalidTraceParentContextError,
    PendingTraceParentContextError,
    TraceParentContextAccessError,
)
from core.ops.unified_trace.parent_context import ProviderParentContext
from models.model import App, Message


def _owned_message(tenant_id: str, app_id: str, message_id: str):
    return (
        Message.id == message_id,
        Message.app_id == app_id,
        select(App.id).where(App.id == app_id, App.tenant_id == tenant_id).exists(),
    )


def save_message_parent(tenant_id: str, app_id: str, context: ProviderParentContext) -> None:
    # First accepted answer wins even if a partially exported trace is retried.
    # This bookkeeping must not change the user-visible answer timestamp.
    try:
        with create_session() as session, session.begin():
            session.execute(
                update(Message)
                .where(*_owned_message(tenant_id, app_id, context.parent_id), Message.otel_parent_context.is_(None))
                .values(otel_parent_context=context.model_dump_json(), updated_at=Message.updated_at)
            )
    except SQLAlchemyError as exc:
        raise TraceParentContextAccessError(f"Could not save OTel parent for message {context.parent_id}") from exc


def load_message_parent(
    tenant_id: str, app_id: str, message_id: str, provider: str, scope: str
) -> ProviderParentContext:
    try:
        with create_session() as session:
            value = session.scalar(
                select(Message.otel_parent_context).where(*_owned_message(tenant_id, app_id, message_id))
            )
    except SQLAlchemyError as exc:
        raise TraceParentContextAccessError(f"Could not read OTel parent for message {message_id}") from exc
    if value is None:
        raise PendingTraceParentContextError(message_id)
    context = ProviderParentContext.model_validate_json(value)
    if context.parent_id != message_id or context.provider != provider or context.scope != scope:
        raise InvalidTraceParentContextError(
            f"Stored OTel parent does not match the destination for message {message_id}"
        )
    return context
