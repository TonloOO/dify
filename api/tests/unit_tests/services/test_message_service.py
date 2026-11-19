import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from queue import Queue
from threading import Thread
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import Span as WireSpan
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, scoped_session

import models.model as model_module
import services.message_service as service_module
from core.app.entities.app_invoke_entities import InvokeFrom
from core.ops import ops_trace_manager
from core.ops.entities.trace_entity import FeedbackTraceInfo, MessageTraceInfo, WorkflowTraceInfo
from core.ops.exceptions import InvalidTraceParentContextError, PendingTraceParentContextError
from core.ops.ops_trace_manager import OpsTraceManager, TraceQueueManager
from core.ops.unified_trace.message_parent import load_message_parent, save_message_parent
from core.ops.unified_trace.otel import OTelTracingConfig, UnifiedOTelTrace
from core.ops.unified_trace.parent_context import ProviderParentContext
from core.ops.unified_trace.trace_builder import RepositoryWorkflowExecutionLoader
from extensions.ext_storage import storage
from extensions.storage.opendal_storage import OpenDALStorage
from graphon.model_runtime.entities.model_entities import ModelType
from models.account import Account, AccountStatus
from models.agent import (
    AgentConfigDraft,
    AgentConfigDraftType,
    AgentConfigSnapshot,
    AgentConfigVersionKind,
    AgentDebugConversation,
    AgentWorkspaceBinding,
)
from models.agent_config_entities import AgentSoulConfig
from models.enums import (
    ConversationFromSource,
    EndUserType,
    FeedbackFromSource,
    FeedbackRating,
)
from models.model import (
    App,
    AppAnnotationSetting,
    AppMode,
    AppModelConfig,
    Conversation,
    EndUser,
    Message,
    MessageFeedback,
)
from models.workflow import Workflow, WorkflowType
from repositories.sqlalchemy_execution_extra_content_repository import SQLAlchemyExecutionExtraContentRepository
from services.errors.message import (
    FirstMessageNotExistsError,
    LastMessageNotExistsError,
    MessageNotExistsError,
    SuggestedQuestionsAfterAnswerDisabledError,
)
from services.message_service import MessageService, attach_message_extra_contents
from tasks.ops_trace_task import process_trace_tasks

SQLITE_MODELS = (
    Conversation,
    Message,
    MessageFeedback,
    AppModelConfig,
    AppAnnotationSetting,
    AgentConfigDraft,
    AgentConfigSnapshot,
    AgentDebugConversation,
    AgentWorkspaceBinding,
)
pytestmark = [
    pytest.mark.usefixtures("sqlite_session"),
    pytest.mark.parametrize("sqlite_session", [SQLITE_MODELS], indirect=True),
]


class _DatabaseBinding:
    """Expose the SQLite engine and shared session through the production DB interface."""

    engine: Engine
    session: scoped_session[Session]

    def __init__(self, engine: Engine, session: scoped_session[Session]) -> None:
        self.engine = engine
        self.session = session


class MessageServiceTestDataFactory:
    """Create real service inputs and persistent message-domain rows."""

    @staticmethod
    def create_app(
        app_id: str = "app-123",
        mode: AppMode = AppMode.ADVANCED_CHAT,
        tenant_id: str = "tenant-123",
    ) -> App:
        return App(
            id=app_id,
            tenant_id=tenant_id,
            name="Test App",
            description="",
            mode=mode,
            enable_site=True,
            enable_api=True,
            max_active_requests=0,
        )

    @staticmethod
    def create_end_user(user_id: str = "user-456") -> EndUser:
        return EndUser(
            id=user_id,
            tenant_id="tenant-123",
            app_id="app-123",
            type=EndUserType.SERVICE_API,
            session_id="session-789",
        )

    @staticmethod
    def create_account(user_id: str = "account-123") -> Account:
        account = Account(name="Admin", email="admin@example.com", status=AccountStatus.ACTIVE)
        account.id = user_id
        return account

    @staticmethod
    def create_workflow(*, features: dict[str, object] | None = None) -> Workflow:
        return Workflow(
            id="workflow-123",
            tenant_id="tenant-123",
            app_id="app-123",
            type=WorkflowType.CHAT,
            version="1",
            graph="{}",
            _features=json.dumps(features or {}),
            created_by="account-123",
        )

    @staticmethod
    def create_conversation(
        conversation_id: str = "conv-001",
        app_id: str = "app-123",
        *,
        app_model_config_id: str | None = None,
        override_model_configs: str | None = None,
    ) -> Conversation:
        conversation = Conversation(
            id=conversation_id,
            app_id=app_id,
            app_model_config_id=app_model_config_id,
            override_model_configs=override_model_configs,
            mode=AppMode.CHAT,
            name="Test conversation",
            status="normal",
            from_source=ConversationFromSource.API,
            from_end_user_id="user-456",
        )
        conversation._inputs = {}
        return conversation

    @staticmethod
    def create_message(
        message_id: str = "msg-001",
        conversation_id: str = "conv-001",
        app_id: str = "app-123",
        *,
        created_at: datetime | None = None,
        from_source: ConversationFromSource = ConversationFromSource.API,
        from_end_user_id: str | None = "user-456",
        from_account_id: str | None = None,
    ) -> Message:
        message = Message(
            id=message_id,
            app_id=app_id,
            conversation_id=conversation_id,
            query="What is AI?",
            message={"role": "user", "content": "What is AI?"},
            answer="AI stands for Artificial Intelligence.",
            message_unit_price=Decimal("0.0001"),
            answer_unit_price=Decimal("0.0002"),
            currency="USD",
            from_source=from_source,
            from_end_user_id=from_end_user_id,
            from_account_id=from_account_id,
        )
        message._inputs = {}
        timestamp = created_at or datetime.now()
        message.created_at = timestamp
        message.updated_at = timestamp
        return message

    @staticmethod
    def create_feedback(
        feedback_id: str,
        message: Message,
        *,
        source: FeedbackFromSource,
        rating: FeedbackRating = FeedbackRating.LIKE,
    ) -> MessageFeedback:
        feedback = MessageFeedback(
            app_id=message.app_id,
            conversation_id=message.conversation_id,
            message_id=message.id,
            rating=rating,
            from_source=source,
            from_end_user_id="user-456" if source == FeedbackFromSource.USER else None,
            from_account_id="account-123" if source == FeedbackFromSource.ADMIN else None,
        )
        feedback.id = feedback_id
        return feedback


@pytest.fixture
def factory() -> MessageServiceTestDataFactory:
    return MessageServiceTestDataFactory()


@pytest.fixture(autouse=True)
def database_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_engine: Engine,
    sqlite_session: Session,
) -> Iterator[None]:
    """Bind global model properties and service-owned factories to the shared SQLite session."""
    sessions = scoped_session(lambda: sqlite_session)
    database = _DatabaseBinding(engine=sqlite_engine, session=sessions)
    monkeypatch.setattr(service_module, "db", database)
    monkeypatch.setattr(model_module, "db", database)
    try:
        yield
    finally:
        sessions.remove()


@pytest.fixture
def empty_extra_content_repository(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    repository = MagicMock()
    repository.get_by_message_ids.side_effect = lambda message_ids: [[] for _ in message_ids]
    monkeypatch.setattr(service_module, "_create_execution_extra_content_repository", lambda: repository)
    return repository


def _persist(session: Session, *records: object) -> None:
    session.add_all(records)
    session.commit()


def _patch_conversation(monkeypatch: pytest.MonkeyPatch, conversation: Conversation) -> MagicMock:
    get_conversation = MagicMock(return_value=conversation)
    monkeypatch.setattr(service_module.ConversationService, "get_conversation", get_conversation)
    return get_conversation


class TestMessageServicePaginationByFirstId:
    """Verify cursor pagination using persisted message timestamps and IDs."""

    @pytest.mark.parametrize(("user", "conversation_id"), [(None, "conv-001"), ("end_user", "")])
    def test_early_return(
        self,
        user: str | None,
        conversation_id: str,
        factory: MessageServiceTestDataFactory,
        sqlite_session: Session,
    ) -> None:
        result = MessageService.pagination_by_first_id(
            app_model=factory.create_app(),
            user=factory.create_end_user() if user else None,
            conversation_id=conversation_id,
            first_id=None,
            limit=10,
            session=sqlite_session,
        )

        assert result.data == []
        assert result.limit == 10
        assert result.has_more is False

    @pytest.mark.parametrize(
        ("order", "expected_ids"),
        [
            ("desc", ["msg-004", "msg-003", "msg-002", "msg-001", "msg-000"]),
            ("asc", ["msg-000", "msg-001", "msg-002", "msg-003", "msg-004"]),
        ],
    )
    def test_orders_persisted_messages(
        self,
        order: str,
        expected_ids: list[str],
        monkeypatch: pytest.MonkeyPatch,
        factory: MessageServiceTestDataFactory,
        sqlite_session: Session,
        empty_extra_content_repository: MagicMock,
    ) -> None:
        conversation = factory.create_conversation()
        messages = [
            factory.create_message(f"msg-{index:03d}", created_at=datetime(2024, 1, 1, 12, index)) for index in range(5)
        ]
        _persist(sqlite_session, conversation, *messages)
        _patch_conversation(monkeypatch, conversation)

        result = MessageService.pagination_by_first_id(
            app_model=factory.create_app(),
            user=factory.create_end_user(),
            conversation_id=conversation.id,
            first_id=None,
            limit=10,
            order=order,
            session=sqlite_session,
        )

        assert [message.id for message in result.data] == expected_ids
        assert result.has_more is False

    def test_first_id_excludes_cursor_and_newer_messages(
        self,
        monkeypatch: pytest.MonkeyPatch,
        factory: MessageServiceTestDataFactory,
        sqlite_session: Session,
        empty_extra_content_repository: MagicMock,
    ) -> None:
        conversation = factory.create_conversation()
        messages = [
            factory.create_message(f"msg-{index:03d}", created_at=datetime(2024, 1, 1, 12, index)) for index in range(7)
        ]
        _persist(sqlite_session, conversation, *messages)
        _patch_conversation(monkeypatch, conversation)

        result = MessageService.pagination_by_first_id(
            app_model=factory.create_app(),
            user=factory.create_end_user(),
            conversation_id=conversation.id,
            first_id="msg-005",
            limit=10,
            order="desc",
            session=sqlite_session,
        )

        assert [message.id for message in result.data] == [f"msg-{index:03d}" for index in range(4, -1, -1)]

    def test_missing_first_id_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        factory: MessageServiceTestDataFactory,
        sqlite_session: Session,
    ) -> None:
        conversation = factory.create_conversation()
        _persist(sqlite_session, conversation)
        _patch_conversation(monkeypatch, conversation)

        with pytest.raises(FirstMessageNotExistsError):
            MessageService.pagination_by_first_id(
                app_model=factory.create_app(),
                user=factory.create_end_user(),
                conversation_id=conversation.id,
                first_id="missing",
                limit=10,
                session=sqlite_session,
            )

    def test_has_more_trims_oldest_extra_row(
        self,
        monkeypatch: pytest.MonkeyPatch,
        factory: MessageServiceTestDataFactory,
        sqlite_session: Session,
        empty_extra_content_repository: MagicMock,
    ) -> None:
        conversation = factory.create_conversation()
        messages = [
            factory.create_message(f"msg-{index:03d}", created_at=datetime(2024, 1, 1, 12, index))
            for index in range(11)
        ]
        _persist(sqlite_session, conversation, *messages)
        _patch_conversation(monkeypatch, conversation)

        result = MessageService.pagination_by_first_id(
            app_model=factory.create_app(),
            user=factory.create_end_user(),
            conversation_id=conversation.id,
            first_id=None,
            limit=10,
            order="desc",
            session=sqlite_session,
        )

        assert len(result.data) == 10
        assert result.has_more is True
        assert result.data[-1].id == "msg-001"

    def test_empty_conversation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        factory: MessageServiceTestDataFactory,
        sqlite_session: Session,
        empty_extra_content_repository: MagicMock,
    ) -> None:
        conversation = factory.create_conversation()
        _persist(sqlite_session, conversation)
        _patch_conversation(monkeypatch, conversation)

        result = MessageService.pagination_by_first_id(
            app_model=factory.create_app(),
            user=factory.create_end_user(),
            conversation_id=conversation.id,
            first_id=None,
            limit=10,
            session=sqlite_session,
        )

        assert result.data == []
        assert result.has_more is False


class TestMessageServicePaginationByLastId:
    """Verify reverse cursor, conversation, and include-ID filtering."""

    def test_no_user(self, factory: MessageServiceTestDataFactory, sqlite_session: Session) -> None:
        result = MessageService.pagination_by_last_id(
            app_model=factory.create_app(), user=None, last_id=None, limit=10, session=sqlite_session
        )
        assert result.data == []
        assert result.limit == 10
        assert result.has_more is False

    def test_without_last_id(self, factory: MessageServiceTestDataFactory, sqlite_session: Session) -> None:
        messages = [
            factory.create_message(f"msg-{index:03d}", created_at=datetime(2024, 1, 1, 12, index)) for index in range(5)
        ]
        _persist(sqlite_session, *messages)

        result = MessageService.pagination_by_last_id(
            app_model=factory.create_app(),
            user=factory.create_end_user(),
            last_id=None,
            limit=10,
            session=sqlite_session,
        )

        assert [message.id for message in result.data] == [f"msg-{index:03d}" for index in range(4, -1, -1)]
        assert result.has_more is False

    def test_last_id_returns_older_rows(self, factory: MessageServiceTestDataFactory, sqlite_session: Session) -> None:
        messages = [
            factory.create_message(f"msg-{index:03d}", created_at=datetime(2024, 1, 1, 12, index)) for index in range(7)
        ]
        _persist(sqlite_session, *messages)

        result = MessageService.pagination_by_last_id(
            app_model=factory.create_app(),
            user=factory.create_end_user(),
            last_id="msg-005",
            limit=10,
            session=sqlite_session,
        )

        assert [message.id for message in result.data] == [f"msg-{index:03d}" for index in range(4, -1, -1)]

    def test_missing_last_id_raises(self, factory: MessageServiceTestDataFactory, sqlite_session: Session) -> None:
        with pytest.raises(LastMessageNotExistsError):
            MessageService.pagination_by_last_id(
                app_model=factory.create_app(),
                user=factory.create_end_user(),
                last_id="missing",
                limit=10,
                session=sqlite_session,
            )

    def test_conversation_filter(
        self,
        monkeypatch: pytest.MonkeyPatch,
        factory: MessageServiceTestDataFactory,
        sqlite_session: Session,
    ) -> None:
        conversation = factory.create_conversation()
        other_conversation = factory.create_conversation("conv-002")
        matching = factory.create_message("matching", conversation_id=conversation.id)
        excluded = factory.create_message("excluded", conversation_id=other_conversation.id)
        _persist(sqlite_session, conversation, other_conversation, matching, excluded)
        get_conversation = _patch_conversation(monkeypatch, conversation)

        result = MessageService.pagination_by_last_id(
            app_model=factory.create_app(),
            user=factory.create_end_user(),
            last_id=None,
            limit=10,
            conversation_id=conversation.id,
            session=sqlite_session,
        )

        assert [message.id for message in result.data] == [matching.id]
        get_conversation.assert_called_once()

    def test_include_ids_filter(self, factory: MessageServiceTestDataFactory, sqlite_session: Session) -> None:
        messages = [
            factory.create_message(f"msg-{index:03d}", created_at=datetime(2024, 1, 1, 12, index)) for index in range(4)
        ]
        _persist(sqlite_session, *messages)

        result = MessageService.pagination_by_last_id(
            app_model=factory.create_app(),
            user=factory.create_end_user(),
            last_id=None,
            limit=10,
            include_ids=["msg-001", "msg-003"],
            session=sqlite_session,
        )

        assert [message.id for message in result.data] == ["msg-003", "msg-001"]

    def test_has_more(self, factory: MessageServiceTestDataFactory, sqlite_session: Session) -> None:
        messages = [
            factory.create_message(f"msg-{index:03d}", created_at=datetime(2024, 1, 1, 12, index))
            for index in range(11)
        ]
        _persist(sqlite_session, *messages)

        result = MessageService.pagination_by_last_id(
            app_model=factory.create_app(),
            user=factory.create_end_user(),
            last_id=None,
            limit=10,
            session=sqlite_session,
        )

        assert len(result.data) == 10
        assert result.has_more is True


class TestMessageServiceUtilities:
    def test_attach_message_extra_contents_empty(self) -> None:
        attach_message_extra_contents([])

    def test_attach_message_extra_contents(
        self,
        monkeypatch: pytest.MonkeyPatch,
        factory: MessageServiceTestDataFactory,
    ) -> None:
        messages = [factory.create_message("msg-1"), factory.create_message("msg-2")]
        content_one = MagicMock()
        content_one.model_dump.return_value = {"key": "value1"}
        content_two = MagicMock()
        content_two.model_dump.return_value = {"key": "value2"}
        repository = MagicMock()
        repository.get_by_message_ids.return_value = [[content_one], [content_two]]
        monkeypatch.setattr(service_module, "_create_execution_extra_content_repository", lambda: repository)

        attach_message_extra_contents(messages)

        assert messages[0].extra_contents == [{"key": "value1"}]
        assert messages[1].extra_contents == [{"key": "value2"}]

    def test_attach_message_extra_contents_missing_list(
        self,
        monkeypatch: pytest.MonkeyPatch,
        factory: MessageServiceTestDataFactory,
    ) -> None:
        message = factory.create_message("msg-1")
        repository = MagicMock()
        repository.get_by_message_ids.return_value = []
        monkeypatch.setattr(service_module, "_create_execution_extra_content_repository", lambda: repository)

        attach_message_extra_contents([message])

        assert message.extra_contents == []

    def test_create_execution_extra_content_repository_uses_sqlite_factory(self, sqlite_engine: Engine) -> None:
        repository = service_module._create_execution_extra_content_repository()

        assert isinstance(repository, SQLAlchemyExecutionExtraContentRepository)
        assert repository._session_maker.kw["bind"] is sqlite_engine
        with repository._session_maker() as session:
            assert isinstance(session, Session)


class TestMessageServiceGetMessage:
    @pytest.mark.parametrize("actor", ["end_user", "account"])
    def test_identity_scoped_success(
        self,
        actor: str,
        factory: MessageServiceTestDataFactory,
        sqlite_session: Session,
    ) -> None:
        if actor == "end_user":
            user: Account | EndUser = factory.create_end_user("end-user-123")
            message = factory.create_message(
                "msg-123", from_end_user_id=user.id, from_account_id=None, from_source=ConversationFromSource.API
            )
        else:
            user = factory.create_account("account-123")
            message = factory.create_message(
                "msg-123",
                from_end_user_id=None,
                from_account_id=user.id,
                from_source=ConversationFromSource.CONSOLE,
            )
        distractor = factory.create_message("wrong-app", app_id="app-456")
        _persist(sqlite_session, message, distractor)

        result = MessageService.get_message(
            app_model=factory.create_app(), user=user, message_id=message.id, session=sqlite_session
        )

        assert result.id == message.id

    def test_not_found(self, factory: MessageServiceTestDataFactory, sqlite_session: Session) -> None:
        with pytest.raises(MessageNotExistsError):
            MessageService.get_message(
                app_model=factory.create_app(),
                user=factory.create_end_user(),
                message_id="missing",
                session=sqlite_session,
            )


@pytest.fixture
def otlp_receiver(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[OTelTracingConfig, list[WireSpan]]]:
    received: list[WireSpan] = []

    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            request = ExportTraceServiceRequest.FromString(self.rfile.read(int(self.headers["Content-Length"])))
            for batch in request.resource_spans:
                for scope in batch.scope_spans:
                    received.extend(scope.spans)
            self.send_response(200)
            self.end_headers()

    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    with HTTPServer(("127.0.0.1", 0), Receiver) as receiver:
        thread = Thread(target=receiver.serve_forever, daemon=True)
        thread.start()
        try:
            yield OTelTracingConfig(endpoint=f"http://127.0.0.1:{receiver.server_port}/v1/traces"), received
        finally:
            receiver.shutdown()
            thread.join()


class TestMessageServiceFeedback:
    def test_feedback_waits_for_answer_and_preserves_first_accepted_parent(
        self,
        factory: MessageServiceTestDataFactory,
        sqlite_session: Session,
        otlp_receiver: tuple[OTelTracingConfig, list[WireSpan]],
    ) -> None:
        app = factory.create_app()
        message = factory.create_message()
        _persist(sqlite_session, app, message)
        updated_at = message.updated_at
        config, received = otlp_receiver
        submission = FeedbackTraceInfo(
            submission_id=uuid4(),
            app_id=app.id,
            tenant_id=app.tenant_id,
            message_id=message.id,
            start_time=datetime.now(UTC),
            rating="dislike",
            content=None,
            metadata={},
        )
        with pytest.raises(PendingTraceParentContextError):
            UnifiedOTelTrace(config).trace(submission)
        assert received == []

        accepted = ProviderParentContext(
            provider="otel",
            scope="saved-destination",
            trace_id="answer",
            parent_id=message.id,
            provider_context={"traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01"},
        )
        save_message_parent(app.tenant_id, app.id, accepted)
        duplicate = accepted.model_copy(
            update={"provider_context": {"traceparent": "00-" + "c" * 32 + "-" + "d" * 16 + "-01"}}
        )
        save_message_parent(app.tenant_id, app.id, duplicate)
        assert load_message_parent(app.tenant_id, app.id, message.id, "otel", accepted.scope) == accepted
        sqlite_session.refresh(message)
        assert message.updated_at == updated_at
        with pytest.raises(InvalidTraceParentContextError):
            UnifiedOTelTrace(config).trace(submission)
        assert received == []

    @pytest.mark.parametrize("chatflow", [False, True], ids=["chat", "chatflow"])
    def test_next_day_feedback_reaches_original_otlp_span(
        self,
        factory: MessageServiceTestDataFactory,
        sqlite_session: Session,
        monkeypatch: pytest.MonkeyPatch,
        otlp_receiver: tuple[OTelTracingConfig, list[WireSpan]],
        tmp_path: Path,
        chatflow: bool,
    ) -> None:
        yesterday = datetime.now(UTC) - timedelta(days=1)
        app = factory.create_app()
        app.tracing = json.dumps({"enabled": True, "tracing_provider": "otel"})
        message = factory.create_message(created_at=yesterday)
        _persist(sqlite_session, app, message)
        config, received = otlp_receiver
        metadata = {"app_id": app.id, "tenant_id": app.tenant_id, "app_name": app.name}
        if chatflow:
            monkeypatch.setattr(RepositoryWorkflowExecutionLoader, "__call__", lambda _self, _info: [])
            info = WorkflowTraceInfo.model_construct(
                message_id=message.id,
                start_time=yesterday,
                end_time=yesterday,
                metadata=metadata,
                workflow_id="workflow",
                workflow_run_id="run",
                workflow_run_status="succeeded",
                query=message.query,
                workflow_run_inputs={},
                workflow_run_outputs={"answer": message.answer},
                total_tokens=3,
            )
        else:
            info = MessageTraceInfo.model_construct(
                message_id=message.id,
                message_data=message,
                start_time=yesterday,
                end_time=yesterday,
                message_tokens=1,
                answer_tokens=2,
                total_tokens=3,
                metadata=metadata,
            )
        UnifiedOTelTrace(config).trace(info)
        # Fresh worker and empty Redis, as after the short-lived parent context expires.
        provider = UnifiedOTelTrace(config)
        files: list[dict[str, str]] = []
        monkeypatch.setattr(OpsTraceManager, "get_ops_trace_instance", lambda _app_id: provider)
        monkeypatch.setattr(ops_trace_manager, "trace_manager_queue", Queue())
        monkeypatch.setattr(TraceQueueManager, "start_timer", lambda _self: None)
        monkeypatch.setattr(TraceQueueManager, "enqueue_persisted_trace", lambda _self, file: files.append(file))
        monkeypatch.setattr(storage, "storage_runner", OpenDALStorage("fs", root=str(tmp_path)), raising=False)
        for rating, content in [(FeedbackRating.LIKE, None), (FeedbackRating.DISLIKE, "请补充来源")]:
            MessageService.create_feedback(
                app_model=app,
                message_id=message.id,
                user=factory.create_end_user(),
                rating=rating,
                content=content,
                session=sqlite_session,
            )
        TraceQueueManager(app_id=app.id).run()
        assert len(files) == 2
        for file in files:
            path = ops_trace_manager.ops_trace_payload_path(file["app_id"], file["file_id"])
            payload = storage.load(path)
            process_trace_tasks.run(file)
        storage.save(path, payload)
        process_trace_tasks.run(files[-1])  # Same queued operation redelivered.

        answer, _, like, dislike, replay = received
        assert like.span_id != dislike.span_id == replay.span_id
        for span in (like, dislike, replay):
            assert span.name == "feedback"
            assert span.trace_id == answer.trace_id
            assert span.parent_span_id == answer.span_id
            assert span.start_time_unix_nano - answer.end_time_unix_nano >= 86_400 * 10**9
        attrs = [{attr.key: attr.value.string_value for attr in span.attributes} for span in (like, dislike)]
        assert attrs[0]["dify.feedback.rating"] == "like"
        assert "dify.feedback.content" not in attrs[0]
        assert attrs[1]["dify.feedback.rating"] == "dislike"
        assert attrs[1]["dify.feedback.content"] == "请补充来源"
        assert attrs[1]["dify.message.id"] == message.id
        with pytest.raises(InvalidTraceParentContextError):
            load_message_parent(app.tenant_id, app.id, message.id, "otel", "different-destination")
        for tenant_id, app_id in [("other-tenant", app.id), (app.tenant_id, "other-app")]:
            with pytest.raises(PendingTraceParentContextError):
                load_message_parent(tenant_id, app_id, message.id, "otel", "different-destination")

    def test_create_new_end_user_feedback(
        self,
        factory: MessageServiceTestDataFactory,
        sqlite_session: Session,
        sqlite_engine: Engine,
    ) -> None:
        user = factory.create_end_user()
        message = factory.create_message("msg-123")
        _persist(sqlite_session, message)

        feedback = MessageService.create_feedback(
            app_model=factory.create_app(),
            message_id=message.id,
            user=user,
            rating=FeedbackRating.LIKE,
            content="Good answer",
            session=sqlite_session,
        )

        with Session(sqlite_engine) as verification_session:
            persisted = verification_session.get(MessageFeedback, feedback.id)
            assert persisted is not None
            assert persisted.rating == FeedbackRating.LIKE
            assert persisted.content == "Good answer"
            assert persisted.from_source == FeedbackFromSource.USER

    def test_update_account_feedback(
        self,
        factory: MessageServiceTestDataFactory,
        sqlite_session: Session,
        sqlite_engine: Engine,
    ) -> None:
        user = factory.create_account()
        message = factory.create_message(
            "msg-123",
            from_source=ConversationFromSource.CONSOLE,
            from_end_user_id=None,
            from_account_id=user.id,
        )
        feedback = factory.create_feedback("feedback-1", message, source=FeedbackFromSource.ADMIN)
        _persist(sqlite_session, message, feedback)

        result = MessageService.create_feedback(
            app_model=factory.create_app(),
            message_id=message.id,
            user=user,
            rating=FeedbackRating.DISLIKE,
            content="Bad answer",
            session=sqlite_session,
        )

        assert result.id == feedback.id
        with Session(sqlite_engine) as verification_session:
            persisted = verification_session.get(MessageFeedback, feedback.id)
            assert persisted is not None
            assert persisted.rating == FeedbackRating.DISLIKE
            assert persisted.content == "Bad answer"

    def test_delete_feedback(
        self,
        factory: MessageServiceTestDataFactory,
        sqlite_session: Session,
        sqlite_engine: Engine,
    ) -> None:
        user = factory.create_end_user()
        message = factory.create_message("msg-123")
        feedback = factory.create_feedback("feedback-1", message, source=FeedbackFromSource.USER)
        _persist(sqlite_session, message, feedback)

        MessageService.create_feedback(
            app_model=factory.create_app(),
            message_id=message.id,
            user=user,
            rating=None,
            content=None,
            session=sqlite_session,
        )

        with Session(sqlite_engine) as verification_session:
            assert verification_session.get(MessageFeedback, feedback.id) is None

    def test_get_all_feedbacks_is_app_scoped_and_paginated(
        self,
        factory: MessageServiceTestDataFactory,
        sqlite_session: Session,
    ) -> None:
        message = factory.create_message("msg-123")
        newest = factory.create_feedback("feedback-new", message, source=FeedbackFromSource.USER)
        oldest = factory.create_feedback("feedback-old", message, source=FeedbackFromSource.USER)
        other_message = factory.create_message("other-msg", app_id="app-456")
        other_app = factory.create_feedback("feedback-other", other_message, source=FeedbackFromSource.USER)
        newest.created_at = datetime(2024, 1, 2)
        oldest.created_at = datetime(2024, 1, 1)
        other_app.created_at = datetime(2024, 1, 3)
        _persist(sqlite_session, newest, oldest, other_app)

        result = MessageService.get_all_messages_feedbacks(
            app_model=factory.create_app(), page=1, limit=1, session=sqlite_session
        )

        assert [record["id"] for record in result] == [newest.id]


class TestMessageServiceSuggestedQuestions:
    @staticmethod
    def _agent_soul(
        *,
        enabled: bool = True,
        prompt: str | None = None,
        model: dict[str, object] | None = None,
    ) -> AgentSoulConfig:
        suggested_questions: dict[str, object] = {"enabled": enabled}
        if prompt is not None:
            suggested_questions["prompt"] = prompt
        if model is not None:
            suggested_questions["model"] = model
        return AgentSoulConfig.model_validate(
            {"app_features": {"suggested_questions_after_answer": suggested_questions}}
        )

    @staticmethod
    def _chat_boundaries(
        monkeypatch: pytest.MonkeyPatch,
        conversation: Conversation,
    ) -> tuple[MagicMock, MagicMock, MagicMock]:
        message = MessageServiceTestDataFactory.create_message(message_id="msg-123", conversation_id=conversation.id)
        monkeypatch.setattr(service_module.MessageService, "get_message", MagicMock(return_value=message))
        monkeypatch.setattr(
            service_module.ConversationService, "get_conversation", MagicMock(return_value=conversation)
        )
        model_manager = MagicMock()
        monkeypatch.setattr(service_module.ModelManager, "for_tenant", MagicMock(return_value=model_manager))
        memory = MagicMock()
        memory.return_value.get_history_prompt_text.return_value = "histories"
        monkeypatch.setattr(service_module, "TokenBufferMemory", memory)
        llm_generator = MagicMock()
        llm_generator.generate_suggested_questions_after_answer.return_value = ["Q1?"]
        monkeypatch.setattr(service_module, "LLMGenerator", llm_generator)
        monkeypatch.setattr(service_module, "TraceQueueManager", MagicMock())
        return model_manager, memory, llm_generator

    def test_user_none(self, factory: MessageServiceTestDataFactory, sqlite_session: Session) -> None:
        with pytest.raises(ValueError, match="user cannot be None"):
            MessageService.get_suggested_questions_after_answer(
                app_model=factory.create_app(),
                user=None,
                message_id="msg-123",
                invoke_from=InvokeFrom.WEB_APP,
                session=sqlite_session,
            )

    def test_advanced_chat_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
        factory: MessageServiceTestDataFactory,
        sqlite_session: Session,
    ) -> None:
        conversation = factory.create_conversation()
        _, _, llm_generator = self._chat_boundaries(monkeypatch, conversation)
        workflow = factory.create_workflow(features={"suggested_questions_after_answer": {"enabled": True}})
        workflow_service = MagicMock()
        workflow_service.return_value.get_published_workflow.return_value = workflow
        monkeypatch.setattr(service_module, "WorkflowService", workflow_service)
        app_config_manager = MagicMock()
        app_config_manager.get_app_config.return_value.additional_features.suggested_questions_after_answer = True
        monkeypatch.setattr(service_module, "AdvancedChatAppConfigManager", app_config_manager)

        result = MessageService.get_suggested_questions_after_answer(
            app_model=factory.create_app(mode=AppMode.ADVANCED_CHAT),
            user=factory.create_end_user(),
            message_id="msg-123",
            invoke_from=InvokeFrom.WEB_APP,
            session=sqlite_session,
        )

        assert result == ["Q1?"]
        llm_generator.generate_suggested_questions_after_answer.assert_called_once()

    @pytest.mark.parametrize("draft_type", [AgentConfigDraftType.DRAFT, AgentConfigDraftType.DEBUG_BUILD])
    def test_agent_debug_uses_matching_draft(
        self,
        draft_type: AgentConfigDraftType,
        monkeypatch: pytest.MonkeyPatch,
        factory: MessageServiceTestDataFactory,
        sqlite_session: Session,
    ) -> None:
        conversation = factory.create_conversation()
        conversation.mode = AppMode.AGENT
        _, _, llm_generator = self._chat_boundaries(monkeypatch, conversation)
        account = factory.create_account()
        draft = AgentConfigDraft(
            tenant_id="tenant-123",
            agent_id="agent-123",
            draft_type=draft_type,
            account_id=account.id if draft_type == AgentConfigDraftType.DEBUG_BUILD else None,
            draft_owner_key=account.id if draft_type == AgentConfigDraftType.DEBUG_BUILD else "",
            base_snapshot_id=None,
            home_snapshot_id=None,
            agent_workspace_binding_id=None,
            config_snapshot=self._agent_soul(
                prompt=f"{draft_type.value} prompt",
                model={
                    "provider": "openai",
                    "name": "gpt-4o-mini",
                    "mode": "chat",
                    "completion_params": {"temperature": 0.1},
                },
            ),
            created_by=account.id,
            updated_by=account.id,
        )
        draft.id = "draft-123"
        mapping = AgentDebugConversation(
            tenant_id="tenant-123",
            agent_id="agent-123",
            app_id="app-123",
            account_id=account.id,
            draft_type=draft_type,
            conversation_id=conversation.id,
        )
        mapping.id = "mapping-123"
        app_model_config = AppModelConfig(
            app_id="app-123",
            suggested_questions_after_answer=json.dumps({"enabled": False}),
        )
        app_model_config.id = "config-123"
        _persist(sqlite_session, draft, mapping, app_model_config)
        roster_service = MagicMock()
        monkeypatch.setattr("services.agent.roster_service.AgentRosterService", roster_service)
        app = factory.create_app(mode=AppMode.AGENT)
        app.app_model_config_id = app_model_config.id

        result = MessageService.get_suggested_questions_after_answer(
            app_model=app,
            user=account,
            message_id="msg-123",
            invoke_from=InvokeFrom.DEBUGGER,
            session=sqlite_session,
        )

        assert result == ["Q1?"]
        roster_service.assert_not_called()
        llm_generator.generate_suggested_questions_after_answer.assert_called_once_with(
            tenant_id="tenant-123",
            histories="histories",
            instruction_prompt=f"{draft_type.value} prompt",
            model_config={
                "provider": "openai",
                "name": "gpt-4o-mini",
                "mode": "chat",
                "completion_params": {"temperature": 0.1},
            },
        )

    def test_agent_published_conversation_uses_bound_snapshot(
        self,
        monkeypatch: pytest.MonkeyPatch,
        factory: MessageServiceTestDataFactory,
        sqlite_session: Session,
    ) -> None:
        conversation = factory.create_conversation()
        conversation.mode = AppMode.AGENT
        conversation.agent_workspace_binding_id = "binding-123"
        _, _, llm_generator = self._chat_boundaries(monkeypatch, conversation)
        snapshot = AgentConfigSnapshot(
            tenant_id="tenant-123",
            agent_id="agent-123",
            version=1,
            config_snapshot=self._agent_soul(prompt="bound snapshot prompt"),
            home_snapshot_id=None,
            summary=None,
            version_note=None,
            created_by="account-123",
        )
        snapshot.id = "snapshot-123"
        binding = AgentWorkspaceBinding(
            tenant_id="tenant-123",
            app_id="app-123",
            workspace_id="workspace-123",
            agent_id="agent-123",
            base_home_snapshot_id=None,
            agent_config_version_id=snapshot.id,
            agent_config_version_kind=AgentConfigVersionKind.SNAPSHOT,
            backend_binding_ref="backend-binding-123",
            session_snapshot=None,
            retired_at=None,
            pending_form_id=None,
            pending_tool_call_id=None,
        )
        binding.id = "binding-123"
        _persist(sqlite_session, snapshot, binding)
        roster_service = MagicMock()
        roster_service.return_value.get_published_agent_soul_for_app.return_value = self._agent_soul(
            prompt="current published prompt"
        )
        monkeypatch.setattr("services.agent.roster_service.AgentRosterService", roster_service)

        result = MessageService.get_suggested_questions_after_answer(
            app_model=factory.create_app(mode=AppMode.AGENT),
            user=factory.create_end_user(),
            message_id="msg-123",
            invoke_from=InvokeFrom.SERVICE_API,
            session=sqlite_session,
        )

        assert result == ["Q1?"]
        roster_service.assert_not_called()
        llm_generator.generate_suggested_questions_after_answer.assert_called_once_with(
            tenant_id="tenant-123",
            histories="histories",
            instruction_prompt="bound snapshot prompt",
            model_config=None,
        )

    def test_agent_without_binding_uses_published_soul(
        self,
        monkeypatch: pytest.MonkeyPatch,
        factory: MessageServiceTestDataFactory,
        sqlite_session: Session,
    ) -> None:
        conversation = factory.create_conversation()
        conversation.mode = AppMode.AGENT
        _, _, llm_generator = self._chat_boundaries(monkeypatch, conversation)
        roster_service = MagicMock()
        roster_service.return_value.get_published_agent_soul_for_app.return_value = self._agent_soul(
            prompt="published prompt"
        )
        monkeypatch.setattr("services.agent.roster_service.AgentRosterService", roster_service)

        result = MessageService.get_suggested_questions_after_answer(
            app_model=factory.create_app(mode=AppMode.AGENT),
            user=factory.create_end_user(),
            message_id="msg-123",
            invoke_from=InvokeFrom.SERVICE_API,
            session=sqlite_session,
        )

        assert result == ["Q1?"]
        roster_service.return_value.get_published_agent_soul_for_app.assert_called_once_with(
            tenant_id="tenant-123",
            app_id="app-123",
        )
        llm_generator.generate_suggested_questions_after_answer.assert_called_once_with(
            tenant_id="tenant-123",
            histories="histories",
            instruction_prompt="published prompt",
            model_config=None,
        )

    def test_agent_unavailable_default_model_returns_empty_questions(
        self,
        monkeypatch: pytest.MonkeyPatch,
        factory: MessageServiceTestDataFactory,
        sqlite_session: Session,
    ) -> None:
        conversation = factory.create_conversation()
        conversation.mode = AppMode.AGENT
        model_manager, memory, llm_generator = self._chat_boundaries(monkeypatch, conversation)
        model_manager.get_default_model_instance.side_effect = ValueError("unsupported default model")
        roster_service = MagicMock()
        roster_service.return_value.get_published_agent_soul_for_app.return_value = self._agent_soul()
        monkeypatch.setattr("services.agent.roster_service.AgentRosterService", roster_service)

        result = MessageService.get_suggested_questions_after_answer(
            app_model=factory.create_app(mode=AppMode.AGENT),
            user=factory.create_end_user(),
            message_id="msg-123",
            invoke_from=InvokeFrom.SERVICE_API,
            session=sqlite_session,
        )

        assert result == []
        memory.assert_not_called()
        llm_generator.generate_suggested_questions_after_answer.assert_not_called()

    def test_historical_agent_without_soul_uses_current_app_model_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
        factory: MessageServiceTestDataFactory,
        sqlite_session: Session,
    ) -> None:
        conversation = factory.create_conversation()
        conversation.mode = AppMode.AGENT
        _, _, llm_generator = self._chat_boundaries(monkeypatch, conversation)
        app_model_config = AppModelConfig(
            app_id="app-123",
            suggested_questions_after_answer=json.dumps({"enabled": True, "prompt": "legacy prompt"}),
        )
        app_model_config.id = "config-123"
        _persist(sqlite_session, app_model_config)
        app = factory.create_app(mode=AppMode.AGENT)
        app.app_model_config_id = app_model_config.id
        roster_service = MagicMock()
        roster_service.return_value.get_published_agent_soul_for_app.return_value = None
        monkeypatch.setattr("services.agent.roster_service.AgentRosterService", roster_service)

        result = MessageService.get_suggested_questions_after_answer(
            app_model=app,
            user=factory.create_end_user(),
            message_id="msg-123",
            invoke_from=InvokeFrom.SERVICE_API,
            session=sqlite_session,
        )

        assert result == ["Q1?"]
        llm_generator.generate_suggested_questions_after_answer.assert_called_once_with(
            tenant_id="tenant-123",
            histories="histories",
            instruction_prompt="legacy prompt",
            model_config=None,
        )

    def test_agent_disabled_raises_disabled_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        factory: MessageServiceTestDataFactory,
        sqlite_session: Session,
    ) -> None:
        conversation = factory.create_conversation()
        conversation.mode = AppMode.AGENT
        self._chat_boundaries(monkeypatch, conversation)
        roster_service = MagicMock()
        roster_service.return_value.get_published_agent_soul_for_app.return_value = self._agent_soul(enabled=False)
        monkeypatch.setattr("services.agent.roster_service.AgentRosterService", roster_service)

        with pytest.raises(SuggestedQuestionsAfterAnswerDisabledError):
            MessageService.get_suggested_questions_after_answer(
                app_model=factory.create_app(mode=AppMode.AGENT),
                user=factory.create_end_user(),
                message_id="msg-123",
                invoke_from=InvokeFrom.SERVICE_API,
                session=sqlite_session,
            )

    @pytest.mark.parametrize(
        ("config", "expected_prompt", "expected_model"),
        [
            ({"enabled": True}, None, None),
            (
                {
                    "enabled": True,
                    "prompt": "custom prompt",
                    "model": {
                        "provider": "openai",
                        "name": "gpt-4o-mini",
                        "completion_params": {"max_tokens": 2048, "temperature": 0.1},
                    },
                },
                "custom prompt",
                {
                    "provider": "openai",
                    "name": "gpt-4o-mini",
                    "completion_params": {"max_tokens": 2048, "temperature": 0.1},
                },
            ),
            (
                {"enabled": True, "model": {"provider": "openai", "name": "invalid-model"}},
                None,
                {"provider": "openai", "name": "invalid-model"},
            ),
        ],
    )
    def test_chat_app_uses_persisted_model_config(
        self,
        config: dict[str, object],
        expected_prompt: str | None,
        expected_model: dict[str, object] | None,
        monkeypatch: pytest.MonkeyPatch,
        factory: MessageServiceTestDataFactory,
        sqlite_session: Session,
    ) -> None:
        app_model_config = AppModelConfig(
            app_id="app-123",
            suggested_questions_after_answer=json.dumps(config),
        )
        app_model_config.id = "config-1"
        conversation = factory.create_conversation(app_model_config_id=app_model_config.id)
        _persist(sqlite_session, app_model_config)
        model_manager, memory, llm_generator = self._chat_boundaries(monkeypatch, conversation)

        result = MessageService.get_suggested_questions_after_answer(
            app_model=factory.create_app(mode=AppMode.CHAT),
            user=factory.create_end_user(),
            message_id="msg-123",
            invoke_from=InvokeFrom.WEB_APP,
            session=sqlite_session,
        )

        assert result == ["Q1?"]
        model_manager.get_default_model_instance.assert_called_once_with(
            tenant_id="tenant-123", model_type=ModelType.LLM
        )
        memory.assert_called_once_with(
            conversation=conversation,
            model_instance=model_manager.get_default_model_instance.return_value,
        )
        llm_generator.generate_suggested_questions_after_answer.assert_called_once_with(
            tenant_id="tenant-123",
            histories="histories",
            instruction_prompt=expected_prompt,
            model_config=expected_model,
        )

    def test_chat_app_uses_compatible_override_model_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
        factory: MessageServiceTestDataFactory,
        sqlite_session: Session,
    ) -> None:
        override = {
            "model": {"provider": "openai", "name": "gpt-4o-mini", "mode": "chat"},
            "suggested_questions_after_answer": {"enabled": True, "prompt": "legacy prompt"},
        }
        conversation = factory.create_conversation(override_model_configs=json.dumps(override))
        _, _, llm_generator = self._chat_boundaries(monkeypatch, conversation)

        result = MessageService.get_suggested_questions_after_answer(
            app_model=factory.create_app(mode=AppMode.CHAT),
            user=factory.create_end_user(),
            message_id="msg-123",
            invoke_from=InvokeFrom.WEB_APP,
            session=sqlite_session,
        )

        assert result == ["Q1?"]
        llm_generator.generate_suggested_questions_after_answer.assert_called_once_with(
            tenant_id="tenant-123",
            histories="histories",
            instruction_prompt="legacy prompt",
            model_config=None,
        )

    def test_disabled_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        factory: MessageServiceTestDataFactory,
        sqlite_session: Session,
    ) -> None:
        conversation = factory.create_conversation()
        self._chat_boundaries(monkeypatch, conversation)
        workflow = factory.create_workflow()
        workflow_service = MagicMock()
        workflow_service.return_value.get_published_workflow.return_value = workflow
        monkeypatch.setattr(service_module, "WorkflowService", workflow_service)
        app_config_manager = MagicMock()
        app_config_manager.get_app_config.return_value.additional_features.suggested_questions_after_answer = False
        monkeypatch.setattr(service_module, "AdvancedChatAppConfigManager", app_config_manager)

        with pytest.raises(SuggestedQuestionsAfterAnswerDisabledError):
            MessageService.get_suggested_questions_after_answer(
                app_model=factory.create_app(mode=AppMode.ADVANCED_CHAT),
                user=factory.create_end_user(),
                message_id="msg-123",
                invoke_from=InvokeFrom.WEB_APP,
                session=sqlite_session,
            )
