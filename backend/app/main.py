"""FastAPI entrypoint exposing the exact assignment API contract."""

from contextlib import asynccontextmanager
import logging
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.database import Database
from app.llm_service import AllProvidersUnavailableError, LLMService
from app.schemas import (
    ChatMetadata,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SessionMessagesResponse,
)


def create_app(
    settings: Settings | None = None,
    llm_service: LLMService | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    app_settings = settings or get_settings()
    db = Database(app_settings.database_url)
    llm = llm_service or LLMService(app_settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.db.init_db()
        logging.getLogger("uvicorn.error").info(
            "Startup config: keys_configured=%d, model=%s, cache_enabled=%s",
            application.state.llm.keys_configured,
            application.state.settings.llm_model,
            application.state.llm.cache_enabled,
        )
        yield

    application = FastAPI(
        title=app_settings.app_name,
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.settings = app_settings
    application.state.db = db
    application.state.llm = llm

    application.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": jsonable_encoder(exc.errors())},
        )

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "internal server error"},
        )

    def get_db(request: Request) -> Database:
        return request.app.state.db

    def get_llm(request: Request) -> LLMService:
        return request.app.state.llm

    def get_cfg(request: Request) -> Settings:
        return request.app.state.settings

    @application.get("/health", response_model=HealthResponse, tags=["health"])
    def health_check(
        database: Database = Depends(get_db),
        service: LLMService = Depends(get_llm),
    ) -> HealthResponse:
        if not database.check_health():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="database unavailable",
            )
        return HealthResponse(
            status="ok",
            keys_configured=service.keys_configured,
            cache_enabled=service.cache_enabled,
        )

    @application.post("/chat", response_model=ChatResponse, tags=["chat"])
    async def chat(
        payload: ChatRequest,
        x_simulate_failure: str | None = Header(default=None, alias="X-Simulate-Failure"),
        database: Database = Depends(get_db),
        service: LLMService = Depends(get_llm),
        cfg: Settings = Depends(get_cfg),
    ) -> ChatResponse:
        stored_system_prompt = database.get_session_system_prompt(payload.session_id)
        effective_system_prompt = (
            payload.system_prompt
            if payload.system_prompt is not None
            else stored_system_prompt or cfg.default_system_prompt
        )

        history = database.get_history_for_llm(
            payload.session_id, limit=cfg.max_history_messages
        )
        llm_messages: list[dict[str, str]] = []
        if effective_system_prompt.strip():
            llm_messages.append({"role": "system", "content": effective_system_prompt.strip()})
        llm_messages.extend(history)
        llm_messages.append({"role": "user", "content": payload.message})

        try:
            llm_reply = await service.generate_reply(
                messages=llm_messages,
                simulate_failure_header=x_simulate_failure,
            )
        except AllProvidersUnavailableError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="all providers unavailable",
            ) from exc

        metadata = ChatMetadata(
            resolution_source=llm_reply.resolution_source,
            model=llm_reply.model,
            latency_ms=llm_reply.latency_ms,
            input_tokens=llm_reply.input_tokens,
            output_tokens=llm_reply.output_tokens,
            key_id=llm_reply.key_id,
            fallback_attempts=llm_reply.fallback_attempts,
        )

        database.save_turn(
            session_id=payload.session_id,
            user_content=payload.message,
            assistant_content=llm_reply.reply,
            metadata=metadata.model_dump(),
            system_prompt=payload.system_prompt,
        )

        return ChatResponse(
            session_id=payload.session_id,
            reply=llm_reply.reply,
            metadata=metadata,
        )

    @application.get(
        "/sessions/{session_id}/messages",
        response_model=SessionMessagesResponse,
        response_model_exclude_none=True,
        tags=["sessions"],
    )
    def get_session_messages(
        session_id: str,
        database: Database = Depends(get_db),
    ) -> SessionMessagesResponse:
        messages = database.get_session_messages(session_id)
        if messages is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="session not found",
            )
        return SessionMessagesResponse(session_id=session_id, messages=messages)

    return application


app = create_app()
