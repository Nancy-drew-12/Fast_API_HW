"""Единый формат ошибок и обработчики исключений."""
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import logger


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Any = None


class ServiceError(Exception):
    """Ошибка сервиса: машинный код, понятное сообщение, HTTP-статус и подробности."""

    def __init__(self, code: str, message: str, status_code: int = 400, details: Any = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details


def error_json(status_code: int, code: str, message: str, details: Any = None) -> JSONResponse:
    body = ErrorResponse(code=code, message=message, details=details)
    return JSONResponse(status_code=status_code, content=body.model_dump())


async def service_error_handler(request: Request, exc: ServiceError):
    logger.warning("%s %s -> %s: %s", request.method, request.url.path, exc.code, exc.message)
    return error_json(exc.status_code, exc.code, exc.message, exc.details)


async def http_error_handler(request: Request, exc: StarletteHTTPException):
    code = "not_found" if exc.status_code == 404 else "http_error"
    return error_json(exc.status_code, code, str(exc.detail))


# Ошибки-«шум» от варианта тела запроса «один клиент ИЛИ список клиентов»
UNION_NOISE = {"list_type", "model_type", "model_attributes_type", "dict_type"}


def clean_field(loc: tuple) -> str:
    """('body', 'FeatureVectorChurn', 'usage_hours') -> 'usage_hours'; ('body', 'list[...]', 0, 'region') -> '0.region'."""
    parts = [str(p) for p in loc if p != "body" and "FeatureVectorChurn" not in str(p)]
    return ".".join(parts) or "body"


async def validation_error_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    meaningful = [err for err in errors if err["type"] not in UNION_NOISE] or errors
    details = [
        {"field": clean_field(err["loc"]), "problem": err["msg"], "input": err.get("input")}
        for err in meaningful
    ]
    logger.warning("%s %s -> validation_error: %s", request.method, request.url.path,
                   [d["field"] for d in details])
    return error_json(422, "validation_error", "Неверные входные данные", details)


async def unexpected_error_handler(request: Request, exc: Exception):
    logger.exception("Unexpected error on %s", request.url.path)
    return error_json(500, "internal_error", "Внутренняя ошибка сервиса. Попробуйте позже")


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ServiceError, service_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, unexpected_error_handler)
