# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnusedCallResult=false, reportUnusedFunction=false, reportUnnecessaryComparison=false

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import socket
import sys
import time
import urllib.request
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
    cast,
)
from urllib.error import HTTPError, URLError
from rich.console import Console
from rich.live import Live
from rich.table import Table

try:
    from typing import Protocol, TypedDict
except ImportError:
    from typing_extensions import Protocol, TypedDict


from . import utils


_ENDPOINT_NOT_SUPPORTED_STATUS_CODES = {404, 405, 410, 501}
_NON_FALLBACK_STATUS_CODES = {401, 403, 429}
_ENDPOINT_NOT_FOUND_ERROR_TYPES = {
    "endpoint_not_found",
    "unsupported_endpoint",
    "unknown_endpoint",
}
_ENDPOINT_NOT_FOUND_MESSAGE_MARKERS = (
    "endpoint_not_found",
    "endpoint not found",
    "unsupported endpoint",
    "unknown endpoint",
    "unknown request url",
    "no route matched",
)
_NON_FALLBACK_ERROR_TYPES = {
    "authentication_error",
    "insufficient_permissions",
    "invalid_api_key",
    "model_not_found",
    "rate_limit_exceeded",
}
_NON_FALLBACK_MESSAGE_MARKERS = (
    "invalid api key",
    "insufficient permissions",
    "model not found",
    "does not exist",
    "rate limit",
    "too many requests",
    "unauthorized",
    "forbidden",
)

_RETRYABLE_HTTP_STATUS_CODES = {408, 429}
_NON_RETRYABLE_HTTP_STATUS_CODES = {400, 401, 403, 404, 405, 410, 422}
_NON_RETRYABLE_HTTP_ERROR_TYPES = {
    "authentication_error",
    "insufficient_permissions",
    "invalid_api_key",
    "invalid_request_error",
    "model_not_found",
}
_NON_RETRYABLE_HTTP_ERROR_CODES = {
    "invalid_api_key",
    "invalid_payload",
    "invalid_request_error",
    "model_not_found",
}
_NON_RETRYABLE_HTTP_MESSAGE_MARKERS = (
    "invalid api key",
    "invalid request",
    "missing required",
    "unsupported parameter",
    "model not found",
)
_DEFAULT_HTTP_TIMEOUT_SECONDS = 30.0

_TEST_STATUS_LABELS = {
    "queued": "🕒 QUEUED",
    "running": "🔄 RUNNING",
    "pass": "✅ PASS",
    "warn": "⚠️ WARN",
    "fail": "❌ FAIL",
    "skip": "⏭️ SKIP",
}
_TEST_DASHBOARD_COLUMNS = ("Step", "Status", "HTTP", "Time(ms)", "Summary")
_TEST_STEP_ORDER = ("network", "auth", "models", "probe")
_TEST_MODE_TO_STEPS: dict[str, tuple[str, ...]] = {
    "network": ("network",),
    "auth": ("network", "auth"),
    "models": ("network", "auth", "models"),
    "probe": _TEST_STEP_ORDER,
    "all": _TEST_STEP_ORDER,
}
_TEST_PROBE_KEYWORDS = ("nano", "flash", "mini")
_TEST_CHEAP_MODEL_ALLOWLIST = (
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "gpt-4o-mini",
    "gpt-4o-mini-search-preview",
    "o4-mini",
)
_DEFAULT_CHAT_MODEL = "gpt-4.1-mini"

_DEFAULT_OPENAI_ENDPOINT = "https://api.openai.com/v1"
_OPENAI_ERROR_CONSOLE = Console(stderr=True)

_OPENAI_EXIT_CODES: dict[str, int] = {
    "success": 0,
    "usage": 2,
    "network": 10,
    "auth": 20,
    "models": 30,
    "probe": 40,
    "chat": 50,
    "internal": 90,
    "interrupt": 130,
}
_AUTH_FAILURE_HTTP_STATUS_CODES = {401, 403}
_AUTH_FAILURE_ERROR_TYPES = {
    "authentication_error",
    "insufficient_permissions",
    "invalid_api_key",
}
_AUTH_FAILURE_ERROR_CODES = {
    "authentication_error",
    "insufficient_permissions",
    "invalid_api_key",
}

_JsonPayload = Optional[Union[Dict[str, object], List[object], str, int, float, bool]]


class _NormalizedError(TypedDict):
    kind: str
    http_status: int | None
    error_type: str | None
    error_code: str | None
    message: str
    retryable: bool


class _RequestResult(TypedDict):
    ok: bool
    attempts: int
    http_status: int | None
    data: _JsonPayload
    error: _NormalizedError | None


class _TestStepRender(TypedDict, total=False):
    status: str
    http: int | str | None
    time_ms: int | float | str | None
    summary: str | None


class _OutputPrecedence(TypedDict):
    quiet: bool
    json: bool
    human: bool


class _ListModel(TypedDict):
    id: str
    owned_by: str
    created: int | None


class _UrlOpenResponse(Protocol):
    def getcode(self) -> int: ...

    def read(self) -> bytes: ...

    def __iter__(self) -> Iterable[bytes]: ...


class _UrlOpenContextManager(Protocol):
    def __enter__(self) -> _UrlOpenResponse: ...

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> object: ...


def _resolve_output_precedence(
    *, quiet: bool, json_requested: bool
) -> _OutputPrecedence:
    quiet_enabled = bool(quiet)
    json_enabled = bool(json_requested) and not quiet_enabled
    human_enabled = not quiet_enabled and not json_enabled
    return {
        "quiet": quiet_enabled,
        "json": json_enabled,
        "human": human_enabled,
    }


def _resolve_openai_exit_code(
    *,
    ok: bool,
    command: str,
    stage: str | None = None,
    error: _NormalizedError | None = None,
    exception: BaseException | None = None,
    usage_error: bool = False,
) -> int:
    if ok:
        return _OPENAI_EXIT_CODES["success"]

    if isinstance(exception, KeyboardInterrupt):
        return _OPENAI_EXIT_CODES["interrupt"]

    if usage_error or isinstance(exception, ValueError):
        return _OPENAI_EXIT_CODES["usage"]

    normalized_command = (_normalize_optional_text(command) or "").lower()
    normalized_stage = (_normalize_optional_text(stage) or "").lower()
    http_status = error["http_status"] if error is not None else None
    error_kind = (
        (_normalize_optional_text(error.get("kind")) or "").lower()
        if error is not None
        else ""
    )
    error_type = (
        (_normalize_optional_text(error.get("error_type")) or "").lower()
        if error is not None
        else ""
    )
    error_code = (
        (_normalize_optional_text(error.get("error_code")) or "").lower()
        if error is not None
        else ""
    )

    if (
        http_status in _AUTH_FAILURE_HTTP_STATUS_CODES
        or error_type in _AUTH_FAILURE_ERROR_TYPES
        or error_code in _AUTH_FAILURE_ERROR_CODES
        or normalized_stage == "auth"
    ):
        return _OPENAI_EXIT_CODES["auth"]

    if error_kind == "transport_error" or normalized_stage == "network":
        return _OPENAI_EXIT_CODES["network"]

    if normalized_command == "list" or normalized_stage == "models":
        return _OPENAI_EXIT_CODES["models"]

    if normalized_stage == "probe":
        return _OPENAI_EXIT_CODES["probe"]

    if normalized_command == "chat":
        return _OPENAI_EXIT_CODES["chat"]

    return _OPENAI_EXIT_CODES["internal"]


def _normalize_optional_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    text = str(value).strip()
    return text or None


def _sanitize_error_message(message: str | None, *, key: str | None) -> str:
    sanitized_message = _normalize_optional_text(message) or "Request failed."
    normalized_key = (key or "").strip()
    if normalized_key != "":
        sanitized_message = sanitized_message.replace(
            f"Bearer {normalized_key}",
            "Bearer [REDACTED]",
        )
        sanitized_message = sanitized_message.replace(normalized_key, "[REDACTED]")
    return sanitized_message


def _parse_json_bytes(raw_bytes: bytes) -> tuple[_JsonPayload | None, str | None]:
    if len(raw_bytes) == 0:
        return None, None
    decoded_body = raw_bytes.decode("utf-8", errors="replace").strip()
    if decoded_body == "":
        return None, None
    try:
        return cast(_JsonPayload, json.loads(decoded_body)), decoded_body
    except json.JSONDecodeError:
        return None, decoded_body


def _extract_error_details(
    payload: object,
    *,
    fallback_message: str | None,
) -> tuple[str | None, str | None, str]:
    error_type = None
    error_code = None
    message = fallback_message

    if isinstance(payload, dict):
        raw_error = payload.get("error")
        if isinstance(raw_error, dict):
            error_type = _normalize_optional_text(raw_error.get("type"))
            error_code = _normalize_optional_text(raw_error.get("code"))
            nested_message = _normalize_optional_text(raw_error.get("message"))
            if nested_message is not None:
                message = nested_message
        elif isinstance(raw_error, str):
            nested_message = _normalize_optional_text(raw_error)
            if nested_message is not None:
                message = nested_message

        if error_type is None:
            error_type = _normalize_optional_text(payload.get("type"))
        if error_code is None:
            error_code = _normalize_optional_text(payload.get("code"))
        root_message = _normalize_optional_text(payload.get("message"))
        if root_message is not None and message is None:
            message = root_message

    normalized_message = _normalize_optional_text(message) or "Request failed."
    return error_type, error_code, normalized_message


def _contains_message_marker(message: str | None, markers: tuple[str, ...]) -> bool:
    normalized_message = _normalize_optional_text(message)
    if normalized_message is None:
        return False
    normalized_message = normalized_message.lower()
    return any(marker in normalized_message for marker in markers)


def _is_obvious_non_retryable_http_error(
    *,
    status_code: int | None,
    error_type: str | None,
    error_code: str | None,
    message: str | None,
) -> bool:
    if status_code in _NON_RETRYABLE_HTTP_STATUS_CODES:
        return True

    normalized_error_type = (_normalize_optional_text(error_type) or "").lower()
    normalized_error_code = (_normalize_optional_text(error_code) or "").lower()
    if normalized_error_type in _NON_RETRYABLE_HTTP_ERROR_TYPES:
        return True
    if normalized_error_code in _NON_RETRYABLE_HTTP_ERROR_CODES:
        return True
    return _contains_message_marker(message, _NON_RETRYABLE_HTTP_MESSAGE_MARKERS)


def _is_retryable_http_error(
    *,
    status_code: int | None,
    error_type: str | None,
    error_code: str | None,
    message: str | None,
) -> bool:
    if _is_obvious_non_retryable_http_error(
        status_code=status_code,
        error_type=error_type,
        error_code=error_code,
        message=message,
    ):
        return False
    if status_code is None:
        return False
    if status_code in _RETRYABLE_HTTP_STATUS_CODES:
        return True
    return 500 <= status_code <= 599


def _is_retryable_transport_error(error: Exception) -> bool:
    if isinstance(error, (TimeoutError, socket.timeout, URLError)):
        return True
    return False


def _build_normalized_error(
    *,
    kind: str,
    http_status: int | None,
    error_type: str | None,
    error_code: str | None,
    message: str | None,
    retryable: bool,
    key: str | None,
) -> _NormalizedError:
    return {
        "kind": kind,
        "http_status": http_status,
        "error_type": _normalize_optional_text(error_type),
        "error_code": _normalize_optional_text(error_code),
        "message": _sanitize_error_message(message, key=key),
        "retryable": retryable,
    }


def _build_json_request(
    *,
    url: str,
    method: str,
    key: str | None,
    payload: _JsonPayload = None,
    headers: dict[str, str] | None = None,
) -> urllib.request.Request:
    data = None
    request_headers = {
        "Accept": "application/json",
        "User-Agent": "TensorNeko",
    }

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"

    normalized_key = (key or "").strip()
    if normalized_key != "":
        request_headers["Authorization"] = f"Bearer {normalized_key}"

    if headers is not None:
        request_headers.update(headers)

    return urllib.request.Request(
        url,
        data=data,
        method=method.strip().upper(),
        headers=request_headers,
    )


def _normalize_http_error(error: HTTPError, *, key: str | None) -> _NormalizedError:
    status_code = int(error.code)
    response_body = b""
    try:
        response_body = error.read()
    except Exception:
        response_body = b""

    parsed_body, decoded_body = _parse_json_bytes(response_body)
    fallback_message = (
        decoded_body
        or _normalize_optional_text(getattr(error, "reason", None))
        or _normalize_optional_text(error.msg)
    )
    error_type, error_code, error_message = _extract_error_details(
        parsed_body,
        fallback_message=fallback_message,
    )
    retryable = _is_retryable_http_error(
        status_code=status_code,
        error_type=error_type,
        error_code=error_code,
        message=error_message,
    )
    return _build_normalized_error(
        kind="http_error",
        http_status=status_code,
        error_type=error_type,
        error_code=error_code,
        message=error_message,
        retryable=retryable,
        key=key,
    )


def _normalize_transport_error(
    error: Exception, *, key: str | None
) -> _NormalizedError:
    error_type = type(error).__name__.lower()
    if isinstance(error, URLError):
        reason = error.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            error_type = "timeout_error"
        else:
            error_type = "url_error"
        message = _normalize_optional_text(reason) or _normalize_optional_text(error)
    elif isinstance(error, (TimeoutError, socket.timeout)):
        error_type = "timeout_error"
        message = _normalize_optional_text(error)
    else:
        message = _normalize_optional_text(error)

    return _build_normalized_error(
        kind="transport_error",
        http_status=None,
        error_type=error_type,
        error_code=None,
        message=message,
        retryable=_is_retryable_transport_error(error),
        key=key,
    )


def _request_json_with_retry(
    *,
    url: str,
    method: str,
    key: str | None,
    payload: _JsonPayload = None,
    timeout: float = _DEFAULT_HTTP_TIMEOUT_SECONDS,
    retries: int = 0,
    headers: dict[str, str] | None = None,
) -> _RequestResult:
    max_retries = max(0, retries)
    normalized_timeout = timeout if timeout > 0 else _DEFAULT_HTTP_TIMEOUT_SECONDS
    normalized_key = (key or "").strip()
    key_for_error = normalized_key or None

    for attempt_idx in range(max_retries + 1):
        attempts = attempt_idx + 1
        request = _build_json_request(
            url=url,
            method=method,
            key=normalized_key,
            payload=payload,
            headers=headers,
        )
        try:
            response_context = cast(
                _UrlOpenContextManager,
                urllib.request.urlopen(request, timeout=normalized_timeout),
            )
            with response_context as response:
                status_code = int(response.getcode())
                response_body = response.read()
            parsed_payload, decoded_body = _parse_json_bytes(response_body)
            if parsed_payload is None and decoded_body is not None:
                normalized_error = _build_normalized_error(
                    kind="response_parse_error",
                    http_status=status_code,
                    error_type="invalid_json_response",
                    error_code=None,
                    message=f"Unable to parse JSON response: {decoded_body}",
                    retryable=False,
                    key=key_for_error,
                )
                return {
                    "ok": False,
                    "attempts": attempts,
                    "http_status": status_code,
                    "data": None,
                    "error": normalized_error,
                }

            return {
                "ok": True,
                "attempts": attempts,
                "http_status": status_code,
                "data": parsed_payload if parsed_payload is not None else {},
                "error": None,
            }
        except HTTPError as error:
            normalized_error = _normalize_http_error(error, key=key_for_error)
            if normalized_error["retryable"] and attempt_idx < max_retries:
                continue
            return {
                "ok": False,
                "attempts": attempts,
                "http_status": normalized_error["http_status"],
                "data": None,
                "error": normalized_error,
            }
        except (URLError, TimeoutError, socket.timeout) as error:
            normalized_error = _normalize_transport_error(error, key=key_for_error)
            if normalized_error["retryable"] and attempt_idx < max_retries:
                continue
            return {
                "ok": False,
                "attempts": attempts,
                "http_status": None,
                "data": None,
                "error": normalized_error,
            }

    fallback_error = _build_normalized_error(
        kind="internal_error",
        http_status=None,
        error_type="retry_loop_exhausted",
        error_code=None,
        message="Retry loop exhausted unexpectedly.",
        retryable=False,
        key=key_for_error,
    )
    return {
        "ok": False,
        "attempts": max_retries + 1,
        "http_status": None,
        "data": None,
        "error": fallback_error,
    }


def _validate_endpoint_and_key_args(
    endpoint: str | None,
    key: str | None,
) -> tuple[str, str]:
    endpoint_value = (endpoint or "").strip()
    key_value = (key or "").strip()

    normalized_endpoint = endpoint_value or _DEFAULT_OPENAI_ENDPOINT
    normalized_key = key_value or ""

    return normalized_endpoint, normalized_key


def _resolve_endpoint_candidates(endpoint: str) -> list[str]:
    normalized_endpoint = endpoint.strip().rstrip("/")
    if normalized_endpoint == "":
        raise ValueError("--endpoint must not be empty.")
    if normalized_endpoint.endswith("/v1"):
        return [normalized_endpoint]
    return [f"{normalized_endpoint}/v1", normalized_endpoint]


def _contains_marker(content: str | None, markers: tuple[str, ...]) -> bool:
    if content is None:
        return False
    normalized_content = content.strip().lower()
    if normalized_content == "":
        return False
    return any(marker in normalized_content for marker in markers)


def _resolve_test_status_label(status_key: str | None) -> str:
    normalized_key = (_normalize_optional_text(status_key) or "queued").lower()
    return _TEST_STATUS_LABELS.get(normalized_key, _TEST_STATUS_LABELS["queued"])


def _format_test_http_value(value: int | str | None) -> str:
    normalized_text = _normalize_optional_text(value)
    if normalized_text is not None:
        return normalized_text
    if isinstance(value, int):
        return str(value)
    return "-"


def _format_test_time_ms(value: int | float | str | None) -> str:
    if isinstance(value, bool):
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    normalized_text = _normalize_optional_text(value)
    return normalized_text or "-"


def _build_test_dashboard_table(
    step_rows: dict[str, _TestStepRender] | None,
) -> Table:
    table = Table(show_header=True, header_style="bold magenta")
    for column in _TEST_DASHBOARD_COLUMNS:
        table.add_column(column)

    rows = step_rows or {}
    for step_name in _TEST_STEP_ORDER:
        raw_step: _TestStepRender = rows.get(step_name, {})
        status_key = "queued"
        http_value: int | str | None = None
        time_ms_value: int | float | str | None = None
        summary_text = ""

        if raw_step.get("status") is not None:
            status_key = cast(str, raw_step.get("status"))
        http_value = raw_step.get("http")
        time_ms_value = raw_step.get("time_ms")
        summary_text = _normalize_optional_text(raw_step.get("summary")) or ""

        table.add_row(
            step_name,
            _resolve_test_status_label(status_key),
            _format_test_http_value(http_value),
            _format_test_time_ms(time_ms_value),
            summary_text,
        )

    return table


def _select_test_render_mode(*, no_live: bool, is_tty: bool | None = None) -> str:
    if no_live:
        return "static"
    terminal_flag = is_tty
    if terminal_flag is None:
        terminal_flag = bool(getattr(utils.console, "is_terminal", False))
    return "live" if terminal_flag else "static"


def _render_test_dashboard_live(
    step_rows: dict[str, _TestStepRender] | None,
    *,
    live: Live | None = None,
) -> None:
    dashboard_table = _build_test_dashboard_table(step_rows)
    if live is not None:
        live.update(dashboard_table, refresh=True)
        return None
    with Live(
        dashboard_table, console=utils.console, refresh_per_second=8, transient=False
    ):
        return None


def _render_test_dashboard_static(step_rows: dict[str, _TestStepRender] | None) -> None:
    utils.console.print(_build_test_dashboard_table(step_rows))


def _render_test_dashboard(
    step_rows: dict[str, _TestStepRender] | None,
    *,
    no_live: bool,
    is_tty: bool | None = None,
    live: Live | None = None,
) -> str:
    mode = _select_test_render_mode(no_live=no_live, is_tty=is_tty)
    if mode == "live":
        _render_test_dashboard_live(step_rows, live=live)
    else:
        _render_test_dashboard_static(step_rows)
    return mode


def _render_chat_plain_text(text: str, *, append: bool = False) -> None:
    if append:
        utils.console.print(text, end="")
        return
    utils.console.print(text)


def _build_human_error_summary(
    *,
    command: str,
    error: _NormalizedError | None,
    endpoint: str | None = None,
    stage: str | None = None,
) -> str:
    normalized_command = _normalize_optional_text(command) or "openai"
    normalized_stage = _normalize_optional_text(stage)
    normalized_endpoint = _normalize_optional_text(endpoint)

    title = f"OpenAI {normalized_command} failed"
    if normalized_stage is not None and normalized_command == "test":
        title = f"{title} during {normalized_stage}"

    if error is None:
        if normalized_endpoint is not None:
            return f"{title} (endpoint={normalized_endpoint})."
        return f"{title}."

    details: list[str] = []
    if normalized_endpoint is not None:
        details.append(f"endpoint={normalized_endpoint}")

    http_status = error.get("http_status")
    if isinstance(http_status, int):
        details.append(f"http={http_status}")

    error_type = _normalize_optional_text(error.get("error_type"))
    if error_type is not None:
        details.append(f"type={error_type}")

    error_code = _normalize_optional_text(error.get("error_code"))
    if error_code is not None and error_code != error_type:
        details.append(f"code={error_code}")

    detail_prefix = f" ({', '.join(details)})" if len(details) > 0 else ""
    message = _normalize_optional_text(error.get("message")) or "Request failed."
    return f"{title}{detail_prefix}: {message}"


def _render_human_error_to_stderr(message: str) -> None:
    _OPENAI_ERROR_CONSOLE.print(f"[bold red]{message}[/bold red]")


def _emit_human_error(
    *,
    output_precedence: _OutputPrecedence,
    command: str,
    error: _NormalizedError | None,
    endpoint: str | None = None,
    stage: str | None = None,
) -> None:
    if not output_precedence["human"] or error is None:
        return
    _render_human_error_to_stderr(
        _build_human_error_summary(
            command=command,
            error=error,
            endpoint=endpoint,
            stage=stage,
        )
    )


def _build_list_table(models: list[_ListModel]) -> Table:
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model ID")
    table.add_column("Owner")
    table.add_column("Created")

    for model in models:
        model_id = _normalize_optional_text(model.get("id")) or "<unknown>"
        owner = _normalize_optional_text(model.get("owned_by")) or "-"
        created = model.get("created")
        created_text = "-" if created is None else str(created)
        table.add_row(model_id, owner, created_text)

    return table


def _build_list_summary_text(
    models: list[_ListModel], *, endpoint: str | None = None
) -> str:
    count = len(models)
    normalized_endpoint = _normalize_optional_text(endpoint) or "-"
    return f"Total models: {count}\nEndpoint: {normalized_endpoint}"


def _render_list_table_and_summary(
    models: list[_ListModel], *, endpoint: str | None = None
) -> None:
    utils.console.print(_build_list_summary_text(models, endpoint=endpoint))
    utils.console.print(_build_list_table(models))


def _should_try_endpoint_fallback(
    status_code: int | None,
    error_type: str | None,
    error_message: str | None,
) -> bool:
    normalized_error_type = (error_type or "").strip().lower()

    if status_code in _NON_FALLBACK_STATUS_CODES:
        return False
    if normalized_error_type in _NON_FALLBACK_ERROR_TYPES:
        return False
    if _contains_marker(error_message, _NON_FALLBACK_MESSAGE_MARKERS):
        return False

    if status_code in _ENDPOINT_NOT_SUPPORTED_STATUS_CODES:
        return True
    if normalized_error_type in _ENDPOINT_NOT_FOUND_ERROR_TYPES:
        return True
    if _contains_marker(error_message, _ENDPOINT_NOT_FOUND_MESSAGE_MARKERS):
        return True
    return False


def _should_retry_raw_endpoint_candidate(
    *,
    endpoint_candidates: list[str],
    candidate_index: int,
    error: _NormalizedError | None,
) -> bool:
    if len(endpoint_candidates) < 2:
        return False
    if candidate_index != 0:
        return False
    if error is None:
        return False

    return _should_try_endpoint_fallback(
        status_code=error.get("http_status"),
        error_type=error.get("error_type"),
        error_message=error.get("message"),
    )


def _resolve_test_mode_steps(mode: str | None) -> tuple[str, tuple[str, ...]]:
    normalized_mode = (_normalize_optional_text(mode) or "all").lower()
    if normalized_mode not in _TEST_MODE_TO_STEPS:
        normalized_mode = "all"
    return normalized_mode, _TEST_MODE_TO_STEPS[normalized_mode]


def _initialize_test_step_rows(
    *,
    active_steps: tuple[str, ...],
    mode: str,
) -> dict[str, _TestStepRender]:
    active_step_set = set(active_steps)
    rows: dict[str, _TestStepRender] = {}
    for step in _TEST_STEP_ORDER:
        if step in active_step_set:
            rows[step] = {
                "status": "queued",
                "http": None,
                "time_ms": None,
                "summary": "Waiting",
            }
        else:
            rows[step] = {
                "status": "skip",
                "http": None,
                "time_ms": None,
                "summary": f"Skipped in {mode} mode.",
            }
    return rows


def _set_test_step(
    step_rows: dict[str, _TestStepRender],
    *,
    step: str,
    status: str,
    summary: str,
    http: int | str | None = None,
    time_ms: int | float | str | None = None,
) -> None:
    row = step_rows.setdefault(step, {})
    row["status"] = status
    row["summary"] = summary
    row["http"] = http
    row["time_ms"] = time_ms


def _mark_remaining_steps_as_skip(
    step_rows: dict[str, _TestStepRender],
    *,
    active_steps: tuple[str, ...],
    after_step: str,
    reason: str,
) -> None:
    should_skip = False
    for step in active_steps:
        if should_skip:
            _set_test_step(
                step_rows,
                step=step,
                status="skip",
                summary=reason,
                http=None,
                time_ms=None,
            )
        if step == after_step:
            should_skip = True


def _render_test_dashboard_if_human(
    step_rows: dict[str, _TestStepRender],
    *,
    no_live: bool,
    output_precedence: _OutputPrecedence,
    live: Live | None = None,
) -> None:
    if not output_precedence["human"]:
        return
    if live is None:
        _render_test_dashboard(step_rows, no_live=no_live)
        return
    _render_test_dashboard(step_rows, no_live=no_live, live=live)


def _is_auth_failure(error: _NormalizedError | None) -> bool:
    if error is None:
        return False

    status_code = error.get("http_status")
    error_type = (_normalize_optional_text(error.get("error_type")) or "").lower()
    error_code = (_normalize_optional_text(error.get("error_code")) or "").lower()

    if status_code in _AUTH_FAILURE_HTTP_STATUS_CODES:
        return True
    if error_type in _AUTH_FAILURE_ERROR_TYPES:
        return True
    if error_code in _AUTH_FAILURE_ERROR_CODES:
        return True
    return False


def _elapsed_ms(start_time: float) -> int:
    return max(0, int((time.perf_counter() - start_time) * 1000))


def _parse_model_created(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)

    normalized_text = _normalize_optional_text(value)
    if normalized_text is None:
        return None
    if normalized_text.lstrip("-").isdigit():
        return int(normalized_text)
    return None


def _extract_list_models(payload: _JsonPayload) -> list[_ListModel]:
    raw_models: list[object]
    if isinstance(payload, dict):
        data_field = payload.get("data")
        if isinstance(data_field, list):
            raw_models = list(data_field)
        else:
            raw_models = []
    elif isinstance(payload, list):
        raw_models = list(payload)
    else:
        raw_models = []

    parsed_models: list[_ListModel] = []
    seen: set[str] = set()
    for raw_model in raw_models:
        model_id: str | None = None
        owned_by = "-"
        created: int | None = None

        if isinstance(raw_model, dict):
            model_id = _normalize_optional_text(raw_model.get("id"))
            owned_by = (
                _normalize_optional_text(
                    raw_model.get("owned_by") or raw_model.get("owner")
                )
                or "-"
            )
            created = _parse_model_created(raw_model.get("created"))
        elif isinstance(raw_model, str):
            model_id = _normalize_optional_text(raw_model)

        if model_id is None:
            continue

        normalized_key = model_id.lower()
        if normalized_key in seen:
            continue
        seen.add(normalized_key)

        parsed_models.append(
            {
                "id": model_id,
                "owned_by": owned_by,
                "created": created,
            }
        )

    return sorted(parsed_models, key=lambda model: model["id"].lower())


def _extract_model_ids(payload: _JsonPayload) -> list[str]:
    raw_models: list[object]
    if isinstance(payload, dict):
        data_field = payload.get("data")
        if isinstance(data_field, list):
            raw_models = list(data_field)
        else:
            raw_models = []
    elif isinstance(payload, list):
        raw_models = list(payload)
    else:
        raw_models = []

    model_ids: list[str] = []
    seen: set[str] = set()
    for raw_model in raw_models:
        model_id: str | None
        if isinstance(raw_model, dict):
            model_id = _normalize_optional_text(raw_model.get("id"))
        elif isinstance(raw_model, str):
            model_id = _normalize_optional_text(raw_model)
        else:
            model_id = None

        if model_id is None:
            continue

        normalized_key = model_id.lower()
        if normalized_key in seen:
            continue
        seen.add(normalized_key)
        model_ids.append(model_id)

    return model_ids


def _select_probe_model(
    *, explicit_model: str | None, models: list[str]
) -> tuple[str | None, str]:
    explicit = _normalize_optional_text(explicit_model)
    if explicit is not None:
        return explicit, "explicit"

    normalized_models = [
        model_id
        for model_id in (_normalize_optional_text(model) for model in models)
        if model_id is not None
    ]
    if len(normalized_models) == 0:
        return None, "none"

    model_lookup = {model_id.lower(): model_id for model_id in normalized_models}
    for allowlisted_model in _TEST_CHEAP_MODEL_ALLOWLIST:
        allowlisted = model_lookup.get(allowlisted_model.lower())
        if allowlisted is not None:
            return allowlisted, "allowlist"

    keyword_matches = sorted(
        (
            model_id
            for model_id in normalized_models
            if any(keyword in model_id.lower() for keyword in _TEST_PROBE_KEYWORDS)
        ),
        key=str.lower,
    )
    if len(keyword_matches) > 0:
        return keyword_matches[0], "keyword"

    fallback_any = sorted(normalized_models, key=str.lower)
    return fallback_any[0], "fallback_any"


def _build_probe_responses_payload(model_id: str) -> dict[str, object]:
    return {
        "model": model_id,
        "input": "Reply with exactly: pong",
        "max_output_tokens": 16,
    }


def _build_probe_chat_payload(model_id: str) -> dict[str, object]:
    return {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly: pong",
            }
        ],
        "max_tokens": 16,
    }


def _probe_single_model(
    *,
    endpoint_base: str,
    key: str,
    model_id: str,
    no_chat_fallback: bool,
) -> tuple[_RequestResult, bool]:
    responses_result = _request_json_with_retry(
        url=f"{endpoint_base}/responses",
        method="POST",
        key=key,
        payload=_build_probe_responses_payload(model_id),
    )
    if responses_result["ok"]:
        return responses_result, False

    normalized_error = responses_result["error"]
    if no_chat_fallback or normalized_error is None:
        return responses_result, False

    should_fallback = _should_try_endpoint_fallback(
        status_code=normalized_error.get("http_status"),
        error_type=normalized_error.get("error_type"),
        error_message=normalized_error.get("message"),
    )
    if not should_fallback:
        return responses_result, False

    chat_result = _request_json_with_retry(
        url=f"{endpoint_base}/chat/completions",
        method="POST",
        key=key,
        payload=_build_probe_chat_payload(model_id),
    )
    return chat_result, True


def _resolve_test_step_ok(status: str) -> bool | None:
    normalized_status = (_normalize_optional_text(status) or "queued").lower()
    if normalized_status == "pass":
        return True
    if normalized_status in {"fail", "warn"}:
        return False
    return None


def _parse_test_step_http_status(value: int | str | None) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    normalized_text = _normalize_optional_text(value)
    if normalized_text is None:
        return None
    if normalized_text.lstrip("-").isdigit():
        return int(normalized_text)
    return None


def _parse_test_step_elapsed_ms(value: int | float | str | None) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    normalized_text = _normalize_optional_text(value)
    if normalized_text is None:
        return None
    try:
        return int(float(normalized_text))
    except ValueError:
        return None


def _build_test_json_steps(
    step_rows: dict[str, _TestStepRender],
    *,
    failure_stage: str | None,
    error: _NormalizedError | None,
) -> list[dict[str, object]]:
    normalized_failure_stage = (_normalize_optional_text(failure_stage) or "").lower()
    steps: list[dict[str, object]] = []

    for step_name in _TEST_STEP_ORDER:
        step_data = step_rows.get(step_name, {})
        status = (_normalize_optional_text(step_data.get("status")) or "queued").lower()
        message = _normalize_optional_text(step_data.get("summary")) or ""
        http_status = _parse_test_step_http_status(step_data.get("http"))
        elapsed_ms = _parse_test_step_elapsed_ms(step_data.get("time_ms"))
        step_error_code: str | None = None

        if normalized_failure_stage == step_name and error is not None:
            step_error_code = _normalize_optional_text(
                error.get("error_code")
            ) or _normalize_optional_text(error.get("error_type"))
            if message == "":
                message = error.get("message")
            if http_status is None:
                error_http_status = error.get("http_status")
                if isinstance(error_http_status, int):
                    http_status = error_http_status

        steps.append(
            {
                "name": step_name,
                "status": status,
                "ok": _resolve_test_step_ok(status),
                "http_status": http_status,
                "error_code": step_error_code,
                "message": message,
                "elapsed_ms": elapsed_ms,
            }
        )

    return steps


def _emit_test_json(
    *,
    output_precedence: _OutputPrecedence,
    ok: bool,
    exit_code: int,
    endpoint: str | None,
    resolved_endpoint_base: str | None,
    mode: str,
    selected_model: str | None,
    selection_strategy: str | None,
    probe_endpoint: str | None,
    failure_stage: str | None,
    step_rows: dict[str, _TestStepRender],
    started_at: float,
    finished_at: float,
    error: _NormalizedError | None,
) -> None:
    if not output_precedence["json"]:
        return

    payload = {
        "command": "test",
        "ok": ok,
        "exit_code": exit_code,
        "endpoint": endpoint,
        "resolved_endpoint_base": resolved_endpoint_base,
        "mode": mode,
        "selected_model": selected_model,
        "model_selection_strategy": selection_strategy,
        "probe_endpoint": probe_endpoint,
        "steps": _build_test_json_steps(
            step_rows,
            failure_stage=failure_stage,
            error=error,
        ),
        "started_at": started_at,
        "finished_at": finished_at,
        "error": error,
    }
    utils.console.print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _add_shared_arguments(parser):
    parser.add_argument("--endpoint", help="OpenAI-compatible API endpoint", type=str)
    parser.add_argument("--key", help="OpenAI-compatible API key", type=str)


def register_subparser(subparsers):
    parser_openai = subparsers.add_parser(
        "openai", help="OpenAI-compatible command family"
    )
    openai_subparsers = parser_openai.add_subparsers(dest="openai_sub_command")

    parser_test = openai_subparsers.add_parser(
        "test", help="Run OpenAI connectivity and capability checks"
    )
    _add_shared_arguments(parser_test)
    parser_test.add_argument(
        "--mode",
        help="Test mode",
        type=str,
        choices=["all", "network", "auth", "models", "probe"],
        default="all",
    )
    parser_test.add_argument(
        "--json", action="store_true", help="Render output as JSON"
    )
    parser_test.add_argument(
        "--no-live",
        action="store_true",
        help="Disable live dashboard rendering",
    )
    parser_test.add_argument(
        "--fail-fast", action="store_true", help="Stop on first test failure"
    )
    parser_test.add_argument("--model", help="Model ID for probing behavior", type=str)
    parser_test.add_argument(
        "--no-chat-fallback",
        action="store_true",
        help="Disable fallback chat probing when model tests fail",
    )
    parser_test.set_defaults(func=run_test)

    parser_chat = openai_subparsers.add_parser("chat", help="Send one chat request")
    _add_shared_arguments(parser_chat)
    parser_chat.add_argument(
        "message", nargs="?", help="Optional message prompt", type=str
    )
    parser_chat.add_argument(
        "--no-stream", action="store_true", help="Disable streaming output"
    )
    parser_chat.add_argument(
        "--json", action="store_true", help="Render output as JSON"
    )
    parser_chat.add_argument("--model", help="Model ID to use for chat", type=str)
    parser_chat.set_defaults(func=run_chat)

    parser_list = openai_subparsers.add_parser("list", help="List available models")
    _add_shared_arguments(parser_list)
    parser_list.add_argument(
        "--json", action="store_true", help="Render output as JSON"
    )
    parser_list.set_defaults(func=run_list)


def run_test(_args: argparse.Namespace) -> int:
    started_at = time.time()
    output_precedence = _resolve_output_precedence(
        quiet=bool(getattr(_args, "quiet", False)),
        json_requested=bool(getattr(_args, "json", False)),
    )
    mode, active_steps = _resolve_test_mode_steps(getattr(_args, "mode", None))
    step_rows = _initialize_test_step_rows(active_steps=active_steps, mode=mode)
    no_live = bool(getattr(_args, "no_live", False))
    active_step_set = set(active_steps)

    endpoint, key = _validate_endpoint_and_key_args(
        getattr(_args, "endpoint", None),
        getattr(_args, "key", None),
    )
    endpoint_candidates: list[str] = []
    endpoint_base: str | None = None
    selected_model: str | None = None
    selection_strategy: str | None = None
    probe_endpoint: str | None = None
    key_for_errors: str | None = key
    current_stage: str | None = None
    dashboard_live: Live | None = None

    def _finalize(
        *,
        ok: bool,
        stage: str | None = None,
        error: _NormalizedError | None = None,
        exception: BaseException | None = None,
        usage_error: bool = False,
    ) -> int:
        finished_at = time.time()
        if no_live:
            _render_test_dashboard_if_human(
                step_rows,
                no_live=no_live,
                output_precedence=output_precedence,
                live=dashboard_live,
            )
        exit_code = _resolve_openai_exit_code(
            ok=ok,
            command="test",
            stage=stage,
            error=error,
            exception=exception,
            usage_error=usage_error,
        )
        _emit_test_json(
            output_precedence=output_precedence,
            ok=ok,
            exit_code=exit_code,
            endpoint=endpoint,
            resolved_endpoint_base=endpoint_base,
            mode=mode,
            probe_endpoint=probe_endpoint,
            failure_stage=stage,
            step_rows=step_rows,
            selected_model=selected_model,
            selection_strategy=selection_strategy,
            started_at=started_at,
            finished_at=finished_at,
            error=error,
        )
        _emit_human_error(
            output_precedence=output_precedence,
            command="test",
            error=error,
            endpoint=endpoint_base,
            stage=stage,
        )
        return exit_code

    with ExitStack() as dashboard_stack:
        render_mode = "static"
        if output_precedence["human"]:
            render_mode = _select_test_render_mode(no_live=no_live)
        if output_precedence["human"] and render_mode == "live":
            dashboard_live = dashboard_stack.enter_context(
                Live(
                    _build_test_dashboard_table(step_rows),
                    console=utils.console,
                    refresh_per_second=8,
                    transient=False,
                )
            )

        def _render_dashboard() -> None:
            if no_live:
                return
            _render_test_dashboard_if_human(
                step_rows,
                no_live=no_live,
                output_precedence=output_precedence,
                live=dashboard_live,
            )

        _render_dashboard()

        try:
            endpoint_candidates = _resolve_endpoint_candidates(endpoint)
            endpoint_base = endpoint_candidates[0]

            current_stage = "network"
            network_start = time.perf_counter()
            _set_test_step(
                step_rows,
                step="network",
                status="running",
                summary="Checking endpoint reachability.",
                http=None,
                time_ms=None,
            )
            _render_dashboard()

            models_result = _request_json_with_retry(
                url=f"{endpoint_base}/models",
                method="GET",
                key=key,
            )
            if not models_result["ok"] and _should_retry_raw_endpoint_candidate(
                endpoint_candidates=endpoint_candidates,
                candidate_index=0,
                error=models_result["error"],
            ):
                endpoint_base = endpoint_candidates[1]
                models_result = _request_json_with_retry(
                    url=f"{endpoint_base}/models",
                    method="GET",
                    key=key,
                )
            network_time = _elapsed_ms(network_start)
            if models_result["ok"]:
                _set_test_step(
                    step_rows,
                    step="network",
                    status="pass",
                    summary="Endpoint reachable.",
                    http=models_result["http_status"],
                    time_ms=network_time,
                )
            else:
                network_error = models_result["error"]
                network_http = models_result["http_status"]
                network_summary = "Request failed."
                if network_error is not None:
                    network_http = network_error.get("http_status")
                    network_summary = network_error.get("message") or network_summary

                if (
                    network_error is not None
                    and network_error.get("kind") == "transport_error"
                ):
                    _set_test_step(
                        step_rows,
                        step="network",
                        status="fail",
                        summary=network_summary,
                        http=network_http,
                        time_ms=network_time,
                    )
                    _mark_remaining_steps_as_skip(
                        step_rows,
                        active_steps=active_steps,
                        after_step="network",
                        reason="Skipped after network failure.",
                    )
                    _render_dashboard()
                    return _finalize(ok=False, stage="network", error=network_error)

                _set_test_step(
                    step_rows,
                    step="network",
                    status="pass",
                    summary=f"Endpoint reachable (HTTP {network_http or '-'})",
                    http=network_http,
                    time_ms=network_time,
                )
            _render_dashboard()

            if "auth" in active_step_set:
                current_stage = "auth"
                auth_start = time.perf_counter()
                _set_test_step(
                    step_rows,
                    step="auth",
                    status="running",
                    summary="Checking authentication status.",
                    http=None,
                    time_ms=None,
                )
                _render_dashboard()

                auth_error = models_result["error"]
                if _is_auth_failure(auth_error):
                    auth_http = (
                        auth_error["http_status"] if auth_error is not None else None
                    )
                    auth_summary = (
                        auth_error["message"]
                        if auth_error is not None
                        else "Authentication failed."
                    )
                    _set_test_step(
                        step_rows,
                        step="auth",
                        status="fail",
                        summary=auth_summary,
                        http=auth_http,
                        time_ms=_elapsed_ms(auth_start),
                    )
                    _mark_remaining_steps_as_skip(
                        step_rows,
                        active_steps=active_steps,
                        after_step="auth",
                        reason="Skipped after auth failure.",
                    )
                    _render_dashboard()
                    return _finalize(ok=False, stage="auth", error=auth_error)

                _set_test_step(
                    step_rows,
                    step="auth",
                    status="pass",
                    summary="Authentication checks passed.",
                    http=models_result["http_status"],
                    time_ms=_elapsed_ms(auth_start),
                )
                _render_dashboard()

            model_ids: list[str] = []
            if "models" in active_step_set:
                current_stage = "models"
                models_start = time.perf_counter()
                _set_test_step(
                    step_rows,
                    step="models",
                    status="running",
                    summary="Fetching model catalog.",
                    http=None,
                    time_ms=None,
                )
                _render_dashboard()

                if not models_result["ok"]:
                    models_error = models_result["error"]
                    if models_error is None:
                        models_error = _build_normalized_error(
                            kind="http_error",
                            http_status=models_result["http_status"],
                            error_type="model_list_failed",
                            error_code=None,
                            message="Failed to fetch model list.",
                            retryable=False,
                            key=key,
                        )
                    _set_test_step(
                        step_rows,
                        step="models",
                        status="fail",
                        summary=models_error["message"],
                        http=models_error["http_status"],
                        time_ms=_elapsed_ms(models_start),
                    )
                    _mark_remaining_steps_as_skip(
                        step_rows,
                        active_steps=active_steps,
                        after_step="models",
                        reason="Skipped after models failure.",
                    )
                    _render_dashboard()
                    return _finalize(ok=False, stage="models", error=models_error)

                model_ids = _extract_model_ids(models_result["data"])
                if len(model_ids) == 0:
                    models_error = _build_normalized_error(
                        kind="response_parse_error",
                        http_status=models_result["http_status"],
                        error_type="no_models_available",
                        error_code=None,
                        message="No models available from endpoint.",
                        retryable=False,
                        key=key,
                    )
                    _set_test_step(
                        step_rows,
                        step="models",
                        status="fail",
                        summary=models_error["message"],
                        http=models_result["http_status"],
                        time_ms=_elapsed_ms(models_start),
                    )
                    _mark_remaining_steps_as_skip(
                        step_rows,
                        active_steps=active_steps,
                        after_step="models",
                        reason="Skipped after models failure.",
                    )
                    _render_dashboard()
                    return _finalize(ok=False, stage="models", error=models_error)

                _set_test_step(
                    step_rows,
                    step="models",
                    status="pass",
                    summary=f"Loaded {len(model_ids)} model(s).",
                    http=models_result["http_status"],
                    time_ms=_elapsed_ms(models_start),
                )
                _render_dashboard()

            if "probe" in active_step_set:
                current_stage = "probe"
                probe_start = time.perf_counter()
                _set_test_step(
                    step_rows,
                    step="probe",
                    status="running",
                    summary="Running single-model smoke probe.",
                    http=None,
                    time_ms=None,
                )
                _render_dashboard()

                selected_model, selection_strategy = _select_probe_model(
                    explicit_model=getattr(_args, "model", None),
                    models=model_ids,
                )
                if selected_model is None:
                    probe_error = _build_normalized_error(
                        kind="response_parse_error",
                        http_status=models_result["http_status"],
                        error_type="no_probe_model",
                        error_code=None,
                        message="No model available for probe.",
                        retryable=False,
                        key=key,
                    )
                    _set_test_step(
                        step_rows,
                        step="probe",
                        status="fail",
                        summary=probe_error["message"],
                        http=models_result["http_status"],
                        time_ms=_elapsed_ms(probe_start),
                    )
                    _render_dashboard()
                    return _finalize(ok=False, stage="probe", error=probe_error)

                probe_result, used_chat_fallback = _probe_single_model(
                    endpoint_base=endpoint_base,
                    key=key,
                    model_id=selected_model,
                    no_chat_fallback=bool(getattr(_args, "no_chat_fallback", False)),
                )
                probe_endpoint = (
                    "chat/completions" if used_chat_fallback else "responses"
                )
                probe_time = _elapsed_ms(probe_start)

                if probe_result["ok"]:
                    probe_path = (
                        "chat/completions fallback"
                        if used_chat_fallback
                        else "responses"
                    )
                    _set_test_step(
                        step_rows,
                        step="probe",
                        status="pass",
                        summary=f"Probe succeeded via {probe_path} ({selection_strategy}).",
                        http=probe_result["http_status"],
                        time_ms=probe_time,
                    )
                    _render_dashboard()
                else:
                    probe_error = probe_result["error"]
                    if probe_error is None:
                        probe_error = _build_normalized_error(
                            kind="http_error",
                            http_status=probe_result["http_status"],
                            error_type="probe_failed",
                            error_code=None,
                            message="Probe request failed.",
                            retryable=False,
                            key=key,
                        )
                    probe_summary = probe_error["message"]
                    if used_chat_fallback:
                        probe_summary = f"Fallback probe failed: {probe_summary}"
                    _set_test_step(
                        step_rows,
                        step="probe",
                        status="fail",
                        summary=probe_summary,
                        http=probe_error["http_status"],
                        time_ms=probe_time,
                    )
                    _render_dashboard()
                    return _finalize(ok=False, stage="probe", error=probe_error)

            return _finalize(ok=True)
        except ValueError as exc:
            usage_error = _build_normalized_error(
                kind="usage_error",
                http_status=None,
                error_type="invalid_arguments",
                error_code=None,
                message=_normalize_optional_text(exc) or "Invalid arguments.",
                retryable=False,
                key=key_for_errors,
            )
            return _finalize(
                ok=False,
                stage=current_stage,
                error=usage_error,
                exception=exc,
                usage_error=True,
            )
        except KeyboardInterrupt as exc:
            interrupt_error = _build_normalized_error(
                kind="interrupt",
                http_status=None,
                error_type="keyboard_interrupt",
                error_code=None,
                message="Interrupted by user.",
                retryable=False,
                key=key_for_errors,
            )
            return _finalize(
                ok=False,
                stage=current_stage,
                error=interrupt_error,
                exception=exc,
            )
        except Exception as exc:
            internal_error = _build_normalized_error(
                kind="internal_error",
                http_status=None,
                error_type=type(exc).__name__.lower(),
                error_code=None,
                message=_normalize_optional_text(exc) or "Unexpected internal error.",
                retryable=False,
                key=key_for_errors,
            )
            return _finalize(
                ok=False,
                stage=current_stage,
                error=internal_error,
                exception=exc,
            )


def _read_stdin_for_chat_prompt() -> tuple[bool, str | None]:
    stdin = sys.stdin
    is_tty = bool(getattr(stdin, "isatty", lambda: True)())
    if is_tty:
        return False, None

    raw_stdin = stdin.read()
    return True, _normalize_optional_text(raw_stdin)


def _resolve_chat_prompt(message: str | None) -> tuple[str, bool]:
    normalized_message = _normalize_optional_text(message)
    stdin_is_piped, stdin_prompt = _read_stdin_for_chat_prompt()

    if normalized_message is not None and stdin_is_piped:
        raise ValueError(
            "Ambiguous input: provide either positional message or piped stdin."
        )

    if normalized_message is not None:
        return normalized_message, stdin_is_piped

    if stdin_is_piped and stdin_prompt is not None:
        return stdin_prompt, stdin_is_piped

    raise ValueError(
        "Missing prompt: provide a positional message or non-empty piped stdin."
    )


def _select_chat_streaming(*, no_stream: bool, is_tty: bool) -> bool:
    if no_stream:
        return False
    return bool(is_tty)


def _build_chat_responses_payload(
    *,
    prompt: str,
    model_id: str,
    stream: bool,
) -> dict[str, object]:
    return {
        "model": model_id,
        "input": prompt,
        "stream": stream,
    }


def _build_chat_completions_payload(
    *,
    prompt: str,
    model_id: str,
    stream: bool,
) -> dict[str, object]:
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
    }


def _extract_stream_delta(payload: _JsonPayload) -> str | None:
    if not isinstance(payload, dict):
        return None

    raw_delta_text = payload.get("delta")
    if isinstance(raw_delta_text, str):
        return raw_delta_text

    choices = payload.get("choices")
    if isinstance(choices, list) and len(choices) > 0 and isinstance(choices[0], dict):
        first_choice = choices[0]
        raw_delta = first_choice.get("delta")
        if isinstance(raw_delta, dict):
            raw_content = raw_delta.get("content")
            if isinstance(raw_content, str):
                return raw_content
        raw_direct_content = first_choice.get("content")
        if isinstance(raw_direct_content, str):
            return raw_direct_content

    return None


def _is_stream_completion_event(payload: _JsonPayload, raw_data: str) -> bool:
    normalized_raw_data = raw_data.strip().lower()
    if normalized_raw_data == "[done]":
        return True

    if isinstance(payload, dict):
        event_type = (_normalize_optional_text(payload.get("type")) or "").lower()
        if event_type in {
            "response.completed",
            "response.complete",
            "message.stop",
            "done",
        }:
            return True

        choices = payload.get("choices")
        if (
            isinstance(choices, list)
            and len(choices) > 0
            and isinstance(choices[0], dict)
        ):
            finish_reason = _normalize_optional_text(choices[0].get("finish_reason"))
            if finish_reason is not None:
                return True

    return False


def _consume_chat_stream_lines(
    lines: Iterable[bytes],
    *,
    on_delta: Callable[[str], None] | None = None,
) -> tuple[str, bool]:
    deltas: list[str] = []
    completed = False

    for raw_line in lines:
        decoded_line = raw_line.decode("utf-8", errors="replace").strip()
        if decoded_line == "" or decoded_line.startswith(":"):
            continue
        if not decoded_line.lower().startswith("data:"):
            continue

        data_text = decoded_line[5:].strip()
        if data_text == "":
            continue

        if data_text == "[DONE]":
            completed = True
            break

        try:
            payload = cast(_JsonPayload, json.loads(data_text))
        except json.JSONDecodeError:
            continue

        delta = _extract_stream_delta(payload)
        if delta is not None:
            deltas.append(delta)
            if on_delta is not None:
                on_delta(delta)

        if _is_stream_completion_event(payload, data_text):
            completed = True
            break

    return "".join(deltas), completed


def _request_chat_stream_once(
    *,
    url: str,
    key: str | None,
    payload: dict[str, object],
    on_delta: Callable[[str], None] | None = None,
    timeout: float = _DEFAULT_HTTP_TIMEOUT_SECONDS,
) -> tuple[bool, str, int | None, _NormalizedError | None]:
    normalized_key = (key or "").strip()
    key_for_error = normalized_key or None
    request = _build_json_request(
        url=url,
        method="POST",
        key=normalized_key,
        payload=payload,
        headers={"Accept": "text/event-stream"},
    )
    normalized_timeout = timeout if timeout > 0 else _DEFAULT_HTTP_TIMEOUT_SECONDS

    try:
        response_context = cast(
            _UrlOpenContextManager,
            urllib.request.urlopen(request, timeout=normalized_timeout),
        )
        with response_context as response:
            status_code = int(response.getcode())
            stream_text, _ = _consume_chat_stream_lines(
                cast(Iterable[bytes], cast(object, response)),
                on_delta=on_delta,
            )
        return True, stream_text, status_code, None
    except HTTPError as error:
        normalized_error = _normalize_http_error(error, key=key_for_error)
        return False, "", normalized_error["http_status"], normalized_error
    except (URLError, TimeoutError, socket.timeout) as error:
        normalized_error = _normalize_transport_error(error, key=key_for_error)
        return False, "", None, normalized_error


def _chat_stream_with_fallback(
    *,
    endpoint_base: str,
    key: str,
    prompt: str,
    model_id: str,
    on_delta: Callable[[str], None] | None = None,
) -> tuple[bool, str, int | None, _NormalizedError | None, bool]:
    responses_payload = _build_chat_responses_payload(
        prompt=prompt,
        model_id=model_id,
        stream=True,
    )
    ok, text, status_code, error = _request_chat_stream_once(
        url=f"{endpoint_base}/responses",
        key=key,
        payload=responses_payload,
        on_delta=on_delta,
    )
    if ok:
        return True, text, status_code, None, False
    if error is None:
        return False, text, status_code, error, False

    should_fallback = _should_try_endpoint_fallback(
        status_code=error.get("http_status"),
        error_type=error.get("error_type"),
        error_message=error.get("message"),
    )
    if not should_fallback:
        return False, text, status_code, error, False

    completions_payload = _build_chat_completions_payload(
        prompt=prompt,
        model_id=model_id,
        stream=True,
    )
    fallback_ok, fallback_text, fallback_status, fallback_error = (
        _request_chat_stream_once(
            url=f"{endpoint_base}/chat/completions",
            key=key,
            payload=completions_payload,
            on_delta=on_delta,
        )
    )
    return fallback_ok, fallback_text, fallback_status, fallback_error, True


def _extract_non_stream_text_from_responses(payload: _JsonPayload) -> str | None:
    if not isinstance(payload, dict):
        return None

    output_text = _normalize_optional_text(payload.get("output_text"))
    if output_text is not None:
        return output_text

    output_field = payload.get("output")
    if not isinstance(output_field, list):
        return None

    chunks: list[str] = []
    for output_item in output_field:
        if not isinstance(output_item, dict):
            continue
        content_list = output_item.get("content")
        if not isinstance(content_list, list):
            continue
        for content_item in content_list:
            if not isinstance(content_item, dict):
                continue
            content_text = _normalize_optional_text(content_item.get("text"))
            if content_text is not None:
                chunks.append(content_text)

    if len(chunks) == 0:
        return None
    return "".join(chunks)


def _extract_non_stream_text_from_chat_completions(payload: _JsonPayload) -> str | None:
    if not isinstance(payload, dict):
        return None

    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        return None

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None

    message = first_choice.get("message")
    if isinstance(message, dict):
        content = _normalize_optional_text(message.get("content"))
        if content is not None:
            return content

    return _normalize_optional_text(first_choice.get("text"))


def _chat_non_stream_with_fallback(
    *,
    endpoint_base: str,
    key: str,
    prompt: str,
    model_id: str,
) -> tuple[_RequestResult, bool]:
    responses_result = _request_json_with_retry(
        url=f"{endpoint_base}/responses",
        method="POST",
        key=key,
        payload=_build_chat_responses_payload(
            prompt=prompt,
            model_id=model_id,
            stream=False,
        ),
    )
    if responses_result["ok"]:
        return responses_result, False

    normalized_error = responses_result["error"]
    if normalized_error is None:
        return responses_result, False

    should_fallback = _should_try_endpoint_fallback(
        status_code=normalized_error.get("http_status"),
        error_type=normalized_error.get("error_type"),
        error_message=normalized_error.get("message"),
    )
    if not should_fallback:
        return responses_result, False

    fallback_result = _request_json_with_retry(
        url=f"{endpoint_base}/chat/completions",
        method="POST",
        key=key,
        payload=_build_chat_completions_payload(
            prompt=prompt,
            model_id=model_id,
            stream=False,
        ),
    )
    return fallback_result, True


def _extract_chat_finish_reason(payload: _JsonPayload) -> str | None:
    if not isinstance(payload, dict):
        return None

    direct_finish_reason = _normalize_optional_text(payload.get("finish_reason"))
    if direct_finish_reason is not None:
        return direct_finish_reason

    status_finish_reason = _normalize_optional_text(payload.get("status"))
    if status_finish_reason is not None:
        return status_finish_reason

    choices = payload.get("choices")
    if isinstance(choices, list) and len(choices) > 0 and isinstance(choices[0], dict):
        return _normalize_optional_text(choices[0].get("finish_reason"))

    return None


def _extract_chat_usage(payload: _JsonPayload) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        return None

    raw_usage = payload.get("usage")
    if not isinstance(raw_usage, dict):
        return None

    return cast(Dict[str, object], dict(raw_usage))


def _emit_chat_json(
    *,
    output_precedence: _OutputPrecedence,
    ok: bool,
    exit_code: int,
    endpoint: str | None,
    resolved_endpoint_base: str | None,
    model: str,
    stream: bool,
    text: str,
    finish_reason: str | None,
    usage: dict[str, object] | None,
    started_at: float,
    finished_at: float,
    error: _NormalizedError | None,
) -> None:
    if not output_precedence["json"]:
        return

    payload = {
        "command": "chat",
        "ok": ok,
        "exit_code": exit_code,
        "endpoint": endpoint,
        "resolved_endpoint_base": resolved_endpoint_base,
        "model": model,
        "stream": stream,
        "text": text,
        "finish_reason": finish_reason,
        "usage": usage,
        "started_at": started_at,
        "finished_at": finished_at,
        "error": error,
    }
    utils.console.print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _emit_list_json(
    *,
    output_precedence: _OutputPrecedence,
    ok: bool,
    exit_code: int,
    endpoint: str | None,
    resolved_endpoint_base: str | None,
    models: list[_ListModel],
    started_at: float,
    finished_at: float,
    error: _NormalizedError | None,
) -> None:
    if not output_precedence["json"]:
        return

    payload = {
        "command": "list",
        "ok": ok,
        "exit_code": exit_code,
        "endpoint": endpoint,
        "resolved_endpoint_base": resolved_endpoint_base,
        "model_count": len(models),
        "models": models,
        "started_at": started_at,
        "finished_at": finished_at,
        "error": error,
    }
    utils.console.print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def run_chat(_args: argparse.Namespace) -> int:
    started_at = time.time()
    output_precedence = _resolve_output_precedence(
        quiet=bool(getattr(_args, "quiet", False)),
        json_requested=bool(getattr(_args, "json", False)),
    )
    current_stage = "chat"
    endpoint, key = _validate_endpoint_and_key_args(
        getattr(_args, "endpoint", None),
        getattr(_args, "key", None),
    )
    endpoint_candidates: list[str] = []
    endpoint_base: str | None = None
    key_for_errors: str | None = key
    model_id = (
        _normalize_optional_text(getattr(_args, "model", None)) or _DEFAULT_CHAT_MODEL
    )
    stdout_is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    stream_enabled = _select_chat_streaming(
        no_stream=bool(getattr(_args, "no_stream", False)),
        is_tty=stdout_is_tty,
    )

    def _finalize(
        *,
        ok: bool,
        stream: bool,
        text: str,
        finish_reason: str | None,
        usage: dict[str, object] | None,
        error: _NormalizedError | None,
        exception: BaseException | None = None,
        usage_error: bool = False,
    ) -> int:
        finished_at = time.time()
        exit_code = _resolve_openai_exit_code(
            ok=ok,
            command="chat",
            stage=current_stage,
            error=error,
            exception=exception,
            usage_error=usage_error,
        )
        _emit_chat_json(
            output_precedence=output_precedence,
            ok=ok,
            exit_code=exit_code,
            endpoint=endpoint,
            resolved_endpoint_base=endpoint_base,
            model=model_id,
            stream=stream,
            text=text,
            finish_reason=finish_reason,
            usage=usage,
            started_at=started_at,
            finished_at=finished_at,
            error=error,
        )
        _emit_human_error(
            output_precedence=output_precedence,
            command="chat",
            error=error,
            endpoint=endpoint_base,
        )
        return exit_code

    try:
        endpoint_candidates = _resolve_endpoint_candidates(endpoint)
        endpoint_base = endpoint_candidates[0]

        prompt, _ = _resolve_chat_prompt(getattr(_args, "message", None))

        if stream_enabled:
            emitted_stream_delta = False
            stream_delta_renderer: Callable[[str], None] | None = None
            if output_precedence["human"]:

                def _render_stream_delta(delta_text: str) -> None:
                    nonlocal emitted_stream_delta
                    emitted_stream_delta = True
                    _render_chat_plain_text(delta_text, append=True)

                stream_delta_renderer = _render_stream_delta

            candidate_index = 0
            endpoint_base = endpoint_candidates[candidate_index]
            ok, text, _http_status, error, _used_fallback = _chat_stream_with_fallback(
                endpoint_base=endpoint_base,
                key=key,
                prompt=prompt,
                model_id=model_id,
                on_delta=stream_delta_renderer,
            )
            if not ok and _should_retry_raw_endpoint_candidate(
                endpoint_candidates=endpoint_candidates,
                candidate_index=candidate_index,
                error=error,
            ):
                candidate_index = 1
                endpoint_base = endpoint_candidates[candidate_index]
                ok, text, _http_status, error, _used_fallback = (
                    _chat_stream_with_fallback(
                        endpoint_base=endpoint_base,
                        key=key,
                        prompt=prompt,
                        model_id=model_id,
                        on_delta=stream_delta_renderer,
                    )
                )
            if not ok:
                if output_precedence["human"] and emitted_stream_delta:
                    _render_chat_plain_text("")
                return _finalize(
                    ok=False,
                    stream=True,
                    text="",
                    finish_reason=None,
                    usage=None,
                    error=error,
                )

            if output_precedence["human"]:
                _render_chat_plain_text("")
            return _finalize(
                ok=True,
                stream=True,
                text=text,
                finish_reason=None,
                usage=None,
                error=None,
            )

        candidate_index = 0
        endpoint_base = endpoint_candidates[candidate_index]
        non_stream_result, used_fallback = _chat_non_stream_with_fallback(
            endpoint_base=endpoint_base,
            key=key,
            prompt=prompt,
            model_id=model_id,
        )
        if not non_stream_result["ok"] and _should_retry_raw_endpoint_candidate(
            endpoint_candidates=endpoint_candidates,
            candidate_index=candidate_index,
            error=non_stream_result["error"],
        ):
            candidate_index = 1
            endpoint_base = endpoint_candidates[candidate_index]
            non_stream_result, used_fallback = _chat_non_stream_with_fallback(
                endpoint_base=endpoint_base,
                key=key,
                prompt=prompt,
                model_id=model_id,
            )
        if not non_stream_result["ok"]:
            return _finalize(
                ok=False,
                stream=False,
                text="",
                finish_reason=None,
                usage=None,
                error=non_stream_result["error"],
            )

        response_payload = non_stream_result["data"]
        text: str | None
        if used_fallback:
            text = _extract_non_stream_text_from_chat_completions(response_payload)
        else:
            text = _extract_non_stream_text_from_responses(response_payload)
        rendered_text = text or ""
        finish_reason = _extract_chat_finish_reason(response_payload)
        usage = _extract_chat_usage(response_payload)

        if output_precedence["human"]:
            _render_chat_plain_text(rendered_text)
        return _finalize(
            ok=True,
            stream=False,
            text=rendered_text,
            finish_reason=finish_reason,
            usage=usage,
            error=None,
        )
    except ValueError as exc:
        usage_error = _build_normalized_error(
            kind="usage_error",
            http_status=None,
            error_type="invalid_arguments",
            error_code=None,
            message=_normalize_optional_text(exc) or "Invalid arguments.",
            retryable=False,
            key=key_for_errors,
        )
        return _finalize(
            ok=False,
            stream=stream_enabled,
            text="",
            finish_reason=None,
            usage=None,
            error=usage_error,
            exception=exc,
            usage_error=True,
        )
    except KeyboardInterrupt as exc:
        interrupt_error = _build_normalized_error(
            kind="interrupt",
            http_status=None,
            error_type="keyboard_interrupt",
            error_code=None,
            message="Interrupted by user.",
            retryable=False,
            key=key_for_errors,
        )
        return _finalize(
            ok=False,
            stream=stream_enabled,
            text="",
            finish_reason=None,
            usage=None,
            error=interrupt_error,
            exception=exc,
        )
    except Exception as exc:
        internal_error = _build_normalized_error(
            kind="internal_error",
            http_status=None,
            error_type=type(exc).__name__.lower(),
            error_code=None,
            message=_normalize_optional_text(exc) or "Unexpected internal error.",
            retryable=False,
            key=key_for_errors,
        )
        return _finalize(
            ok=False,
            stream=stream_enabled,
            text="",
            finish_reason=None,
            usage=None,
            error=internal_error,
            exception=exc,
        )


def run_list(_args: argparse.Namespace) -> int:
    started_at = time.time()
    output_precedence = _resolve_output_precedence(
        quiet=bool(getattr(_args, "quiet", False)),
        json_requested=bool(getattr(_args, "json", False)),
    )
    current_stage = "models"
    endpoint, key = _validate_endpoint_and_key_args(
        getattr(_args, "endpoint", None),
        getattr(_args, "key", None),
    )
    endpoint_candidates: list[str] = []
    endpoint_base: str | None = None
    key_for_errors: str | None = key
    resolved_models: list[_ListModel] = []

    def _finalize(
        *,
        ok: bool,
        error: _NormalizedError | None = None,
        exception: BaseException | None = None,
        usage_error: bool = False,
    ) -> int:
        finished_at = time.time()
        exit_code = _resolve_openai_exit_code(
            ok=ok,
            command="list",
            stage=current_stage,
            error=error,
            exception=exception,
            usage_error=usage_error,
        )
        _emit_list_json(
            output_precedence=output_precedence,
            ok=ok,
            exit_code=exit_code,
            endpoint=endpoint,
            resolved_endpoint_base=endpoint_base,
            models=resolved_models,
            started_at=started_at,
            finished_at=finished_at,
            error=error,
        )
        _emit_human_error(
            output_precedence=output_precedence,
            command="list",
            error=error,
            endpoint=endpoint_base,
        )
        if ok and output_precedence["human"]:
            _render_list_table_and_summary(resolved_models, endpoint=endpoint_base)
        return exit_code

    try:
        endpoint_candidates = _resolve_endpoint_candidates(endpoint)
        endpoint_base = endpoint_candidates[0]

        models_result = _request_json_with_retry(
            url=f"{endpoint_base}/models",
            method="GET",
            key=key,
        )
        if not models_result["ok"] and _should_retry_raw_endpoint_candidate(
            endpoint_candidates=endpoint_candidates,
            candidate_index=0,
            error=models_result["error"],
        ):
            endpoint_base = endpoint_candidates[1]
            models_result = _request_json_with_retry(
                url=f"{endpoint_base}/models",
                method="GET",
                key=key,
            )
        if not models_result["ok"]:
            models_error = models_result["error"]
            if models_error is None:
                models_error = _build_normalized_error(
                    kind="http_error",
                    http_status=models_result["http_status"],
                    error_type="model_list_failed",
                    error_code=None,
                    message="Failed to fetch model list.",
                    retryable=False,
                    key=key,
                )
            return _finalize(ok=False, error=models_error)

        resolved_models = _extract_list_models(models_result["data"])
        if len(resolved_models) == 0:
            models_error = _build_normalized_error(
                kind="response_parse_error",
                http_status=models_result["http_status"],
                error_type="no_models_available",
                error_code=None,
                message="No models available from endpoint.",
                retryable=False,
                key=key,
            )
            return _finalize(ok=False, error=models_error)

        return _finalize(ok=True)
    except ValueError as exc:
        usage_error = _build_normalized_error(
            kind="usage_error",
            http_status=None,
            error_type="invalid_arguments",
            error_code=None,
            message=_normalize_optional_text(exc) or "Invalid arguments.",
            retryable=False,
            key=key_for_errors,
        )
        return _finalize(
            ok=False,
            error=usage_error,
            exception=exc,
            usage_error=True,
        )
    except KeyboardInterrupt as exc:
        interrupt_error = _build_normalized_error(
            kind="interrupt",
            http_status=None,
            error_type="keyboard_interrupt",
            error_code=None,
            message="Interrupted by user.",
            retryable=False,
            key=key_for_errors,
        )
        return _finalize(
            ok=False,
            error=interrupt_error,
            exception=exc,
        )
    except Exception as exc:
        internal_error = _build_normalized_error(
            kind="internal_error",
            http_status=None,
            error_type=type(exc).__name__.lower(),
            error_code=None,
            message=_normalize_optional_text(exc) or "Unexpected internal error.",
            retryable=False,
            key=key_for_errors,
        )
        return _finalize(
            ok=False,
            error=internal_error,
            exception=exc,
        )
