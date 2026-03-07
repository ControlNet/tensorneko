# pyright: reportUnknownMemberType=false, reportMissingImports=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownLambdaType=false

import argparse
import copy
import io
import json
import os
import socket
import unittest
from email.message import Message
from typing import Protocol, cast, override
from urllib.error import HTTPError, URLError
from unittest.mock import MagicMock, call, patch

import tensorneko_tool.openai as openai_module


class _OpenAIRootArgs(Protocol):
    sub_command: str
    openai_sub_command: str | None


class _OpenAITestArgs(_OpenAIRootArgs, Protocol):
    mode: str
    json: bool
    no_live: bool
    fail_fast: bool
    model: str | None
    no_chat_fallback: bool
    endpoint: str | None
    key: str | None
    func: object


class _OpenAIChatArgs(_OpenAIRootArgs, Protocol):
    message: str | None
    no_stream: bool
    json: bool
    model: str | None
    endpoint: str | None
    key: str | None
    func: object


class _OpenAIListArgs(_OpenAIRootArgs, Protocol):
    json: bool
    endpoint: str | None
    key: str | None
    func: object


def _as_root_args(value: argparse.Namespace) -> _OpenAIRootArgs:
    return cast(_OpenAIRootArgs, cast(object, value))


def _as_test_args(value: argparse.Namespace) -> _OpenAITestArgs:
    return cast(_OpenAITestArgs, cast(object, value))


def _as_chat_args(value: argparse.Namespace) -> _OpenAIChatArgs:
    return cast(_OpenAIChatArgs, cast(object, value))


def _as_list_args(value: argparse.Namespace) -> _OpenAIListArgs:
    return cast(_OpenAIListArgs, cast(object, value))


def _table_column_cells(table: openai_module.Table, column_index: int) -> list[str]:
    if column_index >= len(table.columns):
        return []
    return cast(list[str], getattr(table.columns[column_index], "_cells", []))


class TestOpenAIParserTree(unittest.TestCase):
    def _build_parser(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="sub_command")
        openai_module.register_subparser(subparsers)
        return parser, subparsers

    def test_register_subparser_creates_openai_parent_and_nested_commands(self):
        parser, subparsers = self._build_parser()

        self.assertIn("openai", subparsers.choices)

        args_test = _as_test_args(parser.parse_args(["openai", "test"]))
        args_chat = _as_chat_args(parser.parse_args(["openai", "chat"]))
        args_list = _as_list_args(parser.parse_args(["openai", "list"]))

        self.assertEqual(args_test.openai_sub_command, "test")
        self.assertEqual(args_chat.openai_sub_command, "chat")
        self.assertEqual(args_list.openai_sub_command, "list")

    def test_nested_subcommands_bind_handlers_via_set_defaults(self):
        parser, _ = self._build_parser()

        args_test = _as_test_args(parser.parse_args(["openai", "test"]))
        self.assertIs(args_test.func, openai_module.run_test)

        args_chat = _as_chat_args(parser.parse_args(["openai", "chat"]))
        self.assertIs(args_chat.func, openai_module.run_chat)

        args_list = _as_list_args(parser.parse_args(["openai", "list"]))
        self.assertIs(args_list.func, openai_module.run_list)

    def test_openai_without_nested_subcommand_has_deterministic_namespace(self):
        parser, _ = self._build_parser()

        args = _as_root_args(parser.parse_args(["openai"]))

        self.assertEqual(args.sub_command, "openai")
        self.assertIsNone(args.openai_sub_command)
        self.assertFalse(hasattr(args, "func"))


class TestOpenAIParserDefaults(unittest.TestCase):
    def _build_parser(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="sub_command")
        openai_module.register_subparser(subparsers)
        return parser

    def test_test_subcommand_defaults_and_custom_values(self):
        parser = self._build_parser()

        args_default = _as_test_args(parser.parse_args(["openai", "test"]))
        self.assertEqual(args_default.mode, "all")
        self.assertFalse(args_default.json)
        self.assertFalse(args_default.no_live)
        self.assertFalse(args_default.fail_fast)
        self.assertIsNone(args_default.model)
        self.assertFalse(args_default.no_chat_fallback)
        self.assertIsNone(args_default.endpoint)
        self.assertIsNone(args_default.key)

        args_custom = _as_test_args(
            parser.parse_args(
                [
                    "openai",
                    "test",
                    "--mode",
                    "probe",
                    "--json",
                    "--no-live",
                    "--fail-fast",
                    "--model",
                    "gpt-4.1-mini",
                    "--no-chat-fallback",
                    "--endpoint",
                    "https://example.openai.local/v1",
                    "--key",
                    "token-123",
                ]
            )
        )
        self.assertEqual(args_custom.mode, "probe")
        self.assertTrue(args_custom.json)
        self.assertTrue(args_custom.no_live)
        self.assertTrue(args_custom.fail_fast)
        self.assertEqual(args_custom.model, "gpt-4.1-mini")
        self.assertTrue(args_custom.no_chat_fallback)
        self.assertEqual(args_custom.endpoint, "https://example.openai.local/v1")
        self.assertEqual(args_custom.key, "token-123")

    def test_test_subcommand_rejects_invalid_mode(self):
        parser = self._build_parser()

        with self.assertRaises(SystemExit):
            _ = parser.parse_args(["openai", "test", "--mode", "invalid"])

    def test_no_live_help_mentions_dashboard_rendering(self):
        parser = self._build_parser()

        def _find_subparser(
            actions: list[argparse.Action], name: str
        ) -> argparse.ArgumentParser:
            for action in actions:
                choices = getattr(action, "choices", None)
                if isinstance(choices, dict) and name in choices:
                    return choices[name]
            raise AssertionError(f"subparser '{name}' not present")

        openai_parser = _find_subparser(parser._actions, "openai")
        test_parser = _find_subparser(openai_parser._actions, "test")

        no_live_action = next(
            action
            for action in test_parser._actions
            if getattr(action, "dest", None) == "no_live"
        )
        help_text = (no_live_action.help or "").lower()

        self.assertIn("dashboard", help_text)
        self.assertIn("live", help_text)
        self.assertIn("render", help_text)
        self.assertNotIn("network", help_text)

    def test_chat_subcommand_defaults_and_custom_values(self):
        parser = self._build_parser()

        args_default = _as_chat_args(parser.parse_args(["openai", "chat"]))
        self.assertIsNone(args_default.message)
        self.assertFalse(args_default.no_stream)
        self.assertFalse(args_default.json)
        self.assertIsNone(args_default.model)
        self.assertIsNone(args_default.endpoint)
        self.assertIsNone(args_default.key)

        args_custom = _as_chat_args(
            parser.parse_args(
                [
                    "openai",
                    "chat",
                    "hello world",
                    "--no-stream",
                    "--json",
                    "--model",
                    "gpt-4.1",
                    "--endpoint",
                    "https://example.openai.local/v1",
                    "--key",
                    "token-456",
                ]
            )
        )
        self.assertEqual(args_custom.message, "hello world")
        self.assertTrue(args_custom.no_stream)
        self.assertTrue(args_custom.json)
        self.assertEqual(args_custom.model, "gpt-4.1")
        self.assertEqual(args_custom.endpoint, "https://example.openai.local/v1")
        self.assertEqual(args_custom.key, "token-456")

    def test_list_subcommand_defaults_and_custom_values(self):
        parser = self._build_parser()

        args_default = _as_list_args(parser.parse_args(["openai", "list"]))
        self.assertFalse(args_default.json)
        self.assertIsNone(args_default.endpoint)
        self.assertIsNone(args_default.key)

        args_custom = _as_list_args(
            parser.parse_args(
                [
                    "openai",
                    "list",
                    "--json",
                    "--endpoint",
                    "https://example.openai.local/v1",
                    "--key",
                    "token-789",
                ]
            )
        )
        self.assertTrue(args_custom.json)
        self.assertEqual(args_custom.endpoint, "https://example.openai.local/v1")
        self.assertEqual(args_custom.key, "token-789")


class TestOpenAIEndpointResolver(unittest.TestCase):
    def test_resolve_endpoint_candidates_strips_trailing_slash_and_prefers_v1(self):
        candidates = openai_module._resolve_endpoint_candidates(
            "https://api.example.com/"
        )

        self.assertEqual(
            candidates,
            ["https://api.example.com/v1", "https://api.example.com"],
        )

    def test_resolve_endpoint_candidates_keeps_single_v1_candidate(self):
        candidates = openai_module._resolve_endpoint_candidates(
            "https://api.example.com/v1/"
        )

        self.assertEqual(candidates, ["https://api.example.com/v1"])

    def test_validate_endpoint_and_key_normalizes_defaults_without_env_fallback(
        self,
    ):
        with patch.dict(
            os.environ,
            {
                "OPENAI_BASE_URL": "https://env.example.com/v1",
                "OPENAI_API_KEY": "env-key",
            },
            clear=True,
        ):
            default_endpoint, default_key = (
                openai_module._validate_endpoint_and_key_args(
                    endpoint=None,
                    key=None,
                )
            )
            self.assertEqual(
                default_endpoint,
                openai_module._DEFAULT_OPENAI_ENDPOINT,
            )
            self.assertEqual(default_key, "")

            blank_endpoint, blank_key = openai_module._validate_endpoint_and_key_args(
                endpoint="   ",
                key="\n",
            )
            self.assertEqual(
                blank_endpoint,
                openai_module._DEFAULT_OPENAI_ENDPOINT,
            )
            self.assertEqual(blank_key, "")

            explicit_endpoint, missing_key = (
                openai_module._validate_endpoint_and_key_args(
                    endpoint="https://example.openai.local/v1",
                    key=None,
                )
            )
            self.assertEqual(explicit_endpoint, "https://example.openai.local/v1")
            self.assertEqual(missing_key, "")

            default_endpoint_with_key, normalized_key = (
                openai_module._validate_endpoint_and_key_args(
                    endpoint=None,
                    key="  token-with-padding  ",
                )
            )
            self.assertEqual(
                default_endpoint_with_key,
                openai_module._DEFAULT_OPENAI_ENDPOINT,
            )
            self.assertEqual(normalized_key, "token-with-padding")


class TestOpenAIFallbackSignals(unittest.TestCase):
    def test_should_try_endpoint_fallback_allows_endpoint_not_supported_statuses(self):
        for code in (404, 405, 410, 501):
            with self.subTest(status_code=code):
                self.assertTrue(
                    openai_module._should_try_endpoint_fallback(
                        status_code=code,
                        error_type=None,
                        error_message=None,
                    )
                )

    def test_should_try_endpoint_fallback_allows_explicit_endpoint_markers(self):
        self.assertTrue(
            openai_module._should_try_endpoint_fallback(
                status_code=400,
                error_type="invalid_request_error",
                error_message="endpoint_not_found: /chat/completions",
            )
        )

    def test_should_try_endpoint_fallback_disallows_auth_and_rate_limit_failures(self):
        for code in (401, 403, 429):
            with self.subTest(status_code=code):
                self.assertFalse(
                    openai_module._should_try_endpoint_fallback(
                        status_code=code,
                        error_type="invalid_request_error",
                        error_message="request failed",
                    )
                )

    def test_should_try_endpoint_fallback_disallows_model_failures(self):
        self.assertFalse(
            openai_module._should_try_endpoint_fallback(
                status_code=404,
                error_type="model_not_found",
                error_message="The model `gpt-test` does not exist",
            )
        )


def _http_error(
    *,
    status_code: int,
    body: bytes,
    message: str = "request-failed",
) -> HTTPError:
    return HTTPError(
        url="https://api.example.com/v1/test",
        code=status_code,
        msg=message,
        hdrs=Message(),
        fp=io.BytesIO(body),
    )


class _FakeJsonResponse:
    _body: bytes
    _status_code: int

    def __init__(self, *, body: bytes, status_code: int):
        self._body = body
        self._status_code = status_code

    def read(self) -> bytes:
        return self._body

    def getcode(self) -> int:
        return self._status_code


class _FakeResponseContext:
    _response: _FakeJsonResponse

    def __init__(self, response: _FakeJsonResponse):
        self._response = response

    def __enter__(self) -> _FakeJsonResponse:
        return self._response

    def __exit__(
        self,
        _exc_type: object,
        _exc_value: object,
        _traceback: object,
    ) -> None:
        return None


class TestOpenAIHttpLayer(unittest.TestCase):
    @patch("tensorneko_tool.openai.urllib.request.urlopen")
    def test_request_success_returns_parsed_json(self, mock_urlopen):
        response = _FakeJsonResponse(
            body=json.dumps({"ok": True, "id": "resp-1"}).encode("utf-8"),
            status_code=200,
        )
        mock_urlopen.return_value = _FakeResponseContext(response)

        result = openai_module._request_json_with_retry(
            url="https://api.example.com/v1/responses",
            method="POST",
            key="secret-token",
            payload={"model": "gpt-4.1-mini"},
            timeout=7.5,
            retries=0,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(result["http_status"], 200)
        self.assertEqual(result["data"], {"ok": True, "id": "resp-1"})
        self.assertIsNone(result["error"])

        req = mock_urlopen.call_args[0][0]
        timeout = mock_urlopen.call_args[1]["timeout"]
        self.assertEqual(timeout, 7.5)
        self.assertEqual(req.full_url, "https://api.example.com/v1/responses")
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.headers["Authorization"], "Bearer secret-token")
        self.assertEqual(req.headers["Content-type"], "application/json")
        self.assertEqual(req.headers["Accept"], "application/json")
        self.assertEqual(req.headers["User-agent"], "TensorNeko")
        self.assertEqual(
            json.loads(req.data.decode("utf-8")),
            {"model": "gpt-4.1-mini"},
        )

    @patch("tensorneko_tool.openai.urllib.request.urlopen")
    def test_missing_key_omits_authorization_header(self, mock_urlopen):
        def _exercise(blank_key: str | None):
            response = _FakeJsonResponse(
                body=json.dumps({"ok": True}).encode("utf-8"),
                status_code=200,
            )
            mock_urlopen.return_value = _FakeResponseContext(response)

            result = openai_module._request_json_with_retry(
                url="https://api.example.com/v1/responses",
                method="GET",
                key=blank_key,
            )

            self.assertTrue(result["ok"])
            req = mock_urlopen.call_args[0][0]
            self.assertNotIn("Authorization", req.headers)

        for blank_key in (None, "   "):
            with self.subTest(key=blank_key):
                _exercise(blank_key)
                mock_urlopen.reset_mock()

    @patch("tensorneko_tool.openai.urllib.request.urlopen")
    def test_http_error_non_json_body_is_normalized_and_redacted(self, mock_urlopen):
        error = _http_error(
            status_code=500,
            body=b"<html>upstream failed with token secret-token</html>",
            message="server-error",
        )
        self.addCleanup(error.close)
        mock_urlopen.side_effect = error

        result = openai_module._request_json_with_retry(
            url="https://api.example.com/v1/models",
            method="GET",
            key="secret-token",
            timeout=4.0,
            retries=0,
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(result["http_status"], 500)

        normalized_error = result["error"]
        self.assertIsNotNone(normalized_error)
        if normalized_error is None:
            self.fail("expected normalized error")
        self.assertEqual(
            set(normalized_error.keys()),
            {
                "kind",
                "http_status",
                "error_type",
                "error_code",
                "message",
                "retryable",
            },
        )
        self.assertEqual(normalized_error["kind"], "http_error")
        self.assertEqual(normalized_error["http_status"], 500)
        self.assertIsNone(normalized_error["error_type"])
        self.assertIsNone(normalized_error["error_code"])
        self.assertTrue(normalized_error["retryable"])
        self.assertNotIn("secret-token", normalized_error["message"])
        self.assertIn("upstream failed", normalized_error["message"])


class TestOpenAIRetryPolicy(unittest.TestCase):
    @patch("tensorneko_tool.openai.urllib.request.urlopen")
    def test_429_retries_then_returns_retry_exhausted_error(self, mock_urlopen):
        error1 = _http_error(
            status_code=429,
            body=json.dumps({"error": {"message": "too many requests"}}).encode(
                "utf-8"
            ),
            message="rate-limit",
        )
        error2 = _http_error(
            status_code=429,
            body=json.dumps({"error": {"message": "too many requests"}}).encode(
                "utf-8"
            ),
            message="rate-limit",
        )
        error3 = _http_error(
            status_code=429,
            body=json.dumps({"error": {"message": "too many requests"}}).encode(
                "utf-8"
            ),
            message="rate-limit",
        )
        self.addCleanup(error1.close)
        self.addCleanup(error2.close)
        self.addCleanup(error3.close)
        mock_urlopen.side_effect = [error1, error2, error3]

        result = openai_module._request_json_with_retry(
            url="https://api.example.com/v1/models",
            method="GET",
            key="k",
            timeout=3.0,
            retries=2,
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(mock_urlopen.call_count, 3)
        normalized_error = result["error"]
        self.assertIsNotNone(normalized_error)
        if normalized_error is None:
            self.fail("expected normalized error")
        self.assertEqual(normalized_error["http_status"], 429)
        self.assertTrue(normalized_error["retryable"])

    @patch("tensorneko_tool.openai.urllib.request.urlopen")
    def test_auth_error_does_not_retry(self, mock_urlopen):
        error = _http_error(
            status_code=401,
            body=json.dumps(
                {
                    "error": {
                        "message": "Incorrect API key provided",
                        "type": "invalid_api_key",
                        "code": "invalid_api_key",
                    }
                }
            ).encode("utf-8"),
            message="unauthorized",
        )
        self.addCleanup(error.close)
        mock_urlopen.side_effect = error

        result = openai_module._request_json_with_retry(
            url="https://api.example.com/v1/models",
            method="GET",
            key="token",
            retries=4,
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(mock_urlopen.call_count, 1)
        normalized_error = result["error"]
        self.assertIsNotNone(normalized_error)
        if normalized_error is None:
            self.fail("expected normalized error")
        self.assertFalse(normalized_error["retryable"])
        self.assertEqual(normalized_error["error_type"], "invalid_api_key")
        self.assertEqual(normalized_error["error_code"], "invalid_api_key")

    @patch("tensorneko_tool.openai.urllib.request.urlopen")
    def test_invalid_request_shape_error_does_not_retry(self, mock_urlopen):
        error = _http_error(
            status_code=400,
            body=json.dumps(
                {
                    "error": {
                        "message": "Invalid request payload",
                        "type": "invalid_request_error",
                        "code": "invalid_payload",
                    }
                }
            ).encode("utf-8"),
            message="bad-request",
        )
        self.addCleanup(error.close)
        mock_urlopen.side_effect = error

        result = openai_module._request_json_with_retry(
            url="https://api.example.com/v1/responses",
            method="POST",
            key="token",
            payload={"invalid": True},
            retries=4,
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(mock_urlopen.call_count, 1)
        normalized_error = result["error"]
        self.assertIsNotNone(normalized_error)
        if normalized_error is None:
            self.fail("expected normalized error")
        self.assertFalse(normalized_error["retryable"])
        self.assertEqual(normalized_error["error_type"], "invalid_request_error")
        self.assertEqual(normalized_error["error_code"], "invalid_payload")

    @patch("tensorneko_tool.openai.urllib.request.urlopen")
    def test_timeout_error_retries_then_succeeds(self, mock_urlopen):
        timed_out = URLError(socket.timeout("timed out"))
        response = _FakeJsonResponse(
            body=json.dumps({"id": "ok-after-retry"}).encode("utf-8"),
            status_code=200,
        )
        mock_urlopen.side_effect = [
            timed_out,
            _FakeResponseContext(response),
        ]

        result = openai_module._request_json_with_retry(
            url="https://api.example.com/v1/models",
            method="GET",
            key="token",
            retries=1,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(mock_urlopen.call_count, 2)
        self.assertEqual(result["data"], {"id": "ok-after-retry"})

    @patch("tensorneko_tool.openai.urllib.request.urlopen")
    def test_url_error_retries_until_exhausted(self, mock_urlopen):
        mock_urlopen.side_effect = [
            URLError("socket closed"),
            URLError("socket closed"),
        ]

        result = openai_module._request_json_with_retry(
            url="https://api.example.com/v1/models",
            method="GET",
            key="token",
            retries=1,
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(mock_urlopen.call_count, 2)
        normalized_error = result["error"]
        self.assertIsNotNone(normalized_error)
        if normalized_error is None:
            self.fail("expected normalized error")
        self.assertEqual(normalized_error["kind"], "transport_error")
        self.assertTrue(normalized_error["retryable"])


class TestOpenAIRenderContracts(unittest.TestCase):
    def test_canonical_status_vocabulary_is_locked(self):
        self.assertEqual(
            openai_module._TEST_STATUS_LABELS,
            {
                "queued": "🕒 QUEUED",
                "running": "🔄 RUNNING",
                "pass": "✅ PASS",
                "warn": "⚠️ WARN",
                "fail": "❌ FAIL",
                "skip": "⏭️ SKIP",
            },
        )

    def test_dashboard_contract_columns_and_step_order_are_locked(self):
        self.assertEqual(
            openai_module._TEST_DASHBOARD_COLUMNS,
            ("Step", "Status", "HTTP", "Time(ms)", "Summary"),
        )
        self.assertEqual(
            openai_module._TEST_STEP_ORDER,
            ("network", "auth", "models", "probe"),
        )

    def test_build_test_dashboard_table_respects_schema_and_step_order(self):
        table = openai_module._build_test_dashboard_table(
            {
                "probe": {
                    "status": "pass",
                    "http": 200,
                    "time_ms": 31,
                    "summary": "Probe request succeeded",
                },
                "network": {
                    "status": "pass",
                    "http": 200,
                    "time_ms": 12,
                    "summary": "DNS + TCP ok",
                },
                "auth": {
                    "status": "warn",
                    "http": 401,
                    "time_ms": 9,
                    "summary": "Token near expiry",
                },
            }
        )

        self.assertEqual(
            [column.header for column in table.columns],
            ["Step", "Status", "HTTP", "Time(ms)", "Summary"],
        )
        self.assertEqual(
            table.columns[0]._cells, ["network", "auth", "models", "probe"]
        )
        self.assertEqual(table.columns[1]._cells[0], "✅ PASS")
        self.assertEqual(table.columns[1]._cells[1], "⚠️ WARN")
        self.assertEqual(table.columns[1]._cells[2], "🕒 QUEUED")

    @patch("tensorneko_tool.openai.utils.console")
    def test_render_chat_plain_text_uses_shared_console_sink(self, mock_console):
        mock_console.print = MagicMock()
        with patch("builtins.print") as mock_print:
            openai_module._render_chat_plain_text("hello")

        mock_console.print.assert_called_once_with("hello")
        mock_print.assert_not_called()

    @patch("tensorneko_tool.openai.utils.console")
    def test_render_chat_plain_text_append_mode_uses_non_newline_sink(
        self, mock_console
    ):
        mock_console.print = MagicMock()
        openai_module._render_chat_plain_text("hello", append=True)
        mock_console.print.assert_called_once_with("hello", end="")

    @patch("tensorneko_tool.openai._build_list_summary_text")
    @patch("tensorneko_tool.openai._build_list_table")
    @patch("tensorneko_tool.openai.utils.console")
    def test_render_model_list_table_and_summary_flows_through_shared_console(
        self,
        mock_console,
        mock_build_table,
        mock_build_summary,
    ):
        mock_console.print = MagicMock()
        fake_table = object()
        models: list[openai_module._ListModel] = [
            {"id": "gpt-4.1-mini", "owned_by": "openai", "created": None},
            {"id": "gpt-4.1", "owned_by": "openai", "created": None},
        ]
        mock_build_table.return_value = fake_table
        mock_build_summary.return_value = (
            "Total models: 2\nEndpoint: https://api.example.com/v1"
        )

        openai_module._render_list_table_and_summary(models)

        mock_build_table.assert_called_once_with(models)
        mock_build_summary.assert_called_once_with(models, endpoint=None)
        self.assertEqual(mock_console.print.call_count, 2)
        mock_console.print.assert_any_call(fake_table)
        mock_console.print.assert_any_call(
            "Total models: 2\nEndpoint: https://api.example.com/v1"
        )


class TestOpenAIRenderModeSelection(unittest.TestCase):
    def test_select_render_mode_prefers_live_only_for_tty_and_live_enabled(self):
        self.assertEqual(
            openai_module._select_test_render_mode(no_live=False, is_tty=True),
            "live",
        )

    def test_select_render_mode_uses_static_when_no_live_flag_is_on(self):
        self.assertEqual(
            openai_module._select_test_render_mode(no_live=True, is_tty=True),
            "static",
        )

    def test_select_render_mode_uses_static_when_not_tty(self):
        self.assertEqual(
            openai_module._select_test_render_mode(no_live=False, is_tty=False),
            "static",
        )

    @patch("tensorneko_tool.openai._render_test_dashboard_live")
    @patch("tensorneko_tool.openai._render_test_dashboard_static")
    def test_render_test_dashboard_dispatches_to_live_path_when_live_mode(
        self,
        mock_render_static,
        mock_render_live,
    ):
        mode = openai_module._render_test_dashboard(
            {
                "network": {
                    "status": "running",
                    "http": None,
                    "time_ms": None,
                    "summary": "Resolving endpoint",
                }
            },
            no_live=False,
            is_tty=True,
        )

        self.assertEqual(mode, "live")
        mock_render_live.assert_called_once()
        mock_render_static.assert_not_called()

    @patch("tensorneko_tool.openai._render_test_dashboard_live")
    @patch("tensorneko_tool.openai._render_test_dashboard_static")
    def test_render_test_dashboard_dispatches_to_static_when_no_live_or_non_tty(
        self,
        mock_render_static,
        mock_render_live,
    ):
        for no_live, is_tty in ((True, True), (False, False)):
            with self.subTest(no_live=no_live, is_tty=is_tty):
                _ = openai_module._render_test_dashboard(
                    {
                        "network": {
                            "status": "queued",
                            "http": None,
                            "time_ms": None,
                            "summary": "Waiting",
                        }
                    },
                    no_live=no_live,
                    is_tty=is_tty,
                )

        self.assertEqual(mock_render_static.call_count, 2)
        mock_render_live.assert_not_called()

    @patch("tensorneko_tool.openai.Live")
    def test_non_tty_render_path_never_uses_live_refresh(self, mock_live):
        _ = openai_module._render_test_dashboard(
            {
                "network": {
                    "status": "queued",
                    "http": None,
                    "time_ms": None,
                    "summary": "Waiting",
                }
            },
            no_live=False,
            is_tty=False,
        )

        mock_live.assert_not_called()


def _normalized_error(
    *,
    kind: str,
    http_status: int | None = None,
    error_type: str | None = None,
) -> openai_module._NormalizedError:
    return {
        "kind": kind,
        "http_status": http_status,
        "error_type": error_type,
        "error_code": None,
        "message": "request failed",
        "retryable": False,
    }


def _request_ok(
    *,
    http_status: int,
    data: openai_module._JsonPayload,
) -> openai_module._RequestResult:
    return {
        "ok": True,
        "attempts": 1,
        "http_status": http_status,
        "data": data,
        "error": None,
    }


def _request_fail(
    *,
    error: openai_module._NormalizedError,
) -> openai_module._RequestResult:
    return {
        "ok": False,
        "attempts": 1,
        "http_status": error["http_status"],
        "data": None,
        "error": error,
    }


def _make_test_args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "mode": "all",
        "json": False,
        "no_live": True,
        "fail_fast": False,
        "model": None,
        "no_chat_fallback": False,
        "endpoint": "https://api.example.com",
        "key": "test-key",
        "quiet": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _make_chat_args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "message": None,
        "no_stream": False,
        "json": False,
        "model": None,
        "endpoint": "https://api.example.com",
        "key": "test-key",
        "quiet": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _make_list_args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "json": False,
        "endpoint": "https://api.example.com",
        "key": "test-key",
        "quiet": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class _FakeStdin(io.StringIO):
    _is_tty: bool

    def __init__(self, content: str, *, is_tty: bool):
        super().__init__(content)
        self._is_tty = is_tty

    @override
    def isatty(self) -> bool:
        return self._is_tty


def _step_status(
    frame: dict[str, openai_module._TestStepRender],
    step: str,
) -> str | None:
    return cast(str | None, frame.get(step, {}).get("status"))


class TestOpenAIExitCodes(unittest.TestCase):
    def test_exit_code_table_contract_is_frozen(self):
        self.assertEqual(
            openai_module._OPENAI_EXIT_CODES,
            {
                "success": 0,
                "usage": 2,
                "network": 10,
                "auth": 20,
                "models": 30,
                "probe": 40,
                "chat": 50,
                "internal": 90,
                "interrupt": 130,
            },
        )

    def test_resolve_openai_exit_code_maps_success_and_usage(self):
        self.assertEqual(
            openai_module._resolve_openai_exit_code(
                ok=True,
                command="test",
            ),
            0,
        )
        self.assertEqual(
            openai_module._resolve_openai_exit_code(
                ok=False,
                command="test",
                usage_error=True,
            ),
            2,
        )

    def test_resolve_openai_exit_code_maps_network_and_auth(self):
        self.assertEqual(
            openai_module._resolve_openai_exit_code(
                ok=False,
                command="test",
                stage="network",
                error=_normalized_error(kind="transport_error"),
            ),
            10,
        )
        self.assertEqual(
            openai_module._resolve_openai_exit_code(
                ok=False,
                command="test",
                stage="auth",
                error=_normalized_error(
                    kind="http_error",
                    http_status=401,
                    error_type="invalid_api_key",
                ),
            ),
            20,
        )

    def test_resolve_openai_exit_code_maps_models_probe_and_chat_failures(self):
        self.assertEqual(
            openai_module._resolve_openai_exit_code(
                ok=False,
                command="list",
                error=_normalized_error(kind="http_error", http_status=500),
            ),
            30,
        )
        self.assertEqual(
            openai_module._resolve_openai_exit_code(
                ok=False,
                command="test",
                stage="probe",
                error=_normalized_error(kind="http_error", http_status=500),
            ),
            40,
        )
        self.assertEqual(
            openai_module._resolve_openai_exit_code(
                ok=False,
                command="chat",
                error=_normalized_error(kind="http_error", http_status=500),
            ),
            50,
        )

    def test_resolve_openai_exit_code_maps_interrupt_and_internal(self):
        self.assertEqual(
            openai_module._resolve_openai_exit_code(
                ok=False,
                command="chat",
                exception=KeyboardInterrupt(),
                error=_normalized_error(kind="transport_error"),
            ),
            130,
        )
        self.assertEqual(
            openai_module._resolve_openai_exit_code(
                ok=False,
                command="test",
                exception=RuntimeError("unexpected"),
            ),
            90,
        )

    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_omitted_key_is_not_treated_as_usage_error_in_runners(self, mock_request):
        auth_error = _normalized_error(
            kind="http_error",
            http_status=401,
            error_type="invalid_api_key",
        )
        cast(dict[str, object], auth_error)["error_code"] = "invalid_api_key"

        mock_request.return_value = _request_fail(error=auth_error)
        test_code = openai_module.run_test(
            _make_test_args(
                mode="auth",
                no_live=True,
                quiet=True,
                endpoint=None,
                key=None,
            )
        )
        self.assertEqual(test_code, openai_module._OPENAI_EXIT_CODES["auth"])
        self.assertNotEqual(test_code, openai_module._OPENAI_EXIT_CODES["usage"])
        self.assertEqual(mock_request.call_count, 1)
        self.assertEqual(
            mock_request.call_args_list[0].kwargs["url"],
            f"{openai_module._DEFAULT_OPENAI_ENDPOINT}/models",
        )
        self.assertEqual(mock_request.call_args_list[0].kwargs["key"], "")

        network_error = _normalized_error(kind="transport_error")
        mock_request.reset_mock()
        mock_request.return_value = _request_fail(error=network_error)
        chat_args = _make_chat_args(
            message="hello",
            no_stream=True,
            endpoint=None,
            key=None,
            quiet=True,
        )
        with patch("sys.stdin", _FakeStdin("", is_tty=True)):
            chat_code = openai_module.run_chat(chat_args)
        self.assertEqual(chat_code, openai_module._OPENAI_EXIT_CODES["network"])
        self.assertNotEqual(chat_code, openai_module._OPENAI_EXIT_CODES["usage"])
        self.assertEqual(mock_request.call_count, 1)
        self.assertEqual(
            mock_request.call_args_list[0].kwargs["url"],
            f"{openai_module._DEFAULT_OPENAI_ENDPOINT}/responses",
        )
        self.assertEqual(mock_request.call_args_list[0].kwargs["key"], "")

        models_error = _normalized_error(
            kind="http_error",
            http_status=500,
            error_type="server_error",
        )
        mock_request.reset_mock()
        mock_request.return_value = _request_fail(error=models_error)
        list_code = openai_module.run_list(
            _make_list_args(endpoint=None, key=None, quiet=True)
        )
        self.assertEqual(list_code, openai_module._OPENAI_EXIT_CODES["models"])
        self.assertNotEqual(list_code, openai_module._OPENAI_EXIT_CODES["usage"])
        self.assertEqual(mock_request.call_count, 1)
        self.assertEqual(
            mock_request.call_args_list[0].kwargs["url"],
            f"{openai_module._DEFAULT_OPENAI_ENDPOINT}/models",
        )
        self.assertEqual(mock_request.call_args_list[0].kwargs["key"], "")


class TestOpenAIOutputPrecedence(unittest.TestCase):
    def test_output_precedence_quiet_overrides_json(self):
        policy = openai_module._resolve_output_precedence(
            quiet=True, json_requested=True
        )

        self.assertTrue(policy["quiet"])
        self.assertFalse(policy["json"])
        self.assertFalse(policy["human"])

    def test_output_precedence_json_is_enabled_only_when_not_quiet(self):
        policy = openai_module._resolve_output_precedence(
            quiet=False, json_requested=True
        )

        self.assertFalse(policy["quiet"])
        self.assertTrue(policy["json"])
        self.assertFalse(policy["human"])

    def test_output_precedence_human_is_enabled_only_when_not_quiet_or_json(self):
        policy = openai_module._resolve_output_precedence(
            quiet=False, json_requested=False
        )

        self.assertFalse(policy["quiet"])
        self.assertFalse(policy["json"])
        self.assertTrue(policy["human"])


class TestOpenAITestPipeline(unittest.TestCase):
    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_all_mode_runs_full_chain_and_passes(self, mock_request):
        mock_request.side_effect = [
            _request_ok(
                http_status=200,
                data={
                    "data": [
                        {"id": "gpt-4.1-mini"},
                    ]
                },
            ),
            _request_ok(
                http_status=200,
                data={"id": "resp-1"},
            ),
        ]

        frames: list[dict[str, openai_module._TestStepRender]] = []

        def _capture_render(
            step_rows: dict[str, openai_module._TestStepRender],
            *,
            no_live: bool,
            is_tty: bool | None = None,
        ) -> str:
            self.assertTrue(no_live)
            _ = is_tty
            frames.append(copy.deepcopy(step_rows))
            return "static"

        with patch(
            "tensorneko_tool.openai._render_test_dashboard",
            side_effect=_capture_render,
        ):
            result = openai_module.run_test(_make_test_args(mode="all", no_live=True))

        self.assertEqual(result, 0)
        self.assertEqual(mock_request.call_count, 2)
        self.assertEqual(
            mock_request.call_args_list[0].kwargs["url"],
            "https://api.example.com/v1/models",
        )
        self.assertEqual(mock_request.call_args_list[0].kwargs["method"], "GET")
        self.assertEqual(
            mock_request.call_args_list[1].kwargs["url"],
            "https://api.example.com/v1/responses",
        )
        self.assertEqual(mock_request.call_args_list[1].kwargs["method"], "POST")

        self.assertEqual(
            len(frames),
            1,
            "--no-live should render exactly one final dashboard table.",
        )

        final = frames[-1]
        self.assertEqual(_step_status(final, "network"), "pass")
        self.assertEqual(_step_status(final, "auth"), "pass")
        self.assertEqual(_step_status(final, "models"), "pass")
        self.assertEqual(_step_status(final, "probe"), "pass")

    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_no_live_success_path_renders_single_final_table_only(self, mock_request):
        mock_request.side_effect = [
            _request_ok(
                http_status=200,
                data={"data": [{"id": "gpt-4.1-mini"}]},
            ),
            _request_ok(
                http_status=200,
                data={"id": "resp-1"},
            ),
        ]

        frames: list[dict[str, openai_module._TestStepRender]] = []

        def _capture_render(
            step_rows: dict[str, openai_module._TestStepRender],
            *,
            no_live: bool,
            is_tty: bool | None = None,
        ) -> str:
            self.assertTrue(no_live)
            _ = is_tty
            frames.append(copy.deepcopy(step_rows))
            return "static"

        with patch(
            "tensorneko_tool.openai._render_test_dashboard",
            side_effect=_capture_render,
        ):
            code = openai_module.run_test(
                _make_test_args(mode="all", no_live=True, quiet=False)
            )

        self.assertEqual(code, 0)
        self.assertEqual(
            len(frames),
            1,
            "--no-live should render exactly one final dashboard table.",
        )
        final = frames[-1]
        self.assertEqual(_step_status(final, "network"), "pass")
        self.assertEqual(_step_status(final, "auth"), "pass")
        self.assertEqual(_step_status(final, "models"), "pass")
        self.assertEqual(_step_status(final, "probe"), "pass")

    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_no_live_failure_path_renders_single_final_table_only(self, mock_request):
        mock_request.return_value = _request_fail(
            error=_normalized_error(kind="transport_error")
        )
        frames: list[dict[str, openai_module._TestStepRender]] = []

        def _capture_render(
            step_rows: dict[str, openai_module._TestStepRender],
            *,
            no_live: bool,
            is_tty: bool | None = None,
        ) -> str:
            self.assertTrue(no_live)
            _ = is_tty
            frames.append(copy.deepcopy(step_rows))
            return "static"

        with patch(
            "tensorneko_tool.openai._render_test_dashboard",
            side_effect=_capture_render,
        ):
            code = openai_module.run_test(
                _make_test_args(mode="all", no_live=True, quiet=False)
            )

        self.assertEqual(code, openai_module._OPENAI_EXIT_CODES["network"])
        self.assertEqual(
            len(frames),
            1,
            "--no-live should render exactly one final dashboard table.",
        )
        final = frames[-1]
        self.assertEqual(_step_status(final, "network"), "fail")
        self.assertEqual(_step_status(final, "auth"), "skip")
        self.assertEqual(_step_status(final, "models"), "skip")
        self.assertEqual(_step_status(final, "probe"), "skip")

    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_mode_expansion_runs_expected_prefix_of_pipeline(self, mock_request):
        mode_to_expected = {
            "network": {
                "network": "pass",
                "auth": "skip",
                "models": "skip",
                "probe": "skip",
                "request_count": 1,
                "request_sequence": [
                    ("https://api.example.com/v1/models", "GET"),
                ],
            },
            "auth": {
                "network": "pass",
                "auth": "pass",
                "models": "skip",
                "probe": "skip",
                "request_count": 1,
                "request_sequence": [
                    ("https://api.example.com/v1/models", "GET"),
                ],
            },
            "models": {
                "network": "pass",
                "auth": "pass",
                "models": "pass",
                "probe": "skip",
                "request_count": 1,
                "request_sequence": [
                    ("https://api.example.com/v1/models", "GET"),
                ],
            },
            "probe": {
                "network": "pass",
                "auth": "pass",
                "models": "pass",
                "probe": "pass",
                "request_count": 2,
                "request_sequence": [
                    ("https://api.example.com/v1/models", "GET"),
                    ("https://api.example.com/v1/responses", "POST"),
                ],
            },
        }

        for mode, expected in mode_to_expected.items():
            with self.subTest(mode=mode):
                mock_request.reset_mock()
                mock_request.side_effect = [
                    _request_ok(
                        http_status=200,
                        data={
                            "data": [
                                {"id": "gpt-4.1-mini"},
                            ]
                        },
                    ),
                    _request_ok(http_status=200, data={"id": "resp-1"}),
                ]

                frames: list[dict[str, openai_module._TestStepRender]] = []

                def _capture_render(
                    step_rows: dict[str, openai_module._TestStepRender],
                    *,
                    no_live: bool,
                    is_tty: bool | None = None,
                ) -> str:
                    self.assertTrue(no_live)
                    _ = is_tty
                    frames.append(copy.deepcopy(step_rows))
                    return "static"

                with patch(
                    "tensorneko_tool.openai._render_test_dashboard",
                    side_effect=_capture_render,
                ):
                    code = openai_module.run_test(
                        _make_test_args(mode=mode, no_live=True)
                    )

                self.assertEqual(code, 0)
                request_count = cast(int, expected["request_count"])
                self.assertEqual(mock_request.call_count, request_count)
                self.assertEqual(
                    len(frames),
                    1,
                    "--no-live should render exactly one final dashboard table.",
                )
                request_sequence = cast(
                    list[tuple[str, str]], expected["request_sequence"]
                )
                for index, (url, method) in enumerate(request_sequence):
                    self.assertEqual(
                        mock_request.call_args_list[index].kwargs["url"], url
                    )
                    self.assertEqual(
                        mock_request.call_args_list[index].kwargs["method"], method
                    )
                final = frames[-1]
                self.assertEqual(_step_status(final, "network"), expected["network"])
                self.assertEqual(_step_status(final, "auth"), expected["auth"])
                self.assertEqual(_step_status(final, "models"), expected["models"])
                self.assertEqual(_step_status(final, "probe"), expected["probe"])

    @patch("tensorneko_tool.openai.Live")
    @patch("tensorneko_tool.openai._select_test_render_mode", return_value="live")
    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_live_mode_keeps_single_live_lifecycle_with_multiple_updates(
        self,
        mock_request,
        _mock_select_render_mode,
        mock_live,
    ):
        mock_request.side_effect = [
            _request_ok(
                http_status=200,
                data={"data": [{"id": "gpt-4.1-mini"}]},
            ),
            _request_ok(
                http_status=200,
                data={"id": "resp-1"},
            ),
        ]
        lifecycle_counts = {"enter": 0, "exit": 0}
        live_updates: list[tuple[object, bool]] = []

        class _LiveSession:
            def update(self, _renderable: object, *, refresh: bool = False) -> None:
                live_updates.append((_renderable, refresh))

        live_session = _LiveSession()

        class _LiveContext:
            def __enter__(self) -> _LiveSession:
                lifecycle_counts["enter"] += 1
                return live_session

            def __exit__(
                self,
                _exc_type: object,
                _exc_value: object,
                _traceback: object,
            ) -> bool:
                lifecycle_counts["exit"] += 1
                return False

        mock_live.return_value = _LiveContext()

        code = openai_module.run_test(
            _make_test_args(mode="probe", no_live=False, quiet=False)
        )

        self.assertEqual(code, 0)
        mock_live.assert_called_once()
        self.assertEqual(lifecycle_counts["enter"], 1)
        self.assertEqual(lifecycle_counts["exit"], 1)
        self.assertGreaterEqual(len(live_updates), 5)
        self.assertTrue(all(refresh for _table, refresh in live_updates))
        self.assertTrue(
            all(
                isinstance(table, openai_module.Table)
                for table, _refresh in live_updates
            )
        )


class TestOpenAIModelSelection(unittest.TestCase):
    def test_explicit_model_has_top_priority(self):
        model_id, strategy = openai_module._select_probe_model(
            explicit_model="my-explicit-model",
            models=["gpt-4.1-nano", "gpt-4.1-mini"],
        )

        self.assertEqual(model_id, "my-explicit-model")
        self.assertEqual(strategy, "explicit")

    def test_allowlist_is_checked_before_keyword_heuristic(self):
        model_id, strategy = openai_module._select_probe_model(
            explicit_model=None,
            models=["custom-mini-model", "gpt-4.1-nano", "z-model"],
        )

        self.assertEqual(model_id, "gpt-4.1-nano")
        self.assertEqual(strategy, "allowlist")

    def test_keyword_heuristic_selects_nano_flash_or_mini(self):
        model_id, strategy = openai_module._select_probe_model(
            explicit_model=None,
            models=["x-large", "vendor-model-mini-v2", "z-large"],
        )

        self.assertEqual(model_id, "vendor-model-mini-v2")
        self.assertEqual(strategy, "keyword")

    def test_fallback_any_is_deterministic(self):
        model_id, strategy = openai_module._select_probe_model(
            explicit_model=None,
            models=["zeta", "alpha", "gamma"],
        )

        self.assertEqual(model_id, "alpha")
        self.assertEqual(strategy, "fallback_any")

    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_probe_only_attempts_single_model(self, mock_request):
        mock_request.side_effect = [
            _request_ok(
                http_status=200,
                data={
                    "data": [
                        {"id": "model-a"},
                        {"id": "model-b"},
                        {"id": "model-c"},
                    ]
                },
            ),
            _request_ok(http_status=200, data={"id": "resp-1"}),
        ]

        with patch(
            "tensorneko_tool.openai._render_test_dashboard", return_value="static"
        ):
            result = openai_module.run_test(
                _make_test_args(mode="probe", no_live=True, quiet=True)
            )

        self.assertEqual(result, 0)
        self.assertEqual(mock_request.call_count, 2)
        probe_payload = mock_request.call_args_list[1].kwargs["payload"]
        self.assertIsInstance(probe_payload, dict)
        if not isinstance(probe_payload, dict):
            self.fail("expected probe payload to be a dict")
        self.assertIn("model", probe_payload)
        self.assertIsInstance(probe_payload["model"], str)


class TestOpenAIProbeFallback(unittest.TestCase):
    @patch("tensorneko_tool.openai._render_test_dashboard", return_value="static")
    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_missing_endpoint_and_key_run_test_reaches_request_layer(
        self,
        mock_request,
        _mock_render,
    ):
        mock_request.return_value = _request_fail(
            error=_normalized_error(
                kind="http_error",
                http_status=401,
                error_type="invalid_api_key",
            )
        )
        cast(dict[str, object], mock_request.return_value["error"])["error_code"] = (
            "invalid_api_key"
        )

        result = openai_module.run_test(
            _make_test_args(
                mode="auth",
                no_live=True,
                quiet=True,
                endpoint=None,
                key=None,
            )
        )

        self.assertEqual(result, 20)
        self.assertEqual(mock_request.call_count, 1)
        self.assertEqual(
            mock_request.call_args_list[0].kwargs["url"],
            f"{openai_module._DEFAULT_OPENAI_ENDPOINT}/models",
        )
        self.assertEqual(mock_request.call_args_list[0].kwargs["key"], "")

    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_endpoint_not_supported_falls_back_to_chat(self, mock_request):
        mock_request.side_effect = [
            _request_ok(
                http_status=200,
                data={"data": [{"id": "gpt-4.1-mini"}]},
            ),
            _request_fail(
                error={
                    "kind": "http_error",
                    "http_status": 404,
                    "error_type": "invalid_request_error",
                    "error_code": None,
                    "message": "endpoint_not_found: /responses",
                    "retryable": False,
                }
            ),
            _request_ok(http_status=200, data={"id": "chat-1"}),
        ]

        with patch(
            "tensorneko_tool.openai._render_test_dashboard", return_value="static"
        ):
            result = openai_module.run_test(
                _make_test_args(mode="probe", no_live=True, quiet=True)
            )

        self.assertEqual(result, 0)
        self.assertEqual(mock_request.call_count, 3)
        self.assertEqual(
            mock_request.call_args_list[1].kwargs["url"],
            "https://api.example.com/v1/responses",
        )
        self.assertEqual(
            mock_request.call_args_list[2].kwargs["url"],
            "https://api.example.com/v1/chat/completions",
        )

    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_401_does_not_fallback_to_chat(self, mock_request):
        mock_request.side_effect = [
            _request_ok(
                http_status=200,
                data={"data": [{"id": "gpt-4.1-mini"}]},
            ),
            _request_fail(
                error={
                    "kind": "http_error",
                    "http_status": 401,
                    "error_type": "invalid_api_key",
                    "error_code": "invalid_api_key",
                    "message": "Incorrect API key provided",
                    "retryable": False,
                }
            ),
        ]

        with patch(
            "tensorneko_tool.openai._render_test_dashboard", return_value="static"
        ):
            result = openai_module.run_test(
                _make_test_args(mode="probe", no_live=True, quiet=True)
            )

        self.assertEqual(result, 20)
        self.assertEqual(mock_request.call_count, 2)
        urls = [call.kwargs["url"] for call in mock_request.call_args_list]
        self.assertEqual(
            urls,
            [
                "https://api.example.com/v1/models",
                "https://api.example.com/v1/responses",
            ],
        )

    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_no_chat_fallback_flag_disables_fallback(self, mock_request):
        mock_request.side_effect = [
            _request_ok(
                http_status=200,
                data={"data": [{"id": "gpt-4.1-mini"}]},
            ),
            _request_fail(
                error={
                    "kind": "http_error",
                    "http_status": 404,
                    "error_type": "invalid_request_error",
                    "error_code": None,
                    "message": "endpoint_not_found: /responses",
                    "retryable": False,
                }
            ),
        ]

        with patch(
            "tensorneko_tool.openai._render_test_dashboard", return_value="static"
        ):
            result = openai_module.run_test(
                _make_test_args(
                    mode="probe",
                    no_live=True,
                    quiet=True,
                    no_chat_fallback=True,
                )
            )

        self.assertEqual(result, 40)
        self.assertEqual(mock_request.call_count, 2)
        self.assertEqual(
            mock_request.call_args_list[1].kwargs["url"],
            "https://api.example.com/v1/responses",
        )

    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_non_endpoint_support_error_does_not_fallback_to_chat(self, mock_request):
        mock_request.side_effect = [
            _request_ok(
                http_status=200,
                data={"data": [{"id": "gpt-4.1-mini"}]},
            ),
            _request_fail(
                error={
                    "kind": "http_error",
                    "http_status": 500,
                    "error_type": "server_error",
                    "error_code": None,
                    "message": "server overloaded",
                    "retryable": False,
                }
            ),
        ]

        with patch(
            "tensorneko_tool.openai._render_test_dashboard", return_value="static"
        ):
            result = openai_module.run_test(
                _make_test_args(mode="probe", no_live=True, quiet=True)
            )

        self.assertEqual(result, 40)
        self.assertEqual(mock_request.call_count, 2)
        urls = [call.kwargs["url"] for call in mock_request.call_args_list]
        self.assertEqual(
            urls,
            [
                "https://api.example.com/v1/models",
                "https://api.example.com/v1/responses",
            ],
        )


class TestOpenAIEndpointCandidateFallback(unittest.TestCase):
    @patch("tensorneko_tool.openai._render_test_dashboard", return_value="static")
    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_run_test_retries_raw_endpoint_candidate_on_endpoint_not_supported(
        self,
        mock_request,
        _mock_render,
    ):
        endpoint_error = _normalized_error(
            kind="http_error",
            http_status=404,
            error_type="invalid_request_error",
        )
        endpoint_error["message"] = "endpoint_not_found: /models"
        mock_request.side_effect = [
            _request_fail(error=endpoint_error),
            _request_ok(
                http_status=200,
                data={"data": [{"id": "gpt-4.1-mini"}]},
            ),
            _request_ok(http_status=200, data={"id": "resp-1"}),
        ]

        result = openai_module.run_test(
            _make_test_args(mode="probe", no_live=True, quiet=True)
        )

        self.assertEqual(result, 0)
        self.assertEqual(mock_request.call_count, 3)
        urls = [call.kwargs["url"] for call in mock_request.call_args_list]
        self.assertEqual(
            urls,
            [
                "https://api.example.com/v1/models",
                "https://api.example.com/models",
                "https://api.example.com/responses",
            ],
        )

    @patch("tensorneko_tool.openai._render_chat_plain_text")
    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_run_chat_retries_raw_endpoint_candidate_on_endpoint_not_supported(
        self,
        mock_request,
        mock_render,
    ):
        endpoint_error = _normalized_error(
            kind="http_error",
            http_status=404,
            error_type="invalid_request_error",
        )
        endpoint_error["message"] = "endpoint_not_found: /responses"
        mock_request.side_effect = [
            _request_fail(error=endpoint_error),
            _request_fail(error=endpoint_error),
            _request_ok(http_status=200, data={"output_text": "raw endpoint output"}),
        ]

        args = _make_chat_args(message="hello", no_stream=True)
        with patch("sys.stdin", _FakeStdin("", is_tty=True)):
            code = openai_module.run_chat(args)

        self.assertEqual(code, 0)
        self.assertEqual(mock_request.call_count, 3)
        urls = [call.kwargs["url"] for call in mock_request.call_args_list]
        self.assertEqual(
            urls,
            [
                "https://api.example.com/v1/responses",
                "https://api.example.com/v1/chat/completions",
                "https://api.example.com/responses",
            ],
        )
        mock_render.assert_called_once_with("raw endpoint output")

    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_missing_endpoint_and_key_run_chat_reaches_request_layer(
        self, mock_request
    ):
        mock_request.return_value = _request_fail(
            error=_normalized_error(
                kind="http_error",
                http_status=401,
                error_type="invalid_api_key",
            )
        )
        cast(dict[str, object], mock_request.return_value["error"])["error_code"] = (
            "invalid_api_key"
        )

        args = _make_chat_args(
            message="hello",
            no_stream=True,
            endpoint=None,
            key=None,
        )
        with patch("sys.stdin", _FakeStdin("", is_tty=True)):
            code = openai_module.run_chat(args)

        self.assertEqual(code, 20)
        self.assertEqual(mock_request.call_count, 1)
        self.assertEqual(
            mock_request.call_args_list[0].kwargs["url"],
            f"{openai_module._DEFAULT_OPENAI_ENDPOINT}/responses",
        )
        self.assertEqual(mock_request.call_args_list[0].kwargs["key"], "")

    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_run_list_retries_raw_endpoint_candidate_on_endpoint_not_supported(
        self,
        mock_request,
    ):
        endpoint_error = _normalized_error(
            kind="http_error",
            http_status=404,
            error_type="invalid_request_error",
        )
        endpoint_error["message"] = "endpoint_not_found: /models"
        mock_request.side_effect = [
            _request_fail(error=endpoint_error),
            _request_ok(http_status=200, data={"data": [{"id": "gpt-4.1-mini"}]}),
        ]

        code = openai_module.run_list(_make_list_args(quiet=True))

        self.assertEqual(code, 0)
        self.assertEqual(mock_request.call_count, 2)
        urls = [call.kwargs["url"] for call in mock_request.call_args_list]
        self.assertEqual(
            urls,
            [
                "https://api.example.com/v1/models",
                "https://api.example.com/models",
            ],
        )

    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_missing_endpoint_and_key_run_list_reaches_request_layer(
        self, mock_request
    ):
        mock_request.return_value = _request_fail(
            error=_normalized_error(
                kind="http_error",
                http_status=500,
                error_type="server_error",
            )
        )

        code = openai_module.run_list(
            _make_list_args(endpoint=None, key=None, quiet=True)
        )

        self.assertEqual(code, 30)
        self.assertEqual(mock_request.call_count, 1)
        self.assertEqual(
            mock_request.call_args_list[0].kwargs["url"],
            f"{openai_module._DEFAULT_OPENAI_ENDPOINT}/models",
        )
        self.assertEqual(mock_request.call_args_list[0].kwargs["key"], "")

    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_run_list_does_not_retry_raw_endpoint_for_disallowed_signals(
        self,
        mock_request,
    ):
        disallowed_cases = [
            (401, "invalid_api_key", "Incorrect API key provided", 20),
            (403, "insufficient_permissions", "Forbidden", 20),
            (429, "rate_limit_exceeded", "Rate limit exceeded", 30),
            (404, "model_not_found", "The model `gpt-test` does not exist", 30),
        ]

        for status_code, error_type, message, expected_exit_code in disallowed_cases:
            with self.subTest(status_code=status_code, error_type=error_type):
                error = _normalized_error(
                    kind="http_error",
                    http_status=status_code,
                    error_type=error_type,
                )
                error["message"] = message
                if status_code in {401, 403}:
                    error["error_code"] = error_type

                mock_request.reset_mock()
                mock_request.return_value = _request_fail(error=error)

                code = openai_module.run_list(_make_list_args(quiet=True))

                self.assertEqual(code, expected_exit_code)
                self.assertEqual(mock_request.call_count, 1)
                self.assertEqual(
                    mock_request.call_args_list[0].kwargs["url"],
                    "https://api.example.com/v1/models",
                )

    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_run_list_with_explicit_v1_endpoint_keeps_single_candidate(
        self,
        mock_request,
    ):
        endpoint_error = _normalized_error(
            kind="http_error",
            http_status=404,
            error_type="invalid_request_error",
        )
        endpoint_error["message"] = "endpoint_not_found: /models"
        mock_request.return_value = _request_fail(error=endpoint_error)

        code = openai_module.run_list(
            _make_list_args(endpoint="https://api.example.com/v1", quiet=True)
        )

        self.assertEqual(code, 30)
        self.assertEqual(mock_request.call_count, 1)
        self.assertEqual(
            mock_request.call_args_list[0].kwargs["url"],
            "https://api.example.com/v1/models",
        )


class TestOpenAIChatInputArbitration(unittest.TestCase):
    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_positional_and_piped_stdin_returns_usage_error(self, mock_request):
        args = _make_chat_args(message="hello", no_stream=True)
        with patch("sys.stdin", _FakeStdin("stdin prompt\n", is_tty=False)):
            code = openai_module.run_chat(args)

        self.assertEqual(code, 2)
        mock_request.assert_not_called()

    @patch("tensorneko_tool.openai._render_chat_plain_text")
    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_missing_positional_reads_piped_stdin(self, mock_request, mock_render):
        mock_request.return_value = _request_ok(
            http_status=200,
            data={
                "output": [
                    {
                        "content": [
                            {"type": "output_text", "text": "hello from model"},
                        ]
                    }
                ]
            },
        )
        args = _make_chat_args(message=None, no_stream=True)
        with patch("sys.stdin", _FakeStdin("stdin prompt\n", is_tty=False)):
            code = openai_module.run_chat(args)

        self.assertEqual(code, 0)
        self.assertEqual(mock_request.call_count, 1)
        self.assertEqual(
            mock_request.call_args_list[0].kwargs["url"],
            "https://api.example.com/v1/responses",
        )
        payload = mock_request.call_args_list[0].kwargs["payload"]
        self.assertIsInstance(payload, dict)
        if not isinstance(payload, dict):
            self.fail("expected payload dict")
        self.assertEqual(payload["input"], "stdin prompt")
        mock_render.assert_called_once_with("hello from model")

    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_missing_positional_with_interactive_stdin_returns_usage_error(
        self, mock_request
    ):
        args = _make_chat_args(message=None, no_stream=True)
        with patch("sys.stdin", _FakeStdin("", is_tty=True)):
            code = openai_module.run_chat(args)

        self.assertEqual(code, 2)
        mock_request.assert_not_called()

    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_missing_positional_with_empty_piped_stdin_returns_usage_error(
        self, mock_request
    ):
        args = _make_chat_args(message=None, no_stream=True)
        with patch("sys.stdin", _FakeStdin("  \n", is_tty=False)):
            code = openai_module.run_chat(args)

        self.assertEqual(code, 2)
        mock_request.assert_not_called()


class TestOpenAIChatStreaming(unittest.TestCase):
    def test_select_chat_streaming_policy(self):
        self.assertTrue(
            openai_module._select_chat_streaming(no_stream=False, is_tty=True)
        )
        self.assertFalse(
            openai_module._select_chat_streaming(no_stream=False, is_tty=False)
        )
        self.assertFalse(
            openai_module._select_chat_streaming(no_stream=True, is_tty=True)
        )

    def test_consume_chat_stream_lines_parses_delta_and_stops_on_completion(self):
        lines = [
            b"event: response.output_text.delta\n",
            b'data: {"type":"response.output_text.delta","delta":"Hello"}\n',
            b"\n",
            b'data: {"choices":[{"delta":{"content":" world"}}]}\n',
            b'data: {"type":"response.completed"}\n',
            b'data: {"type":"response.output_text.delta","delta":" ignored"}\n',
        ]
        emitted_deltas: list[str] = []

        text, completed = openai_module._consume_chat_stream_lines(
            lines,
            on_delta=emitted_deltas.append,
        )

        self.assertEqual(text, "Hello world")
        self.assertTrue(completed)
        self.assertEqual(emitted_deltas, ["Hello", " world"])

    @patch("tensorneko_tool.openai._request_chat_stream_once")
    def test_stream_with_fallback_uses_chat_only_on_endpoint_not_supported(
        self, mock_stream_request
    ):
        endpoint_error = _normalized_error(
            kind="http_error", http_status=404, error_type="invalid_request_error"
        )
        endpoint_error["message"] = "endpoint_not_found: /responses"

        mock_stream_request.side_effect = [
            (False, "", 404, endpoint_error),
            (True, "fallback stream text", 200, None),
        ]

        ok, text, http_status, error, used_fallback = (
            openai_module._chat_stream_with_fallback(
                endpoint_base="https://api.example.com/v1",
                key="test-key",
                prompt="hello",
                model_id="gpt-4.1-mini",
            )
        )

        self.assertTrue(ok)
        self.assertEqual(text, "fallback stream text")
        self.assertEqual(http_status, 200)
        self.assertIsNone(error)
        self.assertTrue(used_fallback)
        self.assertEqual(mock_stream_request.call_count, 2)
        self.assertEqual(
            mock_stream_request.call_args_list[0].kwargs["url"],
            "https://api.example.com/v1/responses",
        )
        self.assertEqual(
            mock_stream_request.call_args_list[1].kwargs["url"],
            "https://api.example.com/v1/chat/completions",
        )

    @patch("tensorneko_tool.openai._request_chat_stream_once")
    def test_stream_auth_error_does_not_fallback(self, mock_stream_request):
        auth_error = _normalized_error(
            kind="http_error", http_status=401, error_type="invalid_api_key"
        )
        auth_error["error_code"] = "invalid_api_key"
        auth_error["message"] = "Incorrect API key provided"

        mock_stream_request.return_value = (False, "", 401, auth_error)

        ok, text, http_status, error, used_fallback = (
            openai_module._chat_stream_with_fallback(
                endpoint_base="https://api.example.com/v1",
                key="test-key",
                prompt="hello",
                model_id="gpt-4.1-mini",
            )
        )

        self.assertFalse(ok)
        self.assertEqual(text, "")
        self.assertEqual(http_status, 401)
        self.assertIsNotNone(error)
        self.assertFalse(used_fallback)
        self.assertEqual(mock_stream_request.call_count, 1)

    @patch("tensorneko_tool.openai._render_chat_plain_text")
    @patch("tensorneko_tool.openai._chat_stream_with_fallback")
    @patch("tensorneko_tool.openai._select_chat_streaming")
    def test_run_chat_streaming_emits_plain_reply_text_incrementally(
        self,
        mock_select_stream,
        mock_chat_stream,
        mock_render,
    ):
        mock_select_stream.return_value = True

        def _stream_side_effect(**kwargs: object):
            on_delta = kwargs.get("on_delta")
            self.assertIsNotNone(on_delta)
            if not callable(on_delta):
                self.fail("expected streaming callback")
            _ = on_delta("stream ")
            _ = on_delta("output")
            return (True, "stream output", 200, None, False)

        mock_chat_stream.side_effect = _stream_side_effect
        args = _make_chat_args(message="hello", no_stream=False)

        with patch("sys.stdin", _FakeStdin("", is_tty=True)):
            code = openai_module.run_chat(args)

        self.assertEqual(code, 0)
        mock_chat_stream.assert_called_once()
        self.assertEqual(
            mock_render.call_args_list,
            [
                call("stream ", append=True),
                call("output", append=True),
                call(""),
            ],
        )


class TestOpenAIChatNonStreaming(unittest.TestCase):
    @patch("tensorneko_tool.openai._chat_stream_with_fallback")
    @patch("tensorneko_tool.openai._render_chat_plain_text")
    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_no_stream_flag_forces_non_stream(
        self,
        mock_request,
        mock_render,
        mock_stream,
    ):
        mock_request.return_value = _request_ok(
            http_status=200,
            data={"output_text": "non-stream output"},
        )
        args = _make_chat_args(message="hello", no_stream=True)

        with patch("sys.stdin", _FakeStdin("", is_tty=True)):
            code = openai_module.run_chat(args)

        self.assertEqual(code, 0)
        mock_stream.assert_not_called()
        self.assertEqual(mock_request.call_count, 1)
        payload = mock_request.call_args_list[0].kwargs["payload"]
        self.assertIsInstance(payload, dict)
        if not isinstance(payload, dict):
            self.fail("expected payload dict")
        self.assertEqual(payload.get("stream"), False)
        mock_render.assert_called_once_with("non-stream output")

    @patch("tensorneko_tool.openai._render_chat_plain_text")
    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_non_stream_endpoint_not_supported_falls_back_to_chat(
        self,
        mock_request,
        mock_render,
    ):
        endpoint_error = _normalized_error(
            kind="http_error", http_status=404, error_type="invalid_request_error"
        )
        endpoint_error["message"] = "endpoint_not_found: /responses"
        mock_request.side_effect = [
            _request_fail(error=endpoint_error),
            _request_ok(
                http_status=200,
                data={
                    "choices": [
                        {"message": {"role": "assistant", "content": "fallback output"}}
                    ]
                },
            ),
        ]
        args = _make_chat_args(message="hello", no_stream=True)

        with patch("sys.stdin", _FakeStdin("", is_tty=True)):
            code = openai_module.run_chat(args)

        self.assertEqual(code, 0)
        self.assertEqual(mock_request.call_count, 2)
        self.assertEqual(
            mock_request.call_args_list[0].kwargs["url"],
            "https://api.example.com/v1/responses",
        )
        self.assertEqual(
            mock_request.call_args_list[1].kwargs["url"],
            "https://api.example.com/v1/chat/completions",
        )
        mock_render.assert_called_once_with("fallback output")

    @patch("tensorneko_tool.openai._render_chat_plain_text")
    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_non_stream_auth_error_does_not_fallback(self, mock_request, mock_render):
        auth_error = _normalized_error(
            kind="http_error", http_status=401, error_type="invalid_api_key"
        )
        auth_error["error_code"] = "invalid_api_key"
        auth_error["message"] = "Incorrect API key provided"
        mock_request.return_value = _request_fail(error=auth_error)
        args = _make_chat_args(message="hello", no_stream=True)

        with patch("sys.stdin", _FakeStdin("", is_tty=True)):
            code = openai_module.run_chat(args)

        self.assertEqual(code, 20)
        self.assertEqual(mock_request.call_count, 1)
        mock_render.assert_not_called()

    @patch("tensorneko_tool.openai._render_chat_plain_text")
    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_non_stream_server_error_does_not_fallback(self, mock_request, mock_render):
        server_error = _normalized_error(
            kind="http_error", http_status=500, error_type="server_error"
        )
        server_error["message"] = "server overloaded"
        mock_request.return_value = _request_fail(error=server_error)
        args = _make_chat_args(message="hello", no_stream=True)

        with patch("sys.stdin", _FakeStdin("", is_tty=True)):
            code = openai_module.run_chat(args)

        self.assertEqual(code, 50)
        self.assertEqual(mock_request.call_count, 1)
        self.assertEqual(
            mock_request.call_args_list[0].kwargs["url"],
            "https://api.example.com/v1/responses",
        )
        mock_render.assert_not_called()


class TestOpenAIListHumanOutput(unittest.TestCase):
    @patch("tensorneko_tool.openai.utils.console")
    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_list_renders_sorted_table_and_summary(self, mock_request, mock_console):
        mock_console.print = MagicMock()
        mock_request.return_value = _request_ok(
            http_status=200,
            data={
                "data": [
                    {
                        "id": "gpt-4.1-zeta",
                        "owned_by": "zeta-lab",
                        "created": 200,
                    },
                    {
                        "id": "gpt-4.1-alpha",
                        "owned_by": "alpha-lab",
                        "created": 100,
                    },
                ]
            },
        )

        code = openai_module.run_list(_make_list_args())

        self.assertEqual(code, 0)
        self.assertEqual(mock_request.call_count, 1)
        self.assertEqual(
            mock_request.call_args_list[0].kwargs["url"],
            "https://api.example.com/v1/models",
        )
        self.assertEqual(mock_request.call_args_list[0].kwargs["method"], "GET")

        self.assertEqual(mock_console.print.call_count, 2)
        summary_text = cast(str, mock_console.print.call_args_list[0].args[0])
        rendered_table = cast(object, mock_console.print.call_args_list[1].args[0])
        self.assertIsInstance(rendered_table, openai_module.Table)
        if not isinstance(rendered_table, openai_module.Table):
            self.fail("Expected Rich table output for model list")
        table = cast(openai_module.Table, rendered_table)
        self.assertEqual(
            [column.header for column in table.columns],
            ["Model ID", "Owner", "Created"],
        )
        self.assertEqual(
            _table_column_cells(table, 0),
            ["gpt-4.1-alpha", "gpt-4.1-zeta"],
        )
        self.assertEqual(_table_column_cells(table, 1), ["alpha-lab", "zeta-lab"])
        self.assertEqual(_table_column_cells(table, 2), ["100", "200"])

        self.assertIn("Total models: 2", summary_text)
        self.assertIn("Endpoint: https://api.example.com/v1", summary_text)


class TestOpenAIListDataParsing(unittest.TestCase):
    @patch("tensorneko_tool.openai.utils.console")
    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_list_json_output_returns_command_specific_schema(
        self,
        mock_request,
        mock_console,
    ):
        mock_console.print = MagicMock()
        mock_request.return_value = _request_ok(
            http_status=200,
            data=[
                {"id": "gpt-b", "owner": "owner-b", "created": "2"},
                {"id": "gpt-a", "owned_by": "owner-a", "created": 1},
            ],
        )

        code = openai_module.run_list(_make_list_args(json=True))

        self.assertEqual(code, 0)
        self.assertEqual(mock_console.print.call_count, 1)
        payload = cast(
            dict[str, object],
            json.loads(cast(str, mock_console.print.call_args_list[0].args[0])),
        )

        self.assertEqual(payload["command"], "list")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["exit_code"], 0)
        self.assertEqual(payload["endpoint"], "https://api.example.com")
        self.assertEqual(
            payload["resolved_endpoint_base"], "https://api.example.com/v1"
        )
        self.assertEqual(payload["model_count"], 2)
        self.assertEqual(
            payload["models"],
            [
                {"id": "gpt-a", "owned_by": "owner-a", "created": 1},
                {"id": "gpt-b", "owned_by": "owner-b", "created": 2},
            ],
        )
        self.assertIn("started_at", payload)
        self.assertIn("finished_at", payload)
        self.assertIsNone(payload.get("error"))

    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_empty_models_array_returns_failure_code(self, mock_request):
        mock_request.return_value = _request_ok(http_status=200, data={"data": []})

        code = openai_module.run_list(_make_list_args())

        self.assertEqual(code, 30)


_JSON_ERROR_KEYS = {
    "kind",
    "http_status",
    "error_type",
    "error_code",
    "message",
    "retryable",
}


def _assert_timestamp_pair(
    test_case: unittest.TestCase,
    payload: dict[str, object],
) -> None:
    started_at = payload.get("started_at")
    finished_at = payload.get("finished_at")
    test_case.assertIsInstance(started_at, (int, float))
    test_case.assertIsInstance(finished_at, (int, float))
    if isinstance(started_at, bool) or isinstance(finished_at, bool):
        test_case.fail("timestamps must be numeric, not bool")
    if isinstance(started_at, (int, float)) and isinstance(finished_at, (int, float)):
        test_case.assertLessEqual(float(started_at), float(finished_at))


def _assert_json_error_schema(
    test_case: unittest.TestCase,
    error_payload: object,
) -> None:
    test_case.assertIsInstance(error_payload, dict)
    if not isinstance(error_payload, dict):
        return
    test_case.assertEqual(set(error_payload.keys()), _JSON_ERROR_KEYS)


class TestOpenAIJsonTestSchema(unittest.TestCase):
    _TOP_LEVEL_KEYS: set[str] = {
        "command",
        "ok",
        "exit_code",
        "endpoint",
        "resolved_endpoint_base",
        "mode",
        "selected_model",
        "model_selection_strategy",
        "probe_endpoint",
        "steps",
        "started_at",
        "finished_at",
        "error",
    }
    _STEP_KEYS: set[str] = {
        "name",
        "status",
        "ok",
        "http_status",
        "error_code",
        "message",
        "elapsed_ms",
    }

    @patch("tensorneko_tool.openai._render_test_dashboard")
    @patch("tensorneko_tool.openai.utils.console")
    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_test_json_success_schema_stable(
        self,
        mock_request,
        mock_console,
        mock_render,
    ):
        mock_console.print = MagicMock()
        mock_request.return_value = _request_ok(
            http_status=200,
            data={"data": [{"id": "gpt-4.1-mini"}]},
        )

        code = openai_module.run_test(
            _make_test_args(mode="network", json=True, no_live=False, quiet=False)
        )

        self.assertEqual(code, 0)
        mock_render.assert_not_called()
        self.assertEqual(mock_console.print.call_count, 1)

        payload = cast(
            dict[str, object],
            json.loads(cast(str, mock_console.print.call_args_list[0].args[0])),
        )
        self.assertEqual(set(payload.keys()), self._TOP_LEVEL_KEYS)
        self.assertEqual(payload["command"], "test")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["exit_code"], 0)
        self.assertEqual(payload["endpoint"], "https://api.example.com")
        self.assertEqual(
            payload["resolved_endpoint_base"],
            "https://api.example.com/v1",
        )
        self.assertEqual(payload["mode"], "network")
        self.assertIsNone(payload["selected_model"])
        self.assertIsNone(payload["model_selection_strategy"])
        self.assertIsNone(payload["probe_endpoint"])
        _assert_timestamp_pair(self, payload)

        steps = payload["steps"]
        self.assertIsInstance(steps, list)
        if not isinstance(steps, list):
            self.fail("expected steps list")
        self.assertEqual(
            [cast(str, cast(dict[str, object], step)["name"]) for step in steps],
            list(openai_module._TEST_STEP_ORDER),
        )
        for step in steps:
            self.assertIsInstance(step, dict)
            if not isinstance(step, dict):
                self.fail("expected step object")
            self.assertEqual(set(step.keys()), self._STEP_KEYS)

        self.assertIsNone(payload["error"])

    @patch("tensorneko_tool.openai._render_test_dashboard")
    @patch("tensorneko_tool.openai.utils.console")
    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_test_json_error_schema_stable(
        self,
        mock_request,
        mock_console,
        mock_render,
    ):
        mock_console.print = MagicMock()
        network_error = _normalized_error(
            kind="transport_error",
            error_type="url_error",
        )
        network_error["message"] = "socket closed"
        mock_request.return_value = _request_fail(error=network_error)

        code = openai_module.run_test(
            _make_test_args(mode="all", json=True, no_live=False, quiet=False)
        )

        self.assertEqual(code, 10)
        mock_render.assert_not_called()
        self.assertEqual(mock_console.print.call_count, 1)

        payload = cast(
            dict[str, object],
            json.loads(cast(str, mock_console.print.call_args_list[0].args[0])),
        )
        self.assertEqual(set(payload.keys()), self._TOP_LEVEL_KEYS)
        self.assertEqual(payload["command"], "test")
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["exit_code"], 10)
        _assert_timestamp_pair(self, payload)
        _assert_json_error_schema(self, payload["error"])

        steps = payload["steps"]
        self.assertIsInstance(steps, list)
        if not isinstance(steps, list):
            self.fail("expected steps list")
        for step in steps:
            self.assertIsInstance(step, dict)
            if not isinstance(step, dict):
                self.fail("expected step object")
            self.assertEqual(set(step.keys()), self._STEP_KEYS)

        network_step = next(
            cast(dict[str, object], step)
            for step in steps
            if cast(dict[str, object], step)["name"] == "network"
        )
        self.assertEqual(network_step["status"], "fail")
        self.assertEqual(network_step["error_code"], "url_error")


class TestOpenAIJsonChatSchema(unittest.TestCase):
    _TOP_LEVEL_KEYS: set[str] = {
        "command",
        "ok",
        "exit_code",
        "endpoint",
        "resolved_endpoint_base",
        "model",
        "stream",
        "text",
        "finish_reason",
        "usage",
        "started_at",
        "finished_at",
        "error",
    }

    @patch("tensorneko_tool.openai._render_chat_plain_text")
    @patch("tensorneko_tool.openai._chat_stream_with_fallback")
    @patch("tensorneko_tool.openai._select_chat_streaming")
    @patch("tensorneko_tool.openai.utils.console")
    def test_chat_json_streaming_success_schema_stable(
        self,
        mock_console,
        mock_select_stream,
        mock_chat_stream,
        mock_render,
    ):
        mock_console.print = MagicMock()
        mock_select_stream.return_value = True
        mock_chat_stream.return_value = (True, "streamed json reply", 200, None, False)

        args = _make_chat_args(message="hello", no_stream=False, json=True, quiet=False)
        with patch("sys.stdin", _FakeStdin("", is_tty=True)):
            code = openai_module.run_chat(args)

        self.assertEqual(code, 0)
        mock_render.assert_not_called()
        self.assertEqual(mock_console.print.call_count, 1)
        self.assertIsNone(mock_chat_stream.call_args_list[0].kwargs.get("on_delta"))

        payload = cast(
            dict[str, object],
            json.loads(cast(str, mock_console.print.call_args_list[0].args[0])),
        )
        self.assertEqual(set(payload.keys()), self._TOP_LEVEL_KEYS)
        self.assertEqual(payload["command"], "chat")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["exit_code"], 0)
        self.assertEqual(payload["endpoint"], "https://api.example.com")
        self.assertEqual(
            payload["resolved_endpoint_base"],
            "https://api.example.com/v1",
        )
        self.assertEqual(payload["model"], openai_module._DEFAULT_CHAT_MODEL)
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["text"], "streamed json reply")
        self.assertIsNone(payload["finish_reason"])
        self.assertIsNone(payload["usage"])
        _assert_timestamp_pair(self, payload)
        self.assertIsNone(payload["error"])

    @patch("tensorneko_tool.openai._render_chat_plain_text")
    @patch("tensorneko_tool.openai.utils.console")
    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_chat_json_success_schema_stable(
        self,
        mock_request,
        mock_console,
        mock_render,
    ):
        mock_console.print = MagicMock()
        mock_request.return_value = _request_ok(
            http_status=200,
            data={
                "output_text": "json reply",
                "status": "completed",
                "usage": {"input_tokens": 4, "output_tokens": 2},
            },
        )

        args = _make_chat_args(message="hello", no_stream=True, json=True, quiet=False)
        with patch("sys.stdin", _FakeStdin("", is_tty=True)):
            code = openai_module.run_chat(args)

        self.assertEqual(code, 0)
        mock_render.assert_not_called()
        self.assertEqual(mock_console.print.call_count, 1)

        payload = cast(
            dict[str, object],
            json.loads(cast(str, mock_console.print.call_args_list[0].args[0])),
        )
        self.assertEqual(set(payload.keys()), self._TOP_LEVEL_KEYS)
        self.assertEqual(payload["command"], "chat")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["exit_code"], 0)
        self.assertEqual(payload["endpoint"], "https://api.example.com")
        self.assertEqual(
            payload["resolved_endpoint_base"],
            "https://api.example.com/v1",
        )
        self.assertEqual(payload["model"], openai_module._DEFAULT_CHAT_MODEL)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["text"], "json reply")
        self.assertEqual(payload["finish_reason"], "completed")
        self.assertEqual(
            payload["usage"],
            {"input_tokens": 4, "output_tokens": 2},
        )
        _assert_timestamp_pair(self, payload)
        self.assertIsNone(payload["error"])

    @patch("tensorneko_tool.openai._render_chat_plain_text")
    @patch("tensorneko_tool.openai.utils.console")
    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_chat_json_error_schema_stable(
        self,
        mock_request,
        mock_console,
        mock_render,
    ):
        mock_console.print = MagicMock()
        auth_error = _normalized_error(
            kind="http_error",
            http_status=401,
            error_type="invalid_api_key",
        )
        auth_error["error_code"] = "invalid_api_key"
        auth_error["message"] = "Incorrect API key provided"
        mock_request.return_value = _request_fail(error=auth_error)

        args = _make_chat_args(message="hello", no_stream=True, json=True, quiet=False)
        with patch("sys.stdin", _FakeStdin("", is_tty=True)):
            code = openai_module.run_chat(args)

        self.assertEqual(code, 20)
        mock_render.assert_not_called()
        self.assertEqual(mock_console.print.call_count, 1)

        payload = cast(
            dict[str, object],
            json.loads(cast(str, mock_console.print.call_args_list[0].args[0])),
        )
        self.assertEqual(set(payload.keys()), self._TOP_LEVEL_KEYS)
        self.assertEqual(payload["command"], "chat")
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["exit_code"], 20)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["text"], "")
        self.assertIsNone(payload["finish_reason"])
        self.assertIsNone(payload["usage"])
        _assert_timestamp_pair(self, payload)
        _assert_json_error_schema(self, payload["error"])


class TestOpenAIJsonListSchema(unittest.TestCase):
    _TOP_LEVEL_KEYS: set[str] = {
        "command",
        "ok",
        "exit_code",
        "endpoint",
        "resolved_endpoint_base",
        "model_count",
        "models",
        "started_at",
        "finished_at",
        "error",
    }
    _MODEL_KEYS: set[str] = {"id", "owned_by", "created"}

    @patch("tensorneko_tool.openai._render_list_table_and_summary")
    @patch("tensorneko_tool.openai.utils.console")
    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_list_json_success_schema_stable(
        self,
        mock_request,
        mock_console,
        mock_render,
    ):
        mock_console.print = MagicMock()
        mock_request.return_value = _request_ok(
            http_status=200,
            data={
                "data": [
                    {"id": "gpt-b", "owner": "owner-b", "created": "2"},
                    {"id": "gpt-a", "owned_by": "owner-a", "created": 1},
                ]
            },
        )

        code = openai_module.run_list(_make_list_args(json=True, quiet=False))

        self.assertEqual(code, 0)
        mock_render.assert_not_called()
        self.assertEqual(mock_console.print.call_count, 1)

        payload = cast(
            dict[str, object],
            json.loads(cast(str, mock_console.print.call_args_list[0].args[0])),
        )
        self.assertEqual(set(payload.keys()), self._TOP_LEVEL_KEYS)
        self.assertEqual(payload["command"], "list")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["exit_code"], 0)
        self.assertEqual(payload["endpoint"], "https://api.example.com")
        self.assertEqual(
            payload["resolved_endpoint_base"],
            "https://api.example.com/v1",
        )
        self.assertEqual(payload["model_count"], 2)
        _assert_timestamp_pair(self, payload)
        self.assertIsNone(payload["error"])

        models = payload["models"]
        self.assertIsInstance(models, list)
        if not isinstance(models, list):
            self.fail("expected models list")
        self.assertEqual(
            models,
            [
                {"id": "gpt-a", "owned_by": "owner-a", "created": 1},
                {"id": "gpt-b", "owned_by": "owner-b", "created": 2},
            ],
        )
        for model in models:
            self.assertIsInstance(model, dict)
            if not isinstance(model, dict):
                self.fail("expected model object")
            self.assertEqual(set(model.keys()), self._MODEL_KEYS)

    @patch("tensorneko_tool.openai._render_list_table_and_summary")
    @patch("tensorneko_tool.openai.utils.console")
    @patch("tensorneko_tool.openai._request_json_with_retry")
    def test_list_json_error_schema_stable(
        self,
        mock_request,
        mock_console,
        mock_render,
    ):
        mock_console.print = MagicMock()
        models_error = _normalized_error(
            kind="http_error",
            http_status=500,
            error_type="server_error",
        )
        models_error["message"] = "server overloaded"
        mock_request.return_value = _request_fail(error=models_error)

        code = openai_module.run_list(_make_list_args(json=True, quiet=False))

        self.assertEqual(code, 30)
        mock_render.assert_not_called()
        self.assertEqual(mock_console.print.call_count, 1)

        payload = cast(
            dict[str, object],
            json.loads(cast(str, mock_console.print.call_args_list[0].args[0])),
        )
        self.assertEqual(set(payload.keys()), self._TOP_LEVEL_KEYS)
        self.assertEqual(payload["command"], "list")
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["exit_code"], 30)
        self.assertEqual(payload["model_count"], 0)
        self.assertEqual(payload["models"], [])
        _assert_timestamp_pair(self, payload)
        _assert_json_error_schema(self, payload["error"])


if __name__ == "__main__":
    _ = unittest.main()
