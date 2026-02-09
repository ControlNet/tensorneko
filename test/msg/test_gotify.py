import io
import importlib
import json
import unittest
from email.message import Message
from urllib.error import HTTPError, URLError
from unittest.mock import MagicMock, patch

gotify = importlib.import_module("tensorneko_util.msg.gotify")


class TestGotifyPush(unittest.TestCase):
    @patch("tensorneko_util.msg.gotify.socket.gethostname", return_value="host-1")
    @patch("tensorneko_util.msg.gotify.urllib.request.urlopen")
    def test_push_success_with_explicit_url_and_token(
        self, mock_urlopen, _mock_hostname
    ):
        mock_urlopen.return_value.__enter__.return_value = MagicMock()

        gotify.push("hello", url="https://gotify.local/", token="token-1")

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.full_url, "https://gotify.local/message?token=token-1")
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.headers["Content-type"], "application/json")
        self.assertEqual(req.headers["User-agent"], "TensorNeko")
        self.assertEqual(
            json.loads(req.data.decode("utf-8")),
            {"title": "host-1", "message": "hello", "priority": 0},
        )

    @patch("tensorneko_util.msg.gotify.urllib.request.urlopen")
    def test_push_success_with_custom_title_and_priority(self, mock_urlopen):
        mock_urlopen.return_value.__enter__.return_value = MagicMock()

        gotify.push(
            "done",
            url="https://gotify.local",
            token="token-2",
            title="build",
            priority=7,
        )

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(
            json.loads(req.data.decode("utf-8")),
            {"title": "build", "message": "done", "priority": 7},
        )

    @patch("tensorneko_util.msg.gotify.urllib.request.urlopen")
    def test_push_uses_environment_when_url_and_token_not_provided(self, mock_urlopen):
        mock_urlopen.return_value.__enter__.return_value = MagicMock()

        with patch.dict(
            "tensorneko_util.msg.gotify.os.environ",
            {
                "GOTIFY_URL": "https://env-gotify",
                "GOTIFY_TOKEN": "env-token",
                "GOTIFY_TITLE": "env-title",
            },
            clear=True,
        ):
            gotify.push("from-env")

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.full_url, "https://env-gotify/message?token=env-token")
        self.assertEqual(
            json.loads(req.data.decode("utf-8")),
            {"title": "env-title", "message": "from-env", "priority": 0},
        )

    @patch(
        "tensorneko_util.msg.gotify.urllib.request.urlopen",
        side_effect=URLError("down"),
    )
    def test_push_raises_on_network_error(self, _mock_urlopen):
        with self.assertRaises(URLError):
            gotify.push("hello", url="https://gotify.local", token="token")

    @patch("builtins.print")
    @patch("tensorneko_util.msg.gotify.urllib.request.urlopen")
    def test_push_prints_http_error_body(self, mock_urlopen, mock_print):
        error = HTTPError(
            url="https://gotify.local/message?token=token",
            code=500,
            msg="server-error",
            hdrs=Message(),
            fp=io.BytesIO(b'{"error":"boom"}'),
        )
        self.addCleanup(error.close)
        mock_urlopen.side_effect = error

        gotify.push("hello", url="https://gotify.local", token="token")

        mock_print.assert_called_once_with('{"error":"boom"}')

    def test_push_raises_when_url_or_token_missing(self):
        with patch.dict("tensorneko_util.msg.gotify.os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "URL is not provided"):
                gotify.push("hello", token="token")
            with self.assertRaisesRegex(ValueError, "Token is not provided"):
                gotify.push("hello", url="https://gotify.local")


if __name__ == "__main__":
    unittest.main()
