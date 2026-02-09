import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from tensorneko_util.msg import postgres


class TestPostgresExecute(unittest.TestCase):
    @patch("tensorneko_util.msg.postgres.psycopg.Connection.connect")
    def test_execute_returns_rows_for_select_query(self, mock_connect):
        mock_cur = MagicMock()
        mock_cur.description = ("id",)
        mock_cur.fetchall.return_value = [(1, "alice")]

        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        mock_connect.return_value.__enter__.return_value = mock_conn

        result = postgres.execute(
            "SELECT id, name FROM users", db_url="postgresql://db"
        )

        self.assertEqual(result, [(1, "alice")])
        mock_connect.assert_called_once_with("postgresql://db")
        mock_cur.execute.assert_called_once_with("SELECT id, name FROM users")
        mock_conn.commit.assert_called_once()

    @patch("tensorneko_util.msg.postgres.psycopg.Connection.connect")
    def test_execute_returns_none_for_non_select_query(self, mock_connect):
        mock_cur = MagicMock()
        mock_cur.description = None

        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        mock_connect.return_value.__enter__.return_value = mock_conn

        result = postgres.execute(
            "UPDATE users SET active = true", db_url="postgresql://db"
        )

        self.assertIsNone(result)
        mock_conn.commit.assert_called_once()

    @patch("tensorneko_util.msg.postgres.psycopg.Connection.connect")
    def test_execute_uses_environment_db_url_when_not_provided(self, mock_connect):
        mock_cur = MagicMock()
        mock_cur.description = None

        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        mock_connect.return_value.__enter__.return_value = mock_conn

        with patch.dict(
            "tensorneko_util.msg.postgres.os.environ",
            {"DB_URL": "postgresql://env"},
            clear=True,
        ):
            postgres.execute("DELETE FROM users WHERE id = 1")

        mock_connect.assert_called_once_with("postgresql://env")

    @patch(
        "tensorneko_util.msg.postgres.psycopg.Connection.connect",
        side_effect=RuntimeError("connect failed"),
    )
    def test_execute_propagates_connection_error(self, _mock_connect):
        with self.assertRaisesRegex(RuntimeError, "connect failed"):
            postgres.execute("SELECT 1", db_url="postgresql://db")


class TestPostgresExecuteAsync(unittest.IsolatedAsyncioTestCase):
    @patch(
        "tensorneko_util.msg.postgres.psycopg.AsyncConnection.connect",
        new_callable=AsyncMock,
    )
    async def test_execute_async_returns_rows(self, mock_connect):
        mock_cur = AsyncMock()
        mock_cur.description = ("id",)
        mock_cur.fetchall.return_value = [(1, "alice")]

        cursor_cm = AsyncMock()
        cursor_cm.__aenter__.return_value = mock_cur

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = cursor_cm

        conn_cm = AsyncMock()
        conn_cm.__aenter__.return_value = mock_conn

        mock_connect.return_value = conn_cm

        result = await postgres.execute_async(
            "SELECT id, name FROM users", db_url="postgresql://async-db"
        )

        self.assertEqual(result, [(1, "alice")])
        mock_connect.assert_awaited_once_with("postgresql://async-db", autocommit=True)
        mock_cur.execute.assert_awaited_once_with("SELECT id, name FROM users")
        mock_cur.fetchall.assert_awaited_once()

    @patch(
        "tensorneko_util.msg.postgres.psycopg.AsyncConnection.connect",
        new_callable=AsyncMock,
    )
    async def test_execute_async_returns_none_for_non_select_query(self, mock_connect):
        mock_cur = AsyncMock()
        mock_cur.description = None

        cursor_cm = AsyncMock()
        cursor_cm.__aenter__.return_value = mock_cur

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = cursor_cm

        conn_cm = AsyncMock()
        conn_cm.__aenter__.return_value = mock_conn

        mock_connect.return_value = conn_cm

        result = await postgres.execute_async(
            "UPDATE users SET active = true", db_url="postgresql://async-db"
        )

        self.assertIsNone(result)
        mock_cur.execute.assert_awaited_once_with("UPDATE users SET active = true")
        mock_cur.fetchall.assert_not_called()

    @patch(
        "tensorneko_util.msg.postgres.psycopg.AsyncConnection.connect",
        new_callable=AsyncMock,
    )
    async def test_execute_async_uses_environment_db_url(self, mock_connect):
        mock_cur = AsyncMock()
        mock_cur.description = None

        cursor_cm = AsyncMock()
        cursor_cm.__aenter__.return_value = mock_cur

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = cursor_cm

        conn_cm = AsyncMock()
        conn_cm.__aenter__.return_value = mock_conn

        mock_connect.return_value = conn_cm

        with patch.dict(
            "tensorneko_util.msg.postgres.os.environ",
            {"DB_URL": "postgresql://env-async"},
            clear=True,
        ):
            await postgres.execute_async("DELETE FROM users")

        mock_connect.assert_awaited_once_with("postgresql://env-async", autocommit=True)

    async def test_execute_async_raises_when_db_url_missing(self):
        with patch.dict("tensorneko_util.msg.postgres.os.environ", {}, clear=True):
            with self.assertRaisesRegex(
                ValueError, "DB_URL environment variable is not set"
            ):
                await postgres.execute_async("SELECT 1")


if __name__ == "__main__":
    unittest.main()
