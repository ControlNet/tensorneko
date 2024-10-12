import os
from typing import Any, List, Optional, Tuple

import psycopg


def execute(sql: str, db_url: Optional[str] = None) -> Optional[List[Tuple[Any, ...]]]:
    """
    Execute a PostgreSQL database SQL query.

    Args:
        sql (``str``): The SQL query to be executed.
        db_url (``str``, optional): The URL of the PostgreSQL database. The default value is environment variable DB_URL.

    Returns:
        ``list[tuple] | None``: The result of the query. If the query is not a SELECT query, then it will return None.

    Examples::

        result = execute("SELECT * FROM table_name", "postgresql://user:password@localhost:5432/db_name")
        # then the result will be a list of tuples.

        execute("INSERT INTO table_name (column1, column2) VALUES (value1, value2)", "postgresql://user:password@localhost:5432/db_name")
        # then the execute will be executed and committed.
    """
    db_url = db_url or os.environ.get("DB_URL")

    with psycopg.Connection.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
            if cur.description:
                return cur.fetchall()
            return None


async def execute_async(sql: str, db_url: Optional[str] = None) -> Optional[List[Tuple[Any, ...]]]:
    """
    Execute a PostgreSQL database SQL query with async.

    Args:
        sql (``str``): The SQL query to be executed.
        db_url (``str``, optional): The URL of the PostgreSQL database. The default value is environment variable DB_URL.

    Returns:
        ``list[tuple] | None``: The result of the query. If the query is not a SELECT query, then it will return None.

    Examples::

        result = await execute_async("SELECT * FROM table_name", "postgresql://user:password@localhost:5432/db_name")
        # then the result will be a list of tuples.

        await execute_async("INSERT INTO table_name (column1, column2) VALUES (value1, value2)")
        # then the execute will be executed
    """
    db_url = db_url or os.environ.get("DB_URL")

    if db_url is None:
        raise ValueError("DB_URL environment variable is not set.")

    async with await psycopg.AsyncConnection.connect(db_url, autocommit=True) as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql)
            if cur.description:
                return await cur.fetchall()
            return None
