import builtins
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import tensorneko_util.backend.tqdm as tqdm_backend


class TestTqdmBackend(unittest.TestCase):
    def setUp(self):
        self._orig_tqdm_available = tqdm_backend._is_tqdm_available
        self._orig_mita_available = tqdm_backend._is_mita_tqdm_available
        tqdm_backend._is_tqdm_available = None
        tqdm_backend._is_mita_tqdm_available = None

    def tearDown(self):
        tqdm_backend._is_tqdm_available = self._orig_tqdm_available
        tqdm_backend._is_mita_tqdm_available = self._orig_mita_available

    def _import_hook(self, failing=(), replacements=None):
        real_import = builtins.__import__
        failing = set(failing)
        replacements = replacements or {}

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            root_name = name.split(".")[0]
            if name in replacements:
                return replacements[name]
            if root_name in replacements:
                return replacements[root_name]
            if name in failing or root_name in failing:
                raise ImportError(f"mock missing module: {name}")
            return real_import(name, globals, locals, fromlist, level)

        return fake_import

    def test_import_tqdm_success_sets_cache(self):
        fake_tqdm = SimpleNamespace(__name__="tqdm")
        with patch(
            "builtins.__import__",
            side_effect=self._import_hook(replacements={"tqdm": fake_tqdm}),
        ):
            module = tqdm_backend.import_tqdm()

        self.assertIs(module, fake_tqdm)
        self.assertTrue(tqdm_backend._is_tqdm_available)

    def test_import_tqdm_missing_sets_false_then_cached_false_raises_without_import(
        self,
    ):
        with patch(
            "builtins.__import__", side_effect=self._import_hook(failing={"tqdm"})
        ):
            with self.assertRaises(ImportError) as ctx:
                tqdm_backend.import_tqdm()
        self.assertIn("pip install tqdm", str(ctx.exception))
        self.assertFalse(tqdm_backend._is_tqdm_available)

        with patch(
            "builtins.__import__", side_effect=AssertionError("unexpected import")
        ):
            with self.assertRaises(ImportError):
                tqdm_backend.import_tqdm()

    def test_import_tqdm_auto_success_sets_cache(self):
        fake_auto = SimpleNamespace(__name__="auto")
        fake_tqdm = SimpleNamespace(auto=fake_auto)
        with patch(
            "builtins.__import__",
            side_effect=self._import_hook(replacements={"tqdm": fake_tqdm}),
        ):
            module = tqdm_backend.import_tqdm_auto()

        self.assertIs(module, fake_auto)
        self.assertTrue(tqdm_backend._is_tqdm_available)

    def test_import_tqdm_auto_cached_true_branch(self):
        fake_auto = SimpleNamespace(__name__="auto")
        fake_tqdm = SimpleNamespace(auto=fake_auto)
        tqdm_backend._is_tqdm_available = True
        with patch(
            "builtins.__import__",
            side_effect=self._import_hook(replacements={"tqdm": fake_tqdm}),
        ):
            module = tqdm_backend.import_tqdm_auto()
        self.assertIs(module, fake_auto)

    def test_import_mita_tqdm_success_and_cached_true_branch(self):
        fake_mita_tqdm = SimpleNamespace(__name__="mita_tqdm")
        fake_mita_client = SimpleNamespace(mita_tqdm=fake_mita_tqdm)
        with patch(
            "builtins.__import__",
            side_effect=self._import_hook(
                replacements={"mita_client": fake_mita_client}
            ),
        ):
            module = tqdm_backend.import_mita_tqdm()
        self.assertIs(module, fake_mita_tqdm)
        self.assertTrue(tqdm_backend._is_mita_tqdm_available)

        tqdm_backend._is_mita_tqdm_available = True
        with patch(
            "builtins.__import__",
            side_effect=self._import_hook(
                replacements={"mita_client": fake_mita_client}
            ),
        ):
            module = tqdm_backend.import_mita_tqdm()
        self.assertIs(module, fake_mita_tqdm)

    def test_import_mita_tqdm_missing_sets_false_then_cached_false_raises_without_import(
        self,
    ):
        with patch(
            "builtins.__import__",
            side_effect=self._import_hook(failing={"mita_client"}),
        ):
            with self.assertRaises(ImportError) as ctx:
                tqdm_backend.import_mita_tqdm()
        self.assertIn("pip install mita_client", str(ctx.exception))
        self.assertFalse(tqdm_backend._is_mita_tqdm_available)

        with patch(
            "builtins.__import__", side_effect=AssertionError("unexpected import")
        ):
            with self.assertRaises(ImportError):
                tqdm_backend.import_mita_tqdm()

    def test_import_tqdm_cached_true_branch(self):
        """Lines 17-18: cached True branch for import_tqdm."""
        tqdm_backend._is_tqdm_available = True
        module = tqdm_backend.import_tqdm()
        import tqdm

        self.assertIs(module, tqdm)

    def test_import_tqdm_auto_cached_false_raises(self):
        """Lines 30-32: cached False branch for import_tqdm_auto raises."""
        tqdm_backend._is_tqdm_available = False
        with self.assertRaises(ImportError) as ctx:
            tqdm_backend.import_tqdm_auto()
        self.assertIn("pip install tqdm", str(ctx.exception))

    def test_import_tqdm_cached_false_raises(self):
        """Lines 17-20: cached False branch for import_tqdm raises."""
        tqdm_backend._is_tqdm_available = False
        with self.assertRaises(ImportError) as ctx:
            tqdm_backend.import_tqdm()
        self.assertIn("pip install tqdm", str(ctx.exception))

    def test_import_mita_tqdm_cached_false_raises(self):
        """Line 38: cached False branch for import_mita_tqdm raises."""
        tqdm_backend._is_mita_tqdm_available = False
        with self.assertRaises(ImportError) as ctx:
            tqdm_backend.import_mita_tqdm()
        self.assertIn("pip install mita_client", str(ctx.exception))

    def test_import_tqdm_auto_none_import_fails(self):
        """Lines 30-32: import_tqdm_auto with _is_tqdm_available=None and tqdm import fails."""
        tqdm_backend._is_tqdm_available = None
        with patch(
            "builtins.__import__", side_effect=self._import_hook(failing={"tqdm"})
        ):
            with self.assertRaises(ImportError) as ctx:
                tqdm_backend.import_tqdm_auto()
        self.assertIn("pip install tqdm", str(ctx.exception))
        self.assertFalse(tqdm_backend._is_tqdm_available)


if __name__ == "__main__":
    unittest.main()
