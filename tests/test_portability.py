import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import logseq_memory_mcp as mcp
import logseq_memory_shared as shared

REPO = Path(__file__).resolve().parent.parent


class FakeMsvcrt:
    LK_LOCK = 1
    LK_UNLCK = 0

    def __init__(self, fail_first=0):
        self.calls = []
        self.fail_first = fail_first

    def locking(self, fd, mode, nbytes):
        if mode == self.LK_LOCK and self.fail_first:
            self.fail_first -= 1
            raise OSError("busy")
        self.calls.append((mode, nbytes))


class VaultLockTests(unittest.TestCase):
    def test_windows_lock_retries_then_locks_and_unlocks(self):
        fake = FakeMsvcrt(fail_first=2)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(shared, "fcntl", None), \
                mock.patch.object(shared, "msvcrt", fake, create=True):
            with shared.vault_lock(Path(tmp) / "x.lock"):
                self.assertEqual(fake.calls, [(fake.LK_LOCK, 1)])
        self.assertEqual(fake.calls, [(fake.LK_LOCK, 1), (fake.LK_UNLCK, 1)])

    @unittest.skipIf(os.name == "nt", "POSIX flock path")
    def test_posix_lock_still_uses_flock(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(shared.fcntl, "flock", wraps=shared.fcntl.flock) as flock:
            with shared.vault_lock(Path(tmp) / "x.lock"):
                pass
        self.assertEqual(flock.call_count, 2)


class NotifyTests(unittest.TestCase):
    def test_noop_off_macos(self):
        with mock.patch.object(mcp.sys, "platform", "win32"), \
                mock.patch.object(mcp, "_osascript_notify") as notify:
            mcp._notify("t", "m")
        notify.assert_not_called()

    def test_missing_osascript_is_not_fatal(self):
        with mock.patch.object(mcp.sys, "platform", "darwin"), \
                mock.patch.object(mcp, "_osascript_notify", side_effect=FileNotFoundError):
            mcp._notify("t", "m")


class InstallLocationTests(unittest.TestCase):
    def test_scripts_find_their_install_and_vault_without_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "custom home"
            (home / "bin").mkdir(parents=True)
            shutil.copy(REPO / "logseq_memory_shared.py", home / "bin")
            (home / "config.json").write_text(json.dumps({"pages_dir": str(Path(tmp) / "vault")}), encoding="utf-8")
            spec = importlib.util.spec_from_file_location("installed_shared", home / "bin" / "logseq_memory_shared.py")
            module = importlib.util.module_from_spec(spec)
            env = {k: v for k, v in os.environ.items()
                   if k not in ("LOGSEQ_MEMORY_HOME", "LOGSEQ_PAGES_DIR", "LOGSEQ_INDEX_PATH")}
            with mock.patch.dict(os.environ, env, clear=True):
                spec.loader.exec_module(module)
            self.assertEqual(module.MEMORY_HOME, home.resolve())
            self.assertEqual(module.DEFAULT_PAGES_DIR, Path(tmp) / "vault")
            self.assertEqual(module.DEFAULT_INDEX_PATH, home.resolve() / "vault_index.npz")

    def test_find_cli_falls_back_to_recorded_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "claude"
            binary.write_text("", encoding="utf-8")
            with mock.patch.object(shared.shutil, "which", return_value=None), \
                    mock.patch.object(shared, "_home_config", return_value={"claude_bin": str(binary)}):
                self.assertEqual(shared.find_cli("claude"), str(binary))

    def test_offline_defaults_do_not_override_explicit_settings(self):
        with mock.patch.dict(os.environ, {"TRANSFORMERS_OFFLINE": "0"}, clear=False), \
                mock.patch.object(shared, "model_is_cached", return_value=True):
            for name in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE"):
                os.environ.pop(name, None)
            shared.use_offline_models()
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "0")
            # An explicit choice for one variable leaves the others alone.
            self.assertNotIn("HF_DATASETS_OFFLINE", os.environ)
            self.assertNotIn("HF_HUB_OFFLINE", os.environ)

    def test_cached_model_sets_all_offline_vars(self):
        with mock.patch.dict(os.environ, {}, clear=False), \
                mock.patch.object(shared, "model_is_cached", return_value=True):
            for name in shared.OFFLINE_VARS:
                os.environ.pop(name, None)
            shared.use_offline_models()
            for name in shared.OFFLINE_VARS:
                self.assertEqual(os.environ[name], "1")

    def test_stays_online_until_the_model_is_downloaded(self):
        with mock.patch.dict(os.environ, {}, clear=False), \
                mock.patch.object(shared, "model_is_cached", return_value=False):
            for name in shared.OFFLINE_VARS:
                os.environ.pop(name, None)
            shared.use_offline_models()
            for name in shared.OFFLINE_VARS:
                self.assertNotIn(name, os.environ)


class ModelCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = mock.patch.dict(os.environ, {"HF_HOME": str(self.root / "hf"),
                                                "SENTENCE_TRANSFORMERS_HOME": str(self.root / "st")})
        self.env.start()
        for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
            os.environ.pop(name, None)

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_empty_cache_is_not_cached(self):
        self.assertFalse(shared.model_is_cached())

    def test_hub_snapshot_counts_as_cached(self):
        snap = self.root / "hf" / "hub" / f"models--sentence-transformers--{shared.EMBEDDING_MODEL}" / "snapshots" / "abc"
        snap.mkdir(parents=True)
        self.assertTrue(shared.model_is_cached())

    def test_empty_snapshots_dir_is_not_cached(self):
        (self.root / "hf" / "hub" / f"models--sentence-transformers--{shared.EMBEDDING_MODEL}" / "snapshots").mkdir(parents=True)
        self.assertFalse(shared.model_is_cached())

    def test_legacy_sentence_transformers_cache_counts(self):
        legacy = self.root / "st" / f"sentence-transformers_{shared.EMBEDDING_MODEL}"
        legacy.mkdir(parents=True)
        (legacy / "config.json").write_text("{}", encoding="utf-8")
        self.assertTrue(shared.model_is_cached())


if __name__ == "__main__":
    unittest.main()
