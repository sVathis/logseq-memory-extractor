import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import logseq_memory_codex as codex
import logseq_memory_index as indexer
import logseq_memory_mcp as mcp
from tests.test_codex_memory import jsonl, message, session_meta, turn


class ReadPageScopeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.pages = self.root / "graph" / "pages"
        self.pages.mkdir(parents=True)
        self.patch = mock.patch.object(mcp, "LOGSEQ_PAGES_DIR", self.pages)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_reads_files_inside_the_graph(self):
        journal = self.root / "graph" / "journals" / "2026_10_17.md"
        journal.parent.mkdir()
        journal.write_text("- hello")
        self.assertEqual(mcp._read_page(str(journal)), "- hello")

    def test_refuses_files_outside_the_graph(self):
        secret = self.root / "secret.txt"
        secret.write_text("token")
        self.assertTrue(mcp._read_page(str(secret)).startswith("Refused"))
        escape = self.pages / ".." / ".." / "secret.txt"
        self.assertTrue(mcp._read_page(str(escape)).startswith("Refused"))


class ManualBurstGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pages = Path(self.tmp.name) / "pages"
        self.patches = [
            mock.patch.object(mcp, "LOGSEQ_PAGES_DIR", self.pages),
            mock.patch.object(mcp, "INDEX_PATH", Path(self.tmp.name) / "missing.npz"),
            mock.patch.object(mcp, "_MANUAL_WRITES", {}),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf"]

    def write(self, n, **kwargs):
        # Distinct leading words so the slug-prefix check never fires.
        word = self.WORDS[n % len(self.WORDS)] + str(n)
        return mcp._write_insight(
            "decisions", f"{word} unrelated decision", "s", "d",
            "claude-opus-5-5", **kwargs,
        )

    def test_other_processes_manual_pages_do_not_count(self):
        # Five session-less pages from "earlier sessions" the same day.
        for n in range(5):
            self.assertTrue(self.write(n).startswith("Written"))
        mcp._MANUAL_WRITES.clear()  # a new server process / client session
        self.assertTrue(self.write(5).startswith("Written"))

    def test_this_process_is_held_after_threshold(self):
        for n in range(mcp.BURST_THRESHOLD):
            self.assertTrue(self.write(n).startswith("Written"))
        self.assertTrue(self.write(99).startswith("Held"))
        self.assertTrue(self.write(99, force=True).startswith("Written"))


class CodexTitleRedactionTests(unittest.TestCase):
    def test_conversation_title_is_redacted(self):
        raw = jsonl(
            session_meta("/work/project"),
            turn("gpt-5.6-sol"),
            message("user", "deploy with token=abc123secret please"),
            message("assistant", "ok"),
        )
        parsed = codex.parse_transcript(raw)
        self.assertNotIn("abc123secret", parsed.title)
        self.assertIn("[REDACTED]", parsed.title)


class IndexRebuildLockTests(unittest.TestCase):
    def test_rebuild_runs_under_vault_lock(self):
        events = []

        @contextlib.contextmanager
        def fake_lock(*_args, **_kwargs):
            events.append("lock")
            yield
            events.append("unlock")

        def fake_rebuild(*_args, **_kwargs):
            events.append("rebuild")

        fake = mock.MagicMock()
        with (
            mock.patch.dict(
                "sys.modules", {"sentence_transformers": fake, "numpy": fake}
            ),
            mock.patch.object(indexer.shared, "vault_lock", fake_lock),
            mock.patch.object(indexer, "_rebuild", fake_rebuild),
        ):
            indexer.build_index(verbose=False)
        self.assertEqual(events, ["lock", "rebuild", "unlock"])


if __name__ == "__main__":
    unittest.main()
