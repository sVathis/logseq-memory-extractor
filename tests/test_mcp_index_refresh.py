import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np
except ImportError:
    np = None
import logseq_memory_mcp as mcp


@unittest.skipIf(np is None, "numpy not installed")
class IndexRefreshTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.index = self.root / 'index.npz'
        self.pages = self.root / 'decisions'
        self.pages.mkdir()
        self.old = self.pages / 'old.md'
        self.new = self.pages / 'new.md'
        self.old.write_text('old page')
        self.new.write_text('new page')
        self.constructor = mock.Mock()
        self.constructor.return_value.encode.return_value = np.array([[1., 0.]])
        for patch in [
            mock.patch.object(mcp, 'INDEX_PATH', self.index),
            mock.patch.object(mcp, '_model', None),
            mock.patch.object(mcp, '_embeddings', None),
            mock.patch.object(mcp, '_paths', None),
            mock.patch.object(mcp, '_index_signature', None, create=True),
            mock.patch.dict(os.environ, {'LOGSEQ_LOCK_PATH': str(self.root / 'index.lock')}),
            mock.patch.dict('sys.modules', {'sentence_transformers': mock.Mock(SentenceTransformer=self.constructor)}),
        ]:
            patch.start()
            self.addCleanup(patch.stop)
        self.save(self.index, self.old)

    def save(self, target, page):
        np.savez_compressed(target, embeddings=np.array([[1., 0.]], dtype='float32'), paths=np.array([str(page)]))

    def search(self):
        return [r['path'] for r in mcp._search('query', 3, 'decisions')]

    def test_same_process_sees_replaced_index_and_reuses_model(self):
        self.assertEqual(self.search(), [str(self.old)])
        previous = self.index.stat()
        replacement = self.root / 'replacement.npz'
        self.save(replacement, self.new)
        os.utime(replacement, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        replacement.replace(self.index)
        self.assertEqual(self.search(), [str(self.new)])
        self.constructor.assert_called_once_with(mcp.MODEL_NAME)

    def test_same_process_sees_in_place_update(self):
        self.assertEqual(self.search(), [str(self.old)])
        self.save(self.index, self.new)
        self.assertEqual(self.search(), [str(self.new)])

    def test_unchanged_index_is_not_reloaded(self):
        self.assertEqual(self.search(), [str(self.old)])
        with mock.patch.object(np, 'load', wraps=np.load) as load:
            self.assertEqual(self.search(), [str(self.old)])
            load.assert_not_called()
        self.constructor.assert_called_once()

    def test_missing_index_does_not_serve_stale_results_and_recovers(self):
        self.assertEqual(self.search(), [str(self.old)])
        self.index.unlink()
        self.assertEqual(self.search(), [])
        self.save(self.index, self.new)
        self.assertEqual(self.search(), [str(self.new)])

    def test_invalid_update_does_not_publish_partial_state_and_recovers(self):
        self.assertEqual(self.search(), [str(self.old)])
        before = mcp._embeddings, mcp._paths
        np.savez_compressed(self.index, embeddings=np.array([[0., 1.]]))
        self.assertEqual(self.search(), [])
        self.assertIs(mcp._embeddings, before[0])
        self.assertIs(mcp._paths, before[1])
        self.save(self.index, self.new)
        self.assertEqual(self.search(), [str(self.new)])

    def test_index_read_uses_writer_lock(self):
        with mock.patch.object(mcp.shared, 'vault_lock', wraps=mcp.shared.vault_lock) as lock:
            self.assertEqual(self.search(), [str(self.old)])
            lock.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
