#!/usr/bin/env python3
"""
Full rebuild of the semantic vault index.

Run once after installation, then re-run if the vault has drifted significantly
or after a model change. Incremental updates are handled automatically by the
Stop hook (logseq_memory_extractor.py) after each session.

Usage:
    python3 logseq_memory_index.py
    python3 logseq_memory_index.py --quiet   # suppress progress bar

Requirements:
    pip install sentence-transformers numpy
"""

import sys
from pathlib import Path

import logseq_memory_shared as shared

LOGSEQ_PAGES_DIR = shared.DEFAULT_PAGES_DIR
INDEX_PATH = shared.DEFAULT_INDEX_PATH
CATEGORIES = ["patterns", "mistakes", "decisions", "context"]
MODEL_NAME = "all-MiniLM-L6-v2"
MAX_TEXT_CHARS = 500  # chars per file passed to the encoder (title + summary)


def extract_text(path: Path) -> str:
    """Extract title + summary from a Logseq insight page for embedding."""
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""

    title = ""
    summary_parts: list[str] = []
    in_summary = False

    for line in lines:
        if line.startswith("title::"):
            title = line[7:].strip()
        elif "## Summary" in line:
            in_summary = True
        elif in_summary:
            stripped = line.strip()
            if stripped.startswith("- ##"):
                break
            if stripped.startswith("- "):
                summary_parts.append(stripped[2:].strip())
                if len(summary_parts) >= 3:
                    break

    return f"{title}. {' '.join(summary_parts)}".strip(". ")[:MAX_TEXT_CHARS]


def build_index(verbose: bool = True) -> None:
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "Missing dependencies. Install with:\n  pip install sentence-transformers numpy",
            file=sys.stderr,
        )
        sys.exit(1)

    if verbose:
        print(f"Loading model '{MODEL_NAME}'...", flush=True)
    model = SentenceTransformer(MODEL_NAME)

    # Hold the vault lock from scan to save: an incremental update_vault_index()
    # (Stop hook, MCP, Codex) landing mid-rebuild would otherwise be overwritten.
    with shared.vault_lock():
        _rebuild(model, np, verbose)


def _rebuild(model, np, verbose: bool) -> None:
    paths: list[str] = []
    texts: list[str] = []

    for cat in CATEGORIES:
        subdir = shared.memory_root(LOGSEQ_PAGES_DIR) / cat
        if not subdir.exists():
            continue
        for p in sorted(subdir.glob("*.md")):
            text = extract_text(p)
            if text and len(text) > 10:
                paths.append(str(p))
                texts.append(text)

    if texts:
        if verbose:
            print(f"Embedding {len(texts):,} files...", flush=True)
        embeddings = model.encode(
            texts,
            show_progress_bar=verbose,
            batch_size=256,
            normalize_embeddings=True,  # pre-normalised: dot product == cosine similarity
        )
    else:
        # A fresh vault: write an empty index so incremental appends from the
        # writers (which skip when no index exists) start working right away.
        print(
            f"No pages yet under {shared.memory_root(LOGSEQ_PAGES_DIR)}; "
            "writing an empty index.",
            file=sys.stderr,
        )
        embeddings = np.zeros((0, model.get_sentence_embedding_dimension()))

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        INDEX_PATH,
        embeddings=embeddings.astype("float32"),
        paths=np.array(paths),
    )

    if verbose:
        size_mb = INDEX_PATH.stat().st_size / (1024 * 1024)
        print(
            f"Index saved to {INDEX_PATH}\n"
            f"  {len(paths):,} entries  |  {size_mb:.1f} MB",
            flush=True,
        )


if __name__ == "__main__":
    verbose = "--quiet" not in sys.argv
    build_index(verbose=verbose)
