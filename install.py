#!/usr/bin/env python3
"""Install the Logseq memory system for Claude Code and/or Codex.

Stdlib only; run with any Python >= 3.10:

    python3 install.py --vault ~/path/to/graph/pages

What it does (each step is idempotent, every edited file is backed up first):

1. Copies the scripts to ~/.logseq-memory/bin/ and records the vault path in
   ~/.logseq-memory/config.json.
2. Creates a private virtualenv (~/.logseq-memory/venv) with
   sentence-transformers, numpy and mcp, and caches the embedding model.
3. Creates <vault>/memory/ and builds the semantic index.
4. For each agent that is present (or forced with --claude / --codex):
   - Claude Code: Stop + UserPromptSubmit hooks and the logseq-vault MCP
     server in ~/.claude/settings.json (plus the Claude Desktop config when
     the app is installed), and a managed instructions block in
     ~/.claude/CLAUDE.md.
   - Codex: SessionEnd + UserPromptSubmit hooks in ~/.codex/hooks.json,
     the logseq_vault MCP server and features.hooks in ~/.codex/config.toml,
     the $extract-memory skill, and a managed block in ~/.codex/AGENTS.md.

Existing logseq-memory hook entries (from any earlier install location) are
replaced, never duplicated; everything else in those files is kept.
`--uninstall` removes the agent registrations and managed blocks again; the
vault itself is never modified or deleted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent
SCRIPTS = (
    "logseq_memory_shared.py",
    "logseq_memory_extractor.py",
    "logseq_memory_index.py",
    "logseq_memory_retriever.py",
    "logseq_memory_mcp.py",
    "logseq_memory_migrate_links.py",
    "logseq_memory_codex.py",
    "codex_extraction_schema.json",
)
DEPENDENCIES = ("sentence-transformers", "numpy", "mcp")
MODEL_NAME = "all-MiniLM-L6-v2"
CATEGORIES = ("patterns", "mistakes", "decisions", "context", "sessions")
OFFLINE_ENV = {"TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
DEFAULT_MEMORY_HOME = Path("~/.logseq-memory").expanduser()
HOOK_MARKER = "logseq_memory_"
CLAUDE_MCP_NAME = "logseq-vault"
CODEX_MCP_NAME = "logseq_vault"
BLOCK_BEGIN = "<!-- BEGIN logseq-memory (managed by install.py; re-run to update) -->"
BLOCK_END = "<!-- END logseq-memory -->"


class Installer:
    def __init__(
        self,
        *,
        home: Path,
        memory_home: Path,
        pages_dir: Path | None,
        dry_run: bool = False,
        out=print,
        windows: bool | None = None,
    ):
        self.windows = os.name == "nt" if windows is None else windows
        self.home = home
        self.memory_home = memory_home
        self.pages_dir = pages_dir
        self.dry_run = dry_run
        self.out = out
        self.bin = memory_home / "bin"
        self.python = (
            memory_home / "venv" / "Scripts" / "python.exe" if self.windows
            else memory_home / "venv" / "bin" / "python"
        )
        self.claude_dir = home / ".claude"
        self.codex_dir = home / ".codex"
        self.desktop_config = self._desktop_config_dir() / "claude_desktop_config.json"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.backup_dir = memory_home / "backups" / stamp
        self._backed_up: set[Path] = set()

    # ------------------------------------------------------------------ utils
    def _write(self, path: Path, text: str) -> None:
        if path.exists() and path.read_text(errors="replace", encoding="utf-8") == text:
            return
        if self.dry_run:
            self.out(f"  would write {path}")
            return
        if path.exists() and path not in self._backed_up:
            # Flatten the absolute path into one file name. Joining it
            # directly would, on Windows, yield the drive path itself.
            name = "_".join(part.strip("\\/:") for part in path.parts if part.strip("\\/:"))
            target = self.backup_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            self._backed_up.add(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        self.out(f"  wrote {path}")

    def _run(self, cmd: list[str], env: dict | None = None) -> None:
        self.out(f"  $ {' '.join(cmd)}")
        if not self.dry_run:
            subprocess.run(cmd, check=True, env={**os.environ, **(env or {})})

    def _env(self) -> dict[str, str]:
        """Environment for the MCP server entries. Hooks need none: the
        scripts go offline themselves and find their install from bin/."""
        return dict(OFFLINE_ENV)

    def _join(self, argv: list[str]) -> str:
        """One command line for shells that take a string (Codex hooks)."""
        return subprocess.list2cmdline(argv) if self.windows else shlex.join(argv)

    def _desktop_config_dir(self) -> Path:
        if self.windows:
            appdata = os.environ.get("APPDATA")
            base = Path(appdata) if appdata else self.home / "AppData" / "Roaming"
            return base / "Claude"
        if sys.platform == "darwin" or (self.home / "Library").is_dir():
            return self.home / "Library" / "Application Support" / "Claude"
        return self.home / ".config" / "Claude"

    @staticmethod
    def _load_json(path: Path) -> dict:
        if not path.exists() or not path.read_text(encoding="utf-8").strip():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise SystemExit(f"{path} is not a JSON object; refusing to edit it")
        return data

    # ------------------------------------------------------------- detection
    def _apps(self, *names: str) -> bool:
        roots = [Path("/Applications"), self.home / "Applications"]
        if self.windows:
            local = os.environ.get("LOCALAPPDATA")
            local_root = Path(local) if local else self.home / "AppData" / "Local"
            roots = [local_root, local_root / "Programs"]
        return any((root / name).exists() for root in roots for name in names)

    def detect(self) -> dict[str, bool]:
        desktop = self.desktop_config.parent.is_dir() or self._apps(
            "Claude.app", "AnthropicClaude", "Claude"
        )
        return {
            "claude": self.claude_dir.is_dir() or shutil.which("claude") is not None,
            "claude_desktop": desktop,
            "codex": self.codex_dir.is_dir() or shutil.which("codex") is not None
            or self._apps("Codex.app", "ChatGPT.app"),
            "claude_cli": shutil.which("claude") is not None,
            "codex_cli": shutil.which("codex") is not None,
        }

    # ---------------------------------------------------------- shared files
    def install_files(self) -> None:
        self.out("Shared files")
        for name in SCRIPTS:
            self._write(self.bin / name, (REPO / name).read_text(encoding="utf-8"))
        config_path = self.memory_home / "config.json"
        config = self._load_json(config_path)
        config["pages_dir"] = str(self.pages_dir)
        # Hooks started by GUI apps may lack the shell PATH; remember the CLIs.
        for cli in ("claude", "codex"):
            found = shutil.which(cli)
            if found:
                config[f"{cli}_bin"] = found
        self._write(config_path, json.dumps(config, indent=2) + "\n")
        for category in CATEGORIES:
            directory = self.pages_dir / "memory" / category
            if not directory.exists():
                if self.dry_run:
                    self.out(f"  would create {directory}")
                else:
                    directory.mkdir(parents=True, exist_ok=True)

    def install_venv(self, base_python: str) -> None:
        self.out("Python environment")
        if not self.python.exists():
            self._run([base_python, "-m", "venv", str(self.memory_home / "venv")])
        self._run([str(self.python), "-m", "pip", "install", "--quiet", "--upgrade", *DEPENDENCIES])
        # Cache the model once (network); hooks then run offline.
        self._run([
            str(self.python), "-c",
            f"from sentence_transformers import SentenceTransformer; SentenceTransformer({MODEL_NAME!r})",
        ])

    def build_index(self) -> None:
        self.out("Semantic index")
        self._run(
            [str(self.python), str(self.bin / "logseq_memory_index.py"), "--quiet"],
            env={**self._env(), "LOGSEQ_PAGES_DIR": str(self.pages_dir)},
        )

    # ---------------------------------------------------------- claude code
    @staticmethod
    def _drop_memory_hooks(groups: list) -> list:
        kept = []
        for group in groups:
            if not isinstance(group, dict):
                kept.append(group)
                continue
            hooks = [
                h for h in group.get("hooks", [])
                if HOOK_MARKER not in str(h.get("command", "")) + str(h.get("args", ""))
            ]
            if hooks or not group.get("hooks"):
                kept.append({**group, "hooks": hooks})
        return kept

    def _mcp_entry(self) -> dict:
        return {
            "command": str(self.python),
            "args": [str(self.bin / "logseq_memory_mcp.py")],
            "env": self._env(),
        }

    def claude_settings(self, data: dict, remove: bool = False) -> dict:
        hooks = data.setdefault("hooks", {})
        # Exec form (command + args): no shell, so it behaves the same under
        # bash and PowerShell and survives spaces in the paths.
        wanted = {
            "Stop": {
                "type": "command",
                "command": str(self.python),
                "args": [str(self.bin / "logseq_memory_extractor.py")],
            },
            "UserPromptSubmit": {
                "type": "command",
                "command": str(self.python),
                "args": [str(self.bin / "logseq_memory_retriever.py")],
                "timeout": 10,
            },
        }
        for event, hook in wanted.items():
            groups = self._drop_memory_hooks(hooks.get(event, []))
            if not remove:
                groups.append({"matcher": "", "hooks": [hook]})
            if groups:
                hooks[event] = groups
            else:
                hooks.pop(event, None)
        if not hooks:
            data.pop("hooks")
        servers = data.setdefault("mcpServers", {})
        if remove:
            servers.pop(CLAUDE_MCP_NAME, None)
        else:
            servers[CLAUDE_MCP_NAME] = self._mcp_entry()
        if not servers:
            data.pop("mcpServers")
        # Only touch the sandbox when the user already configures one: let
        # in-session index rebuilds and vault edits write where they need to.
        if "sandbox" in data and not remove:
            allow = data["sandbox"].setdefault("filesystem", {}).setdefault("allowWrite", [])
            for path in (f"{self.memory_home}/", f"{self.pages_dir.parent}/"):
                if path not in allow:
                    allow.append(path)
        return data

    def desktop_settings(self, data: dict, remove: bool = False) -> dict:
        servers = data.setdefault("mcpServers", {})
        if remove:
            servers.pop(CLAUDE_MCP_NAME, None)
        else:
            servers[CLAUDE_MCP_NAME] = self._mcp_entry()
        return data

    def install_claude(self, desktop: bool, remove: bool = False) -> None:
        self.out("Claude Code")
        settings = self.claude_dir / "settings.json"
        data = self.claude_settings(self._load_json(settings), remove)
        self._write(settings, json.dumps(data, indent=2) + "\n")
        if desktop:
            data = self.desktop_settings(self._load_json(self.desktop_config), remove)
            self._write(self.desktop_config, json.dumps(data, indent=2) + "\n")
        self._instructions(self.claude_dir / "CLAUDE.md", self._claude_block(), remove)

    # ----------------------------------------------------------------- codex
    def codex_hooks(self, data: dict, remove: bool = False) -> dict:
        template = json.loads((REPO / "codex" / "hooks.json").read_text(encoding="utf-8"))
        data.setdefault("description", template.get("description", ""))
        hooks = data.setdefault("hooks", {})
        for event, groups in template["hooks"].items():
            kept = self._drop_memory_hooks(hooks.get(event, []))
            if not remove:
                for group in groups:
                    group = json.loads(json.dumps(group))
                    for hook in group.get("hooks", []):
                        hook["command"] = self._fill_command(hook["command"])
                    kept.append(group)
            if kept:
                hooks[event] = kept
            else:
                hooks.pop(event, None)
        return data

    def _fill_command(self, template: str) -> str:
        """Expand a "__LOGSEQ_PYTHON__ __LOGSEQ_BIN__/script.py arg" template
        into a correctly quoted command line for this platform."""
        argv = []
        for token in template.split():
            if token == "__LOGSEQ_PYTHON__":
                token = str(self.python)
            elif token.startswith("__LOGSEQ_BIN__/"):
                token = str(self.bin / token.split("/", 1)[1])
            argv.append(token)
        return self._join(argv)

    def _fill(self, text: str) -> str:
        """Fill command templates inside `backticks` in a Markdown file."""
        return re.sub(
            r"`([^`]*__LOGSEQ_[^`]*)`", lambda m: f"`{self._fill_command(m.group(1))}`", text
        )

    def codex_config(self, text: str, remove: bool = False) -> str:
        sections = _toml_sections(text)
        main = f"mcp_servers.{CODEX_MCP_NAME}"
        if remove:
            sections = [s for s in sections if not _is_table(s[0], main)]
            return _toml_render(sections)
        _toml_set(sections, main, {
            "command": json.dumps(str(self.python)),
            "args": json.dumps([str(self.bin / "logseq_memory_mcp.py")]),
        }, defaults={"startup_timeout_sec": "60.0"})
        _toml_set(sections, f"{main}.env", {k: json.dumps(v) for k, v in self._env().items()})
        _toml_set(sections, "features", {"hooks": "true"})
        rendered = _toml_render(sections)
        try:
            import tomllib  # Python 3.11+
            tomllib.loads(rendered)
        except ImportError:
            pass
        except Exception as exc:  # never leave Codex with a broken config
            raise SystemExit(f"refusing to write invalid {self.codex_dir / 'config.toml'}: {exc}")
        return rendered

    def install_codex(self, remove: bool = False) -> None:
        self.out("Codex")
        hooks_path = self.codex_dir / "hooks.json"
        data = self.codex_hooks(self._load_json(hooks_path), remove)
        self._write(hooks_path, json.dumps(data, indent=2) + "\n")
        config = self.codex_dir / "config.toml"
        current = config.read_text(encoding="utf-8") if config.exists() else ""
        self._write(config, self.codex_config(current, remove))
        skill_src = REPO / "codex" / "skills" / "extract-memory"
        skill_dst = self.codex_dir / "skills" / "extract-memory"
        if remove:
            if skill_dst.exists() and not self.dry_run:
                shutil.rmtree(skill_dst)
                self.out(f"  removed {skill_dst}")
        else:
            for src in sorted(p for p in skill_src.rglob("*") if p.is_file()):
                self._write(skill_dst / src.relative_to(skill_src), self._fill(src.read_text(encoding="utf-8")))
        self._instructions(self.codex_dir / "AGENTS.md", self._codex_block(), remove)

    # ---------------------------------------------------------- instructions
    def _memory_lines(self) -> list[str]:
        # Forward slashes on every platform: Windows accepts them, and a
        # single separator keeps the paths readable in Markdown.
        root = (self.pages_dir / "memory").as_posix()
        return [
            "## Persistent memory (Logseq vault)",
            "",
            f"- Insight pages live under `{root}/` in `patterns/`, `mistakes/`,",
            "  `decisions/`, `context/` and `sessions/`.",
            "- A prompt hook searches the vault and injects the closest pages as a",
            "  system reminder. Check those matches before acting.",
            f"- Read `{root}/digest.md` once at the start of each session for recent",
            "  context; do not re-read it every turn.",
        ]

    def _claude_block(self) -> str:
        return "\n".join(self._memory_lines() + [
            f"- Save non-obvious learnings with the `{CLAUDE_MCP_NAME}` MCP server's",
            "  `write_insight` tool (search first to avoid duplicates).",
        ])

    def _codex_block(self) -> str:
        return "\n".join(self._memory_lines() + [
            "- Save non-obvious learnings with `$extract-memory`, which writes through",
            f"  the `{CODEX_MCP_NAME}` MCP server's `write_codex_insight` tool.",
        ])

    def _instructions(self, path: Path, block: str, remove: bool) -> None:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        pattern = re.compile(re.escape(BLOCK_BEGIN) + r".*?" + re.escape(BLOCK_END) + r"\n?", re.S)
        has_block = pattern.search(text) is not None
        outside = pattern.sub("", text)
        if has_block:
            # The block is separated by one blank line; drop it with the block.
            outside = outside.rstrip("\n") + "\n" if outside.strip() else ""
        if remove:
            if has_block:
                self._write(path, outside)
            return
        # Match the digest path written with either separator, absolute or ~/.
        digest = (self.pages_dir / "memory" / "digest.md").as_posix()
        tilde = digest.replace(self.home.as_posix(), "~", 1)
        normalized = outside.replace("\\", "/")
        if digest in normalized or tilde in normalized:
            self.out(f"  {path} already describes the vault; left unchanged")
            if has_block:  # drop a stale managed block next to it
                self._write(path, outside)
            return
        managed = f"{BLOCK_BEGIN}\n{block}\n{BLOCK_END}\n"
        if has_block:
            new = pattern.sub(lambda _m: managed, text, count=1)
        else:
            new = (text.rstrip("\n") + "\n\n" if text.strip() else "") + managed
        self._write(path, new)


# ---------------------------------------------------------------- toml edits
# Minimal line-based editing: enough to set keys in a few known tables while
# leaving every other byte of the user's config.toml as it was.
_HEADER = re.compile(r"^\s*\[\s*([^\[\]]+?)\s*\]\s*(#.*)?$")


def _toml_sections(text: str) -> list[list]:
    sections: list[list] = [[None, []]]
    for line in text.splitlines():
        match = _HEADER.match(line)
        if match and not line.lstrip().startswith("[["):
            sections.append([match.group(1).replace('"', "").replace("'", ""), [line]])
        else:
            sections[-1][1].append(line)
    return sections


def _toml_render(sections: list[list]) -> str:
    lines = [line for _, body in sections for line in body]
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines) + "\n" if lines else ""


def _is_table(name: str | None, table: str) -> bool:
    return name is not None and (name == table or name.startswith(table + "."))


def _toml_set(sections: list[list], table: str, values: dict, defaults: dict | None = None) -> None:
    section = next((s for s in sections if s[0] == table), None)
    if section is None:
        last = sections[-1][1]
        if last and last[-1].strip():
            last.append("")
        section = [table, [f"[{table}]"]]
        # Keep a sub-table next to its parent table when the parent exists.
        parent_index = max(
            (i for i, s in enumerate(sections) if s[0] and _is_table(s[0], table.rsplit(".", 1)[0])),
            default=None,
        )
        if "." in table and parent_index is not None:
            sections.insert(parent_index + 1, section)
        else:
            sections.append(section)
    body = section[1]
    for key, value in {**(defaults or {}), **values}.items():
        key_re = re.compile(rf"^\s*{re.escape(key)}\s*=")
        index = next((i for i, line in enumerate(body) if key_re.match(line)), None)
        if index is not None:
            if key in values:
                body[index] = f"{key} = {value}"
            continue
        insert_at = len(body)
        while insert_at > 1 and not body[insert_at - 1].strip():
            insert_at -= 1
        body.insert(insert_at, f"{key} = {value}")
    if body and body[-1].strip():
        body.append("")


# ---------------------------------------------------------------------- main
def _resolve_pages_dir(arg: str | None, memory_home: Path) -> Path | None:
    if arg:
        return Path(arg).expanduser().resolve()
    if os.environ.get("LOGSEQ_PAGES_DIR"):
        return Path(os.environ["LOGSEQ_PAGES_DIR"]).expanduser().resolve()
    try:
        configured = json.loads((memory_home / "config.json").read_text(encoding="utf-8")).get("pages_dir")
    except (OSError, ValueError, AttributeError):
        configured = None
    if configured:
        return Path(configured).expanduser()
    if sys.stdin.isatty():
        answer = input("Path to your Logseq graph's pages/ directory: ").strip()
        if answer:
            return Path(answer).expanduser().resolve()
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vault", help="Logseq graph pages/ directory (memory pages go in <vault>/memory/)")
    parser.add_argument("--home", default=str(DEFAULT_MEMORY_HOME), help="install directory (default ~/.logseq-memory)")
    parser.add_argument("--python", default=sys.executable, help="interpreter used to create the venv")
    for agent in ("claude", "codex"):
        group = parser.add_mutually_exclusive_group()
        group.add_argument(f"--{agent}", dest=agent, action="store_true", default=None,
                           help=f"register with {agent} even if not detected")
        group.add_argument(f"--no-{agent}", dest=agent, action="store_false",
                           help=f"do not register with {agent}")
    parser.add_argument("--skip-deps", action="store_true", help="do not create/update the venv")
    parser.add_argument("--skip-index", action="store_true", help="do not build the semantic index")
    parser.add_argument("--dry-run", action="store_true", help="show what would change")
    parser.add_argument("--uninstall", action="store_true",
                        help="remove agent registrations and managed blocks (keeps vault, venv and index)")
    args = parser.parse_args(argv)

    memory_home = Path(args.home).expanduser().resolve()
    pages_dir = _resolve_pages_dir(args.vault, memory_home)
    if pages_dir is None and not args.uninstall:
        parser.error("pass --vault /path/to/graph/pages (or set LOGSEQ_PAGES_DIR)")
    installer = Installer(
        home=Path.home(), memory_home=memory_home,
        pages_dir=pages_dir or Path("."), dry_run=args.dry_run,
    )
    found = installer.detect()
    use_claude = found["claude"] if args.claude is None else args.claude
    use_codex = found["codex"] if args.codex is None else args.codex
    print(f"Vault: {pages_dir}  |  install: {memory_home}")
    print(f"Agents: Claude Code={'yes' if use_claude else 'no'}"
          f" (Desktop app={'yes' if use_claude and found['claude_desktop'] else 'no'}),"
          f" Codex={'yes' if use_codex else 'no'}")

    if args.uninstall:
        if use_claude:
            installer.install_claude(found["claude_desktop"], remove=True)
        if use_codex:
            installer.install_codex(remove=True)
        print("Uninstalled agent registrations. Vault, venv and index were left in place.")
        return 0

    if use_claude and not found["claude_cli"]:
        print("Warning: `claude` is not on PATH. Prompt retrieval and the MCP server will "
              "work, but saving insights at session end runs `claude -p`; install the "
              "Claude Code CLI and re-run this installer.")
    if use_codex and not found["codex_cli"]:
        print("Warning: `codex` is not on PATH. Retrieval and the MCP server will work, "
              "but the SessionEnd worker runs `codex exec`; install the Codex CLI "
              "(or set CODEX_BIN) and re-run this installer.")
    if not (use_claude or use_codex):
        print("Neither Claude Code nor Codex found; installing shared files only "
              "(use --claude / --codex to register anyway).")
    installer.install_files()
    if not args.skip_deps:
        installer.install_venv(args.python)
    if not args.skip_index:
        installer.build_index()
    if use_claude:
        installer.install_claude(found["claude_desktop"])
    if use_codex:
        installer.install_codex()
    if installer._backed_up:
        print(f"Backups of edited files: {installer.backup_dir}")
    print("Done. Restart Claude Desktop / Claude Code and Codex"
          + (" (then review and trust the hooks with /hooks in Codex)" if use_codex else "") + ".")
    return 0


if __name__ == "__main__":
    sys.exit(main())
