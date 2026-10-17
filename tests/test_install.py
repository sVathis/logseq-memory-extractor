import json
import os
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import install

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10
    tomllib = None


EXISTING_CODEX_CONFIG = """\
model = "gpt-6-sol"

[mcp_servers.logseq_vault]
args = ['/old/hooks/logseq_memory_mcp.py']
command = '/old/python3'
required = true

[mcp_servers.logseq_vault.env]
HF_DATASETS_OFFLINE = '1'

[mcp_servers.logseq_vault.tools.read_page]
approval_mode = 'approve'

[mcp_servers.other]
command = "other"
"""


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.pages = self.home / "graph" / "pages"
        self.installer = self.make()

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, dry_run=False, windows=False, home=None):
        home = home or self.home
        return install.Installer(
            home=home,
            memory_home=home / ".logseq-memory",
            pages_dir=self.pages,
            dry_run=dry_run,
            out=lambda *_a: None,
            windows=windows,
        )

    def write_json(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def read_json(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    def snapshot(self):
        return {
            p: p.read_text(encoding="utf-8")
            for p in self.home.rglob("*")
            if p.is_file() and "backups" not in p.parts
        }

    # ------------------------------------------------------------ shared files
    def test_install_files_copies_scripts_config_and_vault_dirs(self):
        self.installer.install_files()
        bin_dir = self.home / ".logseq-memory" / "bin"
        for name in install.SCRIPTS:
            self.assertEqual((bin_dir / name).read_text(encoding="utf-8"), (install.REPO / name).read_text(encoding="utf-8"))
        config = self.read_json(self.home / ".logseq-memory" / "config.json")
        self.assertEqual(config["pages_dir"], str(self.pages))
        for category in install.CATEGORIES:
            self.assertTrue((self.pages / "memory" / category).is_dir())

    # ---------------------------------------------------------------- claude
    def test_claude_settings_replace_old_hooks_and_keep_the_rest(self):
        settings = self.home / ".claude" / "settings.json"
        self.write_json(settings, {
            "model": "opus",
            "hooks": {
                "Stop": [
                    {"matcher": "", "hooks": [{"type": "command",
                     "command": "python3 /Users/x/.claude/hooks/logseq_memory_extractor.py"}]},
                    {"matcher": "", "hooks": [{"type": "command", "command": "notify-done"}]},
                ],
                "PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "log"}]}],
            },
            "mcpServers": {"other": {"command": "x"}},
            "sandbox": {"filesystem": {"allowWrite": ["/already/"]}},
        })
        self.installer.install_claude(desktop=False)
        data = self.read_json(settings)
        self.assertEqual(data["model"], "opus")
        self.assertEqual(data["hooks"]["PreToolUse"][0]["hooks"][0]["command"], "log")
        stop_hooks = [h for g in data["hooks"]["Stop"] for h in g["hooks"]]
        self.assertIn("notify-done", [h["command"] for h in stop_hooks])
        memory = [h for h in stop_hooks if "logseq_memory_" in json.dumps(h)]
        self.assertEqual(len(memory), 1)
        # Exec form: no shell, no environment prefix.
        self.assertEqual(memory[0]["command"], str(self.home / ".logseq-memory" / "venv" / "bin" / "python"))
        self.assertEqual(
            memory[0]["args"], [str(self.home / ".logseq-memory" / "bin" / "logseq_memory_extractor.py")]
        )
        self.assertIn("other", data["mcpServers"])
        self.assertEqual(
            data["mcpServers"]["logseq-vault"]["args"],
            [str(self.home / ".logseq-memory" / "bin" / "logseq_memory_mcp.py")],
        )
        allow = data["sandbox"]["filesystem"]["allowWrite"]
        self.assertIn(f"{self.home / '.logseq-memory'}/", allow)
        self.assertIn(f"{self.pages.parent}/", allow)
        backups = list((self.home / ".logseq-memory" / "backups").rglob("*settings.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].parent.parent.name, "backups")  # one flat file name

    def test_claude_without_sandbox_config_gets_no_sandbox_block(self):
        self.installer.install_claude(desktop=False)
        data = self.read_json(self.home / ".claude" / "settings.json")
        self.assertNotIn("sandbox", data)

    def test_desktop_config_gets_mcp_server(self):
        self.write_json(self.installer.desktop_config, {"mcpServers": {"k8s": {"command": "k"}}})
        self.installer.install_claude(desktop=True)
        servers = self.read_json(self.installer.desktop_config)["mcpServers"]
        self.assertEqual(set(servers), {"k8s", "logseq-vault"})

    # ----------------------------------------------------------------- codex
    def test_codex_config_preserves_other_tables_and_approvals(self):
        config = self.home / ".codex" / "config.toml"
        config.parent.mkdir(parents=True)
        config.write_text(EXISTING_CODEX_CONFIG, encoding="utf-8")
        self.installer.install_codex()
        text = config.read_text(encoding="utf-8")
        python = str(self.home / ".logseq-memory" / "venv" / "bin" / "python")
        self.assertIn(f'command = "{python}"', text)
        self.assertNotIn("/old/", text)
        self.assertIn("required = true", text)
        self.assertIn("approval_mode = 'approve'", text)
        self.assertIn('[mcp_servers.other]\ncommand = "other"', text)
        self.assertIn('TRANSFORMERS_OFFLINE = "1"', text)
        if tomllib:
            parsed = tomllib.loads(text)
            server = parsed["mcp_servers"]["logseq_vault"]
            self.assertEqual(server["command"], python)
            self.assertEqual(server["env"]["HF_DATASETS_OFFLINE"], "1")
            self.assertEqual(server["tools"]["read_page"]["approval_mode"], "approve")
            self.assertIs(parsed["features"]["hooks"], True)
            self.assertEqual(parsed["model"], "gpt-6-sol")

    def test_codex_hooks_and_skill_are_filled_in(self):
        hooks_path = self.home / ".codex" / "hooks.json"
        self.write_json(hooks_path, {"hooks": {
            "SessionEnd": [{"hooks": [{"type": "command", "command": "/old/python /old/logseq_memory_codex.py enqueue"}]}],
            "Stop": [{"hooks": [{"type": "command", "command": "unrelated"}]}],
        }})
        self.installer.install_codex()
        data = self.read_json(hooks_path)
        self.assertEqual(data["hooks"]["Stop"][0]["hooks"][0]["command"], "unrelated")
        commands = [h["command"] for e in ("SessionEnd", "UserPromptSubmit")
                    for g in data["hooks"][e] for h in g["hooks"]]
        self.assertEqual(len(commands), 2)
        for command in commands:
            self.assertNotIn("__LOGSEQ", command)
            self.assertIn(str(self.home / ".logseq-memory" / "bin"), command)
        skill = (self.home / ".codex" / "skills" / "extract-memory" / "SKILL.md").read_text(encoding="utf-8")
        self.assertNotIn("__LOGSEQ", skill)
        self.assertTrue((self.home / ".codex" / "skills" / "extract-memory" / "agents" / "openai.yaml").exists())

    def test_codex_commands_are_quoted_for_paths_with_spaces(self):
        spaced = self.home / "First Last"
        spaced.mkdir()
        installer = self.make(home=spaced)
        installer.install_codex()
        data = self.read_json(spaced / ".codex" / "hooks.json")
        command = data["hooks"]["SessionEnd"][0]["hooks"][0]["command"]
        self.assertEqual(shlex.split(command), [
            str(spaced / ".logseq-memory" / "venv" / "bin" / "python"),
            str(spaced / ".logseq-memory" / "bin" / "logseq_memory_codex.py"),
            "enqueue",
        ])
        skill = (spaced / ".codex" / "skills" / "extract-memory" / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn(shlex.join([str(installer.python), str(installer.bin / "logseq_memory_codex.py"),
                                  "resolve-session"]), skill)

    # --------------------------------------------------------------- windows
    def test_windows_paths(self):
        with mock.patch.dict(os.environ, {"APPDATA": "", "LOCALAPPDATA": ""}):
            installer = self.make(windows=True)
        self.assertEqual(installer.python, self.home / ".logseq-memory" / "venv" / "Scripts" / "python.exe")
        self.assertEqual(
            installer.desktop_config,
            self.home / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json",
        )
        with mock.patch.dict(os.environ, {"APPDATA": str(self.home / "Roam")}):
            self.assertEqual(self.make(windows=True).desktop_config.parent, self.home / "Roam" / "Claude")

    def test_windows_codex_command_uses_cmd_quoting(self):
        installer = self.make(windows=True)
        line = installer._fill_command("__LOGSEQ_PYTHON__ __LOGSEQ_BIN__/logseq_memory_retriever.py")
        self.assertEqual(line, install.subprocess.list2cmdline(
            [str(installer.python), str(installer.bin / "logseq_memory_retriever.py")]
        ))

    def test_install_records_cli_paths_for_gui_launched_hooks(self):
        with mock.patch.object(install.shutil, "which", lambda name: f"/opt/{name}/bin/{name}"):
            self.installer.install_files()
        config = self.read_json(self.home / ".logseq-memory" / "config.json")
        self.assertEqual(config["claude_bin"], "/opt/claude/bin/claude")
        self.assertEqual(config["codex_bin"], "/opt/codex/bin/codex")

    # ---------------------------------------------------------- instructions
    def test_managed_block_is_added_once_and_removed_on_uninstall(self):
        claude_md = self.home / ".claude" / "CLAUDE.md"
        claude_md.parent.mkdir(parents=True)
        claude_md.write_text("# Mine\n\nKeep this.\n", encoding="utf-8")
        self.installer.install_claude(desktop=False)
        self.installer.install_claude(desktop=False)
        text = claude_md.read_text(encoding="utf-8")
        self.assertEqual(text.count(install.BLOCK_BEGIN), 1)
        self.assertIn("Keep this.", text)
        self.assertIn(str(self.pages / "memory" / "digest.md"), text)
        self.installer.install_claude(desktop=False, remove=True)
        self.assertEqual(claude_md.read_text(encoding="utf-8"), "# Mine\n\nKeep this.\n")

    def test_existing_hand_written_vault_section_is_left_alone(self):
        agents_md = self.home / ".codex" / "AGENTS.md"
        agents_md.parent.mkdir(parents=True)
        agents_md.write_text(f"Read `{self.pages}/memory/digest.md` at start.\n", encoding="utf-8")
        agents_md.write_text(agents_md.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")  # odd trailing blank lines
        before = agents_md.read_text(encoding="utf-8")
        self.installer.install_codex()
        self.assertEqual(agents_md.read_text(encoding="utf-8"), before)

    # ------------------------------------------------- idempotency / dry run
    def test_second_run_changes_nothing(self):
        self.installer.install_files()
        self.installer.install_claude(desktop=False)
        self.installer.install_codex()
        before = self.snapshot()
        again = self.make()
        again.install_files()
        again.install_claude(desktop=False)
        again.install_codex()
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(again._backed_up)

    def test_dry_run_writes_nothing(self):
        dry = self.make(dry_run=True)
        dry.install_files()
        dry.install_claude(desktop=False)
        dry.install_codex()
        self.assertEqual(self.snapshot(), {})

    def test_uninstall_removes_all_registrations(self):
        config = self.home / ".codex" / "config.toml"
        config.parent.mkdir(parents=True)
        config.write_text(EXISTING_CODEX_CONFIG, encoding="utf-8")
        self.installer.install_claude(desktop=False)
        self.installer.install_codex()
        self.installer.install_claude(desktop=False, remove=True)
        self.installer.install_codex(remove=True)
        settings = self.read_json(self.home / ".claude" / "settings.json")
        self.assertNotIn("logseq", json.dumps(settings))
        hooks = self.read_json(self.home / ".codex" / "hooks.json")
        self.assertNotIn("logseq_memory_", json.dumps(hooks))
        text = config.read_text(encoding="utf-8")
        self.assertNotIn("logseq_vault", text)
        self.assertIn("[mcp_servers.other]", text)
        self.assertFalse((self.home / ".codex" / "skills" / "extract-memory").exists())


if __name__ == "__main__":
    unittest.main()
