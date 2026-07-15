"""Tests for the ``agentomatic agents-guide`` command and its primer source."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from agentomatic.cli.agent_guide import WRITE_TARGETS, render_primer, write_primer
from agentomatic.cli.commands import cli


class TestPrimerSource:
    def test_primer_covers_key_features(self) -> None:
        text = render_primer()
        for needle in (
            "Agentomatic",
            "agentomatic deploy",
            "--profile",
            "minimal",
            "AGENTOMATIC_ENABLE_STUDIO",
            "input_to_state",
            "register_vector_provider",
            "Swagger is always available",
        ):
            assert needle in text, f"primer missing: {needle}"

    def test_skill_target_gets_frontmatter(self) -> None:
        skill = render_primer(".cursor/skills/agentomatic/SKILL.md")
        assert skill.startswith("---\n")
        assert "name: agentomatic" in skill.splitlines()[1] or "name: agentomatic" in skill
        # Non-skill targets have no front matter.
        assert not render_primer("AGENTS.md").startswith("---\n")

    def test_write_primer_creates_and_refuses_overwrite(self, tmp_path: Path) -> None:
        dest = write_primer("AGENTS.md", root=tmp_path)
        assert dest.exists()
        assert dest == tmp_path / "AGENTS.md"

        # Second write without force must refuse.
        try:
            write_primer("AGENTS.md", root=tmp_path)
        except FileExistsError:
            pass
        else:  # pragma: no cover - defensive
            raise AssertionError("expected FileExistsError")

        # With force it overwrites.
        again = write_primer("AGENTS.md", root=tmp_path, force=True)
        assert again.exists()

    def test_write_primer_rejects_unknown_target(self, tmp_path: Path) -> None:
        try:
            write_primer("README.md", root=tmp_path)  # type: ignore[arg-type]
        except ValueError:
            pass
        else:  # pragma: no cover - defensive
            raise AssertionError("expected ValueError")

    def test_skill_write_creates_nested_dirs(self, tmp_path: Path) -> None:
        dest = write_primer(".cursor/skills/agentomatic/SKILL.md", root=tmp_path)
        assert dest.exists()
        assert dest.read_text().startswith("---\n")


class TestAgentsGuideCli:
    def test_prints_primer_to_stdout(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["agents-guide"])
        assert result.exit_code == 0, result.output
        assert "Agentomatic — Agent Primer" in result.output
        assert "--profile" in result.output

    def test_write_creates_file(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["agents-guide", "--write", "AGENTS.md"])
            assert result.exit_code == 0, result.output
            assert Path("AGENTS.md").exists()

    def test_write_refuses_without_force(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("CLAUDE.md").write_text("keep me\n")
            result = runner.invoke(cli, ["agents-guide", "--write", "CLAUDE.md"])
            assert result.exit_code == 1
            assert Path("CLAUDE.md").read_text() == "keep me\n"

    def test_write_force_overwrites(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("CLAUDE.md").write_text("old\n")
            result = runner.invoke(cli, ["agents-guide", "--write", "CLAUDE.md", "--force"])
            assert result.exit_code == 0, result.output
            assert "Agentomatic" in Path("CLAUDE.md").read_text()

    def test_all_write_targets_supported(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            for target in WRITE_TARGETS:
                result = runner.invoke(cli, ["agents-guide", "--write", target, "--force"])
                assert result.exit_code == 0, result.output
                assert Path(target).exists()
