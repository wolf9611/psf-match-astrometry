"""
Tests for the main pipeline orchestrator module.
"""

import main
from main import apply_preset, build_parser, cleanup_old_files
from src.config import (
    PRESETS,
    OutputDirs,
    config,
)


def test_main_callable():
    """The ``main`` entry-point should be a callable."""
    assert callable(main.main)


class TestBuildParser:
    def test_parser_returns_argparse_parser(self):
        """build_parser should return an ArgumentParser."""
        parser = build_parser()
        assert parser is not None
        assert parser.prog is not None

    def test_parser_has_expected_arguments(self):
        """The parser should have all expected CLI arguments."""
        parser = build_parser()
        actions = {a.dest for a in parser._actions}
        for expected in (
            "preset",
            "instrument",
            "image",
            "gaia_catalog",
            "no_milliquas",
            "no_clean",
            "extension",
        ):
            assert expected in actions, f"Missing argument: {expected}"

    def test_parser_preset_choices(self):
        """The --preset argument should include PRESETS keys."""
        parser = build_parser()
        for action in parser._actions:
            if action.dest == "preset":
                assert set(action.choices) <= set(PRESETS.keys())
                break


class TestApplyPreset:
    def test_apply_preset_changes_config(self):
        """apply_preset should update config attributes."""
        # Save original
        original_file = config.image_file
        original_gaia = config.gaia_catalog_file

        if "FORS2" in PRESETS:
            apply_preset("FORS2")
            assert config.image_file == PRESETS["FORS2"]["image_file"]
            assert config.gaia_catalog_file == PRESETS["FORS2"]["gaia_catalog_file"]

        # Restore (use object.__setattr__ since config is frozen)
        object.__setattr__(config, "image_file", original_file)
        object.__setattr__(config, "gaia_catalog_file", original_gaia)

    def test_apply_preset_preserves_other_config(self):
        """apply_preset should only change preset-specific fields."""
        original_gain = config.gain
        if "FORS2" in PRESETS:
            apply_preset("FORS2")
            assert config.gain == original_gain
        object.__setattr__(config, "gain", original_gain)


class TestCleanupOldFiles:
    def test_cleanup_old_files_no_crash(self, tmp_path):
        """cleanup_old_files should not raise when directories are empty."""
        dirs = OutputDirs(
            catalogs=tmp_path / "catalogs",
            plots=tmp_path / "plots",
            models=tmp_path / "models",
            logs=tmp_path / "logs",
        )
        for d in [dirs.catalogs, dirs.plots, dirs.models, dirs.logs]:
            d.mkdir(parents=True, exist_ok=True)

        # Should not raise
        cleanup_old_files(dirs)
