from pathlib import Path

import pytest

from src.config import (
    INPUT_DIR,
    OUTPUT_DIR,
    PRESETS,
    PROJECT_ROOT,
    OutputDirs,
    PipelineConfig,
    get_output_dirs,
)


class TestPipelineConfig:
    def test_default_values(self):
        """PipelineConfig should have sensible defaults."""
        cfg = PipelineConfig()
        assert cfg.image_file == INPUT_DIR / "DECam" / "image.fits"
        assert cfg.gaia_catalog_file == INPUT_DIR / "DECam" / "gaia_catalog.csv"
        assert cfg.gain == 1.0
        assert cfg.readnoise == 10.0
        assert cfg.saturation_level == 60000.0
        assert cfg.extension == 0
        assert cfg.gaia_bright_limit == 6.0
        assert cfg.gaia_faint_limit == 21.5
        assert cfg.max_separation_arcsec == 2.0
        assert cfg.clean_old_outputs is True
        assert cfg.use_milliquas is False

    def test_validate_raises_on_missing_image(self, tmp_path):
        """validate() should raise FileNotFoundError when inputs are missing."""
        cfg = PipelineConfig(
            image_file=tmp_path / "nonexistent.fits",
            gaia_catalog_file=tmp_path / "gaia.csv",
            auto_download_gaia=False,
        )
        with pytest.raises(FileNotFoundError, match="Required inputs are missing"):
            cfg.validate()

    def test_validate_skips_image_in_xy_only(self, tmp_path):
        """validate() should skip the image file check when process_xy_only is True."""
        gaia = tmp_path / "gaia.csv"
        gaia.write_text("ra,dec,pmra,pmdec,ref_epoch\n10.0,20.0,0,0,2016.0\n")
        xy_file = tmp_path / "xy.txt"
        xy_file.write_text("1 2\n")
        cfg = PipelineConfig(
            image_file=tmp_path / "nonexistent.fits",
            gaia_catalog_file=gaia,
            xy_file=xy_file,
            process_xy_only=True,
            auto_download_gaia=False,
        )
        cfg.validate()

    def test_validate_raises_on_missing_xy_file(self, tmp_path):
        """validate() should raise when xy_file is missing and process_xy_only is True."""
        cfg = PipelineConfig(
            xy_file=tmp_path / "nonexistent.txt",
            process_xy_only=True,
            auto_download_gaia=False,
        )
        with pytest.raises(FileNotFoundError, match="XY file"):
            cfg.validate()

    def test_config_is_frozen(self):
        """PipelineConfig should be a frozen dataclass (attempted setattr raises)."""
        cfg = PipelineConfig()
        with pytest.raises((AttributeError, TypeError, ValueError)):
            cfg.gain = 99.0


class TestOutputDirs:
    def test_dataclass_fields(self):
        """OutputDirs should store the four directory paths."""
        dirs = OutputDirs(
            catalogs=Path("/a/cat"),
            plots=Path("/a/plot"),
            models=Path("/a/mod"),
            logs=Path("/a/log"),
        )
        assert dirs.catalogs == Path("/a/cat")
        assert dirs.plots == Path("/a/plot")
        assert dirs.models == Path("/a/mod")
        assert dirs.logs == Path("/a/log")

    def test_get_output_dirs_creates_directories(self, tmp_path):
        """get_output_dirs should create the output directory tree."""
        original_output = OUTPUT_DIR
        import src.config as cfg_mod

        cfg_mod.OUTPUT_DIR = tmp_path / "output"
        try:
            dirs = get_output_dirs("TestInst")
            assert dirs.catalogs.exists()
            assert dirs.plots.exists()
            assert dirs.models.exists()
            assert dirs.logs.exists()
        finally:
            cfg_mod.OUTPUT_DIR = original_output

    def test_get_output_dirs_sanitizes_name(self, tmp_path):
        """get_output_dirs should lowercase and normalize instrument names."""
        original_output = OUTPUT_DIR
        import src.config as cfg_mod

        cfg_mod.OUTPUT_DIR = tmp_path / "output"
        try:
            dirs = get_output_dirs("My Camera 2")
            assert dirs.catalogs.parent.name == "my_camera_2"
        finally:
            cfg_mod.OUTPUT_DIR = original_output

    def test_get_output_dirs_default_on_empty(self, tmp_path):
        """get_output_dirs should return default dirs for empty/unknown names."""
        original_output = OUTPUT_DIR
        import src.config as cfg_mod

        cfg_mod.OUTPUT_DIR = tmp_path / "output"
        try:
            dirs = get_output_dirs("")
            assert dirs.catalogs.parent.name == "default"
            dirs2 = get_output_dirs("UNKNOWN")
            assert dirs2.catalogs.parent.name == "default"
        finally:
            cfg_mod.OUTPUT_DIR = original_output


class TestPresets:
    def test_presets_contains_fors2(self):
        """PRESETS should include at least FORS2."""
        assert "FORS2" in PRESETS

    def test_fors2_preset_keys(self):
        """The FORS2 preset should have the expected keys."""
        preset = PRESETS["FORS2"]
        assert "image_file" in preset
        assert "gaia_catalog_file" in preset
        assert "use_milliquas" in preset
        assert preset["use_milliquas"] is False

    def test_presets_paths_are_relative_to_input(self):
        """Preset file paths should live under INPUT_DIR."""
        preset = PRESETS["FORS2"]
        for key in ("image_file", "gaia_catalog_file"):
            path = preset[key]
            assert str(path).startswith(str(INPUT_DIR))


class TestConstants:
    def test_project_root_is_absolute(self):
        assert PROJECT_ROOT.is_absolute()
        assert (PROJECT_ROOT / "src").is_dir()

    def test_input_dir_exists(self):
        assert INPUT_DIR.is_dir()
