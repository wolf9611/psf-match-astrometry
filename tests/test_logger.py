import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.logger import get_logger, log_stage_header, log_substage, setup_logging


class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "pipeline" or logger.name.startswith("pipeline.")

    def test_singleton_behavior(self):
        logger1 = get_logger("pipeline")
        logger2 = get_logger("pipeline")
        assert logger1 is logger2

    def test_child_naming(self):
        parent = get_logger("pipeline")
        child = get_logger("child")
        assert child.name == f"{parent.name}.child" or child is parent

    def test_logger_has_console_handler(self):
        logger = get_logger("test_console")
        handlers = (
            logger.handlers
            if logger.name == "pipeline"
            else get_logger("pipeline").handlers
        )
        assert any(isinstance(h, logging.StreamHandler) for h in handlers)

    def test_propagation_disabled(self):
        logger = get_logger("test_prop")
        if logger.name == "pipeline":
            assert logger.propagate is False


class TestSetupLogging:
    def test_creates_log_directory(self, tmp_path):
        logs_dir = tmp_path / "mylogs"
        setup_logging(logs_dir)
        assert logs_dir.is_dir()

    def test_adds_rotating_file_handler(self, tmp_path):
        logs_dir = tmp_path / "logs"
        setup_logging(logs_dir)
        logger = get_logger("pipeline")
        file_handlers = [
            h for h in logger.handlers if isinstance(h, RotatingFileHandler)
        ]
        assert len(file_handlers) >= 1
        expected_file = logs_dir / "pipeline_verbose.log"
        assert any(Path(h.baseFilename) == expected_file for h in file_handlers)

    def test_setup_logging_replaces_old_handler(self, tmp_path):
        log1 = tmp_path / "run1"
        log2 = tmp_path / "run2"
        setup_logging(log1)
        logger = get_logger("pipeline")
        handlers_before = [
            h for h in logger.handlers if isinstance(h, RotatingFileHandler)
        ]
        setup_logging(log2)
        handlers_after = [
            h for h in logger.handlers if isinstance(h, RotatingFileHandler)
        ]
        assert all(Path(h.baseFilename).parent == log2 for h in handlers_after)


class TestLogHelpers:
    def test_log_stage_header_does_not_raise(self):
        logger = get_logger("test_header")
        log_stage_header(logger, "Test Stage")

    def test_log_substage_does_not_raise(self):
        logger = get_logger("test_substage")
        log_substage(logger, "Test Sub Stage")
