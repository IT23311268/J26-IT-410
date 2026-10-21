"""Package-scoped JSON logging; leaves root logging configuration alone."""

import json
import logging
from datetime import UTC, datetime
from typing import TextIO


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            event["exception"] = self.formatException(record.exc_info)
        return json.dumps(event, ensure_ascii=False)


def configure_logging(level: str = "INFO", stream: TextIO | None = None) -> logging.Logger:
    logger = logging.getLogger("ci_hgnn_rgd")
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        if getattr(handler, "_ci_hgnn_rgd_owned", False):
            logger.removeHandler(handler)
            handler.close()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    handler._ci_hgnn_rgd_owned = True
    logger.addHandler(handler)
    return logger
