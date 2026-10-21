import json
import logging
import random
from io import StringIO
from unittest.mock import patch

import pytest

from ci_hgnn_rgd.logging_utils import configure_logging
from ci_hgnn_rgd.runtime import detect_device, set_random_seeds


def test_python_random_seed_reproduces_sequence():
    original = random.getstate()
    try:
        report = set_random_seeds(42)
        first = [random.random() for _ in range(5)]
        set_random_seeds(42)
        assert first == [random.random() for _ in range(5)]
        assert report["applied_to"] == ["python.random"]
        assert "nondeterministic" in report["limitations"]
    finally:
        random.setstate(original)


@pytest.mark.parametrize("seed", [-1, True, 2**32, 4.2])
def test_seed_validation(seed):
    with pytest.raises(ValueError):
        set_random_seeds(seed)


def test_cpu_fallback_and_explicit_unavailable_accelerator():
    with patch("ci_hgnn_rgd.runtime.util.find_spec", return_value=None):
        assert detect_device("auto").device == "cpu"
        with pytest.raises(ValueError, match="not installed"):
            detect_device("cuda")
    with pytest.raises(ValueError):
        detect_device("unknown")


def test_json_logging_is_not_duplicated_and_preserves_host_handlers():
    root_handlers = list(logging.getLogger().handlers)
    logger = logging.getLogger("ci_hgnn_rgd")
    stream = StringIO()
    external = logging.NullHandler()
    logger.addHandler(external)
    try:
        configure_logging(stream=stream)
        configure_logging(stream=stream).info("Synthetic software-test event")
        events = stream.getvalue().splitlines()
        assert len(events) == 1
        assert json.loads(events[0])["message"] == "Synthetic software-test event"
        assert external in logger.handlers
        assert root_handlers == logging.getLogger().handlers
    finally:
        logger.removeHandler(external)
        for handler in list(logger.handlers):
            if getattr(handler, "_ci_hgnn_rgd_owned", False):
                logger.removeHandler(handler)
                handler.close()
