"""Opt-in integration check against a running MJPEG input, plus a distinct file."""
import os
import time

import cv2
import numpy as np
import pytest

from fire_service.config import SourceConfig
from fire_service.detector import PassthroughDetector
from fire_service.pipeline import Service


@pytest.mark.skipif(not os.environ.get('FIRE_TEST_HTTP_INPUT'), reason='requires a running HTTP MJPEG source')
def test_actual_http_input_and_distinct_local_video(tmp_path):
    path = tmp_path / 'local.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (128, 72))
    assert writer.isOpened()
    for number in range(30):
        writer.write(np.full((72, 128, 3), number * 5, dtype=np.uint8))
    writer.release()
    service = Service([
        SourceConfig('network', os.environ['FIRE_TEST_HTTP_INPUT'], timeout_ms=2000),
        SourceConfig('file', str(path), loop=True)
    ], PassthroughDetector())
    service.start()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not all(s.snapshot()['processed'] >= 3 for s in service.sources.values()):
            time.sleep(.05)
        network = service.sources['network'].results.get()
        local = service.sources['file'].results.get()
        assert network is not None and local is not None
        assert network.metadata['source_id'] == 'network'
        assert local.metadata['source_id'] == 'file'
        assert local.metadata['width'] == 128
        assert network.jpeg != local.jpeg
    finally:
        assert service.stop() == []
