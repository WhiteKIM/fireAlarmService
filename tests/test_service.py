import json
import threading
import time

import cv2
import numpy as np
import pytest

from fire_service.config import SourceConfig, load_sources
from fire_service.detector import IoUTracker, PassthroughDetector, YoloDetector
from fire_service.pipeline import Frame, Latest, Result, Service
from fire_service.web import create_app


def wait_until(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    assert predicate(), 'condition timed out'


class Capture:
    def __init__(self, count=20, opened=True, color=0):
        self.count, self.opened, self.color = count, opened, color
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, key):
        return 60

    def read(self):
        if self.count <= 0:
            return False, None
        self.count -= 1
        return True, np.full((72, 128, 3), self.color, dtype=np.uint8)

    def release(self):
        self.released = True


class FakeDetector:
    mode = 'test'

    def __init__(self):
        self.seen = []

    def process(self, key, frame):
        self.seen.append(key)
        return frame, [dict(name=key, class_id=0, confidence=.9, bbox=[1, 2, 30, 40])]


@pytest.mark.parametrize('override', [dict(id='../x'), dict(uri=True), dict(fps=0), dict(fps=float('nan')),
                                    dict(jpeg_quality=100), dict(max_width=100.5), dict(loop='false'),
                                    dict(timeout_ms=0), dict(uri='ftp://host/video'), dict(uri=-1)])
def test_invalid_config(override):
    with pytest.raises(ValueError):
        SourceConfig(**(dict(id='a', uri='video.mp4') | override))


def test_config_relative_paths_and_duplicates(tmp_path):
    path = tmp_path / 'sources.json'
    path.write_text(json.dumps(dict(sources=[dict(id='file', uri='clip.mp4'), dict(id='usb', uri=0)])))
    a, b = load_sources(path)
    assert a.uri == str(tmp_path / 'clip.mp4')
    assert b.uri == 0 and b.live
    path.write_text(json.dumps(dict(sources=[dict(id='a', uri=0), dict(id='a', uri=1)])))
    with pytest.raises(ValueError, match='duplicate'):
        load_sources(path)


def test_latest_skips_old_frames_and_wakes_on_close():
    slot = Latest()
    for number in range(1, 1001):
        slot.publish(Frame(number, 0, 0, None))
    assert slot.wait(1).sequence == 1000
    waiter = threading.Thread(target=lambda: slot.wait(1000, timeout=10))
    waiter.start()
    slot.close()
    waiter.join(.2)
    assert not waiter.is_alive()


def test_two_sources_are_isolated_and_capture_released():
    captures = []
    def factory(config):
        cap = Capture(color=20 if config.id == 'a' else 200)
        captures.append(cap)
        return cap
    detector = FakeDetector()
    service = Service([SourceConfig('a', 'a.mp4', fps=30), SourceConfig('b', 'b.mp4', fps=30)], detector, factory)
    service.start()
    try:
        wait_until(lambda: all(s.results.closed for s in service.sources.values()))
        for key, source in service.sources.items():
            assert source.results.get().metadata['detections'][0]['name'] == key
            assert source.results.get().sequence == 20
        assert set(detector.seen) == {'a', 'b'}
    finally:
        assert service.stop() == []
    assert all(c.released for c in captures)


def test_reconnect_does_not_block_other_source():
    attempts = []
    def factory(config):
        if config.id == 'bad':
            attempts.append(1)
            return Capture(opened=False)
        return Capture(count=15)
    service = Service([SourceConfig('bad', 'rtsp://user:secret@host/video', reconnect_seconds=.1),
                       SourceConfig('good', 'local.mp4')], FakeDetector(), factory)
    service.start()
    try:
        wait_until(lambda: len(attempts) >= 2 and service.sources['good'].results.get() is not None)
        status = create_app(service).test_client().get('/api/sources').text
        assert 'secret' not in status and 'rtsp' not in status
        assert service.sources['bad'].snapshot()['reconnects'] >= 2
    finally:
        assert service.stop() == []


def test_live_source_recovers_after_open_failure():
    attempts = []
    def factory(config):
        attempts.append(1)
        return Capture(count=100, opened=len(attempts) > 1)
    service = Service([SourceConfig('a', 0, reconnect_seconds=.1)], FakeDetector(), factory)
    service.start()
    try:
        wait_until(lambda: service.sources['a'].results.get() is not None)
        assert len(attempts) > 1
    finally:
        assert service.stop() == []


def result(sequence, captured_at=None):
    ok, jpeg = cv2.imencode('.jpg', np.zeros((32, 32, 3), dtype=np.uint8))
    assert ok
    return Result(sequence, jpeg.tobytes(), dict(source_id='a', sequence=sequence,
                  captured_at=time.time() if captured_at is None else captured_at,
                  detections=[dict(name='fire', bbox=[1, 2, 3, 4])]))


def test_http_contract_slow_reader_and_stream_limit():
    service = Service([SourceConfig('a', 'x')], FakeDetector())
    source = service.sources['a']
    source.results.publish(result(1))
    app = create_app(service, max_streams=2)
    client = app.test_client()
    assert client.get('/healthz').status_code == 200
    assert client.get('/readyz').status_code == 503
    source.update(status='capturing')
    assert client.get('/readyz').status_code == 200
    assert client.get('/api/sources/a/detections').json['sequence'] == 1
    assert client.get('/api/sources/a/snapshot.jpg').data.startswith(b'\xff\xd8')
    assert client.get('/get').json == {'Data':[dict(name='fire', camInfo='a', Points=dict(X=1, Y=2, Z=3, W=4))]}
    slow = client.get('/api/sources/a/stream.mjpg', buffered=False)
    fast = client.get('/api/sources/a/stream.mjpg', buffered=False)
    assert b'X-Sequence: 1' in next(slow.response)
    assert b'X-Sequence: 1' in next(fast.response)
    assert client.get('/api/sources/a/stream.mjpg').status_code == 429
    for number in range(2, 51):
        source.results.publish(result(number))
        assert f'X-Sequence: {number}\r\n'.encode() in next(fast.response)
    assert b'X-Sequence: 50\r\n' in next(slow.response)
    slow.close()
    fast.close()
    assert source.snapshot()['subscribers'] == 0
    again = client.get('/api/sources/a/stream.mjpg', buffered=False)
    again.close()
    assert source.snapshot()['subscribers'] == 0
    assert client.get('/api/sources/missing/detections').status_code == 404


def test_empty_and_stale_sources_return_503():
    service = Service([SourceConfig('a', 'x')], FakeDetector())
    client = create_app(service).test_client()
    for suffix in ['detections', 'snapshot.jpg', 'stream.mjpg']:
        assert client.get('/api/sources/a/' + suffix).status_code == 503
    service.sources['a'].results.publish(result(1, time.time() - 30))
    for suffix in ['detections', 'snapshot.jpg', 'stream.mjpg']:
        assert client.get('/api/sources/a/' + suffix).status_code == 503
    assert client.get('/get').json == {'Data': []}


def test_iou_ids_match_class_expire_and_remain_source_local():
    a, b = IoUTracker(), IoUTracker()
    obj = dict(bbox=[0, 0, 10, 10], confidence=.9, class_id=0)
    assert a.update([obj], now=0)[0]['track_id'] == 1
    assert a.update([obj], now=1)[0]['track_id'] == 1
    assert b.update([obj], now=1)[0]['track_id'] == 1
    assert a.update([dict(obj, class_id=1)], now=1)[0]['track_id'] == 2
    assert a.update([obj], now=5)[0]['track_id'] == 3
    duplicate = a.update([obj, obj], now=6)
    assert len({d['track_id'] for d in duplicate}) == 2


def test_model_absence_fails_and_test_mode_is_explicit():
    with pytest.raises(ValueError, match='weights not found'):
        YoloDetector('/missing/model.pt')
    detector = PassthroughDetector()
    image, objects = detector.process('a', np.zeros((72, 800, 3), dtype=np.uint8))
    assert objects == [] and image.any() and detector.mode == 'passthrough'


def test_processing_failure_isolated_from_healthy_source():
    class FailingDetector(FakeDetector):
        def process(self, key, frame):
            if key == 'bad':
                raise RuntimeError('model failure')
            return super().process(key, frame)
    service = Service([SourceConfig('bad', 'x'), SourceConfig('good', 'y')],
                      FailingDetector(), lambda config: Capture(count=5))
    service.start()
    try:
        wait_until(lambda: all(s.results.closed for s in service.sources.values()))
        assert service.sources['bad'].snapshot()['error'] == 'processing failed'
        assert service.sources['good'].results.get().metadata['source_id'] == 'good'
    finally:
        assert service.stop() == []


def test_capture_error_not_cleared_by_pending_successful_inference():
    started, unblock = threading.Event(), threading.Event()
    class BlockingDetector(FakeDetector):
        def process(self, key, frame):
            started.set()
            assert unblock.wait(2)
            return super().process(key, frame)
    service = Service([SourceConfig('a', 'x', fps=60)], BlockingDetector(), lambda config: Capture(count=3))
    service.start()
    try:
        assert started.wait(2)
        service.sources['a'].update(error='capture unavailable')
        unblock.set()
        wait_until(lambda: service.sources['a'].results.get() is not None)
        assert service.sources['a'].snapshot()['error'] == 'capture unavailable'
    finally:
        unblock.set()
        assert service.stop() == []


def test_file_loop_is_paced_and_reopens_with_monotonic_sequences():
    captures = []
    def factory(config):
        cap = Capture(count=1)
        captures.append(cap)
        return cap
    service = Service([SourceConfig('a', 'x', fps=30, loop=True)], FakeDetector(), factory)
    service.start()
    try:
        wait_until(lambda: service.sources['a'].frames.get() is not None and
                   service.sources['a'].frames.get().sequence >= 3)
        assert service.sources['a'].frames.get().sequence >= 3
        assert len(captures) < 20
    finally:
        assert service.stop() == []
    assert all(c.released for c in captures)


def test_slow_inference_keeps_capture_bounded_and_sources_fair():
    class SlowDetector(FakeDetector):
        def process(self, key, frame):
            time.sleep(.04)
            return super().process(key, frame)
    detector = SlowDetector()
    service = Service([SourceConfig('a', 'x', fps=60), SourceConfig('b', 'y', fps=60)],
                      detector, lambda config: Capture(count=30))
    service.start()
    try:
        wait_until(lambda: all(s.results.closed for s in service.sources.values()))
        for source in service.sources.values():
            assert source.results.get().sequence == 30
            assert source.snapshot()['skipped'] > 0
            assert source.snapshot()['processed'] + source.snapshot()['skipped'] == 30
        assert abs(detector.seen.count('a') - detector.seen.count('b')) <= 1
    finally:
        assert service.stop() == []
