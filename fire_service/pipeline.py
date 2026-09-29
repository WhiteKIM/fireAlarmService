"""Capture independently; infer fairly; publish once; never queue client frames."""
from dataclasses import dataclass
import logging
import math
import threading
import time

import cv2

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Frame:
    sequence: int
    captured_at: float
    captured_mono: float
    image: object


@dataclass(frozen=True)
class Result:
    sequence: int
    jpeg: bytes
    metadata: dict


class Latest:
    """One immutable value shared by readers. Slow readers skip overwritten values."""
    def __init__(self):
        self.condition = threading.Condition()
        self.value = None
        self.closed = False

    def publish(self, value):
        with self.condition:
            if not self.closed:
                self.value = value
                self.condition.notify_all()

    def get(self):
        with self.condition:
            return self.value

    def wait(self, after=0, timeout=10):
        with self.condition:
            self.condition.wait_for(
                lambda: self.closed or (self.value is not None and self.value.sequence > after), timeout)
            if self.value is not None and self.value.sequence > after:
                return self.value
            return None

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


def open_capture(config):
    if isinstance(config.uri, str) and '://' in config.uri:
        return cv2.VideoCapture(config.uri, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, config.timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, config.timeout_ms])
    return cv2.VideoCapture(config.uri)


class Source:
    def __init__(self, config):
        self.config = config
        self.frames = Latest()
        self.results = Latest()
        self.lock = threading.Lock()
        self.stats = dict(status='starting', error=None, processing_error=None, captured=0, processed=0,
                          skipped=0, reconnects=0, subscribers=0, encode_count=0)

    def update(self, **values):
        with self.lock:
            self.stats.update(values)

    def increment(self, name, amount=1):
        with self.lock:
            self.stats[name] += amount

    def snapshot(self):
        with self.lock:
            stats = dict(self.stats)
        stats['error'] = stats['error'] or stats['processing_error']
        result = self.results.get()
        return dict(id=self.config.id, **stats,
                    frame_age_seconds=None if result is None else round(time.time() - result.metadata['captured_at'], 3),
                    latest=None if result is None else result.metadata)


class Service:
    def __init__(self, configs, detector, capture_factory=open_capture):
        self.sources = {c.id: Source(c) for c in configs}
        self.detector = detector
        self.capture_factory = capture_factory
        self.stop_event = threading.Event()
        self.wake = threading.Event()
        self.threads = []
        self.started = False

    def start(self):
        if self.started:
            raise RuntimeError('service can only be started once')
        self.started = True
        for source in self.sources.values():
            self.threads.append(threading.Thread(target=self._capture, args=(source,), daemon=True,
                                                 name=f'capture-{source.config.id}'))
        self.threads.append(threading.Thread(target=self._process, daemon=True, name='inference'))
        for thread in self.threads:
            thread.start()

    def stop(self):
        self.stop_event.set()
        self.wake.set()
        for source in self.sources.values():
            source.frames.close()
            source.results.close()
        deadline = time.monotonic() + 2 + max(c.config.timeout_ms for c in self.sources.values()) / 1000
        for thread in self.threads:
            thread.join(max(0, deadline - time.monotonic()))
        alive = [t.name for t in self.threads if t.is_alive()]
        if alive:
            log.warning('Workers still exiting (backend read/inference may block): %s', alive)
        return alive

    def _capture(self, source):
        config = source.config
        sequence = 0
        due = time.monotonic()
        try:
            while not self.stop_event.is_set():
                capture = None
                failed = False
                try:
                    capture = self.capture_factory(config)
                    if not capture.isOpened():
                        raise RuntimeError('capture open failed')
                    source.update(status='capturing', error=None)
                    native_fps = capture.get(cv2.CAP_PROP_FPS)
                    period = 1 / (native_fps if math.isfinite(native_fps) and native_fps > 0 else config.fps)
                    read_any = False
                    while not self.stop_event.is_set():
                        ok, image = capture.read()
                        if not ok:
                            if config.live or not read_any:
                                raise RuntimeError('capture read failed')
                            break
                        read_any = True
                        # Files follow their media clock; network/camera reads are drained immediately.
                        if not config.live:
                            if self.stop_event.wait(max(0, due - time.monotonic())):
                                break
                            due = max(due + period, time.monotonic())
                        if image.shape[1] > config.max_width:
                            scale = config.max_width / image.shape[1]
                            image = cv2.resize(image, (config.max_width, max(1, round(image.shape[0] * scale))))
                        sequence += 1
                        source.frames.publish(Frame(sequence, time.time(), time.monotonic(), image))
                        source.increment('captured')
                        self.wake.set()
                except Exception as exc:
                    failed = True
                    # Do not include URI credentials or backend exception text in public status.
                    source.update(status='reconnecting' if config.live else 'error', error='capture unavailable')
                    log.warning('Capture unavailable for %s (%s)', config.id, type(exc).__name__)
                finally:
                    if capture is not None:
                        capture.release()
                if self.stop_event.is_set():
                    break
                if not config.live and (failed or not config.loop):
                    if not failed:
                        source.update(status='ended')
                    break
                if failed:
                    source.increment('reconnects')
                    if self.stop_event.wait(config.reconnect_seconds):
                        break
        finally:
            source.frames.close()
            self.wake.set()

    def _process(self):
        seen = {key: 0 for key in self.sources}
        due = {key: 0.0 for key in self.sources}
        while not self.stop_event.is_set():
            self.wake.clear()
            for key, source in self.sources.items():
                if self.stop_event.is_set():
                    break
                frame = source.frames.get()
                if frame is None or frame.sequence <= seen[key]:
                    if source.frames.closed:
                        source.results.close()
                    continue
                if time.monotonic() < due[key]:
                    continue
                source.increment('skipped', frame.sequence - seen[key] - 1)
                seen[key] = frame.sequence
                started = time.monotonic()
                due[key] = started + 1 / source.config.fps
                try:
                    # Copy because the capture slot may still be observed by other code.
                    annotated, objects = self.detector.process(key, frame.image.copy())
                    inferred = time.monotonic()
                    ok, encoded = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, source.config.jpeg_quality])
                    if not ok:
                        raise RuntimeError('JPEG encoding failed')
                    jpeg = encoded.tobytes()
                    source.increment('encode_count')
                    metadata = dict(source_id=key, sequence=frame.sequence, captured_at=frame.captured_at,
                                    published_at=time.time(), mode=self.detector.mode, detections=objects,
                                    width=annotated.shape[1], height=annotated.shape[0], jpeg_bytes=len(jpeg),
                                    inference_ms=round((inferred - started) * 1000, 2),
                                    latency_ms=round((time.monotonic() - frame.captured_mono) * 1000, 2))
                    source.results.publish(Result(frame.sequence, jpeg, metadata))
                    source.increment('processed')
                    source.update(processing_error=None)
                except Exception as exc:
                    source.update(processing_error='processing failed')
                    log.error('Processing failed for %s (%s)', key, type(exc).__name__)
            self.wake.wait(.01)
