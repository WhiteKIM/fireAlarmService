"""Measure the running service using real HTTP clients; no model accuracy claim."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np
import requests


def read_stream(base, source, frames, delay=0):
    sequences, sizes, arrivals = [], [], []
    with requests.get(f'{base}/api/sources/{source}/stream.mjpg', stream=True, timeout=(5, 10)) as response:
        response.raise_for_status()
        reader = response.raw
        while len(sequences) < frames:
            line = reader.readline()
            if not line:
                raise AssertionError('stream ended early')
            if line != b'--frame\r\n':
                continue
            headers = {}
            while (line := reader.readline()) != b'\r\n':
                if not line:
                    raise AssertionError('incomplete headers')
                key, value = line.decode().strip().split(': ', 1)
                headers[key] = value
            length = int(headers['Content-Length'])
            jpeg = reader.read(length)
            assert len(jpeg) == length
            image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            assert image is not None
            sequences.append(int(headers['X-Sequence']))
            sizes.append(length)
            arrivals.append(time.monotonic())
            if delay:
                time.sleep(delay)
    assert all(b > a for a, b in zip(sequences, sequences[1:]))
    return dict(source=source, frames=len(sequences), slow_reader=bool(delay),
                observed_fps=round((len(arrivals) - 1) / (arrivals[-1] - arrivals[0]), 2),
                average_jpeg_bytes=round(statistics.mean(sizes)),
                first_sequence=sequences[0], last_sequence=sequences[-1])


def compare_encoding(video):
    capture = cv2.VideoCapture(video)
    rows = []
    try:
        for _ in range(60):
            ok, frame = capture.read()
            if not ok:
                break
            start = time.perf_counter()
            ok, old = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            assert ok
            old_bytes = len(base64.b64encode(old))
            old_ms = (time.perf_counter() - start) * 1000
            if frame.shape[1] > 960:
                frame = cv2.resize(frame, (960, round(frame.shape[0] * 960 / frame.shape[1])))
            start = time.perf_counter()
            ok, new = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            assert ok
            new_bytes = len(new.tobytes())
            rows.append((old_bytes, new_bytes, old_ms, (time.perf_counter() - start) * 1000))
    finally:
        capture.release()
    assert rows
    means = [statistics.mean(column) for column in zip(*rows)]
    return dict(frames=len(rows), old_jpeg90_base64_bytes=round(means[0]), new_jpeg75_bytes=round(means[1]),
                payload_reduction_percent=round((1 - means[1] / means[0]) * 100, 1),
                old_encode_base64_ms=round(means[2], 3), new_encode_ms=round(means[3], 3),
                note='Component comparison only; quality/size differ. Not an end-to-end legacy benchmark.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:9050')
    parser.add_argument('--video', default='project.avi')
    parser.add_argument('--output', default='artifacts/verification/http.json')
    args = parser.parse_args()
    base = args.base_url.rstrip('/')
    before = requests.get(base + '/api/sources', timeout=5).json()
    sources = [s['id'] for s in before['sources']]
    assert len(sources) >= 2
    assert requests.get(base + '/readyz', timeout=5).status_code == 200
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs = [pool.submit(read_stream, base, source, 60) for source in sources[:2] for _ in range(2)]
        jobs.append(pool.submit(read_stream, base, sources[0], 20, .3))
        clients = [job.result() for job in jobs]
    elapsed = time.monotonic() - start
    after = requests.get(base + '/api/sources', timeout=5).json()
    counters = []
    for old, new in zip(before['sources'], after['sources']):
        processed = new['processed'] - old['processed']
        encodes = new['encode_count'] - old['encode_count']
        assert processed == encodes and processed > 0
        assert new['latest']['source_id'] == new['id']
        assert new['frame_age_seconds'] < 2
        snap = requests.get(f"{base}/api/sources/{new['id']}/snapshot.jpg", timeout=5)
        assert snap.status_code == 200 and snap.content.startswith(b'\xff\xd8')
        counters.append(dict(source=new['id'], processed=processed, encodes=encodes,
                             processed_fps=round(processed / elapsed, 2),
                             latency_ms=new['latest']['latency_ms']))
    report = dict(mode=after['mode'], elapsed_seconds=round(elapsed, 2), clients=clients,
                  counters=counters, encoding_comparison=compare_encoding(args.video))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
