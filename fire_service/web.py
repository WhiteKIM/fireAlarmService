import threading
import time
from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template


def create_app(service, max_streams=16, stale_seconds=10):
    app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent.parent / 'templates'))
    permits = threading.BoundedSemaphore(max_streams)

    def source_or_404(source_id):
        if source_id not in service.sources:
            abort(404)
        return service.sources[source_id]

    def fresh_result(source):
        result = source.results.get()
        if result is None or time.time() - result.metadata['captured_at'] > stale_seconds:
            abort(503, description='No recent frame available')
        return result

    @app.after_request
    def no_cache(response):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.get('/')
    def index():
        return render_template('monitor.html', mode=service.detector.mode)

    @app.get('/healthz')
    def health():
        return jsonify(status='ok', mode=service.detector.mode)

    @app.get('/readyz')
    def ready():
        states = [s.snapshot() for s in service.sources.values()]
        ready = bool(states) and all(s['frame_age_seconds'] is not None and
                                    s['frame_age_seconds'] <= stale_seconds and
                                    s['status'] == 'capturing' and not s['error'] for s in states)
        return jsonify(ready=ready, mode=service.detector.mode), 200 if ready else 503

    @app.get('/api/sources')
    def sources():
        return jsonify(mode=service.detector.mode, sources=[s.snapshot() for s in service.sources.values()])

    @app.get('/api/sources/<source_id>/detections')
    def detections(source_id):
        source = source_or_404(source_id)
        return jsonify(fresh_result(source).metadata)

    @app.get('/api/sources/<source_id>/snapshot.jpg')
    def snapshot(source_id):
        result = fresh_result(source_or_404(source_id))
        return Response(result.jpeg, mimetype='image/jpeg', headers={'X-Sequence': str(result.sequence)})

    @app.get('/api/sources/<source_id>/stream.mjpg')
    def stream(source_id):
        source = source_or_404(source_id)
        fresh_result(source)
        if not permits.acquire(blocking=False):
            return jsonify(error='stream capacity reached'), 429
        source.increment('subscribers')
        released = False
        release_lock = threading.Lock()

        def release():
            nonlocal released
            with release_lock:
                if not released:
                    source.increment('subscribers', -1)
                    permits.release()
                    released = True

        def generate():
            sequence = 0
            try:
                while not service.stop_event.is_set():
                    result = source.results.wait(sequence, timeout=1)
                    if result is None:
                        latest = source.results.get()
                        if source.results.closed or latest is None or time.time() - latest.metadata['captured_at'] > stale_seconds:
                            break
                        continue
                    if time.time() - result.metadata['captured_at'] > stale_seconds:
                        break
                    sequence = result.sequence
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
                           str(len(result.jpeg)).encode() + b'\r\nX-Sequence: ' + str(sequence).encode() +
                           b'\r\n\r\n' + result.jpeg + b'\r\n')
                yield b'--frame--\r\n'
            finally:
                release()

        response = Response(generate(), content_type='multipart/x-mixed-replace; boundary=frame',
                            headers={'X-Accel-Buffering': 'no'})
        response.call_on_close(release)
        return response

    @app.get('/get')
    def legacy_metadata():
        items = []
        for source in service.sources.values():
            result = source.results.get()
            if result is None or time.time() - result.metadata['captured_at'] > stale_seconds:
                continue
            for obj in result.metadata['detections']:
                items.append(dict(name=obj['name'], camInfo=source.config.id,
                                  Points=dict(zip(('X', 'Y', 'Z', 'W'), obj['bbox']))))
        return jsonify(Data=items)

    return app
