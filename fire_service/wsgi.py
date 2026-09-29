"""Gunicorn factory: exactly one worker, threads > FIRE_MAX_STREAMS, no preload."""
import atexit
import os

from .config import load_sources
from .detector import PassthroughDetector, YoloDetector
from .pipeline import Service
from .web import create_app


def application():
    sources = load_sources(os.environ.get('FIRE_CONFIG', 'sources.json'))
    mode = os.environ.get('FIRE_DETECTOR', 'yolov7')
    if mode not in {'yolov7', 'passthrough'}:
        raise ValueError('FIRE_DETECTOR must be yolov7 or passthrough')
    max_streams = int(os.environ.get('FIRE_MAX_STREAMS', '16'))
    if not 1 <= max_streams <= 256:
        raise ValueError('FIRE_MAX_STREAMS must be between 1 and 256')
    detector = (PassthroughDetector() if mode == 'passthrough' else
                YoloDetector(os.environ.get('FIRE_WEIGHTS', 'weights/bestofbest.pt'),
                             device=os.environ.get('FIRE_DEVICE', '')))
    service = Service(sources, detector)
    app = create_app(service, max_streams=max_streams)
    app.extensions['fire_service'] = service
    service.start()
    atexit.register(service.stop)
    return app
