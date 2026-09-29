import argparse
import logging
import signal

from .config import load_sources
from .detector import PassthroughDetector, YoloDetector
from .pipeline import Service
from .web import create_app


def main():
    parser = argparse.ArgumentParser(description='Multi-source detection + MJPEG/JSON server')
    parser.add_argument('--config', default='sources.json')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--detector', choices=['yolov7', 'passthrough'], default='yolov7')
    parser.add_argument('--weights', default='weights/bestofbest.pt')
    parser.add_argument('--device', default='', help='auto CUDA/CPU, cpu, or CUDA index e.g. 0')
    parser.add_argument('--confidence', type=float, default=.25)
    parser.add_argument('--classes', type=int, nargs='+', default=[0])
    parser.add_argument('--image-size', type=int, default=640)
    parser.add_argument('--max-streams', type=int, default=16)
    args = parser.parse_args()
    if not 0 < args.confidence <= 1 or not 32 <= args.image_size <= 2048 or not 1 <= args.max_streams <= 256:
        parser.error('invalid confidence, image size, or max streams')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        sources = load_sources(args.config)
        detector = (PassthroughDetector() if args.detector == 'passthrough' else
                    YoloDetector(args.weights, args.device, args.confidence, args.classes, args.image_size))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.detector == 'passthrough':
        logging.warning('TRANSPORT TEST MODE: no fire detection is performed')
    service = Service(sources, detector)
    def terminate(signum, frame):
        raise KeyboardInterrupt
    previous_handler = signal.signal(signal.SIGTERM, terminate)
    service.start()
    try:
        create_app(service, args.max_streams).run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == '__main__':
    main()
