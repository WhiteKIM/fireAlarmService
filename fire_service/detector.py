"""Model adapter plus independent per-source IoU tracking (no ReID network)."""
from pathlib import Path
import time

import cv2


class PassthroughDetector:
    mode = 'passthrough'

    def process(self, source_id, frame):
        cv2.putText(frame, 'TRANSPORT TEST - NO DETECTION', (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
        return frame, []


def iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0 else 0


class IoUTracker:
    def __init__(self, threshold=.3, max_age=2):
        self.threshold, self.max_age = threshold, max_age
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, now=None):
        now = time.monotonic() if now is None else now
        self.tracks = {key: value for key, value in self.tracks.items() if now - value[1] <= self.max_age}
        unmatched = set(self.tracks)
        output = []
        for detection in sorted(detections, key=lambda d: d['confidence'], reverse=True):
            candidates = [(iou(detection['bbox'], self.tracks[key][0]['bbox']), key)
                          for key in unmatched if detection['class_id'] == self.tracks[key][0]['class_id']]
            score, key = max(candidates, default=(0, 0))
            if score < self.threshold:
                key = self.next_id
                self.next_id += 1
            else:
                unmatched.remove(key)
            item = dict(detection, track_id=key)
            self.tracks[key] = (item, now)
            output.append(item)
        return output


class YoloDetector:
    mode = 'yolov7'

    def __init__(self, weights, device='', confidence=.25, classes=(0,), image_size=640):
        if not Path(weights).is_file():
            raise ValueError(f'Model weights not found: {weights}. Supply --weights or explicitly use --detector passthrough for transport testing.')
        # Lazy import keeps the transport service/test suite independent of ML dependencies.
        from detection_helpers import Detector
        self.detector = Detector(conf_thres=confidence, classes=list(classes), device=device)
        self.detector.load_model(str(weights), img_size=image_size, trace=False)
        self.trackers = {}

    def process(self, source_id, frame):
        raw = self.detector.detect(frame, plot_bb=False)
        objects = []
        if raw is not None:
            for x1, y1, x2, y2, confidence, class_id in raw:
                class_id = int(class_id)
                objects.append(dict(class_id=class_id, name=str(self.detector.names[class_id]),
                                    confidence=round(float(confidence), 4),
                                    bbox=[int(x1), int(y1), int(x2), int(y2)]))
        objects = self.trackers.setdefault(source_id, IoUTracker()).update(objects)
        for obj in objects:
            x1, y1, x2, y2 = obj['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 80, 255), 2)
            cv2.putText(frame, f"{obj['name']} #{obj['track_id']} {obj['confidence']:.2f}",
                        (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 80, 255), 2)
        return frame, objects
