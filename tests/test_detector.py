"""Optional ML checks: synthetic YOLO weights validate code, never accuracy."""
import numpy as np
import pytest


def test_current_torch_loads_and_runs_local_yolov7_checkpoint(tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('torchvision')
    from models.yolo import Model
    from fire_service.detector import YoloDetector

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = Model('cfg/training/yolov7-tiny.yaml', ch=3, nc=1).eval()
        model.names = ['synthetic-test']
        path = tmp_path / 'synthetic.pt'
        torch.save({'model': model}, path)
        detector = YoloDetector(path, device='cpu', confidence=.99, image_size=64)
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        for source in ['a', 'b']:
            output, objects = detector.process(source, frame.copy())
            assert output.shape == frame.shape
            assert isinstance(objects, list)
        assert detector.trackers['a'] is not detector.trackers['b']
    finally:
        torch.set_num_threads(previous_threads)


def test_yolo_adapter_draws_and_keeps_source_local_ids():
    from fire_service.detector import YoloDetector

    class RawDetector:
        names = ['fire']
        def detect(self, frame, plot_bb):
            return np.array([[5, 5, 40, 40, .9, 0]])

    detector = YoloDetector.__new__(YoloDetector)
    detector.detector = RawDetector()
    detector.trackers = {}
    for source in ['a', 'a', 'b']:
        image, objects = detector.process(source, np.zeros((64, 64, 3), dtype=np.uint8))
        assert image.any()
        assert objects[0]['track_id'] == 1
        assert objects[0]['name'] == 'fire'
