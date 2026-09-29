# 다중 영상 화재 탐지 서비스

YOLOv7 탐지 결과를 입력별 영상 스트림과 JSON API로 제공합니다. 신규 실행 경로는 `fire_service`입니다. 이전 `main_dectection.py`, `start_api_server.py`, `server.py/js`와 `test/client.*`는 과거 구현으로 보존했으며 새 서비스와 함께 실행하지 않습니다.

## 빠른 실행

프로젝트 루트에서 Python 3.11 이상으로 실행합니다. 현재 macOS에서는 Python 3.14.7로 검증했습니다. 저장소에 들어 있는 `.venv`는 오래된 Windows 환경이므로 재사용하지 않습니다.

```bash
python3 -m venv .venv-service
.venv-service/bin/python -m pip install -r requirements-service.txt
# 모델 없이 입력/전송만 시험: 화면과 API에 passthrough 명시
.venv-service/bin/python -m fire_service --config sources.json --detector passthrough --port 9050
```

[로컬 모니터](http://127.0.0.1:9050)에서 두 입력을 확인합니다. 예제 설정은 동일한 `project.avi`를 독립 입력 두 개로 엽니다. 이 영상은 Git에서 제외되어 있으므로 없는 환경에서는 자신의 파일/카메라로 설정을 바꿔야 합니다. 서비스는 모델이 없을 때 자동으로 시험 모드로 전환하지 않습니다.

실제 탐지:

```bash
.venv-service/bin/python -m pip install -r requirements-detector.txt
.venv-service/bin/python -m fire_service --config sources.local.json \
  --weights weights/bestofbest.pt --device cpu --port 9050
# NVIDIA 환경: CUDA에 맞는 torch/torchvision 설치 후 --device 0
```

`weights/bestofbest.pt`는 저장소에 없습니다. 기존에 사용한 본인의 신뢰할 수 있는 화재 모델 파일을 준비해야 합니다. 기본 탐지 클래스는 `0`; 모델의 라벨 순서에 맞게 `--classes 0 1`, `--confidence 0.25`, `--image-size 640`으로 조정합니다. MPS 최적화는 포함하지 않았습니다.

## 입력 설정

`sources.json`을 `sources.local.json`으로 복사해 변경합니다. 로컬 설정은 Git에서 제외합니다. 파일 상대 경로는 **설정 파일이 있는 디렉터리** 기준이며 USB 번호는 문자열이 아닌 정수입니다. 설정 변경 후 서비스를 재시작합니다.

```json
{
  "sources": [
    {"id": "entrance", "uri": "rtsp://camera-host/live", "fps": 10, "max_width": 960, "jpeg_quality": 75, "timeout_ms": 5000, "reconnect_seconds": 2},
    {"id": "warehouse", "uri": "http://camera-host/video.mjpg", "fps": 8},
    {"id": "usb", "uri": 0, "fps": 10},
    {"id": "recording", "uri": "videos/fire.mp4", "fps": 10, "loop": true}
  ]
}
```

| 설정 | 기본값 / 의미 |
|---|---|
| `id` | 필수, 영문·숫자·`_`·`-`, 최대 64자, 중복 불가 |
| `uri` | 필수, 파일·USB 번호·RTSP/RTSPS·HTTP/HTTPS 영상 주소 |
| `fps` | 10, 탐지·출력의 최대 FPS; 모델 부하에 따라 낮아짐 |
| `max_width` | 960, 입력 프레임의 최대 가로 폭; 축소된 좌표계로 탐지 결과 반환 |
| `jpeg_quality` | 75, 20–95 |
| `loop` | false, 파일 반복 재생 여부 |
| `reconnect_seconds` | 2, 실시간 입력 장애 시 재시도 간격 |
| `timeout_ms` | 5000, FFmpeg 네트워크 열기/읽기 제한 시간 |

파일은 원본 FPS에 맞춰 읽고, 카메라/네트워크 입력은 계속 읽어 최신 프레임으로 교체합니다. 파일 EOF는 `ended`, 잘못된 파일은 `error`, 실시간 연결 실패는 `reconnecting`으로 표시합니다. 네트워크 시간 제한은 FFmpeg 백엔드 대상이며 USB 장치 드라이버의 정지/읽기 시간까지 보장하지 않습니다. 관련 동작은 [OpenCV 문서](https://docs.opencv.org/4.12.0/d4/d15/group__videoio__flags__base.html)를 따릅니다.

## 외부 수신 API

기본 전달 방식은 외부 클라이언트가 서비스에 연결해 받는 HTTP pull입니다. Socket.IO 이벤트 중계나 지정 서버로의 자동 push는 새 경로에 포함하지 않았습니다.

| 경로 | 응답 |
|---|---|
| `/` | 다중 영상 모니터 |
| `/healthz` | 프로세스 응답 여부와 탐지 모드 |
| `/readyz` | 모든 소스에 최근 프레임이 있고 입력/처리 정상일 때 200, 그 외 503 |
| `/api/sources` | 소스별 상태, 처리/건너뜀/재연결/인코딩/수신자 수, 최신 메타데이터 |
| `/api/sources/{id}/stream.mjpg` | `multipart/x-mixed-replace` JPEG 연속 스트림 |
| `/api/sources/{id}/snapshot.jpg` | 최신 JPEG |
| `/api/sources/{id}/detections` | 최신 탐지 JSON |
| `/get` | 기존 `Data`, `name`, `camInfo`, `Points` 형식. `camInfo`는 실제 소스 ID |

```html
<img src="http://SERVER:9050/api/sources/entrance/stream.mjpg" alt="입구 탐지 영상">
```

JSON에는 `source_id`, `sequence`, `captured_at`/`published_at`(Unix 초), `mode`, `width`, `height`, `jpeg_bytes`, `inference_ms`, `latency_ms`, `detections`가 포함됩니다. 각 탐지는 `class_id`, `name`, `confidence`, `bbox: [x1,y1,x2,y2]`, `track_id`를 가집니다. `track_id`는 소스와 프로세스 수명 내에서만 식별되므로 `(source_id, track_id)`로 사용합니다. 프레임 번호는 버린 프레임만큼 건너뜁니다.

스냅샷/JSON/스트림 최초 연결에서 프레임이 없거나 10초 이상 오래되면 503, 미등록 소스는 404, 동시 영상 수신 제한 초과는 429입니다. 이미 열린 스트림도 오래된 입력이 지속되면 닫히므로 수신자는 재연결해야 합니다. 모니터는 오래된 연결을 해제하고 입력이 복구되면 연결합니다. `/get`은 오래된 결과를 제외합니다.

## 전송·탐지 구조와 한계

```text
파일 / USB / RTSP / HTTP (각각 독립 capture 스레드)
  → 소스당 최신 원본 프레임 1개
  → 공유 YOLO 모델 하나가 소스들을 순환 처리
  → 소스별 IoU 추적 + JPEG 인코딩 1회
  → 소스당 최신 JPEG/JSON 1개
  → 여러 HTTP 수신자 (느리면 중간 프레임 건너뜀)
```

- 무한 파일 쓰기/TCP 반복 전송, 전역 결과 충돌, Base64 변환과 영상 중계 서버를 서비스 경로에서 제거했습니다.
- JPEG는 수신자 수와 관계없이 처리 프레임당 한 번 만듭니다. 애플리케이션에 프레임 이력 큐를 쌓지 않습니다. 대역폭은 수신자 수에 비례합니다.
- TCP/프록시/브라우저 내부 버퍼까지 없애는 것은 아닙니다. 저속 회선 수신자의 화면 지연은 발생할 수 있습니다. 모든 프레임 기록·정확한 전송 완료 보장 서비스가 아닙니다.
- 여러 입력은 동시에 읽지만 GPU 추론은 순차 공유합니다. 한 추론이 오래 걸리면 다른 입력의 탐지도 기다립니다. 배치 추론/다중 GPU/분산 워커는 구현하지 않았습니다.
- TensorFlow DeepSORT ReID 대신 소스별 IoU 추적을 사용합니다. 겹치는 박스의 ID 유지만 제공하며 가림 이후 재식별은 DeepSORT와 동등하지 않습니다. 객체 ID보다 탐지 영상 전달을 우선한 선택입니다.
- 출력 축소는 탐지 입력에도 적용됩니다. 작은 불꽃의 탐지 품질은 실제 모델과 현장 영상으로 검증 후 `max_width`, `image-size`, 임계값을 조정해야 합니다.
- MJPEG에는 소리가 없습니다. 장거리/다수 수신자용 H.264/WebRTC는 별도 확장 대상입니다.

## 운영 서버 실행

빠른 실행의 Flask 내장 서버는 개발 확인용입니다. Linux/macOS에서는 단일 Gunicorn 워커로 모델과 소스 상태를 공유하고, 영상 제한보다 많은 스레드를 둡니다. **워커 수는 1, `--preload`는 사용하지 않습니다.** 워커를 늘리면 카메라와 모델을 중복으로 엽니다. [Gunicorn 스레드·preload 설정](https://gunicorn.org/reference/settings/) 참고.

```bash
.venv-service/bin/python -m pip install -r requirements-server.txt -r requirements-detector.txt
FIRE_CONFIG=sources.local.json FIRE_WEIGHTS=weights/bestofbest.pt FIRE_DEVICE=0 \
  .venv-service/bin/gunicorn 'fire_service.wsgi:application()' \
  --bind 127.0.0.1:9051 --workers 1 --worker-class gthread --threads 24 \
  --timeout 30 --graceful-timeout 10
```

모델 없는 운영 서버 전달 시험은 `FIRE_DETECTOR=passthrough`로 실행합니다. 기본 동시 스트림 제한은 16이며 `FIRE_MAX_STREAMS`로 조정합니다. API 응답용 여유 스레드를 남깁니다. `FIRE_CONFIG`, `FIRE_WEIGHTS` 기본값은 각각 `sources.json`, `weights/bestofbest.pt`입니다.

기본 바인딩은 로컬 전용입니다. LAN 접근은 `--bind 0.0.0.0:9051`(CLI는 `--host 0.0.0.0`)로 변경합니다. 인증/TLS는 서비스에 내장하지 않았으므로 외부 공개 시 인증 프록시 뒤에서 운영해야 합니다. Nginx에서는 영상 경로에 `proxy_buffering off; proxy_read_timeout 60s;`를 적용합니다. 소스 주소/비밀번호는 API에 노출하지 않습니다. 다른 출처의 웹 앱에서 JSON을 읽을 때는 같은 출처의 프록시로 연결합니다.

모델은 기존 YOLO 객체를 포함한 체크포인트라 `weights_only=False`로 읽습니다. [PyTorch 문서](https://docs.pytorch.org/docs/2.14/generated/torch.load.html)에 따라 신뢰할 수 있는 본인의 파일만 사용해야 합니다.

## 검증

```bash
.venv-service/bin/python -m pip install -r requirements-test.txt
.venv-service/bin/python -m pytest -q
# 실행 중인 서버에 실제 수신자 5개 연결, JPEG 디코딩/프레임 순서/인코딩 재사용 확인
.venv-service/bin/python scripts/verify_http.py --base-url http://127.0.0.1:9051
# 실제 HTTP 입력 + 별도 파일 입력을 동시에 처리하는 테스트도 포함
FIRE_TEST_HTTP_INPUT=http://127.0.0.1:9051/api/sources/camera-a/stream.mjpg \
  .venv-service/bin/python -m pytest -q
```

ML 의존성이 없으면 합성 YOLO 테스트 하나를 건너뜁니다. HTTP 입력 URL을 지정하지 않으면 네트워크 통합 테스트 하나를 건너뜁니다. 합성 모델은 코드 호환성 시험이며 실제 화재 탐지 정확도 시험이 아닙니다. 현재 검증한 패키지 스냅샷은 `requirements-verified.txt`, 측정 결과와 미검증 사항은 [VERIFICATION.md](VERIFICATION.md), 작업 경과는 [WORK_LOG.md](WORK_LOG.md)에 있습니다. 기존 학습 환경의 `requirements.txt`는 그대로 보존했습니다.

## 원본 프로젝트

YOLOv7 + DeepSORT 기반 화재 탐지 서비스에서 출발했습니다. 원본 참고: https://github.com/deshwalmahesh/yolov7-deepsort-tracking

기존 시연: https://youtu.be/LzrBnzF2Fzw
