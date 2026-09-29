"""Configuration is local/operator-owned; input URLs are never accepted over HTTP."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit


@dataclass(frozen=True)
class SourceConfig:
    id: str
    uri: str | int
    fps: float = 10
    max_width: int = 960
    jpeg_quality: int = 75
    loop: bool = False
    reconnect_seconds: float = 2
    timeout_ms: int = 5000

    def __post_init__(self):
        if not isinstance(self.id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', self.id):
            raise ValueError('source id must use 1–64 letters, digits, _ or -')
        if isinstance(self.uri, bool) or not isinstance(self.uri, (str, int)):
            raise ValueError('uri must be a file/URL string or camera integer')
        if isinstance(self.uri, int):
            if self.uri < 0:
                raise ValueError('camera index must be non-negative')
        elif not self.uri.strip():
            raise ValueError('uri must not be empty')
        elif '://' in self.uri and urlsplit(self.uri).scheme not in {'rtsp', 'rtsps', 'http', 'https'}:
            raise ValueError('supported URL schemes: rtsp, rtsps, http, https')
        for name, low, high in [('fps', .1, 60), ('max_width', 64, 3840),
                                ('jpeg_quality', 20, 95), ('reconnect_seconds', .1, 60),
                                ('timeout_ms', 100, 30000)]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'{name} must be between {low} and {high}')
        for name in ('max_width', 'jpeg_quality', 'timeout_ms'):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f'{name} must be an integer')
        if not isinstance(self.loop, bool):
            raise ValueError('loop must be boolean')

    @property
    def live(self):
        return isinstance(self.uri, int) or '://' in self.uri


def load_sources(path):
    path = Path(path).resolve()
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or set(document) != {'sources'}:
        raise ValueError('config must contain only a sources array')
    entries = document['sources']
    if not isinstance(entries, list) or not 1 <= len(entries) <= 32:
        raise ValueError('configure 1–32 sources')
    sources = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('each source must be an object')
        try:
            source = SourceConfig(**entry)
        except TypeError as exc:
            raise ValueError('unknown or missing source setting') from exc
        if source.id in {s.id for s in sources}:
            raise ValueError(f'duplicate source id: {source.id}')
        if not source.live:
            data = dict(entry)
            data['uri'] = str((path.parent / source.uri).resolve())
            source = SourceConfig(**data)
        sources.append(source)
    return sources
