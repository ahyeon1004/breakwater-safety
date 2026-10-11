"""Teachable Machine(이미지 프로젝트) Keras 모델 로드와 추론.

Teachable Machine 내보내기 형식:
- keras_model.h5: 입력 (None, 224, 224, 3), 출력 softmax (None, 클래스 수)
- labels.txt: 한 줄에 "번호 라벨" (예: "0 낙상"). 번호가 출력 배열의 인덱스.
- 전처리: 정사각형으로 가운데 자르기 → 224×224 → RGB → (픽셀 / 127.5) - 1  (값 범위 -1~1)
"""
import logging
import threading
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("models")
_tf_lock = threading.Lock()  # 여러 카메라 스레드가 모델을 동시에 부르지 않도록


def read_labels(path: Path):
    labels = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        idx, _, name = line.partition(" ")
        labels[int(idx)] = name.strip()
    return [labels[i] for i in range(len(labels))]


def load_keras(path: Path):
    import tensorflow as tf

    # 일부 Keras 버전은 Teachable Machine 모델의 DepthwiseConv2D 'groups' 인자를 거부하므로 무시하도록 감쌈
    class DepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
        def __init__(self, *args, groups=None, **kwargs):
            super().__init__(*args, **kwargs)

    return tf.keras.models.load_model(str(path), compile=False, custom_objects={"DepthwiseConv2D": DepthwiseConv2D})


def preprocess(frame_bgr, size):
    h, w = frame_bgr.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    img = cv2.resize(frame_bgr[y:y + s, x:x + s], size, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    return (img / 127.5 - 1.0)[None]


class TMClassifier:
    """모델 하나. 로드에 실패해도 예외로 서버를 멈추지 않고 error 를 남긴다."""

    def __init__(self, key, folder: Path):
        self.key = key
        self.folder = folder
        self.labels = []
        self.model = None
        self.size = (224, 224)
        self.error = ""
        try:
            self.labels = read_labels(folder / "labels.txt")
            self.model = load_keras(folder / "keras_model.h5")
            _, h, w, c = self.model.input_shape
            out = self.model.output_shape[-1]
            if c != 3 or out != len(self.labels):
                raise ValueError(f"입력 {self.model.input_shape} / 출력 {out}개가 라벨 {len(self.labels)}개와 맞지 않음")
            self.size = (w, h)
            with _tf_lock:
                self.model(np.zeros((1, h, w, 3), np.float32), training=False)  # 첫 추론 지연을 미리 처리
            log.info("모델 로드: %s (%s) 입력 %s 라벨 %s", key, folder.name, self.model.input_shape, self.labels)
        except Exception as e:  # 파일 없음, 형식 오류 등
            self.model = None
            self.error = f"{type(e).__name__}: {e}"
            log.error("모델 로드 실패: %s (%s) — %s", key, folder, self.error)

    @property
    def ok(self):
        return self.model is not None

    def predict(self, frame_bgr):
        """확률 배열 (labels 순서)."""
        x = preprocess(frame_bgr, self.size)
        with _tf_lock:
            return self.model(x, training=False).numpy()[0]

    def info(self):
        return {"folder": str(self.folder.name), "loaded": self.ok, "labels": self.labels, "error": self.error,
                "input": list(self.size) + [3] if self.ok else None}


class Smoother:
    """최근 N번 추론 중 다수결로 안정된 결과를 낸다.
    - state 'pending': 아직 결과가 적음 (판별 중)
    - state 'uncertain': 의견이 갈리거나 신뢰도가 낮음 (감지 불확실)
    - state 'ok': 안정된 결과
    창 크기 8, 0.25초 간격이면 실제 변화는 약 1~2초 안에 반영된다."""

    def __init__(self, window=8, min_agreement=0.6, min_confidence=0.7):
        self.hist = deque(maxlen=window)
        self.min_agreement = min_agreement
        self.min_confidence = min_confidence

    def push(self, label, confidence):
        self.hist.append((label, float(confidence)))

    def clear(self):
        self.hist.clear()

    def result(self):
        if len(self.hist) < min(3, self.hist.maxlen):
            return {"state": "pending", "label": None, "confidence": None}
        label, votes = Counter(l for l, _ in self.hist).most_common(1)[0]
        conf = float(np.mean([c for l, c in self.hist if l == label]))
        agree = votes / len(self.hist)
        ok = agree >= self.min_agreement and conf >= self.min_confidence
        return {"state": "ok" if ok else "uncertain", "label": label if ok else None,
                "candidate": label, "confidence": round(conf, 3), "agreement": round(agree, 2)}
