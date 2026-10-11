"""사람 수 감지 모듈.

지금은 Teachable Machine '사람수' 모델(이미지 전체 분류: 사람 0~4명)을 쓴다.
이 모델은 사람 하나하나를 찾는 객체 탐지가 아니라 장면 전체를 다섯 가지 중 하나로 분류하므로,
사람이 겹치거나 촬영 각도·거리가 학습 때와 다르면 인원수가 틀릴 수 있고 5명 이상은 셀 수 없다.

나중에 YOLO 같은 객체 탐지 모델로 바꿀 때는 PeopleCounter 를 상속해 count() 만 구현하고
app.py 의 make_people_counter() 에서 그 클래스를 돌려주면 된다. 반환 형식만 맞추면 웹 화면 쪽은 수정할 필요가 없다.
"""
import re

import numpy as np


class PeopleCounter:
    """count(frame) → {"count": 정수 또는 None, "label": 원래 라벨, "confidence": 0~1, "probs": [...]}"""
    kind = "base"
    labels = []

    def count(self, frame_bgr):
        raise NotImplementedError


def label_to_count(label):
    m = re.search(r"(\d+)\s*명", label)
    return int(m.group(1)) if m else None


class TMPeopleCounter(PeopleCounter):
    kind = "teachable-machine-classifier"

    def __init__(self, classifier):
        self.clf = classifier
        self.labels = classifier.labels
        self.counts = [label_to_count(l) for l in self.labels]
        if None in self.counts:
            raise ValueError(f"사람수 라벨에서 인원을 읽을 수 없음: {self.labels}")

    def count(self, frame_bgr):
        probs = self.clf.predict(frame_bgr)
        i = int(np.argmax(probs))
        return {"count": self.counts[i], "label": self.labels[i], "confidence": float(probs[i]), "probs": probs}


# 예시 (아직 구현 안 함):
# class YoloPeopleCounter(PeopleCounter):
#     kind = "yolo-detector"
#     def __init__(self, weights): from ultralytics import YOLO; self.m = YOLO(weights)
#     def count(self, frame_bgr):
#         boxes = [b for b in self.m(frame_bgr)[0].boxes if int(b.cls) == 0 and float(b.conf) > 0.5]
#         return {"count": len(boxes), "label": f"사람 {len(boxes)}명", "confidence": ..., "boxes": ...}
