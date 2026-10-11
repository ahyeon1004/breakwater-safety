"""사람 객체 탐지 + 추적 모듈 (YOLO, Ultralytics).

Teachable Machine '사람수' 모델(장면 전체 분류)과는 별개다. 이 모듈은 사람 하나하나의 위치(바운딩 박스)를 찾고
ByteTrack 추적으로 같은 사람이 움직여도 같은 번호(PERSON 1, PERSON 2 …)가 유지되게 한다.
번호는 발표 시연용 임시 추적 ID이며 사람의 신원을 식별하지 않는다.

- COCO 사전 학습 모델의 'person'(클래스 0)만 탐지한다.
- 박스 좌표는 원본 프레임 기준 0~1 비율로 내보낸다 → 화면은 영상이 실제로 그려진 영역에 맞춰 변환한다.
- 한두 프레임 놓쳐도 박스가 깜빡이지 않도록 hold_sec 동안 마지막 박스를 유지한다.
- 추적기 내부 ID(1, 7, 12 …)를 그대로 쓰지 않고, 처음 나타날 때 비어 있는 가장 작은 번호를 붙인다.
"""
import logging
import time
from pathlib import Path

log = logging.getLogger("person")


class PersonDetector:
    """카메라 하나 전용 (추적 상태를 카메라마다 따로 가져야 하므로 카메라마다 인스턴스를 만든다)."""

    def __init__(self, cfg, base: Path):
        self.cfg = cfg
        self.conf = float(cfg.get("confidence", 0.35))
        self.imgsz = int(cfg.get("imgsz", 640))
        self.device = cfg.get("device", "cpu")
        self.tracker = cfg.get("tracker", "bytetrack.yaml")
        self.hold = float(cfg.get("hold_sec", 0.6))
        self.weights = (base / cfg.get("weights", "models/yolo/yolo11n.pt")).resolve()
        self.model = None
        self.error = ""
        self.tracks = {}   # 추적기 ID → {"no": 표시 번호, "box": [x,y,w,h], "conf": 신뢰도, "seen": 마지막 감지 시각}
        try:
            if not self.weights.exists():
                raise FileNotFoundError(f"가중치 파일 없음: {self.weights} — `./run.sh prepare_models.py --yolo` 로 내려받으세요")
            from ultralytics import YOLO
            self.model = YOLO(str(self.weights))
            names = self.model.names
            if names.get(0) != "person":
                raise ValueError(f"클래스 0이 person 이 아님: {names.get(0)}")
            log.info("사람 탐지 모델 로드: %s (신뢰도 기준 %.2f, 입력 %d, 장치 %s)", self.weights.name, self.conf, self.imgsz, self.device)
        except Exception as e:
            self.model = None
            self.error = f"{type(e).__name__}: {e}"
            log.error("사람 탐지 모델을 사용할 수 없음 — %s", self.error)

    @property
    def ok(self):
        return self.model is not None

    def reset(self):
        """영상이 끊겼다가 다시 들어오면 이전 추적을 이어 붙이지 않도록 초기화."""
        self.tracks.clear()
        if self.model is not None and getattr(self.model, "predictor", None) is not None:
            self.model.predictor = None  # 추적기 상태 초기화

    def _number(self):
        used = {t["no"] for t in self.tracks.values()}
        n = 1
        while n in used:
            n += 1
        return n

    def update(self, frame_bgr):
        """프레임 하나를 처리하고 현재 표시할 사람 목록을 돌려준다."""
        h, w = frame_bgr.shape[:2]
        res = self.model.track(frame_bgr, persist=True, classes=[0], conf=self.conf, imgsz=self.imgsz,
                               device=self.device, tracker=self.tracker, verbose=False)[0]
        now = time.time()
        boxes = res.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)
            for (x1, y1, x2, y2), c, tid in zip(xyxy, confs, ids):
                if tid is None:
                    continue  # 추적기가 아직 확정하지 않은 박스는 번호 없이 띄우지 않음
                box = [float(x1 / w), float(y1 / h), float((x2 - x1) / w), float((y2 - y1) / h)]
                t = self.tracks.get(tid)
                if t is None:
                    t = self.tracks[tid] = {"no": self._number()}
                t.update(box=box, conf=float(c), seen=now)
        # 오래 안 보인 추적은 제거 (사람이 화면에서 사라지면 박스도 사라짐)
        for tid in [k for k, t in self.tracks.items() if now - t["seen"] > self.hold]:
            del self.tracks[tid]
        return sorted(({"id": t["no"], "label": f"PERSON {t['no']}", "box": [round(v, 4) for v in t["box"]],
                        "conf": round(t["conf"], 3), "held": now - t["seen"] > 0.01} for t in self.tracks.values()),
                      key=lambda p: p["id"])
