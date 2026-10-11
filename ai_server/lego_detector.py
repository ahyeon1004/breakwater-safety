"""레고 피규어 개별 탐지 (OpenCV 배경 비교) + 간단한 추적.

방법: 카메라와 방파제 모형은 고정되어 있으므로, 레고가 없는 '빈 배경' 사진을 한 번 저장해 두고
현재 화면과 비교해 달라진 덩어리(= 새로 놓인 물체)를 레고 피규어로 본다.

- 색과 상관없이 동작한다 (노란 얼굴·여러 색 옷 모두).
- 그림자에 덜 민감하도록 밝기 차이(L)는 낮은 가중치, 색 차이(a·b)는 높은 가중치로 비교한다.
- 피규어의 머리·몸·다리가 따로 잡히지 않도록 가까운 조각은 하나로 합친다.
- 두 프레임 이상 연속으로 확인된 덩어리만 표시하고(잡음으로 가짜 박스가 생기지 않게),
  잠깐 놓친 피규어는 hold_sec 동안 박스를 유지한다.
- 같은 피규어는 위치를 이어 붙여 같은 번호(PERSON n)를 유지한다. 번호는 시연용 임시 번호다.

한계 (정직하게):
- 레고가 아닌 물체(손, 다른 소품)가 들어와도 '새 물체'로 잡힌다.
- 피규어 둘이 붙어 있으면 한 덩어리(1명)로 잡힐 수 있다.
- 카메라가 움직이거나 조명이 크게 바뀌면 빈 배경을 다시 저장해야 한다
  (화면 대부분이 달라지면 박스를 그리지 않고 '배경 다시 저장 필요' 상태를 낸다).
- 더 정확하게 하려면 레고 사진에 박스를 그려 YOLO를 추가 학습해야 한다 (README 참고).
"""
import logging
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("lego")


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class SimpleTracker:
    """박스를 프레임 사이에 이어 붙여 같은 물체에 같은 번호를 준다."""

    def __init__(self, hold_sec=0.8, min_hits=2, max_jump=0.15):
        self.hold = hold_sec
        self.min_hits = min_hits
        self.max_jump = max_jump  # 한 번에 이 거리(화면 비율) 이상 튀면 다른 물체로 봄
        self.tracks = {}          # 내부 id → {"no", "box", "conf", "seen", "hits"}
        self.next_id = 1

    def reset(self):
        self.tracks.clear()

    def _number(self):
        used = {t["no"] for t in self.tracks.values() if t["no"]}
        n = 1
        while n in used:
            n += 1
        return n

    def update(self, boxes, now=None):
        now = now or time.time()
        center = lambda b: (b[0] + b[2] / 2, b[1] + b[3] / 2)
        # 가까운 순서대로 기존 추적과 짝짓기
        pairs = []
        for tid, t in self.tracks.items():
            for j, (b, c) in enumerate(boxes):
                (tx, ty), (bx, by) = center(t["box"]), center(b)
                dist = ((tx - bx) ** 2 + (ty - by) ** 2) ** 0.5
                score = iou(t["box"], b)
                if score > 0.1 or dist < self.max_jump:
                    pairs.append((-score, dist, tid, j))
        pairs.sort()
        used_t, used_b = set(), set()
        for _, _, tid, j in pairs:
            if tid in used_t or j in used_b:
                continue
            used_t.add(tid)
            used_b.add(j)
            t = self.tracks[tid]
            b, c = boxes[j]
            t["box"] = [0.6 * nb + 0.4 * ob for nb, ob in zip(b, t["box"])]  # 박스 떨림 완화
            t.update(conf=c, seen=now, hits=t["hits"] + 1)
        for j, (b, c) in enumerate(boxes):
            if j not in used_b:
                self.tracks[self.next_id] = {"no": 0, "box": list(b), "conf": c, "seen": now, "hits": 1}
                self.next_id += 1
        for tid in [k for k, t in self.tracks.items() if now - t["seen"] > self.hold]:
            del self.tracks[tid]
        # 확인된(연속 min_hits 번 이상) 추적에만 번호를 붙여 표시
        for t in self.tracks.values():
            if not t["no"] and t["hits"] >= self.min_hits:
                t["no"] = self._number()
        shown = [t for t in self.tracks.values() if t["no"]]
        return sorted(({"id": t["no"], "label": f"PERSON {t['no']}", "box": [round(v, 4) for v in t["box"]],
                        "conf": None if t["conf"] is None else round(t["conf"], 3), "held": t["seen"] < now} for t in shown), key=lambda p: p["id"])


class BackgroundLegoDetector:
    kind = "background"

    def __init__(self, cfg, base: Path, cam_id: str):
        self.cfg = cfg
        self.cam_id = cam_id
        self.work_w = int(cfg.get("work_width", 640))
        self.thr = float(cfg.get("diff_threshold", 32))
        self.min_area = float(cfg.get("min_area", 0.0005))
        self.max_area = float(cfg.get("max_area", 0.2))
        self.merge_px = int(cfg.get("merge_px", 9))
        self.learn = float(cfg.get("learn_rate", 0.002))
        self.max_fg = float(cfg.get("max_foreground", 0.35))
        self.path = (base / cfg.get("background_dir", "backgrounds") / f"{cam_id}.png").resolve()
        self.tracker = SimpleTracker(float(cfg.get("hold_sec", 0.8)), int(cfg.get("min_hits", 2)))
        self.bg = None      # 작업 크기의 float32 배경
        self.error = ""
        if self.path.exists():
            img = cv2.imread(str(self.path))
            if img is not None:
                self.bg = img.astype(np.float32)
                log.info("%s: 저장된 빈 배경 불러옴 (%s)", cam_id, self.path.name)

    ok = True  # OpenCV 만 쓰므로 모델 로드 실패가 없음

    @property
    def ready(self):
        return self.bg is not None

    def _small(self, frame):
        h, w = frame.shape[:2]
        if w != self.work_w:
            frame = cv2.resize(frame, (self.work_w, round(h * self.work_w / w)), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(frame, (5, 5), 0)

    def set_background(self, frame):
        small = self._small(frame)
        self.bg = small.astype(np.float32)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.path), small)
        self.tracker.reset()
        log.info("%s: 빈 배경 저장 → %s", self.cam_id, self.path)

    def reset(self):
        self.tracker.reset()

    def update(self, frame, now=None):
        """반환: (상태, 사람 목록, 오류 메시지). 상태는 ok / no_background / error. now 는 테스트용 시각."""
        if self.bg is None:
            return "no_background", [], "빈 배경이 저장되지 않음 — 레고를 모두 치운 뒤 ‘빈 배경 저장’을 누르세요"
        small = self._small(frame)
        if small.shape != self.bg.shape:
            return "error", [], "카메라 해상도가 저장한 빈 배경과 다름 — 빈 배경을 다시 저장하세요"
        lab_now = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.int16)
        lab_bg = cv2.cvtColor(self.bg.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.int16)
        d = np.abs(lab_now - lab_bg).astype(np.float32)
        diff = np.maximum(d[..., 0] * 0.6, np.maximum(d[..., 1], d[..., 2]) * 1.5)
        mask = (diff > self.thr).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        k = max(1, self.merge_px)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        fg = float(mask.mean() / 255)
        if fg > self.max_fg:
            self.tracker.reset()
            return "error", [], f"화면의 {fg:.0%}가 빈 배경과 다름 (카메라 이동·조명 변화) — 빈 배경을 다시 저장하세요"

        h, w = mask.shape
        boxes = []
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            frac = area / (w * h)
            if self.min_area <= frac <= self.max_area:
                filled = area / float(bw * bh)  # 덩어리가 박스를 얼마나 채우는지 (가는 선·잡음 거르기)
                if filled >= 0.15:
                    boxes.append(([x / w, y / h, bw / w, bh / h], None))  # 배경 비교는 모델 신뢰도가 없음

        # 조명이 천천히 바뀌는 것에 맞춰, 물체가 없는 곳의 배경만 아주 조금씩 갱신
        if self.learn > 0:
            still = cv2.dilate(mask, np.ones((15, 15), np.uint8)) == 0
            self.bg[still] = self.bg[still] * (1 - self.learn) + small[still].astype(np.float32) * self.learn
        return "ok", self.tracker.update(boxes, now), ""
