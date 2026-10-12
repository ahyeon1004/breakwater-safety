"""레고 피규어 개별 탐지 (노란 머리 색으로 레고 확인 + 빈 배경 보조) + 간단한 추적.

- 1단계: 레고 노란 머리 색(HSV)이 있는 덩어리만 레고 후보로 본다 → 방파제·배경은 PERSON 이 되지 않는다.
- 2단계(빈 배경이 있을 때): 밝기 보정·색 비율 비교로 원래 있던 노란 물체를 빼고, 머리와 이어진 몸까지 박스를 넓힌다.
- 3번 연속 확인된 것만 번호(PERSON n)를 붙이고, 잠깐 놓치면 hold_sec 동안 유지한다. 번호는 시연용 임시 번호다.
- 불확실하면(화면 대부분이 바뀜, 노란 덩어리가 너무 많음) 박스를 그리지 않는다.
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
    """레고 피규어 탐지.

    1단계(레고 확인): 레고 피규어의 노란 머리 색(HSV)을 찾는다. 방파제·테트라포드·책상처럼 노란색이 아닌 것은
            빈 배경과 달라져도 PERSON 으로 표시하지 않는다. 머리 색은 config 의 lego.head_* 로 바꿀 수 있다.
    2단계(보조, 빈 배경이 저장된 경우만): 빈 배경과 비교해
        - 원래부터 있던 노란 물체(배경에도 있는 것)는 제외하고
        - 머리 아래로 이어진 몸 부분까지 박스를 넓힌다.
        비교할 때 화면 전체 밝기 변화(카메라 자동 노출)를 보정하고, 그림자처럼 밝기만 바뀐 곳은 무시한다(색 비율로 비교).
    결과가 불확실하면(화면 대부분이 바뀜, 노란 덩어리가 너무 많음) 박스를 그리지 않고 'uncertain' 을 낸다.
    """
    kind = "background"
    ok = True  # OpenCV 만 쓰므로 모델 로드 실패가 없음

    def __init__(self, cfg, base: Path, cam_id: str):
        self.cfg = cfg
        self.cam_id = cam_id
        self.work_w = int(cfg.get("work_width", 640))
        self.h_lo, self.h_hi = cfg.get("head_hue", [18, 34])      # OpenCV 색상(0~180). 레고 노란색 ≈ 21~28
        self.s_min = int(cfg.get("head_sat_min", 150))              # 레고 노랑 채도 ≈ 175~215, 사람 피부 ≈ 60~140
        self.v_min = int(cfg.get("head_val_min", 80))
        self.head_min = float(cfg.get("head_min_area", 0.00012))  # 화면 대비 머리 최소 넓이
        self.head_max = float(cfg.get("head_max_area", 0.01))
        self.chroma_thr = float(cfg.get("chroma_threshold", 0.06))  # 색 비율 차이 기준 (그림자 무시)
        self.max_fg = float(cfg.get("max_foreground", 0.25))
        self.max_figures = int(cfg.get("max_figures", 12))
        self.path = (base / cfg.get("background_dir", "backgrounds") / f"{cam_id}.png").resolve()
        self.tracker = SimpleTracker(float(cfg.get("hold_sec", 0.8)), int(cfg.get("min_hits", 3)))
        self.bg = None
        self.error = ""
        if self.path.exists():
            img = cv2.imread(str(self.path))
            if img is not None:
                self.bg = img.astype(np.float32)
                log.info("%s: 저장된 빈 배경 불러옴 (%s)", cam_id, self.path.name)

    @property
    def ready(self):
        return True  # 빈 배경 없이도 머리 색으로 탐지 (빈 배경은 정확도 보조)

    def _small(self, frame):
        h, w = frame.shape[:2]
        if w != self.work_w:
            frame = cv2.resize(frame, (self.work_w, round(h * self.work_w / w)), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(frame, (3, 3), 0)

    def set_background(self, frame):
        small = self._small(frame)
        self.bg = small.astype(np.float32)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.path), small)
        self.tracker.reset()
        log.info("%s: 빈 배경 저장 → %s", self.cam_id, self.path)

    def reset(self):
        self.tracker.reset()

    def _foreground(self, small):
        """빈 배경과 비교한 '새로 생긴 물체' 영역. 밝기 보정 + 색 비율 비교(그림자 무시). (마스크, 바뀐 비율)"""
        cur = small.astype(np.float32) + 1.0
        bg = self.bg + 1.0
        gain = np.median(cur.reshape(-1, 3), axis=0) / np.median(bg.reshape(-1, 3), axis=0)  # 카메라 자동 노출 보정
        bg = bg * gain
        cn = cur / cur.sum(axis=2, keepdims=True)   # 색 비율 (밝기와 무관)
        bn = bg / bg.sum(axis=2, keepdims=True)
        chroma = np.abs(cn - bn).max(axis=2)
        lum = np.abs(cur.mean(axis=2) - bg.mean(axis=2))
        # 색이 달라졌거나(레고 옷 등), 밝기가 아주 크게 달라진 곳만 (그림자·조명은 밝기만 조금 바뀜)
        mask = ((chroma > self.chroma_thr) | (lum > 70)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return mask, float(mask.mean() / 255)

    def update(self, frame, now=None):
        """반환: (상태, 사람 목록, 메시지). 상태: ok / uncertain / error."""
        small = self._small(frame)
        h, w = small.shape[:2]
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        ymask = cv2.inRange(hsv, (int(self.h_lo), self.s_min, self.v_min), (int(self.h_hi), 255, 255))
        # 끊어진 노란 조각을 먼저 이어 붙인 뒤 잡음 제거 (얇은 연필이 짧은 조각으로 쪼개져 머리처럼 보이지 않게)
        ymask = cv2.morphologyEx(ymask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        ymask = cv2.morphologyEx(ymask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

        fg = None
        if self.bg is not None and self.bg.shape == small.shape:
            fg, ratio = self._foreground(small)
            if ratio > self.max_fg:
                self.tracker.reset()
                return "uncertain", [], f"화면의 {ratio:.0%}가 빈 배경과 다름 (카메라 이동·조명 변화) — 빈 배경을 다시 저장하세요"

        # 1단계: 노란 머리 후보
        heads = []
        n, _, stats, _ = cv2.connectedComponentsWithStats(ymask)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            frac = area / (w * h)
            if not (self.head_min <= frac <= self.head_max):
                continue
            # 레고 머리는 거의 동그란 원기둥 모양: 가로:세로 0.6~1.7, 꽉 찬 덩어리 (길쭉한 연필·막대는 제외)
            if not (0.6 <= bw / bh <= 1.7) or area / (bw * bh) < 0.45 or bw < 4 or bh < 4:
                continue
            if fg is not None and fg[y:y + bh, x:x + bw].mean() / 255 < 0.3:
                continue  # 빈 배경에도 원래 있던 노란 물체 → 레고 아님
            heads.append([x, y, bw, bh, area])
        # 같은 피규어의 손(작은 노란 덩어리)은 머리에 합침: 큰 것부터 머리로 보고, 바로 아래·옆의 작은 것은 제외
        heads.sort(key=lambda b: -b[4])
        figs = []
        for x, y, bw, bh, area in heads:
            cx = x + bw / 2
            if any(abs(cx - (f[0] + f[2] / 2)) < 1.6 * f[2] and f[1] - 0.5 * f[3] <= y <= f[1] + 4 * f[3] for f in figs):
                continue
            figs.append([x, y, bw, bh, area])
        if len(figs) > self.max_figures:
            self.tracker.reset()
            return "uncertain", [], f"노란 덩어리가 {len(figs)}개로 너무 많음 — 조명·머리 색 설정 확인 필요"

        boxes = []
        lab = None
        if fg is not None:
            # 새로 생긴 물체 영역(+머리)을 덩어리로 나눔 → 머리와 이어진 덩어리만 그 피규어의 몸으로 봄
            lab = cv2.connectedComponents(cv2.dilate(cv2.bitwise_or(fg, ymask), np.ones((3, 3), np.uint8)))[1]
        for x, y, bw, bh, _ in figs:
            x0, y0, x1, y1 = x, y, x + bw, y + bh
            if lab is not None:
                ids = np.unique(lab[y:y + bh, x:x + bw])
                ids = ids[ids > 0]
                if len(ids):
                    # 머리 크기 기준으로 피규어가 있을 수 있는 범위(서 있거나 누운 경우) 안에서만 넓힘
                    rx0, rx1 = max(0, int(x - 4 * bw)), min(w, int(x + 5 * bw))
                    ry0, ry1 = max(0, int(y - 2 * bh)), min(h, int(y + 5 * bh))
                    ys, xs = np.nonzero(np.isin(lab[ry0:ry1, rx0:rx1], ids))
                    if len(xs):
                        x0, x1 = min(x0, rx0 + xs.min()), max(x1, rx0 + xs.max() + 1)
                        y0, y1 = min(y0, ry0 + ys.min()), max(y1, ry0 + ys.max() + 1)
            boxes.append(([x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h], None))

        if self.bg is not None and fg is not None:
            still = cv2.dilate(fg, np.ones((15, 15), np.uint8)) == 0
            self.bg[still] = self.bg[still] * 0.998 + small[still].astype(np.float32) * 0.002
        return "ok", self.tracker.update(boxes, now), ""
