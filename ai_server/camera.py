"""카메라 캡처 관리.

- 물리 카메라 하나당 캡처 스레드 하나만 연다. 같은 장치를 쓰는 화면(share)은 그 캡처의 최신 프레임을 같이 본다.
- 한 컴퓨터에서 AI 서버가 두 개 떠서 같은 카메라를 동시에 여는 일을 파일 잠금으로 막는다.
- 장치를 못 열면 검은 화면 대신 status='error' 와 원인을 남긴다. 몇 번(retries)만 다시 시도하고 멈춘 뒤,
  화면의 '재연결' 버튼(POST /api/reconnect)을 누를 때 다시 시도한다 (끝없는 재시도 방지).
"""
import fcntl
import json
import logging
import subprocess
import threading
import time
from pathlib import Path

import cv2

log = logging.getLogger("camera")
MAX_INDEX = 5  # 자동 탐색할 장치 번호 범위 0~4


def mac_camera_names():
    """macOS가 인식한 카메라 이름 목록 (system_profiler 순서).
    OpenCV 장치 번호와 순서가 대체로 같지만 보장되지는 않으므로 '추정'으로만 쓴다."""
    try:
        out = subprocess.run(["system_profiler", "SPCameraDataType", "-json"], capture_output=True, text=True, timeout=10).stdout
        return [c.get("_name", "") for c in json.loads(out).get("SPCameraDataType", [])]
    except Exception:
        return []


def try_open(index, width=None, height=None, timeout=4.0):
    """장치를 열고 실제 프레임이 들어오는지 확인한다. 성공하면 (cap, (w, h)), 실패하면 (None, 원인).
    index 가 문자열이면 영상 파일 경로로 보고 연다 (카메라 없이 시연 연습·테스트용)."""
    if isinstance(index, str):
        if not Path(index).expanduser().exists():
            return None, f"영상 파일 없음: {index}"
        cap = cv2.VideoCapture(str(Path(index).expanduser()))
    else:
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap.release()
        return None, "장치를 열 수 없음 (없거나 카메라 권한 없음)"
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    end = time.time() + timeout
    while time.time() < end:
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            return cap, (frame.shape[1], frame.shape[0])
        time.sleep(0.1)
    cap.release()
    return None, "장치는 열렸지만 영상 프레임이 들어오지 않음"


class ProcessLock:
    """다른 AI 서버 프로세스가 이미 카메라를 쓰고 있으면 시작하지 않도록 잠금."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w")

    def acquire(self):
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False


class Capture(threading.Thread):
    """물리 카메라 하나. 최신 프레임과 연결 상태를 들고 있다."""

    def __init__(self, cam_cfg, taken):
        super().__init__(daemon=True)
        self.cfg = cam_cfg
        self.id = cam_cfg["id"]
        self.taken = taken  # 다른 캡처가 이미 쓰는 장치 번호 (같은 장치를 두 번 열지 않음)
        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0
        self.frame_ts = 0.0
        self.jpeg = (None, -1)
        self.status = "connecting"
        self.error = ""
        self.index = None
        self.is_file = False
        self.name = ""
        self.resolution = None
        self.fps = 0.0
        self.running = True
        self.retries = int(cam_cfg.get("retries", 3))
        self.wake = threading.Event()   # 재연결 요청
        self.max_width = int(cam_cfg.get("stream_max_width", 960))

    def candidates(self):
        dev = self.cfg.get("device", "auto")
        names = mac_camera_names()
        order = []
        if isinstance(dev, str) and dev != "auto":
            return [dev], names  # 영상 파일
        if isinstance(dev, int):
            order.append(dev)
        else:
            hint = (self.cfg.get("name_hint") or "").lower()
            order += [i for i, n in enumerate(names) if hint and hint in n.lower()]
        if dev == "auto" or self.cfg.get("fallback", True):
            order += list(range(MAX_INDEX))
        seen = []
        for i in order:
            if i not in seen and i not in self.taken:
                seen.append(i)
        return seen, names

    def open(self):
        order, names = self.candidates()
        hint = self.cfg.get("name_hint")
        if hint and not any(hint.lower() in n.lower() for n in names):
            log.warning("%s: '%s' 이름의 카메라가 macOS 목록에 없습니다. 인식된 카메라: %s", self.id, hint, names or "없음")
        reasons = []
        for i in order:
            cap, info = try_open(i, self.cfg.get("width"), self.cfg.get("height"))
            if cap:
                self.index = i
                self.is_file = isinstance(i, str)
                self.name = f"영상 파일 {Path(i).name}" if self.is_file else names[i] if i < len(names) else f"장치 {i}"
                self.resolution = info
                self.taken.add(i)
                log.info("%s: 장치 %s 열기 성공 — %s%s · %dx%d", self.id, i, self.name, "" if self.is_file else "(이름은 추정)", *info)
                return cap
            reasons.append(f"{i}: {info}")
        self.error = "카메라를 찾지 못함 — " + "; ".join(reasons) if reasons else "시도할 장치 없음"
        log.error("%s: %s", self.id, self.error)
        return None

    def reconnect(self):
        self.wake.set()

    def run(self):
        attempts = 0
        while self.running:
            self.status, self.error = "connecting", ""
            cap = self.open()
            if not cap:
                attempts += 1
                self.status = "error"
                if attempts >= self.retries:
                    self.error += f" — {attempts}번 시도 후 중단. 카메라를 확인한 뒤 화면의 ‘재연결’을 누르세요"
                    log.error("%s: 재시도를 멈춤 (재연결 요청을 기다림)", self.id)
                    self.wake.wait()
                    attempts = 0
                else:
                    self.wake.wait(5)
                self.wake.clear()
                continue
            attempts = 0
            self.status = "live"
            fails, t0, n = 0, time.time(), 0
            file_fps = (cap.get(cv2.CAP_PROP_FPS) or 15) if self.is_file else 0
            while self.running:
                ok, frame = cap.read()
                if self.is_file:
                    if not ok:  # 영상 파일은 끝나면 처음부터 반복
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                    time.sleep(1 / file_fps)
                if not ok or frame is None:
                    fails += 1
                    if fails > 30:
                        self.status, self.error = "error", "영상이 끊김 (케이블·권한 확인)"
                        log.error("%s: %s", self.id, self.error)
                        break
                    time.sleep(0.05)
                    continue
                fails = 0
                with self.lock:
                    self.frame = frame
                    self.seq += 1
                    self.frame_ts = time.time()
                n += 1
                if time.time() - t0 >= 2:
                    self.fps, t0, n = n / (time.time() - t0), time.time(), 0
            cap.release()
            self.taken.discard(self.index)
            time.sleep(2)

    def latest(self):
        with self.lock:
            return (None if self.frame is None else self.frame), self.seq, self.frame_ts

    def latest_jpeg(self):
        """최신 프레임 JPEG (같은 프레임은 한 번만 인코딩해 CAM-01·CAM-02 화면이 같이 씀)."""
        with self.lock:
            if self.frame is None:
                return None, -1
            if self.jpeg[1] != self.seq:
                # 화면 전송용은 너비를 줄여 보냄 (브라우저 디코딩·전송 부담 감소). 분석은 원본 프레임으로 함.
                img = self.frame
                if img.shape[1] > self.max_width:
                    img = cv2.resize(img, (self.max_width, round(img.shape[0] * self.max_width / img.shape[1])), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    self.jpeg = (buf.tobytes(), self.seq)
            return self.jpeg

    def info(self):
        live = self.status == "live" and time.time() - self.frame_ts < 3
        return {
            "status": "live" if live else ("error" if self.status == "error" or self.frame_ts else self.status),
            "error": self.error if not live else "",
            "device": self.index,
            "device_name": self.name,
            "resolution": self.resolution,
            "fps": round(self.fps, 1),
        }


class CameraManager:
    def __init__(self, cameras_cfg):
        self.taken = set()
        self.captures = {}   # 물리 캡처 id → Capture
        self.views = {}      # 화면 id → (Capture, 복제 원본 id 또는 None)
        for c in cameras_cfg:
            if "share" not in c:
                cap = Capture(c, self.taken)
                self.captures[c["id"]] = cap
                self.views[c["id"]] = (cap, None)
        for c in cameras_cfg:
            if "share" in c:
                src = self.captures.get(c["share"])
                if not src:
                    raise ValueError(f"{c['id']}: share 대상 {c['share']} 가 없습니다")
                self.views[c["id"]] = (src, c["share"])

    def start(self):
        # 자동 탐색이 겹치지 않도록 하나씩 차례로 연 뒤 다음 캡처 시작
        for cap in self.captures.values():
            cap.start()
            end = time.time() + 15
            while cap.status == "connecting" and time.time() < end:
                time.sleep(0.2)

    def info(self):
        out = []
        for vid, (cap, dup) in self.views.items():
            out.append({"id": vid, "duplicate_of": dup, "source": cap.id, **cap.info()})
        return out
