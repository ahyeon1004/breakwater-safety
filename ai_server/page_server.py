"""관제 화면 서버 (카메라는 쓰지 않음).

- 프로젝트 폴더(index.html, web_models 등)를 http://127.0.0.1:8000/ 으로 보여 준다.
- USB 로 꽂힌 아두이노 온습도 센서를 자동으로 찾아(여러 대) 계속 읽고, 화면은 /api/sensors 로 값을 가져간다.
  → 화면에서 '아두이노 연결' 버튼을 누를 필요가 없다. 나중에 꽂아도 3초 안에 찾는다.
- 아두이노가 보내는 줄 예: "T=23.4,H=56.0", "습도:56% 온도:23도", "T:23.4 H:56", {"t":23.4,"h":56}, "23.4,56"

실행: 부우산 시작.command 를 더블클릭 (또는 ./run.sh page_server.py)
"""
import json
import os
import re
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.dont_write_bytecode = True  # 프로젝트 폴더에 캐시 파일을 만들지 않음

import serial
from serial.tools import list_ports

ROOT = Path(__file__).resolve().parent.parent
PORT = int(os.environ.get("PAGE_PORT", 8000))
BAUD = int(os.environ.get("SENSOR_BAUD", 9600))
SCAN_SEC = 3

T_RE = re.compile(r'(?:^|[^a-z가-힣])(?:temperature|temp|t|온도|기온)\s*["=:]*\s*(-?\d+(?:\.\d+)?)', re.I)
H_RE = re.compile(r'(?:^|[^a-z가-힣])(?:humidity|hum|h|습도)\s*["=:]*\s*(\d+(?:\.\d+)?)', re.I)
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse(line):
    """한 줄에서 온도·습도를 읽는다. 둘 중 하나만 있어도 돌려준다. 못 읽으면 None."""
    out = {}
    t, h = T_RE.search(line), H_RE.search(line)
    if t:
        out["t"] = float(t.group(1))
    if h:
        out["h"] = float(h.group(1))
    if not out and not re.search(r"[a-zA-Z가-힣]", line):
        nums = NUM_RE.findall(line)
        if len(nums) == 2:
            out = {"t": float(nums[0]), "h": float(nums[1])}
    if "t" in out and not -40 < out["t"] < 85:
        del out["t"]
    if "h" in out and not 0 <= out["h"] <= 100:
        del out["h"]
    return out or None


def is_arduino(p):
    name = (p.device or "").lower()
    return (p.vid in (0x2341, 0x2A03, 0x1A86, 0x10C4, 0x0403)  # Arduino, CH340, CP210x, FTDI
            or any(k in name for k in ("usbmodem", "usbserial", "wchusbserial")))


class Reader(threading.Thread):
    def __init__(self, device):
        super().__init__(daemon=True)
        self.device = device
        self.t = self.h = None
        self.at = 0.0
        self.error = ""
        self.alive = True
        self.last_line = ""      # 마지막으로 받은 원본 줄 (원인 확인용)
        self.last_line_at = 0.0

    def run(self):
        try:
            with serial.Serial(self.device, BAUD, timeout=1) as s:
                self.error = ""
                buf = b""
                while self.alive:
                    buf += s.read(256)
                    *lines, buf = buf.split(b"\n")
                    for raw in lines:
                        text = raw.decode("utf-8", "replace").strip()
                        if text:
                            self.last_line, self.last_line_at = text[:120], time.time()
                        r = parse(text)
                        if not r:
                            # 줄은 오는데 온습도 형식이 아니고, 최근 10초 동안 정상 값도 없으면 → 다른 스케치일 가능성
                            if text and time.time() - self.at > 10:
                                self.error = f"온습도 형식이 아님: '{text[:20]}' (다른 스케치가 올라가 있는지 확인)"
                            continue
                        if r.get("t") == 0 and r.get("h") == 0:
                            # 온도·습도가 모두 0 이면 센서를 못 읽는 것 (배선·핀 번호 확인)
                            self.error = "센서 값이 0 — 센서 배선·핀 번호 확인"
                            continue
                        self.t = r.get("t", self.t)
                        self.h = r.get("h", self.h)
                        self.at = time.time()
                        self.error = ""
                    if len(buf) > 1000:
                        buf = b""
        except (serial.SerialException, OSError) as e:
            busy = "busy" in str(e).lower() or getattr(e, "errno", None) == 16
            self.error = ("다른 프로그램이 사용 중 (Chrome 의 '아두이노 연결' 또는 Arduino IDE 시리얼 모니터를 닫으세요)"
                          if busy else f"읽기 실패: {e}")
        self.alive = False


readers = {}
lock = threading.Lock()


def scanner():
    """USB 아두이노를 주기적으로 찾아 새 것은 읽기 시작, 뽑힌 것은 정리, 실패한 것은 다시 시도."""
    while True:
        present = {p.device for p in list_ports.comports() if is_arduino(p)}
        with lock:
            for dev in list(readers):
                if dev not in present:
                    readers[dev].alive = False
                    del readers[dev]
            for dev in present:
                r = readers.get(dev)
                if r is None or (not r.alive and time.time() - r.at > 5):
                    old_err = r.error if r else ""
                    nr = Reader(dev)
                    nr.error = old_err
                    readers[dev] = nr
                    nr.start()
        time.sleep(SCAN_SEC)


def sensors_json():
    with lock:
        devs = sorted(readers)
        return [{"id": dev, "name": f"아두이노 {i + 1}", "t": readers[dev].t, "h": readers[dev].h,
                 "at": int(readers[dev].at * 1000), "error": readers[dev].error,
                 "last_line": readers[dev].last_line, "last_line_at": int(readers[dev].last_line_at * 1000)}
                for i, dev in enumerate(devs)]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")  # 모델을 바꿔도 강력 새로고침 없이 반영
        super().end_headers()

    def do_GET(self):
        if self.path.split("?")[0] == "/api/sensors":
            body = json.dumps({"ok": True, "time": int(time.time() * 1000), "baud": BAUD, "sensors": sensors_json()},
                              ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, *args):
        pass  # 요청마다 출력하지 않음


if __name__ == "__main__":
    threading.Thread(target=scanner, daemon=True).start()
    print(f"관제 화면: http://127.0.0.1:{PORT}/index.html  (아두이노 자동 연결, {BAUD}bps)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
