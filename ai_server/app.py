"""부우산 방파제 안전 관제 — AI 서버 (Flask).

웹캠 영상을 OpenCV로 받아 Teachable Machine 모델 3개(사람수·위험도·추락)로 분석하고,
기존 관제 화면(index.html)에 결과와 영상을 전달한다.

  http://127.0.0.1:5050/            기존 관제 화면 (프로젝트의 index.html 그대로)
  GET  /api/status                  카메라 연결 상태 + 모델 상태 + 카메라별 AI 분석 결과 (화면이 0.5초마다 조회)
  GET  /api/frame/<CAM-01|CAM-02>   최신 사진 한 장 (관제 화면이 이 방식으로 영상을 받음)
  GET  /video/<CAM-01|CAM-02>       MJPEG 영상 (브라우저에서 직접 볼 때용)
  GET  /api/persons                 사람 탐지(YOLO) 박스·PERSON 번호만 빠르게 조회 (화면이 0.15초마다 조회)
  POST /api/speak {"text": ...}     Mac 음성 합성(say)으로 방송 문구 재생 (외부 스피커 장비 제어 아님)
  POST /api/log   {...}             화면의 상황 기록을 logs/events.jsonl 에 저장 (서버를 껐다 켜도 남음)
  GET  /api/log?limit=200           저장된 상황 기록 조회

실행:  cd ai_server && ./run.sh
"""
import json
import logging
import os
import sys

sys.dont_write_bytecode = True  # .pyc 캐시 파일을 만들지 않음 (프로젝트 폴더 파일 변경 → Live Server 새로고침 방지)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # TensorFlow 의 불필요한 콘솔 출력 줄이기
import shutil
import subprocess
import threading
import time
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_from_directory

from camera import CameraManager, ProcessLock
from models import Smoother, TMClassifier
from people_counter import TMPeopleCounter, label_to_count
from lego_detector import BackgroundLegoDetector
from paths import data_dir
from person_detector import PersonDetector


def make_detector(pcfg, cam_id):
    """개별 탐지기 선택 (config.json person_detector.mode).
    - "background": 빈 배경과 비교해 레고 피규어를 찾음 (기본, 레고 시연용)
    - "yolo": 사전 학습 YOLO 의 person 클래스 (실제 사람용. 레고로 추가 학습한 가중치를 weights 에 넣어도 됨)"""
    if pcfg.get("mode", "background") == "yolo":
        return PersonDetector(pcfg.get("yolo", pcfg), HERE)
    return BackgroundLegoDetector(pcfg.get("lego", pcfg.get("background", {})), LOGS, cam_id)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# 다른 설정 파일로 실행하려면: AI_CONFIG=test_config.json ./run.sh
CFG_PATH = Path(os.environ.get("AI_CONFIG", HERE / "config.json"))
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
LOGS = data_dir(CFG)  # 기록·배경 사진 저장 위치 (프로젝트 밖 — Live Server 새로고침 방지, paths.py 참고)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ai")
for noisy in ("werkzeug",):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def make_people_counter(clf):
    """사람 수 감지기. YOLO 등으로 바꿀 때 여기만 바꾸면 된다 (people_counter.py 참고)."""
    return TMPeopleCounter(clf)


class Analyzer(threading.Thread):
    """물리 카메라 하나의 최신 프레임을 일정 간격으로 세 모델에 넣고, 안정화된 결과를 보관한다."""

    def __init__(self, capture, models, people, acfg, mcfg):
        super().__init__(daemon=True)
        self.cap = capture
        self.models = models
        self.people = people
        self.interval = acfg.get("interval_sec", 0.25)
        mk = lambda key: Smoother(acfg.get("window", 8), acfg.get("min_agreement", 0.6), mcfg[key].get("min_confidence", 0.7))
        self.smooth = {k: mk(k) for k in ("people", "danger", "fall")}
        self.positive = {k: mcfg[k].get("positive_label") for k in ("danger", "fall")}
        self.result = {}
        self.last_seq = -1
        self.last_state = {}
        self.error = ""

    def run(self):
        while True:
            t0 = time.time()
            frame, seq, ts = self.cap.latest()
            if frame is None or seq == self.last_seq or time.time() - ts > 3:
                if time.time() - ts > 3:
                    for s in self.smooth.values():
                        s.clear()  # 영상이 끊기면 이전 판단을 지금 결과처럼 쓰지 않음
                time.sleep(0.1)
                continue
            self.last_seq = seq
            # 검은 화면(카메라 가림·신호 없음)이면 위험도·추락·인원을 판정하지 않음
            if float(frame.mean()) < 12 and int(frame.max()) < 40:
                for s in self.smooth.values():
                    s.clear()
                self.result = {"ts": time.time(), "frame_ts": ts, "error": "", "dark": True,
                               **{k: {"state": "no_signal", "label": None, "confidence": None} for k in ("people", "danger", "fall")}}
                time.sleep(self.interval)
                continue
            raw = {}
            try:
                if self.people:
                    r = self.people.count(frame)
                    self.smooth["people"].push(r["label"], r["confidence"])
                    raw["people"] = {"label": r["label"], "confidence": round(r["confidence"], 3)}
                for key in ("danger", "fall"):
                    m = self.models.get(key)
                    if m and m.ok:
                        p = m.predict(frame)
                        i = int(p.argmax())
                        self.smooth[key].push(m.labels[i], p[i])
                        raw[key] = {"label": m.labels[i], "confidence": round(float(p[i]), 3)}
                self.error = ""
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
                log.exception("%s 분석 오류", self.cap.id)
            self.publish(raw, ts)
            time.sleep(max(0.0, self.interval - (time.time() - t0)))

    def publish(self, raw, frame_ts):
        out = {"ts": time.time(), "frame_ts": frame_ts, "error": self.error}
        for key in ("people", "danger", "fall"):
            m = self.models.get(key)
            if not (m and m.ok):
                out[key] = {"state": "error", "error": m.error if m else "모델 없음"}
                continue
            r = self.smooth[key].result()
            r["raw"] = raw.get(key)
            if key == "people":
                r["count"] = label_to_count(r["label"]) if r["label"] else None
            else:
                r["positive"] = (r["label"] == self.positive[key]) if r["label"] else None
            out[key] = r
            # 안정된 판단이 바뀔 때만 서버 기록에 남김 (매 프레임 기록하지 않음)
            sig = (r["state"], r.get("label"))
            if self.last_state.get(key) != sig:
                self.last_state[key] = sig
                append_jsonl("ai_events.jsonl", {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "camera": self.cap.id,
                                                 "model": key, "state": r["state"], "label": r.get("label"),
                                                 "confidence": r.get("confidence")})
        self.result = out


class PersonTracker(threading.Thread):
    """물리 카메라 하나의 개별 탐지·추적 (레고 배경 비교 또는 YOLO). Teachable Machine 분석과 별도 스레드.
    초당 3~5회로 제한하고, 화면은 그 사이를 부드럽게 이어 그린다."""

    def __init__(self, capture, pcfg, analyzer):
        super().__init__(daemon=True)
        self.cap = capture
        self.det = make_detector(pcfg, capture.id)
        self.mode = getattr(self.det, "kind", "yolo")
        self.analyzer = analyzer  # TM 사람수 분류 결과를 비교 기록하려고 참조
        self.interval = float(pcfg.get("interval_sec", 0.25))
        self.result = {"state": "error" if not self.det.ok else "pending", "error": self.det.error, "count": None, "boxes": []}
        self.last_seq = -1
        self.logged = None
        self.pending = (None, 0.0)

    def run(self):
        if not self.det.ok:
            return
        while True:
            t0 = time.time()
            frame, seq, ts = self.cap.latest()
            if frame is None or time.time() - ts > 3:
                if self.result.get("state") == "ok":
                    self.det.reset()  # 영상이 끊기면 박스·번호를 지우고, 다시 들어오면 새로 추적
                self.result = {"state": "pending", "error": "", "count": None, "boxes": [], "ts": time.time()}
                time.sleep(0.2)
                continue
            if seq == self.last_seq:
                time.sleep(0.01)
                continue
            self.last_seq = seq
            try:
                if self.mode == "background":
                    state, people, err = self.det.update(frame)
                else:
                    state, people, err = "ok", self.det.update(frame), ""
                h, w = frame.shape[:2]
                self.result = {"state": state, "mode": self.mode, "error": err, "count": len(people) if state == "ok" else None,
                               "boxes": people if state == "ok" else [], "frame_size": [w, h], "frame_ts": ts, "ts": time.time()}
                if state == "ok":
                    self.log_change(len(people))
            except Exception as e:
                self.result = {"state": "error", "mode": self.mode, "error": f"{type(e).__name__}: {e}", "count": None, "boxes": [], "ts": time.time()}
                log.exception("%s 사람 탐지 오류", self.cap.id)
                time.sleep(1)
            time.sleep(max(0.0, self.interval - (time.time() - t0)))

    def log_change(self, n):
        """탐지 인원이 1초 이상 같은 값으로 바뀌었을 때만 기록 (TM 분류 결과와 비교용으로 함께 남김)."""
        now = time.time()
        if n != self.pending[0]:
            self.pending = (n, now)
            return
        if n != self.logged and now - self.pending[1] >= 1.0:
            self.logged = n
            tm = (self.analyzer.result or {}).get("people", {}) if self.analyzer else {}
            append_jsonl("ai_events.jsonl", {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "camera": self.cap.id,
                                             "model": f"detector:{self.mode}", "count": n,
                                             "tm_people": tm.get("label") or tm.get("state"),
                                             "match": tm.get("count") == n if tm.get("state") == "ok" else None})


_file_lock = threading.Lock()


def append_jsonl(name, obj):
    LOGS.mkdir(exist_ok=True)
    with _file_lock, open(LOGS / name, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------------- 시작 ----------------
lock = ProcessLock(LOGS / ".camera.lock")
if not lock.acquire():
    sys.exit("다른 AI 서버가 이미 실행 중이라 카메라를 열지 않습니다. 기존 서버를 먼저 종료하세요. (같은 카메라 동시 사용 방지)")

MODELS = {key: TMClassifier(key, HERE / m["path"]) for key, m in CFG["models"].items()}
try:
    PEOPLE = make_people_counter(MODELS["people"]) if MODELS["people"].ok else None
except Exception as e:
    MODELS["people"].error = str(e)
    PEOPLE = None

CAMS = CameraManager(CFG["cameras"])
CAMS.start()
ANALYZERS = {}
TRACKERS = {}
PCFG = CFG.get("person_detector", {"enabled": False})
for cid, cap in CAMS.captures.items():
    a = Analyzer(cap, MODELS, PEOPLE, CFG.get("analysis", {}), CFG["models"])
    a.start()
    ANALYZERS[cid] = a
    if PCFG.get("enabled", True):
        t = PersonTracker(cap, PCFG, a)
        t.start()
        TRACKERS[cid] = t


def persons_of(view):
    """화면(view)에 표시할 사람 탐지 결과. 복제 화면은 원본 카메라 결과를 그대로 쓴다."""
    if not PCFG.get("enabled", True):
        return {"state": "disabled", "error": "config.json 에서 person_detector 가 꺼져 있음", "count": None, "boxes": []}
    t = TRACKERS.get(view["source"])
    r = t.result if t else {"state": "error", "error": "탐지기 없음", "count": None, "boxes": []}
    if view["status"] != "live" or (r.get("state") in ("ok", "no_background") and time.time() - r.get("ts", 0) > 2):
        return {"state": "pending", "error": "", "count": None, "boxes": []}  # 오래된 박스는 내보내지 않음
    return r

STREAM_FPS = float(CFG.get("stream_fps", 12))  # 화면 전송 프레임 수 (분석 주기와 별개)
app = Flask(__name__)
speak_state = {"proc": None, "text": "", "at": 0}


@app.after_request
def cors(resp):
    # GitHub Pages 등 다른 주소에서 연 화면도 이 서버에 접속할 수 있게 허용 (로컬 서버 전용)
    if request.path.startswith(("/api/", "/video/")):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.get("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.get("/web_models/<path:sub>")
def web_models(sub):
    """브라우저 분석용 티처블 머신 모델 (TensorFlow.js 형식)."""
    return send_from_directory(ROOT / "web_models", sub)


@app.get("/obs.json")
def obs():
    return send_from_directory(ROOT, "obs.json")


@app.get("/api/status")
def status():
    views = CAMS.info()
    for v in views:
        a = ANALYZERS.get(v["source"])
        r = a.result if a else {}
        # 영상이 3초 넘게 끊겼으면 분석 결과를 내보내지 않음 (오래된 결과를 현재처럼 보이지 않게)
        v["analysis"] = r if r and v["status"] == "live" and time.time() - r.get("ts", 0) < 3 else None
        v["persons"] = persons_of(v)
    return jsonify({
        "ok": True, "time": time.time(),
        "cameras": views,
        "models": {k: m.info() for k, m in MODELS.items()},
        "people_counter": PEOPLE.kind if PEOPLE else None,
        "overlap": CFG.get("overlap", {"mode": "unknown"}),  # 두 카메라 촬영 영역 겹침 설정 (중복 집계 판단용)
        "person_detector": {"enabled": PCFG.get("enabled", True), "mode": PCFG.get("mode", "background"),
                            "error": next((t.det.error for t in TRACKERS.values() if getattr(t.det, "error", "")), "")},
        "speech": {"available": bool(shutil.which("say")), "speaking": bool(speak_state["proc"] and speak_state["proc"].poll() is None)},
    })


@app.post("/api/reconnect")
def reconnect():
    """화면의 '재연결' 버튼: 재시도를 멈춘 카메라를 다시 연결 시도."""
    for cap in CAMS.captures.values():
        if cap.status != "live":
            cap.reconnect()
    return jsonify({"ok": True})


@app.post("/api/background/<cid>")
def save_background(cid):
    """화면의 '빈 배경 저장' 버튼: 레고가 없는 지금 화면을 배경으로 저장 (복제 화면이면 원본 카메라에 저장)."""
    if cid not in CAMS.views:
        abort(404)
    cap, _ = CAMS.views[cid]
    t = TRACKERS.get(cap.id)
    if not t or getattr(t.det, "kind", "") != "background":
        return jsonify({"ok": False, "error": "레고 배경 비교 모드가 아닙니다 (config.json person_detector.mode)"}), 400
    frame, _, ts = cap.latest()
    if frame is None or time.time() - ts > 3:
        return jsonify({"ok": False, "error": "카메라 영상이 없습니다"}), 409
    t.det.set_background(frame)
    return jsonify({"ok": True, "camera": cap.id})


@app.get("/api/persons")
def persons():
    return jsonify({"time": time.time(), "cameras": [{"id": v["id"], "persons": persons_of(v)} for v in CAMS.info()]})


@app.get("/api/frame/<cid>")
def frame(cid):
    """CCTV 최신 사진 한 장 (화면이 초당 약 10번 가져감). 연결을 붙잡지 않아 브라우저 동시 연결 한도에 걸리지 않음."""
    if cid not in CAMS.views:
        abort(404)
    cap, _ = CAMS.views[cid]
    jpg, seq = cap.latest_jpeg()
    if jpg is None:
        return ("", 204)
    return Response(jpg, mimetype="image/jpeg", headers={"Cache-Control": "no-store", "X-Frame-Seq": str(seq)})


@app.get("/video/<cid>")
def video(cid):
    if cid not in CAMS.views:
        abort(404)
    cap, _ = CAMS.views[cid]

    def gen():
        last = -1
        while True:
            jpg, seq = cap.latest_jpeg()
            if jpg is None or seq == last:
                time.sleep(0.03)
                continue
            last = seq
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
            time.sleep(1 / STREAM_FPS)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-store"})


def korean_voice():
    """설치된 한국어 음성 (Yuna 우선, 없으면 첫 번째 ko_KR 음성, 그것도 없으면 기본 음성)."""
    lines = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout.splitlines()
    names = [l.split("(")[0].split("  ")[0].strip() for l in lines if "ko_KR" in l]
    pick = "Yuna" if "Yuna" in names else names[0] if names else None
    return ["-v", pick] if pick else []


@app.post("/api/speak")
def speak():
    text = str((request.get_json(silent=True) or {}).get("text", "")).strip()[:500]
    if not text:
        return jsonify({"ok": False, "error": "문구 없음"}), 400
    if not shutil.which("say"):
        return jsonify({"ok": False, "error": "이 컴퓨터에 say 명령이 없습니다"}), 501
    p = speak_state["proc"]
    if p and p.poll() is None:
        return jsonify({"ok": False, "error": "이미 방송 음성을 재생 중입니다"}), 409
    # 같은 문구를 20초 안에 다시 요청하면 거절 (발표 중 반복 재생 방지)
    if text == speak_state["text"] and time.time() - speak_state["at"] < 20:
        return jsonify({"ok": False, "error": "같은 방송을 방금 재생했습니다"}), 429
    voice = korean_voice()
    speak_state.update(proc=subprocess.Popen(["say", *voice, text]), text=text, at=time.time())
    return jsonify({"ok": True, "voice": voice[1] if voice else "기본"})


@app.post("/api/log")
def save_log():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False}), 400
    append_jsonl("events.jsonl", data)
    return jsonify({"ok": True})


@app.get("/api/log")
def read_log():
    limit = min(int(request.args.get("limit", 200)), 2000)
    path = LOGS / "events.jsonl"
    if not path.exists():
        return jsonify([])
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    return jsonify([json.loads(l) for l in lines if l.strip()])


if __name__ == "__main__":
    s = CFG["server"]
    for v in CAMS.info():
        log.info("화면 %s: %s%s", v["id"], v["status"], f" ({v['duplicate_of']} 복제 영상)" if v["duplicate_of"] else "")
    log.info("기록·빈 배경 저장 위치: %s", LOGS)
    log.info("관제 화면: http://%s:%d/", s["host"], s["port"])
    app.run(host=s["host"], port=s["port"], threaded=True, use_reloader=False)
