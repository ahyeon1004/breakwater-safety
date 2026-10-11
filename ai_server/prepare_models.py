"""Teachable Machine 모델 ZIP을 ai_server/models/<이름>/ 에 한 번만 풀어 둔다.

사용법:
    ./run.sh prepare_models.py               # ~/Downloads 에서 낙상.zip·사람수.zip·위험도.zip 을 찾음
    ./run.sh prepare_models.py --src 폴더경로   # ZIP이 다른 곳에 있을 때
    ./run.sh prepare_models.py --force         # 이미 풀려 있어도 다시 풀기
    ./run.sh prepare_models.py --yolo          # 사람 탐지용 YOLO 가중치(yolo11n.pt)도 내려받기 (인터넷 필요, 약 5MB)

서버(app.py)는 ZIP을 직접 열지 않고 여기서 풀어 둔 keras_model.h5 / labels.txt 만 읽는다.
파일명과 라벨은 ZIP 안의 것을 그대로 쓴다.
"""
import argparse
import os
import unicodedata
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = ["낙상", "사람수", "위험도"]
NEED = {"keras_model.h5", "labels.txt"}


def find_zip(src: Path, name: str):
    # macOS 파일명은 한글이 자모 분리(NFD)로 저장될 수 있어 정규화해서 비교
    want = unicodedata.normalize("NFC", name + ".zip")
    for p in src.iterdir():
        if unicodedata.normalize("NFC", p.name) == want:
            return p
    return None


def download_yolo(name="yolo11n.pt"):
    """Ultralytics 사전 학습 가중치를 models/yolo/ 에 한 번 내려받는다 (이미 있으면 건너뜀)."""
    dest = HERE / "models" / "yolo"
    if (dest / name).exists():
        print(f"[건너뜀] YOLO: 이미 준비됨 → {dest / name}")
        return True
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[실패] ultralytics 가 설치되지 않았습니다: ./setup.sh 를 실행하세요")
        return False
    cwd = os.getcwd()
    try:
        os.chdir(dest)  # Ultralytics 는 현재 폴더에 가중치를 내려받음
        YOLO(name)
    except Exception as e:
        print(f"[실패] YOLO 가중치 다운로드: {e}")
        return False
    finally:
        os.chdir(cwd)
    print(f"[완료] YOLO: {dest / name}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() / "Downloads"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--yolo", action="store_true", help="사람 탐지용 YOLO 가중치도 내려받기")
    args = ap.parse_args()
    src = Path(args.src).expanduser()

    ok = True
    for name in NAMES:
        dest = HERE / "models" / name
        if not args.force and all((dest / f).exists() for f in NEED):
            print(f"[건너뜀] {name}: 이미 준비됨 → {dest}")
            continue
        z = find_zip(src, name)
        if not z:
            print(f"[실패] {name}.zip 을 {src} 에서 찾지 못했습니다.")
            ok = False
            continue
        with zipfile.ZipFile(z) as zf:
            members = [m for m in zf.namelist() if Path(m).name in NEED]
            if {Path(m).name for m in members} != NEED:
                print(f"[실패] {z.name} 안에 keras_model.h5 와 labels.txt 가 모두 있어야 합니다. (내용: {zf.namelist()})")
                ok = False
                continue
            dest.mkdir(parents=True, exist_ok=True)
            for m in members:
                (dest / Path(m).name).write_bytes(zf.read(m))
        labels = (dest / "labels.txt").read_text(encoding="utf-8").strip().splitlines()
        print(f"[완료] {name}: {z} → {dest}  라벨 {labels}")
    if args.yolo:
        ok = download_yolo() and ok
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
