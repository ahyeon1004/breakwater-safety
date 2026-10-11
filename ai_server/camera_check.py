"""macOS 카메라 장치 번호 확인 도구.

    ./run.sh camera_check.py

macOS가 인식한 카메라 이름을 보여 주고, OpenCV 장치 번호 0~4를 하나씩 열어
실제 영상이 들어오는지·해상도는 얼마인지 출력한다. 각 장치의 사진을 ~/Library/Application Support/BuusanAI/camera_<번호>.jpg 로 저장하므로
사진을 열어 보고 피파인 K420이 몇 번인지 확인한 뒤 config.json 의 "device" 에 그 번호를 넣으면 된다.
(AI 서버가 실행 중이면 카메라가 잠겨 있으므로 서버를 끄고 실행하세요.)
"""
from pathlib import Path

import cv2

from camera import MAX_INDEX, mac_camera_names, try_open

from paths import data_dir

out = data_dir()

names = mac_camera_names()
print("macOS가 인식한 카메라:", names or "없음")
if not any("k420" in n.lower() for n in names):
    print("  ⚠ 'K420' 이름의 카메라가 목록에 없습니다. USB 연결을 확인하세요.")
print("(OpenCV 장치 번호와 위 목록 순서는 대체로 같지만 다를 수 있으니 저장된 사진으로 확인하세요)\n")

for i in range(MAX_INDEX):
    cap, info = try_open(i, 1280, 720, timeout=3)
    if not cap:
        print(f"장치 {i}: 실패 — {info}")
        continue
    ok, frame = cap.read()
    cap.release()
    path = out / f"camera_{i}.jpg"
    if ok:
        cv2.imwrite(str(path), frame)
    print(f"장치 {i}: 성공 — {info[0]}x{info[1]} · 이름 추정: {names[i] if i < len(names) else '알 수 없음'} · 사진: {path if ok else '저장 실패'}")
