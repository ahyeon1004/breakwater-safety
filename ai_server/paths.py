"""AI 서버가 실행 중에 쓰는 파일(상황 기록, 빈 배경 사진, 카메라 확인 사진)의 저장 위치.

프로젝트 폴더 안에 쓰면 VS Code Live Server 가 파일 변경을 감지해 웹페이지를 계속 새로고침하므로,
기본값은 프로젝트 밖인 ~/Library/Application Support/BuusanAI 이다.
다른 곳에 두려면 config.json 의 "data_dir" 또는 환경변수 AI_DATA_DIR 로 바꾼다.
"""
import os
from pathlib import Path

DEFAULT = "~/Library/Application Support/BuusanAI"


def data_dir(cfg=None):
    raw = os.environ.get("AI_DATA_DIR") or (cfg or {}).get("data_dir") or DEFAULT
    p = Path(raw).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p
