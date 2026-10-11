#!/bin/bash
# 처음 한 번: Python 3.11 가상환경을 프로젝트 밖에 만들고 라이브러리 설치 + 모델 준비
# (프로젝트 안에 두면 VS Code Live Server 가 수만 개 파일 변화를 감지해 웹페이지를 계속 새로고침함)
set -e
cd "$(dirname "$0")"
VENV="$HOME/Library/Application Support/BuusanAI/venv"
uv venv --python 3.11 "$VENV"
uv pip install --python "$VENV/bin/python" -r requirements.txt
"$VENV/bin/python" prepare_models.py --yolo
echo "준비 완료. 실행: ./run.sh"
