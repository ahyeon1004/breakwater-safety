#!/bin/bash
# AI 서버 실행:            ./run.sh
# 카메라 번호 확인:        ./run.sh camera_check.py
# 다른 파이썬 파일 실행:   ./run.sh 파일이름.py
cd "$(dirname "$0")"
VENV="$HOME/Library/Application Support/BuusanAI/venv"
if [ ! -x "$VENV/bin/python" ]; then
  echo "가상환경이 없습니다. 먼저 ./setup.sh 를 실행하세요."; exit 1
fi
exec "$VENV/bin/python" "${@:-app.py}"
