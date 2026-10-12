#!/bin/bash
# 더블클릭하면 관제 화면을 엽니다.
# - 브라우저가 직접 AI 분석 (AI 서버 불필요)
# - USB 로 꽂힌 아두이노 온습도 센서(여러 대)를 자동으로 읽음 (버튼 필요 없음)
# 이 창(터미널)을 닫으면 화면 서버도 꺼집니다.
cd "$(dirname "$0")"
PORT=8000
PY="$HOME/Library/Application Support/BuusanAI/venv/bin/python"
if lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
  echo "화면 서버가 이미 켜져 있습니다."
elif [ -x "$PY" ] && "$PY" -c "import serial" >/dev/null 2>&1; then
  "$PY" ai_server/page_server.py &
else
  echo "⚠ Python 환경이 없어 아두이노 자동 연결 없이 화면만 엽니다. (ai_server/setup.sh 실행 필요)"
  python3 -m http.server $PORT --bind 127.0.0.1 >/dev/null 2>&1 &
fi
URL="http://127.0.0.1:$PORT/index.html"
# 서버가 실제로 응답할 때까지 기다린 뒤 브라우저를 엶 (최대 20초)
for i in $(seq 1 40); do
  curl -s -o /dev/null "$URL" && break
  sleep 0.5
done
if ! curl -s -o /dev/null "$URL"; then
  echo "⚠ 화면 서버가 켜지지 않았습니다. 이 창의 오류 메시지를 확인하세요."
fi
if [ -d "/Applications/Google Chrome.app" ]; then open -a "Google Chrome" "$URL"; else open "$URL"; fi
echo "관제 화면: $URL"
echo "끝내려면 이 창을 닫으세요."
wait
