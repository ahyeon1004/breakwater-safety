"""기상청 API허브 AWS 매분자료에서 부산 관측소 값을 받아 obs.json으로 저장한다."""
import json, os, sys, time, urllib.request
from datetime import datetime, timedelta, timezone

STATIONS = {"937": "해운대", "942": "부산남구", "904": "사하", "921": "가덕도", "910": "영도", "159": "부산"}
KST = timezone(timedelta(hours=9))
key = os.environ["KMA_KEY"]

def get(url):
    """기상청 서버가 해외(GitHub)에서 접속이 자주 끊겨 몇 번 다시 시도한다."""
    for attempt in range(4):
        try:
            return urllib.request.urlopen(url, timeout=40).read()
        except OSError:
            if attempt == 3:
                raise
            time.sleep(5)

def val(text):
    v = float(text)
    return None if v <= -50 else v  # -50 이하는 결측

BUOYS = {"22459": "오륙도", "22511": "기장", "22460": "다대포", "22513": "이수도"}

def waves():
    """해상 관측자료에서 파고부이의 최근 1시간 중 가장 늦은 유의파고를 고른다. 실패해도 풍속 자료는 저장한다."""
    tm = datetime.now(KST).strftime("%Y%m%d%H%M")
    url = f"https://apihub.kma.go.kr/api/typ01/url/sea_obs.php?tm={tm}&stn=0&help=0&authKey={key}"
    out = {}
    try:
        raw = get(url).decode("euc-kr", "ignore")
    except OSError as e:
        print("파고 요청 실패:", type(e).__name__)
        return out
    for line in raw.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) > 7 and f[2] in BUOYS and val(f[6]) is not None:
            if f[2] not in out or f[1] > out[f[2]]["time"]:
                out[f[2]] = {"name": BUOYS[f[2]], "wave": val(f[6]), "time": f[1]}
    return out

# 10분 단위 시각만 요청한다 (그 외 시각을 전체 관측소로 요청하면 응답이 오지 않음)
now = datetime.now(KST) - timedelta(minutes=3)
base = now.replace(minute=now.minute // 10 * 10, second=0, microsecond=0)
for step in range(3):  # 최신 자료가 아직 없으면 10분씩 앞 시각으로 재시도
    tm = (base - timedelta(minutes=10 * step)).strftime("%Y%m%d%H%M")
    url = f"https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min?tm2={tm}&stn=0&disp=1&help=0&authKey={key}"
    try:
        raw = get(url).decode("euc-kr", "ignore")
    except OSError as e:
        print(tm, "요청 실패:", type(e).__name__)
        continue
    out = {}
    for line in raw.splitlines():
        f = line.split(",")
        if len(f) > 12 and f[1] in STATIONS:
            out[f[1]] = {"name": STATIONS[f[1]], "wind": val(f[7]), "gust": val(f[5]), "rain": val(f[11]), "temp": val(f[8]), "hum": val(f[14])}
    if out:
        json.dump({"time": tm, "stations": out, "waves": waves()}, open("obs.json", "w"), ensure_ascii=False)
        print(tm, out)
        sys.exit(0)
sys.exit("관측값을 받지 못했습니다")
