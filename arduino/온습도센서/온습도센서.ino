// 부우산 관제 화면용 온습도 센서 (DHT11 또는 DHT22)
//
// 준비: Arduino IDE → 라이브러리 매니저에서 "DHT sensor library"(Adafruit) 설치
//       (함께 설치하라고 나오는 "Adafruit Unified Sensor"도 설치)
// 배선: 센서 VCC → 5V,  GND → GND,  DATA(OUT) → 디지털 2번 핀
//       (3핀 모듈이 아닌 4핀 센서 단품이면 DATA 와 VCC 사이에 10kΩ 저항)
// 출력: 2초마다 "T=23.4,H=56.0" 한 줄 (9600bps) → 관제 화면의 '🔌 아두이노 연결'로 읽음
// 주의: 관제 화면에 연결할 때는 Arduino IDE 의 시리얼 모니터를 닫아야 합니다 (포트를 한 프로그램만 쓸 수 있음)

#include <DHT.h>

#define DHTPIN 2        // DATA 핀 번호
#define DHTTYPE DHT11   // DHT22 를 쓰면 DHT22 로 바꾸기

DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(9600);   // 관제 화면 설정의 보드레이트와 같아야 함
  dht.begin();
}

void loop() {
  delay(2000);          // DHT 센서는 2초에 한 번 정도 읽는 것이 안정적
  float h = dht.readHumidity();
  float t = dht.readTemperature();   // 섭씨
  if (isnan(h) || isnan(t)) {
    Serial.println("ERR sensor read failed");   // 배선 확인 (화면에는 '센서 응답 없음')
    return;
  }
  Serial.print("T=");
  Serial.print(t, 1);
  Serial.print(",H=");
  Serial.println(h, 1);
}
