# 🛡️ WatchBot: 감정 인식 기반 실시간 보호자 알림 시스템

## 🔍 프로젝트 개요

**WatchBot**은 실시간 감정 인식, 음성 인식(STT), TTS 응답, 객체 감지를 통해  
사용자의 상태를 분석하고 상황에 따라 음성 안내 및 보호자에게 자동 알림을 제공하는  
지능형 AI 모니터링 시스템입니다.

## 🎯 주요 기능 구성

| 모듈 | 설명 |
|------|------|
| `emotion_detection/` | RTSP 영상 기반 얼굴 감정 인식 (YOLO/TensorRT 기반) |
| `stt/` | 키워드 기반 웨이크워드 STT 감지 (`whisp.py`) |
| `tts/` | DevDive TTS API + LLM 응답 생성 및 음성 재생 (`flow2.py`) |
| `obj_detection/` | 객체 인식 ONNX 실행 스크립트 |
| `models/` | 변환된 Keras 및 ONNX 모델 저장용 |
| `haarcascade_files/` | 얼굴 인식 XML |
| `emotion_videos/` | 감정별 반응 영상 저장 |

## 🧠 모델 변환 방법

```bash
# 모델 구조 정의 → .h5 저장
python export_model.py

# 결과:
# - emotion_model.h5
# - saved_model/
# - emotion_model.onnx
```

TensorRT 변환은 Jetson 환경에서 아래처럼 진행합니다:
```bash
trtexec --onnx=emotion_model.onnx --saveEngine=emotion_model.engine --fp16
```

## ▶️ 실행 방법 예시

```bash
# 키워드 감지 및 대화 흐름 실행
python software/stt/whisp.py
python software/tts/flow2.py

# 감정 인식 실행
python software/emotion_detection/detect_faces_trt.py
```

## 📁 폴더 구조 요약

```
Watchbot/
├── models/                  # 모델 저장 (.h5, .onnx)
├── haarcascade_files/       # 얼굴 검출 XML
├── emotion_videos/          # 감정별 영상
├── software/
│   ├── emotion_detection/
│   ├── stt/
│   ├── tts/
│   └── obj_detection/
├── cnn.py
├── export_model.py
└── README.md
```

## 🙋 기여자

- **오황우** – 프로젝트 구조 설계, 하드웨어 통합, 음성 인터페이스 구현
