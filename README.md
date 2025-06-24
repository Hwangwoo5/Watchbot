
# 🤖 WatchBot - 감정 인식 기반 모니터링 시스템

**WatchBot**은 RTSP 카메라 영상 스트리밍을 통해 사람 얼굴을 감지하고,  
딥러닝 기반 감정 인식 모델로 실시간으로 사용자의 감정 상태를 파악하는 AI 시스템입니다.  
감정에 따라 영상/음성 반응을 제공하며, 위급 상황 알림과 TTS 응답 시스템을 지원합니다.

---

## 🚀 주요 기능

- 실시간 얼굴 감지 (OpenCV + HaarCascade)
- TensorRT 최적화된 감정 인식 모델 사용
- 감정 변화에 따라 반응 영상 자동 재생
- TTS/STT 기반 대화형 응답
- RTSP 영상 입력 기반 처리

---

## 🎯 감정 분류 클래스

다음 7가지 감정 클래스를 분류합니다:

| 클래스 ID | 감정 이름     |
|-----------|---------------|
| 0         | angry (화남)  |
| 1         | disgust (혐오)|
| 2         | scared (공포) |
| 3         | happy (행복)  |
| 4         | sad (슬픔)    |
| 5         | surprised (놀람) |
| 6         | neutral (중립)  |

---

## 📦 모델 정보 (models/)

이 프로젝트에는 사전 학습된 감정 인식 모델이 포함되어 있습니다:

| 파일명              | 설명                                       |
|---------------------|--------------------------------------------|
| `emotion_model.h5`  | Keras 기반 학습 모델                        |
| `emotion_model.onnx`| ONNX 포맷 모델 (Jetson에서 TensorRT 변환 가능) |

### ▶️ 모델 변환 흐름

1. 학습 모델 저장 → `emotion_model.h5`
2. ONNX 변환 (`tf2onnx`) → `emotion_model.onnx`
3. Jetson 변환 (선택) → `emotion_model.engine` (`trtexec` 명령 사용)

📁 **모델 경로**: [`/models`](./models)

---

## 📂 프로젝트 구조

```
Watchbot/
├── models/
│   ├── emotion_model.h5
│   └── emotion_model.onnx
├── software/
│   ├── emotion_detection/
│   ├── obj_detection/
│   ├── stt/
│   └── tts/
├── emotion_videos/
├── haarcascade_files/
├── README.md
└── .gitignore
```

---

## 🧠 참고 사항

- 입력 이미지 크기: `64x64`, grayscale (1채널)
- 모델 엔진 생성 시: `trtexec --onnx=emotion_model.onnx --saveEngine=emotion_model.engine --fp16`

---

## 📜 라이선스

MIT License  
본 프로젝트는 연구 및 교육 목적의 자유로운 활용을 허용합니다.
