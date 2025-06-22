import cv2
import numpy as np
import time
import pyttsx3
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # CUDA context 초기화
import imutils
import subprocess

# 모델 및 리소스 경로
ENGINE_PATH = "/home/xavier/watchbot_clean/Emotion-recognition-master/models/emotion_model.engine"
HAAR_PATH = "/home/xavier/watchbot_clean/Emotion-recognition-master/haarcascade_files/haarcascade_frontalface_default.xml"
RTSP_URL = "rtsp://192.168.0.196:8554/cam"

EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# 감정별 영상 매핑
EMOTION_VIDEO_MAP = {
    "angry": "/home/xavier/watchbot_clean/emotion_videos/angry.mp4",
    "disgust": "/home/xavier/watchbot_clean/emotion_videos/disgust.mp4",
    "scared": "/home/xavier/watchbot_clean/emotion_videos/scared.mp4",
    "happy": "/home/xavier/watchbot_clean/emotion_videos/happy.mp4",
    "sad": "/home/xavier/watchbot_clean/emotion_videos/sad.mp4",
    "surprised": "/home/xavier/watchbot_clean/emotion_videos/surprised.mp4",
    "neutral": "/home/xavier/watchbot_clean/emotion_videos/neutral.mp4"
}

video_process = None

def play_emotion_video(label):
    global video_process
    if video_process is not None:
        video_process.terminate()
        video_process = None

    video_path = EMOTION_VIDEO_MAP.get(label)
    if video_path:
        video_process = subprocess.Popen([
            "ffplay", "-loglevel", "quiet", "-autoexit", "-fs", video_path
        ])

# TensorRT 로드 함수
def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# 추론 함수
def infer(engine, input_data):
    context = engine.create_execution_context()
    input_shape = (1, 1, 64, 64)
    input_data = input_data.astype(np.float32)
    h_input = np.ascontiguousarray(input_data.reshape(input_shape))
    h_output = np.empty((1, 7), dtype=np.float32)

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    bindings = [int(d_input), int(d_output)]

    context.set_binding_shape(0, input_shape)
    cuda.memcpy_htod(d_input, h_input)
    context.execute_v2(bindings)
    cuda.memcpy_dtoh(h_output, d_output)

    return h_output[0]

# 초기화
engine = load_engine(ENGINE_PATH)
face_detection = cv2.CascadeClassifier(HAAR_PATH)
tts_engine = pyttsx3.init()
last_label = None

# 스트림 연결
camera = cv2.VideoCapture(RTSP_URL)
cv2.namedWindow("your_face")
print("📦 TensorRT 엔진 로딩 완료")

while True:
    ret, frame = camera.read()
    if not ret:
        print("❌ 프레임 수신 실패")
        continue

    frame = imutils.resize(frame, width=480)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    preds = np.zeros(len(EMOTIONS))

    if len(faces) > 0:
        (fX, fY, fW, fH) = sorted(faces, reverse=True, key=lambda x: x[2] * x[3])[0]

        roi = gray[fY:fY+fH, fX:fX+fW]
        try:
            roi = cv2.resize(roi, (64, 64))
        except:
            continue

        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=0)  # (1, 64, 64)
        roi = np.expand_dims(roi, axis=0)  # (1, 1, 64, 64)

        preds = infer(engine, roi)
        label = EMOTIONS[np.argmax(preds)]

        if label != last_label:
            play_emotion_video(label)
            last_label = label

        cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX+fW, fY+fH), (0, 0, 255), 2)

    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
        text = f"{emotion}: {prob * 100:.2f}%"
        w = int(prob * 300)
        cv2.rectangle(canvas, (7, i*35 + 5), (w, i*35 + 35), (0, 0, 255), -1)
        cv2.putText(canvas, text, (10, i*35 + 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 2)

    cv2.imshow("your_face", frameClone)
    cv2.imshow("Probabilities", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
