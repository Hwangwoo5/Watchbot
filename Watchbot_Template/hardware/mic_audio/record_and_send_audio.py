import pyaudio, wave, requests

FILENAME = "recorded.wav"
CHUNK, FORMAT, CHANNELS, RATE, SECONDS = 1024, pyaudio.paInt16, 1, 16000, 5

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
frames = [stream.read(CHUNK) for _ in range(int(RATE / CHUNK * SECONDS))]
stream.stop_stream(); stream.close(); p.terminate()

wf = wave.open(FILENAME, 'wb')
wf.setnchannels(CHANNELS); wf.setsampwidth(p.get_sample_size(FORMAT)); wf.setframerate(RATE)
wf.writeframes(b''.join(frames)); wf.close()

url = 'http://<서버주소>:포트/upload_audio'
with open(FILENAME, 'rb') as f:
    requests.post(url, files={'file': f})
