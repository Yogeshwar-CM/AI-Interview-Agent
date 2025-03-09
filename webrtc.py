import logging
import os
import markdown
from datetime import datetime
from multiprocessing import Queue  
from flask import Flask, render_template, request, jsonify
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaPlayer
import cv2
import dlib
import numpy as np

app = Flask(__name__)
logger = logging.getLogger("webrtc-agent")
log_queue = Queue()

shape_predictor_path = "/Users/yogeshwarcm/Desktop/HydHackathon/Project/agent/shape_predictor_68_face_landmarks.dat"
alert_sound_path = "/Users/yogeshwarcm/Desktop/HydHackathon/Project/agent/alert.wav"

if not os.path.exists(shape_predictor_path):
    raise FileNotFoundError(f"Shape predictor file not found: {shape_predictor_path}")

if not os.path.exists(alert_sound_path):
    raise FileNotFoundError(f"Alert sound file not found: {alert_sound_path}")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def is_suspicious(eye_aspect_ratio, threshold=0.2):
    return eye_aspect_ratio < threshold

def calculate_face_angle(landmarks):
    left_eye_center = np.mean(landmarks[36:42], axis=0)
    right_eye_center = np.mean(landmarks[42:48], axis=0)
    nose_center = landmarks[30]

    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return angle

def is_face_within_view(face, frame_width, frame_height):
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    return x > 0 and y > 0 and (x + w) < frame_width and (y + h) < frame_height

def load_company_info(data_folder):
    company_info = ""
    for filename in os.listdir(data_folder):
        if filename.endswith(".md"):
            filepath = os.path.join(data_folder, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                md_content = file.read()
                company_info += markdown.markdown(md_content)
    return company_info

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/offer', methods=['POST'])
async def offer():
    params = request.json
    offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on('datachannel')
    def on_datachannel(channel):
        @channel.on('message')
        def on_message(message):
            print(f"Received message: {message}")

    @pc.on('iceconnectionstatechange')
    def on_iceconnectionstatechange():
        if pc.iceConnectionState == 'failed':
            pc.close()
            pcs.discard(pc)

    @pc.on('track')
    def on_track(track):
        if track.kind == 'video':
            local_video = VideoStreamTrack()
            pc.addTrack(local_video)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return jsonify({
        'sdp': pc.localDescription.sdp,
        'type': pc.localDescription.type
    })

class VideoStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)

    async def recv(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        frame_height, frame_width = frame.shape[:2]
        face_detected = False

        for face in faces:
            face_detected = True
            landmarks = predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            ear = (left_ear + right_ear) / 2.0

            if is_suspicious(ear):
                cv2.putText(frame, "Suspicious activity detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
            cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

            angle = calculate_face_angle(landmarks)
            cv2.putText(frame, f"Face angle: {angle:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if not is_face_within_view(face, frame_width, frame_height):
                cv2.putText(frame, "Please stay within the camera view!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                playsound(alert_sound_path)

        if not face_detected:
            cv2.putText(frame, "No face detected! Please stay in front of the camera.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            playsound(alert_sound_path)

        return frame

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)