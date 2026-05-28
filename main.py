import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import dlib
import numpy as np
import os
import requests
import bz2
import base64
from datetime import datetime
from imutils import face_utils

from src.processor import calculate_ear, calculate_mar, FatigueMonitor

st.set_page_config(page_title="Vigía v1.0 - Driver Monitoring", layout="wide")


def download_model():
    model_path = "models/shape_predictor_68_face_landmarks.dat"
    if os.path.exists(model_path):
        return model_path

    if not os.path.exists("models"):
        os.makedirs("models")

    placeholder = st.empty()
    placeholder.info("Descargando modelo de predicción facial (esto solo ocurrirá una vez)...")

    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    response = requests.get(url, stream=True)

    compressed_file = "models/shape_predictor_68_face_landmarks.dat.bz2"
    with open(compressed_file, "wb") as f:
        f.write(response.content)

    placeholder.info("Descomprimiendo modelo... por favor espera.")
    with bz2.BZ2File(compressed_file) as fr, open(model_path, "wb") as fw:
        fw.write(fr.read())

    os.remove(compressed_file)
    placeholder.success("Modelo listo y cargado.")
    return model_path


def get_audio_base64(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None


predictor_path = download_model()


class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        self.monitor = FatigueMonitor()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.EAR_THRESHOLD = 0.25
        self.MAR_THRESHOLD = 0.5
        self.CONSECUTIVE_FRAMES = 15
        self.COUNTER = 0
        self.ALARM_ON = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        rects = self.detector(gray, 0)

        color_marco = (0, 255, 0)
        ear, mar, perclos_val = 0.0, 0.0, 0.0

        for rect in rects:
            shape = self.predictor(gray, rect)
            shape_np = face_utils.shape_to_np(shape)

            eye_left = shape_np[36:42]
            eye_right = shape_np[42:48]
            mouth = shape_np[48:68]

            ear = (calculate_ear(eye_left) + calculate_ear(eye_right)) / 2.0
            mar = calculate_mar(mouth)

            self.monitor.update(ear < self.EAR_THRESHOLD)
            perclos_val = self.monitor.get_perclos()

            if ear < self.EAR_THRESHOLD:
                self.COUNTER += 1
                if self.COUNTER >= self.CONSECUTIVE_FRAMES:
                    color_marco = (0, 0, 255)
                    self.ALARM_ON = True
                    cv2.putText(img, "!!! ALERTA DE SOMNOLENCIA !!!", (50, 50),
                                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
            else:
                self.COUNTER = 0
                self.ALARM_ON = False

            if mar > self.MAR_THRESHOLD:
                color_marco = (0, 255, 255)
                cv2.putText(img, "BOSTEZO DETECTADO", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            for hull_points in [eye_left, eye_right, mouth]:
                hull = cv2.convexHull(hull_points)
                cv2.drawContours(img, [hull], -1, color_marco, 1)

        cv2.rectangle(img, (0, img.shape[0] - 40), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
        cv2.putText(img, f"EAR: {round(ear, 2)} | MAR: {round(mar, 2)} | PERCLOS: {round(perclos_val, 1)}%",
                    (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        st.session_state["alarm_status"] = self.ALARM_ON

        return frame.from_ndarray(img, format="bgr24")


st.title("Sistema Monitor de Conducción")

placeholder_audio = st.empty()

ctx = webrtc_streamer(
    key="vigia-stream",
    video_processor_factory=DrowsinessProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

if st.session_state.get("alarm_status"):
    st.error("⚠️ ¡¡¡SOMNOLENCIA DETECTADA!!! ⚠️")

    audio_b64 = get_audio_base64("assets/freesound_community-alarm-26718.wav")

    if audio_b64:
        audio_html = f"""
            <audio autoplay loop>
                <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
            </audio>
        """
    else:
        audio_html = """
            <audio autoplay loop>
                <source src="https://www.soundjay.com/buttons/beep-01a.mp3" type="audio/mpeg">
            </audio>
        """

    placeholder_audio.markdown(audio_html, unsafe_allow_html=True)
else:
    placeholder_audio.empty()