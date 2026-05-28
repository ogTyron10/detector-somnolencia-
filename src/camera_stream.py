import cv2
import imutils

class CameraStream:
    def __init__(self):
        self.stream = cv2.VideoCapture(0)

    def get_frame(self):
        (grabbed, frame) = self.stream.read()
        if not grabbed:
            return None
        frame = imutils.resize(frame, width=450)
        return frame

    def stop(self):
        self.stream.release()