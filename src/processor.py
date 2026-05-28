from scipy.spatial import distance as dist
import time

class FatigueMonitor:
    def __init__(self):
        self.history = []
        self.window_seconds = 60

    def update(self, is_closed):
        current_time = time.time()
        self.history.append((current_time, 1 if is_closed else 0))

        self.history = [h for h in self.history if current_time - h[0] <= self.window_seconds]

    def get_perclos(self):
        if not self.history: return 0.0
        closed_frames = sum([h[1] for h in self.history])
        return (closed_frames / len(self.history)) * 100
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_mar(mouth):
    A = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[12], mouth[16])
    mar = A / C
    return mar