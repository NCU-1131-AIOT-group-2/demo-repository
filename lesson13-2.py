from picamera2 import Picamera2
import cv2
import numpy as np

class PiCameraCapture:
    def __init__(self, resolution=(640, 480)):
        self.picam2 = Picamera2()
        camera_config = self.picam2.create_preview_configuration(main={"size": resolution})
        self.picam2.configure(camera_config)
        self.picam2.start()
        self.opened = True
        self.resolution = resolution

    def isOpened(self):
        return self.opened

    def read(self):
        if not self.opened:
            return False, None
        frame = self.picam2.capture_array()
        return True, frame

    def release(self):
        if self.opened:
            self.picam2.stop()
            self.opened = False

cap = PiCameraCapture(resolution=(640, 480))
if not cap.isOpened():
    print("Error: Unable to open camera.")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('video.mp4', fourcc, 30, (400, 300))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (400, 300))
    frame = cv2.flip(frame, 1)
    cv2.imshow('frame', frame)
    out.write(frame)

    if cv2.waitKey(1) == ESC:
        break

cap.release()
out.release()
cv2.destroyAllWindows()