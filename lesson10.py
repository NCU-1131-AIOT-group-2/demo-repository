#libcamera & OpenCV

# 匯入必要模組
import cv2
import subprocess
import numpy as np

# 啟動 libcamera 並獲取影像
subprocess.run(["libcamera-vid", "-t", "0", "--inline"])

# 打開攝像頭
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("無法啟動攝像頭")
    exit()

# 連續讀取影像並顯示
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下 'q' 鍵退出
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()

#picamera2

# 匯入必要模組
from picamera2 import Picamera2
import cv2

# 初始化 picamera2
picam2 = Picamera2()
picam2.start()

# 持續讀取並顯示影像
while True:
    frame = picam2.capture_array()  # 直接獲取影像陣列
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下 'q' 鍵退出
        break

# 停止攝像頭並關閉視窗
cv2.destroyAllWindows()
picam2.stop()