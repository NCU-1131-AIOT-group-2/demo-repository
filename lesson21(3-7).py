from picamera2 import Picamera2
import cv2
import numpy as np

# 定義感興趣區域（ROI）的範圍
RECT = ((220, 20), (370, 190))  # 左上角和右下角座標
(left, top), (right, bottom) = RECT


# 函式：取出 ROI 區域
def roiarea(frame):
    return frame[top:bottom, left:right]


# 函式：將處理後的 ROI 貼回原影像
def replaceroi(frame, roi):
    frame[top:bottom, left:right] = roi
    return frame


# 初始化 Picamera2
picam2 = Picamera2()
# 設定相機解析度和影像格式
camera_config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(camera_config)
picam2.start()

# 定義影像的寬和高
WIDTH = 400
HEIGHT = 300  # 調整以符合 4:3 的比例

while True:
    # 捕捉影像
    frame = picam2.capture_array()
    # 調整影像大小
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    # 水平翻轉影像（鏡像效果）
    frame = cv2.flip(frame, 1)

    # 提取 ROI
    roi = roiarea(frame)
    # 將 ROI 從 BGR 色彩空間轉換為 HSV 色彩空間
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 將處理後的 ROI 貼回原影像
    frame = replaceroi(frame, roi)

    # 在原影像中用紅色矩形標出 ROI 範圍
    cv2.rectangle(frame, RECT[0], RECT[1], (0, 0, 255), 2)
    # 顯示影像
    cv2.imshow('frame', frame)

    # 按下 Esc 鍵退出
    if cv2.waitKey(1) == 27:
        break

# 關閉視窗並停止相機
cv2.destroyAllWindows()
picam2.stop()