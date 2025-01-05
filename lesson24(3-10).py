# 匯入必要模組
import cv2
import numpy as np
from picamera2 import Picamera2

# 定義顏色範圍 (HSV)
color = ((16, 59, 0), (47, 255, 255))  # 範例範圍
lower = np.array(color[0], dtype="uint8")
upper = np.array(color[1], dtype="uint8")

# 初始化 Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()

# 設定影像尺寸
WIDTH = 400
HEIGHT = int(WIDTH * (480 / 640))  # 維持比例

# 持續捕捉影像
while True:
    # 捕捉畫面並調整尺寸與方向
    frame = picam2.capture_array()
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    frame = cv2.flip(frame, 1)

    # 轉換為 HSV 並進行高斯模糊
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (11, 11), 0)

    # 根據顏色範圍生成掩碼
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # 在掩碼中尋找輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 選擇最大輪廓
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 100:
            continue  # 忽略小輪廓

        # 繪製邊框
        x, y, w, h = cv2.boundingRect(cnt)
        p1 = (x - 2, y - 2)
        p2 = (x + w + 4, y + h + 4)
        cv2.rectangle(frame, p1, p2, (0, 255, 255), 2)

        # 高亮顯示目標
        out = cv2.bitwise_and(hsv, hsv, mask=mask)
        cv2.rectangle(hsv, p1, p2, (0, 255, 255), 2)
        cv2.rectangle(out, p1, p2, (0, 255, 255), 2)

        # 合併結果
        frame = cv2.hconcat([frame, hsv, out])

    # 顯示畫面
    cv2.imshow("frame", frame)

    # 按下 'Esc' 鍵退出
    if cv2.waitKey(1) == 27:
        break

# 清理資源
cv2.destroyAllWindows()
picam2.stop()