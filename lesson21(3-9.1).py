# 匯入 OpenCV 模組
import cv2

# 開啟影片檔案
cap = cv2.VideoCapture('vtest.avi')
bg = None  # 初始化背景

# 持續讀取影片幀
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 將畫面轉換為灰階並模糊化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (17, 17), 0)

    # 初始化背景
    if bg is None:
        bg = gray
        continue

    # 計算背景差分並進行二值化處理
    diff = cv2.absdiff(gray, bg)
    diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]
    diff = cv2.erode(diff, None, iterations=2)
    diff = cv2.dilate(diff, None, iterations=2)

    # 尋找輪廓
    cnts, hierarchy = cv2.findContours(
        diff,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for c in cnts:
        # 忽略小輪廓
        if cv2.contourArea(c) < 500:
            continue

        # 繪製矩形框
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 顯示畫面
    cv2.imshow("frame", frame)
    if cv2.waitKey(100) == 27:  # 按下 ESC 鍵退出
        cv2.destroyAllWindows()
        break