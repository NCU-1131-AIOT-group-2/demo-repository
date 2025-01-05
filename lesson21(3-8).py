# 匯入 OpenCV 模組
import cv2

# 打開影片檔案
cap = cv2.VideoCapture('vtest.avi')

# 初始化 CSRT 追蹤器
tracker = cv2.TrackerCSRT_create()

# 初始化感興趣區域 (ROI)
roi = None

# 讀取影片幀
while True:
    ret, frame = cap.read()

    # 如果 ROI 尚未設定，讓用戶選擇
    if roi is None:
        roi = cv2.selectROI('frame', frame)  # 用戶選擇感興趣區域
        if roi != (0, 0, 0, 0):  # 如果 ROI 合法，初始化追蹤器
            tracker.init(frame, roi)

    # 更新追蹤器
    success, rect = tracker.update(frame)
    if success:
        # 如果追蹤成功，繪製矩形框
        (x, y, w, h) = [int(i) for i in rect]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 顯示更新後的幀
    cv2.imshow('frame', frame)

    # 按 ESC 鍵退出
    if cv2.waitKey(66) == 27:
        cv2.destroyAllWindows()
        break