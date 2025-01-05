# 匯入必要模組
import cv2
import numpy as np
import time
from picamera2 import Picamera2

# 初始化 YOLO 模型
def initNet():
    CONFIG = 'yolov4-tiny.cfg'  # YOLO 配置檔案
    WEIGHT = 'yolov4-tiny.weights'  # 預訓練權重
    NAMES = 'coco.names'  # 類別名稱檔案

    # 讀取物件名稱與設定外框顏色
    with open(NAMES, 'r') as f:
        names = [line.strip() for line in f.readlines()]  # 讀取類別名稱
        colors = np.random.uniform(0, 255, size=(len(names), 3))  # 為每個類別隨機生成顏色

    # 初始化 YOLO 神經網路
    net = cv2.dnn.readNet(CONFIG, WEIGHT)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255.0)  # 輸入尺寸與縮放
    model.setInputSwapRB(True)  # YOLO 使用 RGB 格式，需對調顏色

    return model, names, colors

# 執行 YOLOv4-tiny 檢測
def nnProcess(image, model):
    # 使用 YOLO 模型執行檢測
    classes, confs, boxes = model.detect(image, 0.6, 0.3)  # 信心值閾值為 0.6，非極大值抑制為 0.3
    return classes, confs, boxes

# 繪製檢測框與標籤
def drawBox(image, classes, confs, boxes, names, colors):
    new_image = image.copy()
    for (classid, conf, box) in zip(classes, confs, boxes):
        x, y, w, h = box  # 框的座標與大小
        label = '{}: {:.2f}'.format(names[int(classid)], float(conf))  # 類別名稱與信心值
        color = colors[int(classid)]  # 該類別對應的顏色
        cv2.rectangle(new_image, (x, y), (x + w, y + h), color, 2)  # 繪製矩形框
        cv2.putText(new_image, label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2  # 在框上方顯示標籤
        )
    return new_image

# 初始化 YOLO 模型
model, names, colors = initNet()

# 初始化 Picamera2
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size": (640, 480)})  # 設定相機預覽尺寸
picam2.configure(camera_config)
picam2.start()  # 啟動相機

# 設定顯示窗口大小
WIDTH = 800
FONT = cv2.FONT_HERSHEY_SIMPLEX  # 設定文字字體

# 實時檢測
while True:
    begin_time = time.time()  # 記錄當前時間以計算 FPS

    # 從 Picamera2 獲取一幀圖像
    frame = picam2.capture_array()
    ratio = frame.shape[1] / frame.shape[0]  # 計算影像的寬高比
    HEIGHT = int(WIDTH / ratio)  # 根據寬度計算高度，保持比例
    frame = cv2.resize(frame, (WIDTH, HEIGHT))  # 調整影像大小

    # 如果影像是 RGBA 格式，轉換為 BGR 格式
    if frame.shape[2] == 4:  # 檢查影像是否為 4 通道
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    # 執行 YOLOv4-tiny 檢測
    classes, confs, boxes = nnProcess(frame, model)

    # 繪製檢測結果
    frame = drawBox(frame, classes, confs, boxes, names, colors)

    # 計算與顯示 FPS
    fps = 'fps: {:.2f}'.format(1 / (time.time() - begin_time))
    cv2.putText(frame, fps, (10, 30),
        FONT, 0.7, (0, 204, 255), 2  # 在左上角顯示 FPS
    )

    # 顯示處理後的影像
    cv2.imshow('video', frame)

    # 按下 ESC 鍵退出
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()  # 關閉所有 OpenCV 窗口
        picam2.stop()  # 停止相機
        break