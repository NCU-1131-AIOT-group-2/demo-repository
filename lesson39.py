from picamera2 import Picamera2, Preview
from ultralytics import YOLO
import cv2

# 初始化 YOLO 模型
model = YOLO('yolov8n.pt')  # 使用輕量級 YOLO Nano 模型

# 初始化 PiCamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)

# 啟動攝影機
picam2.start()

print("按下 'Ctrl + C' 退出程式")

try:
    while True:
        # 獲取攝影機畫面
        frame = picam2.capture_array()

        # 使用 YOLO 模型進行推理
        results = model(frame)

        # 繪製檢測結果
        annotated_frame = results[0].plot()

        # 使用 OpenCV 顯示結果
        cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

        # 按下 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n程式已中止")

# 停止攝影機並釋放資源
picam2.stop()
cv2.destroyAllWindows()