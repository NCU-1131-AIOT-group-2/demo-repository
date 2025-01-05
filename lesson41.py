from flask import Flask, Response
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# 初始化攝像頭
camera = cv2.VideoCapture(0)  # 0 表示默認的 PC 攝像頭

# 加載 YOLO 模型 (需下載模型文件)
model = YOLO("yolov8n.pt")  # 使用輕量級 YOLO 模型 (yolov8n.pt)


def generate_frames():
    while True:
        # 讀取攝像頭畫面
        success, frame = camera.read()
        if not success:
            break

        # YOLO 偵測處理
        results = model.predict(source=frame, stream=True)

        for result in results:
            # 解析 YOLO 偵測結果
            boxes = result.boxes.xyxy.numpy()  # 每個框的左上角和右下角座標
            classes = result.boxes.cls.numpy()  # 偵測到的類別索引
            confidences = result.boxes.conf.numpy()  # 偵測置信度

            # 繪製偵測框
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = map(int, box)
                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 將 YOLO 處理後的影像轉為 JPEG 格式
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # 返回幀數據
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    # 提供視頻流路由
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    # 主頁面，嵌入視頻流
    return '''
    <html>
    <head>
        <title>YOLO Real-Time Detection</title>
    </head>
    <body>
        <h1>YOLO Real-Time Detection</h1>
        <img src="/video_feed" width="800">
    </body>
    </html>
    '''


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)