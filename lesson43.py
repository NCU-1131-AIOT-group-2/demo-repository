from flask import Flask, Response, render_template_string, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2
import threading
import time

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Global variables for control and information sharing
is_paused = False
lock = threading.Lock()
latest_info = []


# Function to initialize and start the camera
def initialize_camera():
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
    picam2.start()
    return picam2


# Generate video frames for streaming
def generate_frames():
    global is_paused, latest_info
    while True:
        with lock:
            if is_paused:
                time.sleep(0.1)  # Sleep briefly to prevent tight loop when paused
                continue
        try:
            with initialize_camera() as camera:
                frame = camera.capture_array()

                # Check if the image has 4 channels and convert to 3 channels if necessary
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                results = model.predict(source=frame, stream=True)
                temp_info = []
                for result in results:
                    boxes = result.boxes.xyxy.numpy()
                    for box, cls, conf in zip(boxes, result.boxes.cls.numpy(), result.boxes.conf.numpy()):
                        x1, y1, x2, y2 = map(int, box)
                        label = f"{model.names[int(cls)]}: {conf:.2f}"
                        temp_info.append(label)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                with lock:
                    latest_info = temp_info
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except RuntimeError as e:
            print(f"RuntimeError: {e}")
            time.sleep(1)  # Wait before retrying to avoid rapid repeated failures


# Main page route
@app.route('/')
def index():
    return render_template_string('''
        <!doctype html>
        <html>
        <head>
            <title>Live YOLOv8 Object Detection</title>
            <script>
                setInterval(() => {
                    fetch('/latest_info')
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('info').innerText = "Detected: " + data.join(', ');
                        });
                }, 1000);

                function pause() {
                    fetch('/pause', { method: 'POST' }).then(() => alert('Paused'));
                }

                function resume() {
                    fetch('/resume', { method: 'POST' }).then(() => alert('Resumed'));
                }
            </script>
        </head>
        <body>
            <h1>Live YOLOv8 Object Detection</h1>
            <img src="/video_feed" width="640" height="480">
            <div id="info">Detection Info: </div>
            <button onclick="pause()">Pause</button>
            <button onclick="resume()">Resume</button>
        </body>
        </html>
    ''')


# Video feed route
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Pause route
@app.route('/pause', methods=['POST'])
def pause():
    global is_paused
    with lock:
        is_paused = True
    return jsonify({"status": "paused"})


# Resume route
@app.route('/resume', methods=['POST'])
def resume():
    global is_paused
    with lock:
        is_paused = False
    return jsonify({"status": "resumed"})


# Latest info route
@app.route('/latest_info')
def latest_info_route():
    with lock:
        return jsonify(latest_info)


# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)