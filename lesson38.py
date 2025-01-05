from ultralytics import YOLO
import cv2

# 加載 YOLO 模型
model = YOLO('yolov8n.pt')  # 使用 Nano 模型（輕量快速）

# 開啟攝影機（0 表示默認攝影機）
cap = cv2.VideoCapture(0)

# 檢查攝影機是否成功打開
if not cap.isOpened():
    print("無法啟動攝影機！")
    exit()

# 獲取攝影機畫面尺寸
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"攝影機畫面大小：{frame_width}x{frame_height}")

# 即時處理攝影機畫面
while True:
    ret, frame = cap.read()  # 從攝影機捕獲畫面
    if not ret:
        print("無法讀取攝影機畫面！")
        break

    # 使用 YOLO 模型進行推理
    results = model(frame)

    # 獲取帶檢測結果的畫面
    annotated_frame = results[0].plot()

    # 顯示畫面
    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    # 按下 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝影機和關閉視窗
cap.release()
cv2.destroyAllWindows()