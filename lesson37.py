from ultralytics import YOLO
import cv2
import os

# 加載 YOLO 模型
model = YOLO('yolov8n.pt')  # 使用 Nano 模型（輕量快速）

# 輸入影片路徑
video_path = 'bikes.mp4'  # 替換為您的影片路徑
output_path = 'output_video.avi'  # 輸出的影片檔案路徑

# 開啟影片文件
cap = cv2.VideoCapture(video_path)

# 確認影片是否成功打開
if not cap.isOpened():
    print("無法打開影片文件！請檢查路徑。")
    exit()

# 獲取影片資訊
fps = int(cap.get(cv2.CAP_PROP_FPS))  # 幀率
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 寬度
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 高度
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 總幀數

print(f"影片資訊: {fps} FPS, 寬度: {frame_width}, 高度: {frame_height}, 幀數: {frame_count}")

# 初始化影片寫入器
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 編碼格式
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 逐幀處理
frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 推理
    results = model(frame)

    # 獲取帶有檢測結果的圖片
    annotated_frame = results[0].plot()

    # 寫入影片
    out.write(annotated_frame)

    # 在終端打印處理進度
    frame_index += 1
    print(f"處理幀 {frame_index}/{frame_count}", end="\r")

# 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"處理完成！影片已保存到: {output_path}")