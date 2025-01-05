from ultralytics import YOLO
import cv2

# 加載 YOLOv8 模型
model = YOLO('yolov8n.pt')  # Nano 模型

# 圖片檢測
image_path = 'image.jpg'  # 替換為您的圖片路徑
results = model(image_path)

# 顯示檢測結果
annotated_image = results[0].plot()
cv2.imshow('YOLOv8 Detection', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()