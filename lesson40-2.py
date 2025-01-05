from ultralytics import YOLO
import cv2

# 加載 YOLO 模型
model = YOLO('yolov8n.pt')

# 多張圖片路徑
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']

# 批量處理圖片
for image_path in image_paths:
    results = model(image_path)
    annotated_image = results[0].plot()
    output_path = f'detected_{image_path}'
    cv2.imwrite(output_path, annotated_image)
    print(f"檢測完成，結果已保存至 {output_path}")