from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt

# 創建保存結果的目錄
save_dir = 'results/'
os.makedirs(save_dir, exist_ok=True)

# 加載 YOLO 模型 (使用 COCO 預訓練權重)
model = YOLO('yolov8n.pt')  # 使用 Nano 模型

# 推理：對單張圖片進行目標檢測
image_path = 'image.jpg'  # 替換為您的圖片路徑
results = model(image_path)  # 執行目標檢測

# 1. 可視化檢測結果
annotated_image = results[0].plot()  # 繪製帶有檢測結果的圖片

# 使用 Matplotlib 顯示結果
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# 2. 保存檢測結果圖片
cv2.imwrite(os.path.join(save_dir, 'detected_image.jpg'), annotated_image)
print(f"Saved detection result to {os.path.join(save_dir, 'detected_image.jpg')}")

# 3. 打印檢測到的所有類別名稱
detected_classes = set()  # 用集合儲存類別，避免重複
for box in results[0].boxes:
    cls_id = int(box.cls[0])  # 類別 ID
    class_name = model.names[cls_id]  # 類別名稱
    detected_classes.add(class_name)

print("Detected Classes:")
for cls in detected_classes:
    print(f"- {cls}")