# 匯入必要模組
import cv2
import numpy as np

# 初始化圖像與標籤的列表
images = []
labels = []

# 定義固定的圖像大小
fixed_size = (100, 100)

# 加載圖像並設置標籤
for index in range(100):
    filename = f'images/h0/{index:03d}.pgm'
    print(f'Reading {filename}')
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f'Error: {filename} could not be read!')
        continue

    # 調整圖像大小
    img_resized = cv2.resize(img, fixed_size)
    images.append(img_resized)
    labels.append(0)

# 將列表轉換為 NumPy 陣列
images_array = np.array(images, dtype=np.uint8)
labels_array = np.array(labels, dtype=np.int32)

# 訓練 LBPH 臉部識別器
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images_array, labels_array)
model.save('faces.data')
print('Training done and model saved.')