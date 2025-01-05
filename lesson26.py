# 匯入必要模組
import cv2
import numpy as np

# 加載圖像並調整尺寸
src = cv2.imread('cup.jpg', -1)
src = cv2.resize(src, (403, 302))

# 將圖像轉換為灰階並應用高斯模糊
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# 使用霍夫圓變換檢測圓形
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,  # 檢測方法，目前僅支援 HOUGH_GRADIENT
    1,                  # 輸入圖像與檢測圖像的比例，通常設為 1
    20,                 # 各圓心間的最小距離
    None,               # 固定為 None
    10,                 # Canny 演算法的高閾值
    75,                 # 檢測圓形的累加閾值
    3,                  # 最小圓半徑
    75                  # 最大圓半徑
)

# 將檢測結果轉換為整數
circles = circles.astype(int)
if len(circles) > 0:
    out = src.copy()
    for x, y, r in circles[0]:
        # 繪製圓形
        cv2.circle(out, (x, y), r, (0, 0, 255), 3)
        # 繪製圓心
        cv2.circle(out, (x, y), 2, (0, 255, 0), 3)
    # 合併原圖與結果
    src = cv2.hconcat([src, out])

# 顯示結果
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', src)
cv2.waitKey(0)
cv2.destroyAllWindows()