# 匯入必要模組
import cv2

# 常量表示輪廓索引
RECT, HEXAGON = 0, 1

# 加載圖像並轉換為灰階
frame = cv2.imread('poly.png')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# 使用 Canny 邊緣檢測並進行膨脹處理
edged = cv2.Canny(gray, 50, 150)
edged = cv2.dilate(edged, None, iterations=1)

# 檢測輪廓
contours, hierarchy = cv2.findContours(
    edged,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# 輸出處理前的點數量
print('=== 處理前')
print('矩形點數量：{}'.format(len(contours[RECT])))
print('六邊形點數量：{}'.format(len(contours[HEXAGON])))

# 使用多邊形近似簡化輪廓
approx_rect = cv2.approxPolyDP(contours[RECT], 30, True)
approx_hex = cv2.approxPolyDP(contours[HEXAGON], 30, True)

# 輸出處理後的點數量
print('=== 處理後')
print('矩形點數量：{}'.format(len(approx_rect)))
print('六邊形點數量：{}'.format(len(approx_hex)))

# 繪製簡化後的輪廓
cv2.drawContours(frame, [approx_rect], -1, (0, 0, 255), 5)
cv2.drawContours(frame, [approx_hex], -1, (0, 0, 255), 5)

# 顯示結果
cv2.imshow('frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()