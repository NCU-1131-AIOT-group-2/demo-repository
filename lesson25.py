# 匯入 OpenCV 模組
import cv2

# 加載靜態圖像
cap = cv2.VideoCapture('coin.jpg')
ret, frame = cap.read()

# 將圖像轉換為灰階並應用高斯模糊
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (13, 13), 0)

# 使用 Canny 邊緣檢測
edged = cv2.Canny(gray, 50, 150)

# 檢測輪廓
contours, hierarchy = cv2.findContours(
    edged.copy(),
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# 繪製輪廓
out = frame.copy()
cv2.drawContours(out, contours, -1, (0, 255, 128), 2)

# 合併原圖與輪廓圖
frame = cv2.hconcat([frame, out])

# 顯示結果
cv2.imshow('frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()