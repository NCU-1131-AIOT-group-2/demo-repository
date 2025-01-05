# 匯入 OpenCV 模組
import cv2

# 使用 cv2.imread 函數讀取圖片檔案 'demo.jpeg' 並轉換為灰階格式
frame = cv2.imread('demo.jpeg', cv2.IMREAD_GRAYSCALE)

# 使用 cv2.imshow 函數顯示讀取的灰階圖片
cv2.imshow('image', frame)

# 使用 cv2.waitKey 函數等待用戶按下任意鍵，參數為 0 表示無限等待
cv2.waitKey(0)

# 使用 cv2.destroyAllWindows 函數關閉所有開啟的視窗
cv2.destroyAllWindows()
