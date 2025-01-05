import cv2

#測試OpenCV環境
print("OpenCV 版本:", cv2.__version__)

# 讀取圖片
image = cv2.imread('image.jpg')

# 檢查是否成功讀取
if image is None:
    print("無法讀取圖片，請檢查檔案路徑。")
else:
    # 顯示圖片
    cv2.imshow('顯示圖片', image)
    cv2.waitKey(0)  # 等待按下任意鍵
    cv2.destroyAllWindows()
