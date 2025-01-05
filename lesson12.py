# 匯入必要模組
from picamera2 import Picamera2
import cv2

# 初始化相機
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()

# 顯示相機畫面
while True:
    frame = picam2.capture_array()  # 獲取畫面陣列
    cv2.imshow("Preview", frame)  # 顯示畫面
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下 'q' 鍵退出
        break

# 停止相機並釋放資源
picam2.stop()
cv2.destroyAllWindows()