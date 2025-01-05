# 匯入 OpenCV 與 NumPy 模組
import cv2 
import numpy as np

# 建立一個 512x512 大小的白色畫布
gc = np.zeros((512, 512, 3), dtype=np.uint8)
gc.fill(255)

# 在畫布上繪製文字 'OpenCV'
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(gc, 'OpenCV', (10,200), font, 4, (0,0,0), 2, cv2.LINE_AA)

# 顯示畫布
cv2.imshow('draw', gc) 
cv2.waitKey(0)
cv2.destroyAllWindows()