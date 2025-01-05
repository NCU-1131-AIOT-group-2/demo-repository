# 匯入必要模組
import cv2

# 加載圖像並轉換為灰階
frame = cv2.imread('star.png')
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

# 獲取第一個輪廓並進行多邊形近似
cnt = contours[0]
cnt = cv2.approxPolyDP(cnt, 30, True)

# 計算凸包與凹缺點
hull = cv2.convexHull(cnt, returnPoints=False)
defects = cv2.convexityDefects(cnt, hull)

# 輸出凸點與凹點數量
print('凸點數量：{}'.format(len(hull)))
print('凹點數量：{}'.format(len(defects)))

# 繪製凹缺點與凸包邊界
for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(frame, start, end, (0, 255, 0), 2)
    cv2.circle(frame, far, 5, (0, 0, 255), -1)

# 顯示結果
cv2.imshow('frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()