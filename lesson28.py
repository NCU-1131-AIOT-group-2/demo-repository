# 匯入必要模組
import cv2


# 加載圖片
img1 = cv2.imread('box.png')
img2 = cv2.imread('box_in_scene.png')

# 初始化特徵檢測器
feature = cv2.xfeatures2d.SURF_create()

# 檢測並計算特徵點與描述子
kp1, des1 = feature.detectAndCompute(img1, None)
kp2, des2 = feature.detectAndCompute(img2, None)

# 使用 BFMatcher 進行特徵匹配
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# 篩選優良匹配點
good = []
for m, n in matches:
    if m.distance < 0.55 * n.distance:
        good.append(m)

print('Matching points :{}'.format(len(good)))

# 繪製匹配結果
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, [good], outImg=None,
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# 調整輸出圖片大小
width, height, channel = img3.shape
ratio = float(width) / float(height)
img3 = cv2.resize(img3, (1024, int(1024 * ratio)))

# 顯示結果
cv2.imshow('video', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()