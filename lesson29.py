# 匯入必要模組
import cv2

# 加載圖片
img1 = cv2.imread('box.png')
img2 = cv2.imread('box_in_scene.png')

# 初始化 ORB 特徵檢測器
orb = cv2.ORB_create()

# 檢測並計算特徵點與描述子
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 初始化 BFMatcher 並匹配特徵
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# 按距離排序匹配結果
matches = sorted(matches, key=lambda x: x.distance)

# 繪製前 10 個匹配點
img3 = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    matches[:10],
    outImg=None,
    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
)

# 調整輸出圖像大小
width, height, channel = img3.shape
ratio = float(width) / float(height)
img3 = cv2.resize(img3, (1024, int(1024 * ratio)))

# 顯示匹配結果
cv2.imshow('image', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()