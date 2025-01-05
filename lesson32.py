#匯入必要模組
import cv2
import argparse

# 解析命令列參數
ap = argparse.ArgumentParser()
ap.add_argument('img', nargs='+', help='input images')
args = ap.parse_args()

# 加載所有輸入圖片
img_arr = []
for filename in args.img:
    image = cv2.imread(filename)
    img_arr.append(image)

# 初始化拼接器並進行拼接
stitcher = cv2.Stitcher_create()
status, pano = stitcher.stitch(img_arr)

# 驗證拼接結果
if status == cv2.Stitcher_OK:
    # 顯示與保存全景圖
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', pano)
    cv2.imwrite('final.jpg', pano)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('done')
else:
    print('error: {}'.format(status))