from picamera2 import Picamera2
import cv2
import time

# 初始化 Picamera2
picam2 = Picamera2()

# 配置相機的預覽模式，設定主要圖像大小為 640x480
camera_config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(camera_config)
picam2.start()  # 啟動相機

# 加載 DNN 模型
net = cv2.dnn.readNet(
    'opencv_face_detector.pbtxt',  # 模型結構檔案
    'opencv_face_detector_uint8.pb'  # 模型權重檔案
)

# 將模型封裝為檢測模型，設置輸入參數
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(300, 300), scale=1.0)  # 模型輸入尺寸和縮放係數

# 設定影像顯示窗口的寬度及其他相關參數
WIDTH = 600  # 顯示窗口寬度
FONT = cv2.FONT_HERSHEY_SIMPLEX  # 文字字體

while True:  # 進行連續影像處理
    begin_time = time.time()  # 記錄當前時間以計算 FPS

    # 從 Picamera2 獲取一幀圖像，作為 NumPy 陣列
    frame = picam2.capture_array()
    ratio = frame.shape[1] / frame.shape[0]  # 計算原始影像的寬高比
    HEIGHT = int(WIDTH / ratio)  # 調整顯示影像的高度以保持比例
    frame = cv2.resize(frame, (WIDTH, HEIGHT))  # 調整影像尺寸
    frame = cv2.flip(frame, 1)  # 鏡像翻轉，讓視覺效果更自然

    # 如果影像是 RGBA 格式（具有 4 通道），則轉換為 BGR 格式
    if frame.shape[2] == 4:  # 檢查影像的通道數是否為 4
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)  # 將 RGBA 轉換為 BGR

    # 使用模型進行人臉檢測
    classes, confs, boxes = model.detect(frame, 0.5)  # 偵測，0.5 為信心值閾值
    for (classid, conf, box) in zip(classes, confs, boxes):  # 遍歷所有檢測到的框
        x, y, w, h = box  # 取得檢測框的位置與尺寸
        fps = 1 / (time.time() - begin_time)  # 計算 FPS
        text = "fps: {:.1f} {:.2f}%".format(fps, float(conf) * 100)  # FPS 與信心值

        # 確保文字框不超出影像邊界
        if y - 20 < 0:
            y1 = y + 20
        else:
            y1 = y - 10

        # 繪製檢測框與文字
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # 繪製矩形框
        cv2.putText(frame, text, (x, y1), FONT, 0.5, (0, 204, 255), 2)  # 顯示 FPS 與信心值

    # 在窗口中顯示處理後的影像
    cv2.imshow("video", frame)

    # 按下 ESC 鍵（ASCII 碼為 27）以退出
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()  # 關閉所有 OpenCV 窗口
        picam2.stop()  # 停止相機
        break  # 結束程式