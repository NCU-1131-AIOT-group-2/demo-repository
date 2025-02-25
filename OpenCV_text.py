import cv2
import numpy as np

gc = np.zeros((512, 512, 3) , dtype=np.uint8)
gc.fill (255)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(gc, "OpenCV", (10,200), font , 4, (0, 0, 0), 2, cv2.LINE_AA)

cv2.imshow("draw", gc)
cv2.waitKey(0)
cv2.destroyAllWindows()