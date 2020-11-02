import sys
import cv2
import numpy as np

src = cv2.imread(sys.path[0]+'/rice.png', cv2.IMREAD_GRAYSCALE)

if src is None:
	print('Image load failed!')
	sys.exit()

# Local-Binarization
lb_img = np.zeros(src.shape, np.uint8)

dh = src.shape[0] // 4
dw = src.shape[1] // 4


