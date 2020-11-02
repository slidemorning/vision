import sys
import cv2
import numpy as np

def onChange(pos):
	global src
	bsize = pos
	if bsize % 2 == 0:
		bsize = bsize - 1
	if bsize < 3:
		bsize = 3
	dst = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bsize, 5)
	cv2.imshow('Trackbar:Local-binarization', dst)


src = cv2.imread(sys.path[0]+'/rice.png', cv2.IMREAD_GRAYSCALE)

if src is None:
	print('Image load failed!')
	sys.exit()

# Global-Binarization
gb_img = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Local-Binarization
lb_img = np.zeros(src.shape, np.uint8)

dh = src.shape[0] // 4
dw = src.shape[1] // 4

for h in range(4):
	for w in range(4):
		w_src = src[h*dh:(h+1)*dh, w*dw:(w+1)*dw]
		w_dst = lb_img[h*dh:(h+1)*dh, w*dw:(w+1)*dw] # reference lb_img
		cv2.threshold(w_src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU, w_dst)

cv2.namedWindow('Trackbar:Local-binarization')
cv2.createTrackbar('Block Size', 'Trackbar:Local-binarization', 0, 200, onChange)
cv2.setTrackbarPos('Block Size', 'Trackbar:Local-binarization', 13)
cv2.imshow('src', src)
cv2.imshow('global binarization', gb_img)
cv2.imshow('local binarization', lb_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



