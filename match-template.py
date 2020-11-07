import sys
import cv2
import numpy as np
import os

IMAGE_PATH = os.getcwd() + '/image'

def main():
	
	image = cv2.imread(IMAGE_PATH + '/circuit.bmp', cv2.IMREAD_GRAYSCALE)
	templ = cv2.imread(IMAGE_PATH + '/crystal.bmp', cv2.IMREAD_GRAYSCALE)
	
	if	image is None or templ is None:
		print('Fail to read image')
		sys.exit()

	# Add gaussian noise
	gnoise = np.zeros(image.shape, np.int32)
	cv2.randn(gnoise, 50, 10)
	image = cv2.add(image, gnoise, dtype=cv2.CV_8U)

	# Match
	res = cv2.matchTemplate(image, templ, cv2.TM_CCOEFF_NORMED)

	#
	res_normed = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

	_, maxval, _, maxloc = cv2.minMaxLoc(res)
	print('maxval:', maxval)
	print('maxloc', maxloc)

	th, tw = templ.shape[:2]
	dst = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	cv2.rectangle(dst, maxloc, (maxloc[0] + tw, maxloc[1] + th), (0, 0, 255), 2)

	cv2.imshow('dst', dst)
	cv2.imshow('res_normed', res_normed)
	cv2.imshow('templ', templ)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	print('OpenCV', cv2.__version__)
	main()
