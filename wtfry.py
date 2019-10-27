import numpy as np
import argparse
import cv2
 
def wtfry(image):
	# split the image into its BGR components
	(B, G, R) = cv2.split(image)
  Max = np.maximum(np.maximum(R, G), B)
	R[R < Max] = 0
	G[G < Max] = 0
	B[B < Max] = 0
 
	return cv2.merge([B, G, R])
  ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
 
image = cv2.imread(args["image"])
fry = wtfry(image)
cv2.imshow("Images", np.hstack([image, fry]))
cv2.waitKey(0)
