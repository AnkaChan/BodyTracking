import cv2

def preprocessImg(inImgFile):
    # convert to Rgb
    #
    img = cv2.imread(inImgFile, cv2.IMREAD_GRAYSCALE)
    imgColor = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2BGR_EA )
	return imgColor