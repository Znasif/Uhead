import numpy as np
import cv2

img = cv2.imread("no.jpg", 0)
'''
#blur = cv2.GaussianBlur(img, (5, 5), 0)
# ret3, img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
img = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
'''
kernel = np.ones((7, 7), np.uint8)
img = cv2.dilate(img, kernel, iterations=15)
img = cv2.erode(img, kernel, iterations=15)
cv2.imwrite("m_Closing.png", img)
