import numpy as np
import matplotlib.pyplot as plt
import cv2

def plts(title, image, map='gray'):
    plt.imshow(image, cmap=map)
    plt.title(title)
    plt.show()


img = cv2.imread('SymbolSet.png',0)

ret, img = cv2.threshold(img, 180, 255, 0)

im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

m, n = img.shape[:2]
print(m, n)
for cnt in contours:
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    print(rightmost)
    p = int(input())
    if p == 0:
        break

'''
make numbers more eroded
use template matching on the nums.jpg
grow numbers through region growing and make [0-9] templates
http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_template.html
http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.match_template
save the templates[0-9] to DS and find matches
empty = ~(empty ^ spare)
empty = cv2.dilate(empty,kernel,iterations=1)
empty = cv2.erode(empty,kernel,iterations=2)

print(max1,min1)
cv2.imshow('New',empty)
'''