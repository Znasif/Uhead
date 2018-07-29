import cv2
import json
import random
import skimage.draw

f = open("contours.json", "r")
s = f.read()
annotations = json.loads(s)

dir = "Extracted/ALL/"
n = 20
fls = 10
color = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(fls)]
for i in range(n):
    a = cv2.imread(dir + "Gen/" + str(i) + ".tif")
    for j in annotations[str(i)]["regions"].values():
        num = j["region_attributes"]
        ann_x = j["shape_attributes"]["all_points_x"]
        ann_y = j["shape_attributes"]["all_points_y"]
        rr, cc = skimage.draw.polygon(ann_y, ann_x)
        a[rr, cc] = color[num]
    cv2.imwrite(dir + "Gen/" + str(i) + "a.tif", a)
