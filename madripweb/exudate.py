import cv2
import numpy as np
import sys
from subprocess import run,PIPE

imgName = sys.argv[1]
img = cv2.imread(imgName)  

imS = cv2.resize(img, (512, 512)) 

green_image = imS.copy()
green_image[:,:,0] = 0
green_image[:,:,2] = 0

gray = cv2.cvtColor(green_image, cv2.COLOR_BGR2GRAY)
ret, thresh1 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY) 

newName = imgName.split('.')
newName.insert(1,'_exudate.')
newImgName = ""
for i in newName:
    newImgName += i
print(newImgName)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) 

img_dilation = cv2.dilate(thresh1, kernel, iterations=2) 
img_erosion = cv2.erode(img_dilation, kernel, iterations=2) 

cv2.imshow('Input', thresh1) 
cv2.imshow('Dilation', img_dilation) 
cv2.imshow('Erosion', img_erosion)

newName = imgName.split('.')

cv2.imwrite(newImgName,img_erosion) 