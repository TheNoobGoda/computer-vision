import cv2
import numpy as np

def remove_color(img):
    new_img = img.copy()
    new_img[np.where((new_img < [50,50,50]).all(axis=2))] = [0,0,0]
    return new_img

img = cv2.imread('img/akai2.jpeg')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_gray = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2BGR)
black = remove_color(img_gray)
cv2.imwrite('img/lineDetection/black.jpg',black)


img_blur = cv2.GaussianBlur(img,(3,3), sigmaX=0, sigmaY=0)

cv2.imwrite('img/lineDetection/gray.jpg',img_gray)

edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) 
sobel = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
cv2.imwrite('img/lineDetection/sobel.jpg',sobel)
cv2.imwrite('img/lineDetection/canny.jpg',edges)
 
lines = cv2.HoughLines(edges,1,np.pi/180,200)

for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite('img/lineDetection/lines.jpg',img)