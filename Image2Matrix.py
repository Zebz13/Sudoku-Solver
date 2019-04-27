#DONT FUCKING FORGET TO COMMENT AFTER YOU WAKE UP

import cv2
import numpy as np
import pytesseract as tes


def show(x):
    cv2.imshow("Out",x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read(x):
    file=cv2.imread(x)
    file=cv2.resize(file,(400,400))
    return file


def process(x):
    y=x.copy()
    gray=cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
    cv2.bitwise_not(gray,gray)
    kernel = np.ones((1,1),np.uint8)
    dilated=cv2.dilate(gray,kernel,iterations=1)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    for j in range(len(lines)):
        for rho,theta in lines[j]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(x,(x1,y1),(x2,y2),(0,0,255),2)
    return (x,y)


def region(x,y):
    grey=cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(grey,127,255,0)
    kernel = np.ones((5,5), np.uint8) 
    img_erosion = cv2.erode(thresh, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(img_erosion, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(y, contours, -1, (255,0,0), 2)
    return (y,contours)


def matrix(list):
    list.reverse()
    k=0
    w, h = 9,9
    matr = [[0 for x in range(w)] for y in range(h)]
    for i in range(9):
        for j in range(9):  
            if (list[k].isnumeric()):
                matr[i][j]=int(list[k])
            else:
                matr[i][j]=0
            k+=1
    return matr      



def digit(org,cont):
    a=np.zeros(org.shape,np.uint8)
    long_list=[]
    for i in range(81):
        a=cont[i][0][0]
        b=cont[i][2][0]
        crop_img=org[a[1]:b[1], a[0]:b[0]]
        value=tes.image_to_string(crop_img,config='--psm 10')
        long_list.append(value)
    return long_list

(Processed,original)=process(read('/home/zebin/Code/Python/Self Driving/Sudok.png'))
(drawn,boundaries)=region(Processed,original)
listed=digit(original,boundaries)
outp=matrix(listed)
outp=np.array(outp)
print(outp)
