# shape detection script, to add: pentagon, hexagon, heptagon
import cv2
import numpy as np
import math

thresh1 = 110
thresh2 = 180

def stackImages(scale,imgArray,lables=[]):
    sizeW= imgArray[0][0].shape[1]
    sizeH = imgArray[0][0].shape[0]
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (sizeW, sizeH), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (sizeW, sizeH), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, img = cap.read()
    imgCan = cv2.Canny(img, thresh1, thresh2)
    kernel = np.ones((3, 3))
    imgDil = cv2.dilate(imgCan, kernel, iterations=1)
    cnts, hierarchy = cv2.findContours(
        imgDil.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        sides = len(approx)
        area = cv2.contourArea(c)
        if (area > 500):
            ratio = math.sqrt(area)/peri
            slack = 0.05
            cv2.drawContours(img, c, -1, (255, 0, 0), 7)
            # detects triangle
            if(ratio>0.14433756729-slack and ratio < 0.14433756729+slack and sides == 3):
                cv2.putText(img, "triangle", [
                    x + w - 100, y + 60], cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            # detects square // problem: triangle is detected but square is also recognized for some reason idk, might have to implement anti triangle formula idk
            elif (ratio>0.25-slack and ratio < 0.25+slack and sides == 4):
                cv2.putText(img, "square", [
                    x + w - 100, y + 60], cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(img, str(sides), [
                    x + w - 100, y + 60], cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    imgStack = stackImages(0.3,([imgDil],[img]))
    cv2.imshow('img', imgStack)
    k = cv2.waitKey(27) & 0xff
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows