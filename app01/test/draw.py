import cv2
# import HandTrackingModule as htm
from app01.test import HandTrackingModule as htm
import os
import numpy as np

color = [255, 0, 0]

# 画笔粗细
brushThickness = 15
eraserThickness = 40

cap = cv2.VideoCapture(0)  # 打开摄像头
# cap.set(3, 1280)
# cap.set(4, 720)

detector = htm.handDetector()

xl, yl = 0, 0
xr, yr = 0, 0

imgCanvas = np.ones((600, 450, 3), np.uint8)  # 画布
imgCanvas.fill(255)  # 背景设置为白色

imgInv = np.zeros((480, 640, 3), np.uint8)
imgInv.fill(255)

while True:

    success, img = cap.read()
    if img is None:
        break
    img = cv2.flip(img, 1)  # 翻转
    img.flags.writeable = True  # 将图像矩阵修改为读写模式
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 将图像变回BGR形式

    img = detector.findHands(img)
    lmListLeft, lmListRight = detector.findPosition(img, draw=True)
    dicthand = detector.findLabel(img)

    # 橡皮擦
    if len(lmListLeft) != 0:
        xl0, yl0 = lmListLeft[8][1:]
        dicthand = detector.findLabel(img)
        leftlabel = dicthand['Left']
        if leftlabel == "one":
            cv2.circle(img, (xl0, yl0), 15, [255, 255, 255], cv2.FILLED)
            if xl == 0 and yl == 0:
                xl, yl = xl0, yl0
            cv2.line(imgInv, (xl, yl), (xl0, yl0), [255, 255, 255], eraserThickness)
            cv2.line(imgCanvas, (int(xl * 1.5), int(yl * 1.5)), (int(xl0 * 1.5), int(yl0 * 1.5)), [255, 255, 255],
                     int(eraserThickness * 1.5))
        xl, yl = xl0, yl0

    # 画笔
    if len(lmListRight) != 0:
        xr0, yr0 = lmListRight[8][1:]
        dicthand = detector.findLabel(img)
        rightlabel = dicthand['Right']
        if rightlabel == "one":
            cv2.circle(img, (xr0, yr0), 15, [255, 0, 0], cv2.FILLED)
            if xr == 0 and yr == 0:
                xr, yr = xr0, yr0
            cv2.line(imgInv, (xr, yr), (xr0, yr0), color, brushThickness)
            cv2.line(imgCanvas, (int(xr * 1.5), int(yr * 1.5)), (int(xr0 * 1.5), int(yr0 * 1.5)), color,
                     int(brushThickness * 1.5))
        xr, yr = xr0, yr0

    # 实时显示画笔轨迹的实现
    imgGray = cv2.cvtColor(imgInv, cv2.COLOR_BGR2GRAY)
    # _, imgand = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    _, imgand = cv2.threshold(imgGray, 254, 255, cv2.THRESH_BINARY_INV)
    imgand = cv2.cvtColor(imgand, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_or(img, imgand)
    img = cv2.bitwise_and(img, imgInv)

    # print(imgInv.size)
    # print(img.size)
    # print(imgCanvas.size)


    cv2.imshow("Image", img)  # 摄像头显示 -
    cv2.imshow("Canvas", imgCanvas)  # 画布显示


    # cv2.imshow("Inv", imgInv)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
