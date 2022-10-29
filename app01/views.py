from threading import Thread
from django.http import StreamingHttpResponse
from django.shortcuts import render, HttpResponse

import cv2
import numpy as np

from app01.test.main import draw, s
from app01.test import HandTrackingModule as htm


# 开始
def index(request):
    return render(request, 'index.html')


# 主备页面
def add_room(request):
    return render(request, 'addToRoom.html')


# 准备页面
def ready(request):
    return render(request, 'ready.html')


# 开始页面
def start(request):
    return render(request, 'start.html')


color = [0, 0, 255]
# color = [255, 192, 203]
allow = True


# 画布---》图片----》转换为byte类型的，存储在迭代器中
def gen_display_2(imgCanvas):
    """
    视频流生成器功能。
    """
    # while True:
    # 将图片进行解码
    ret, frame = cv2.imencode('.jpeg', imgCanvas)
    if ret:
        # print('ret', ret)
        # 转换为byte类型的，存储在迭代器中
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')


# 定义一个变量，解决一个接口只能返回一种视频流
t_yield = ''


# 视频---》图片----》转换为byte类型的，存储在迭代器中
def gen_display(cap):
    # 画笔粗细
    brushThickness = 15
    eraserThickness = 40
    # 打开摄像头

    detector = htm.handDetector()
    xl, yl = 0, 0
    xr, yr = 0, 0

    imgCanvas = np.ones((720, 960, 3), np.uint8)  # 画布
    imgCanvas.fill(255)  # 背景设置为白色

    imgInv = np.zeros((480, 640, 3), np.uint8)
    imgInv.fill(255)
    while True:
        # 读取图片
        success, img = cap.read()
        if success:
            # 进行处理
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
                    cv2.circle(img, (xl0, yl0), 15, [255, 0, 0], cv2.FILLED)
                    if xl == 0 and yl == 0:
                        xl, yl = xl0, yl0
                    cv2.line(imgInv, (xl, yl), (xl0, yl0), [255, 255, 255], eraserThickness)
                    cv2.line(imgCanvas, (int(xl * 1.5), int(yl * 1.5)), (int(xl0 * 1.5), int(yl0 * 1.5)),
                             [255, 255, 255],
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

            # 将图片进行解码
            ret, frame = cv2.imencode('.jpeg', img)
            # 画布也解析
            # 保证保护与视频同步即可
            # cv2.imshow("Canvascccccccc", imgCanvas)  # 画布显示
            # 转成全局变量，解决只能返回一种视频的问题
            global t_yield
            t_yield = gen_display_2(imgCanvas)
            # 将视频流  # 转换为byte类型的，存储在迭代器中
            if ret:
                # print('ret', ret)
                # 转换为byte类型的，存储在迭代器中
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')


# 视频流函数
def draw2(request):
    cap = cv2.VideoCapture(0)  # 打开摄像头
    # 开启线程
    t2 = Thread(target=s)
    t2.start()
    # 返回视频流，本质：就是一个图片，由于不停的向前端发送图片，产生动态效果
    return StreamingHttpResponse(gen_display(cap), content_type='multipart/x-mixed-replace; boundary=frame')


# 画布流函数
def draw3(request):
    # 把画布返回
    return StreamingHttpResponse(t_yield, content_type='multipart/x-mixed-replace; boundary=frame')
