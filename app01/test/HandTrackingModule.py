import cv2
import mediapipe as mp
import time
import math

#求解二维向量的角度
def vector_2d_angle(v1,v2):
    v1_x=v1[0]
    v1_y=v1[1]
    v2_x=v2[0]
    v2_y=v2[1]
    try:
        angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ =65535.
    if angle_ > 180.:
        angle_ = 65535.
    return angle_


#获取对应手相关向量的二维角度,根据角度确定手势
def hand_angle(hand_):
    angle_list = []

    #---------------------------- thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)

    #---------------------------- index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
        )
    angle_list.append(angle_)

    #---------------------------- middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
        ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
        )
    angle_list.append(angle_)

    #---------------------------- ring 无名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
        ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
        )
    angle_list.append(angle_)

    #---------------------------- pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
        ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
        )
    angle_list.append(angle_)

    return angle_list


# 二维约束的方法定义手势
def h_gesture(angle_list):
    thr_angle = 65.
    thr_angle_thumb = 53.
    thr_angle_s = 49.
    gesture_str = None

    if 65535. not in angle_list:
        if (angle_list[0]>thr_angle_thumb) and (angle_list[1]>thr_angle) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "zero"
        elif (angle_list[0]<thr_angle_s) and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]<thr_angle_s) and (angle_list[4]<thr_angle_s):
            gesture_str = "five"
        elif (angle_list[0]>thr_angle_thumb)  and (angle_list[1]<thr_angle_s) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "one"
        elif (angle_list[0]>thr_angle_thumb)  and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]<thr_angle_s) and (angle_list[4]<thr_angle_s):
            gesture_str = "four"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]>thr_angle) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "thumb"
    return gesture_str


def ges(hand_landmarks,image):
    hand_local = []
    for i in range(21):
        x = hand_landmarks.landmark[i].x*image.shape[1]
        y = hand_landmarks.landmark[i].y*image.shape[0]
        hand_local.append((x,y))
    if hand_local:
        angle_list = hand_angle(hand_local)
        gesture_str = h_gesture(angle_list)
    return gesture_str

class handDetector():
    def __init__(self, mode=False, maxHands=2,model_complexity = 0, min_detection_confidence=0.5, min_tracking_confidence=0.8):
        self.mode = mode
        self.maxHands = maxHands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity,self.min_detection_confidence, self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.mpDrawingStyles = mp.solutions.drawing_styles

    #手部检测
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # 获取检测结果中的左右手标签并打印
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    #输出手势标签
    def findLabel(self,img):
        self.dict_handnumber = {}
        if self.results.multi_hand_landmarks:
            if len(self.results.multi_handedness) == 2:  # 如果检测到两只手
                for i in range(len(self.results.multi_handedness)):
                    label = self.results.multi_handedness[i].classification[0].label  # 获得Label判断是哪几手
                    index = self.results.multi_handedness[i].classification[0].index  # 获取左右手的索引号
                    hand_landmarks = self.results.multi_hand_landmarks[index]  # 根据相应的索引号获取xyz值
                    self.mpDraw.draw_landmarks(
                                    img,
                                    hand_landmarks,
                                    self.mpHands.HAND_CONNECTIONS,
                                    self.mpDrawingStyles.get_default_hand_landmarks_style(),
                                    self.mpDrawingStyles.get_default_hand_connections_style())
                    gesresult = ges(hand_landmarks,img) # 传入21个关键点集合，返回手势标签
                    self.dict_handnumber[label] = gesresult # 与对应的手进行保存为字典
                    #print(label)
            elif len(self.results.multi_handedness) == 1:
                label = self.results.multi_handedness[0].classification[0].label  # 获得Label判断是哪几手
                hand_landmarks = self.results.multi_hand_landmarks[0]
                self.mpDraw.draw_landmarks(img,hand_landmarks,self.mpHands.HAND_CONNECTIONS,self.mpDrawingStyles.get_default_hand_landmarks_style(),self.mpDrawingStyles.get_default_hand_connections_style())
                gesresult = ges(hand_landmarks,img) # 传入21个关键点集合，返回手势标签
                self.dict_handnumber[label] = gesresult # 与对应的手进行保存为字典
                #print(label)
            else:
                hand_landmarks = self.results.multi_hand_landmarks[0]
        return self.dict_handnumber

    #手势标签文字
    def text(self,dict_handnumber):
        if len(dict_handnumber) == 2:  # 如果有两只手，则进入
            leftnumber = dict_handnumber['Right']  
            rightnumber = dict_handnumber['Left']
            s = 'Righthand :{0}\nLefthand :{1}'.format(rightnumber,leftnumber)  # 图像上的文字内容
        elif len(dict_handnumber) == 1 :  # 如果仅有一只手则进入
            labelvalue = list(dict_handnumber.keys())[0]  # 判断检测到的是哪只手
            if labelvalue == 'Left': 
                number = list(dict_handnumber.values())[0]
                s = 'Righthand:{0}\nLefthand:0'.format(number)
            else:  # 右手
                number = list(dict_handnumber.values())[0]
                s = 'Righthand:0\nLefthand:{0}'.format(number)
        else:# 如果没有检测到则只显示帧率
            s = 'Righthand:0\nLefthand:0'
        return s

    #分别获得两只手的坐标列表
    def findPosition(self, img, draw=True):
        self.lmListLeft = []
        self.lmListRight = []
        if self.results.multi_hand_landmarks:
            if len(self.results.multi_handedness) == 2:  # 如果检测到两只手
                for i in range(len(self.results.multi_handedness)):
                    label = self.results.multi_handedness[i].classification[0].label  # 获得Label判断是哪几手
                    if label=='Right':
                        for handLms in self.results.multi_hand_landmarks:
                            for id, lm in enumerate(handLms.landmark):
                                h, w, c = img.shape
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                # print(id, cx, cy)
                                self.lmListRight.append([id, cx, cy])
                                if draw:
                                    cv2.circle(img, (cx, cy), 12, (255, 0, 255), cv2.FILLED)
                    elif label=='Left':
                        for handLms in self.results.multi_hand_landmarks:
                            for id, lm in enumerate(handLms.landmark):
                                h, w, c = img.shape
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                # print(id, cx, cy)
                                self.lmListLeft.append([id, cx, cy])
                                if draw:
                                    cv2.circle(img, (cx, cy), 12, (255, 0, 255), cv2.FILLED)
            elif len(self.results.multi_handedness) == 1:
                label = self.results.multi_handedness[0].classification[0].label  # 获得Label判断是哪几手
                if label=='Right':
                    for handLms in self.results.multi_hand_landmarks:
                        for id, lm in enumerate(handLms.landmark):
                            h, w, c = img.shape
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            # print(id, cx, cy)
                            self.lmListRight.append([id, cx, cy])
                            if draw:
                                cv2.circle(img, (cx, cy), 12, (255, 0, 255), cv2.FILLED)
                elif label=='Left':
                    for handLms in self.results.multi_hand_landmarks:
                        for id, lm in enumerate(handLms.landmark):
                            h, w, c = img.shape
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            # print(id, cx, cy)
                            self.lmListLeft.append([id, cx, cy])
                            if draw:
                                cv2.circle(img, (cx, cy), 12, (255, 0, 255), cv2.FILLED) 
        return self.lmListLeft,self.lmListRight




def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    
    y0,dy = 50,25  # 文字放置初始坐标
    while True:
        s='Righthand:0\nLefthand:0'
        success, img = cap.read()
        img = detector.findHands(img)        # 检测手势并画上骨架信息

        lmListLeft,lmListRight = detector.findPosition(img)  # 获取得到坐标点的列表
        if len(lmListLeft) != 0:
            print(detector.findLabel(img)['Left'])
            print(lmListLeft[4])
        if len(lmListRight) != 0:
            print(detector.findLabel(img)['Right'])
            print(lmListRight[4]) 

        dicthand=detector.findLabel(img)
        if len(dicthand) != 0:
            s=detector.text(dicthand)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        img = cv2.flip(img,1) # 反转图像
        test='fps:' + str(int(fps))+'\n'+s
        for i ,a in enumerate(test.split('\n')): # 根据\n来竖向排列文字
            y = y0 + i*dy
            cv2.putText(img, a, (50,y), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 3)
        cv2.imshow('Image', img)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()
