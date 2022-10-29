draw.py：画图功能，显示摄像头拍摄内容和画布
HandTrackingModule.py：手部关键点检测和手势识别
Speech.py：语音识别功能，调用百度api（https://console.bce.baidu.com/ai/?_=1665978155638&fromai=1#/ai/speech/overview/index），短语音识别接口
	    修改文件中APP_ID、API_KEY和SECRET_KEY即可
main.py：多线程调用画图和语音功能，目前只实现了三个画笔颜色的切换（红色、蓝色、绿色），还在进行优化和添加新的控制语音。