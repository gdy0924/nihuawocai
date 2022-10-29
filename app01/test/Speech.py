import logging
import speech_recognition as sr


from aip import AipSpeech

""" 你的 APPID AK SK """
APP_ID = '27890564'
API_KEY = 'DI81G6bvUTBVUGgDvzWK7Y57'
SECRET_KEY = 'InNxrgQQGKaGeIvPRdG9KPGZzF9pxMEV'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

# 读取文件



def get_file_content(file_path):
    with open(file_path, 'rb') as fp:
         return fp.read()

class Speech():
    def __init__(self, APP_ID = '27890564', API_KEY = 'DI81G6bvUTBVUGgDvzWK7Y57',SECRET_KEY = 'InNxrgQQGKaGeIvPRdG9KPGZzF9pxMEV'):
        self.APP_ID = APP_ID
        self.API_KEY = API_KEY
        self.SECRET_KEY = SECRET_KEY
        self.client=AipSpeech(APP_ID, API_KEY, SECRET_KEY)

    def recognition(self,wav_num):
        result = self.client.asr(get_file_content(f"00{wav_num}.wav"), 'wav', 16000, {
             'dev_pid': 1537,  # 默认1537（普通话 输入法模型），dev_pid参数见本节开头的表格
        })
        if result["err_msg"] == "success.":
            return result['result']
        else:
            return -1
            


def main():
    speechr=Speech()
    wav_num = 0
    logging.basicConfig(level=logging.INFO)
    while True:
        r = sr.Recognizer()
        #启用麦克风
        mic = sr.Microphone()
        logging.info('录音中...')
        with mic as source:
            #降噪
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
        with open(f"00{wav_num}.wav", "wb") as f:
            #将麦克风录到的声音保存为wav文件
            f.write(audio.get_wav_data(convert_rate=16000))
        logging.info('录音结束，识别中...')
        # 识别本地文件
        r=speechr.recognition(wav_num)
        if r==-1:
            print("内容获取失败，退出语音识别")
            break
        else:
            print(r[0])
            #print(type(r[0]))
        wav_num += 1

if __name__ == "__main__":
    main()
