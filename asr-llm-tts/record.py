import pyaudio,wave
import numpy as np
import requests
import time

ASR_LLM_TTS_URL = "http://a21d398aa95ed4f9f9bce28be0c05fa6-320423173.ap-northeast-1.elb.amazonaws.com:8000"
LLM_URL = "http://ac8322d6a0cb94430ad8c2f98837b5be-1580160200.ap-northeast-1.elb.amazonaws.com:11434/api/generate"
audio_file_path = "test.wav"

def listen():
    temp = 20
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 2
    WAVE_OUTPUT_FILENAME = 'test.wav'

    mindb=2000    #最小声音，大于则开始录音，否则结束
    delayTime=2  #小声1.3秒后自动终止
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    #snowboydecoder.play_audio_file()
    print("开始!计时")

    frames = []
    flag = False            # 开始录音节点
    stat = True				#判断是否继续录音
    stat2 = False			#判断声音小了

    tempnum = 0				#tempnum、tempnum2、tempnum3为时间
    tempnum2 = 0

    while stat:
        data = stream.read(CHUNK,exception_on_overflow = False)
        frames.append(data)
        audio_data = np.frombuffer(data, dtype=np.short)
        temp = np.max(audio_data)
        if temp > mindb and flag==False:
            flag =True
            print("开始录音")
            tempnum2=tempnum

        if flag:

            if(temp < mindb and stat2==False):
                stat2 = True
                tempnum2 = tempnum
                print("声音小，且之前是是大的或刚开始，记录当前点")
            if(temp > mindb):
                stat2 =False
                tempnum2 = tempnum
                #刷新

            if(tempnum > tempnum2 + delayTime*15 and stat2==True):
                print("间隔%.2lfs后开始检测是否还是小声"%delayTime)
                if(stat2 and temp < mindb):
                    stat = False
                    #还是小声，则stat=True
                    print("小声！")
                else:
                    stat2 = False
                    print("大声！")


        print(str(temp)  +  "      " +  str(tempnum))
        tempnum = tempnum + 1
        if tempnum > 150:				#超时直接退出
            stat = False
    print("录音结束")

    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def play():
    chunk = 1024  
    wf = wave.open(r"test.wav", 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(),
                    rate=wf.getframerate(), output=True)
   
    data = wf.readframes(chunk)  # 读取数据
    #print(data)
    while data != b'':  # 播放
        stream.write(data)
        data = wf.readframes(chunk)
        #print('while循环中！')
        #print(data)
    stream.stop_stream()  # 停止数据流
    stream.close()
    p.terminate()  # 关闭 PyAudio

def main():
    try:       
        listen()

        for i in range (11):
        #-------------------------------------------------Combine-----------------------------------------------
            start_time = time.time()

            with open(audio_file_path, 'rb') as audio_file:
                files = {'audio': (audio_file_path, audio_file, 'audio/wav')}
                response = requests.post(f"{ASR_LLM_TTS_URL}/asr-llm-tts", files=files)

                if response.status_code != 200:
                    print(f"Failed to upload file: {response.status_code}")      
            end_time = time.time()
            print(f"{i}: Calling ASR-TTS and LLM api takes: {end_time - start_time:.2f} seconds")
            time.sleep(8)
        

        #---------------------------------------------------Separate----------------------------------------------
            # #ASR
            # asr_start_time = time.time()
            # with open(audio_file_path, 'rb') as audio_file:
            #     files = {'audio': (audio_file_path, audio_file, 'audio/wav')}
            #     asr_response = requests.post(f"{ASR_LLM_TTS_URL}/asr", files=files)

            # if asr_response.status_code == 200:
            #     print(asr_response.text)
            # else:
            #     print(f"Failed to upload file: {asr_response.status_code}")
                  
            # asr_end_time = time.time()
            # print(f"{i}: ASR takes: {asr_end_time - asr_start_time:.2f} seconds")

            # #LLM
            # llm_start_time = time.time()
            # data = {
            #     "model": "mistral-nemo",
            #     "prompt": asr_response.text,
            #     "stream": False,
            #     "options": {
            #         "num_predict": 50,
            #         "seed": 42
            #     }
            # }
            # llm_response = requests.post(LLM_URL, json=data)
            # if llm_response.status_code == 200:
            #     print(llm_response.json())
            # else:
            #     print(f"Request failed with status code: {llm_response.status_code}")
            # llm_end_time = time.time()
            # print(f"{i}: Ollama api takes: {llm_end_time - llm_start_time:.2f} seconds")

            # #TTS
            # tts_start_time = time.time()
            # data = {
            #     "text": llm_response.json()['response']
            # }
            # tts_response = requests.post(f"{ASR_LLM_TTS_URL}/tts", json=data)
            # if tts_response.status_code == 200:
            #     print(tts_response.json())
            # else:
            #     print(f"Request failed with status code: {tts_response.status_code}")
            # tts_end_time = time.time()
            # print(f"{i}: TTS takes: {tts_end_time - tts_start_time:.2f} seconds")
            

            # time.sleep(8)

    except Exception as e:
        print("Error loading TTS model:", e)
        raise e

        

if __name__ == "__main__":
    main()