from contextlib import asynccontextmanager
from pydantic import BaseModel
import torch
from TTS.api import TTS
from TTS.utils.radam import RAdam
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import whisper
import os
import uuid
import base64
import pdb
# 載入 Transformers 套件（假設 Mistral-Nemo 模型在 Hugging Face 上有對應實作）
from transformers import AutoTokenizer, AutoModelForCausalLM
from ollama import AsyncClient




# 設定運算設備（GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SynthesisRequest(BaseModel):
    text: str

# 定義 LLM 請求結構
class LLMRequest(BaseModel):
    prompt: str
    max_length: int = 250  # 可依需求調整生成的最大 token 長度

# 使用安全上下文管理器，允許 XttsAudioConfig 全域被反序列化
tts_model="tts_models/multilingual/multi-dataset/xtts_v2"
ollama_url = "http://ollama.default.svc:11434"

@asynccontextmanager
async def lifespan(app: FastAPI):
   
    # 應用啟動時載入 TTS 模型
    try:
        # 載入 Whisper 模型（可根據需要選擇 "tiny", "base", "small", "medium", "large"）
        app.state.asr_model = whisper.load_model("medium")
        print("Whisper model loaded.")
    except Exception as e:
        print("Error loading Whisper model:", e)
        raise e
    try:
        with torch.serialization.safe_globals([RAdam]):
            app.state.tts_model = TTS(tts_model).to(device)
        print("TTS model loaded.")
        
    except Exception as e:
        print("Error loading TTS model:", e)
        raise e


    yield
    # 應用關閉時，可清理資源
    if hasattr(app.state, "tts_model"):
        del app.state.tts_model
    print("TTS model cleaned up.")
    if hasattr(app.state, "asr_model"):
        del app.state.asr_model
    print("ASR model cleaned up.")

app = FastAPI(lifespan=lifespan)


@app.post("/tts/")
async def synthesize(request: SynthesisRequest):

    # 產生一個隨機的 UUID 並轉成字串
    uuid_str = str(uuid.uuid4())
    out_path = f"tts_{uuid_str}.wav"    

    try:
        app.state.tts_model.tts_to_file(text=request.text,speaker_wav="really.wav",language="zh-cn", file_path=out_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {e}")

    if os.path.exists(out_path):
        try:
            with open(out_path, "rb") as audio_file:
                audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error encoding file: {e}")

        return {"audio_data": audio_base64}    

    else:
        raise HTTPException(status_code=500, detail="Output file not found.")
    


@app.post("/asr/")
async def transcribe(audio: UploadFile = File(...), llm_model: str = "mistral-nemo"):
    """
    接收上傳的音訊檔案，使用 Whisper 模型進行轉錄
    """
    # 儲存上傳檔案到暫存路徑（以避免直接在記憶體中處理大型檔案）
    try:
        # 產生唯一檔名
        file_extension = os.path.splitext(audio.filename)[1] or ".mp3"
        temp_filename = f"temp_{uuid.uuid4().hex}{file_extension}"
        with open(temp_filename, "wb") as f:
            content = await audio.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    try:
        # 使用模型進行轉錄
        result = app.state.asr_model.transcribe(temp_filename)
        transcription = result.get("text", "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR transcription failed: {e}")
    finally:
        # 刪除暫存檔案
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    return JSONResponse(content={"text": transcription})



@app.post("/asr-llm-tts/")
async def transcribe(audio: UploadFile = File(...), llm_model: str = "mistral-nemo"):
    """
    1. 接收上傳的音訊檔案，使用 Whisper 模型進行轉錄
    2. 將轉錄結果傳到ollama api處理
    3. 將ollama模型回傳的結果傳到TSS轉成speech
    """
    
    # 1. 接收上傳的音訊檔案，使用 Whisper 模型進行轉錄
    # 儲存上傳檔案到暫存路徑（以避免直接在記憶體中處理大型檔案）
    try:  
        
        file_extension = os.path.splitext(audio.filename)[1] or ".mp3"
        temp_filename = f"temp_{uuid.uuid4().hex}{file_extension}"
        with open(temp_filename, "wb") as f:
            content = await audio.read()
            f.write(content)        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")
    
    try:
        # pdb.set_trace()
        result = app.state.asr_model.transcribe(temp_filename)
        transcription = result.get("text", "").strip()
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR transcription failed: {e}")
    
    # 2. 將轉錄結果傳到ollama api處理
    try:
        # pdb.set_trace()
        ollama_client = AsyncClient(host = ollama_url)
        response = await ollama_client.generate(model = llm_model, prompt = transcription, stream = False)
        ollama_response = response.get('response', 'No response found')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama chat failed: {e}")
    
    # 3. 將ollama模型回傳的結果傳到TSS轉成speech
    uuid_str = str(uuid.uuid4())
    out_path = f"tts_{uuid_str}.wav"    

    try:
        app.state.tts_model.tts_to_file(text=ollama_response,speaker_wav="really.wav",language="zh-cn", file_path=out_path)
        return {"status": "OK"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {e}")

    # if os.path.exists(out_path):
    #     try:
    #         with open(out_path, "rb") as audio_file:
    #             audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
    #     except Exception as e:
    #         raise HTTPException(status_code=500, detail=f"Error encoding file: {e}")

    #     return {"audio_data": audio_base64}    

    # else:
    #     raise HTTPException(status_code=500, detail="Output file not found.")


# 若要在本地測試，請使用以下指令啟動服務:
# uvicorn web_tts_asr:app --host 0.0.0.0 --port 8000


