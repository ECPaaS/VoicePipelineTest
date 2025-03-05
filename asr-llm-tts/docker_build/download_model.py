# web_tts.py
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
# 載入 Transformers 套件（假設 Mistral-Nemo 模型在 Hugging Face 上有對應實作）
from transformers import AutoTokenizer, AutoModelForCausalLM



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
llm_model = "mistralai/Mistral-7B-v0.1"  # 模型名稱,需要在 Hugging Face 上有對應存取權限


# 應用啟動時載入 TTS 模型
try:
    # 載入 Whisper 模型（可根據需要選擇 "tiny", "base", "small", "medium", "large"）
    whisper.load_model("medium")
    print("Whisper model loaded.")
except Exception as e:
    print("Error loading Whisper model:", e)
    raise e
try:
    with torch.serialization.safe_globals([RAdam]):
        TTS(tts_model).to(device)
    print("TTS model loaded.")
except Exception as e:
    print("Error loading TTS model:", e)
    raise e
