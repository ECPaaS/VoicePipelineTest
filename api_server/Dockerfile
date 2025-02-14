FROM bitnami/python:3.11.11
 
RUN apt-get update -y && apt-get install -y ffmpeg
 
COPY . /api
 
WORKDIR /api

RUN pip install --no-cache-dir -r requirements.txt 
RUN pip install --no-cache-dir uvicorn pydantic fastapi torch

RUN sed -i '16 s/$/os.environ["COQUI_TOS_AGREED"] = "1"/' download_model.py
RUN sed -i '16 s/$/os.environ["COQUI_TOS_AGREED"] = "1"/' web_tts_asr.py
RUN sed -i '51 s/.$/, weights_only=False)/' /opt/bitnami/python/lib/python3.11/site-packages/TTS/utils/io.py
RUN sed -i '54 s/.$/, weights_only=False)/' /opt/bitnami/python/lib/python3.11/site-packages/TTS/utils/io.py

RUN python download_model.py
 
ENTRYPOINT ["python", "-m", "uvicorn", "web_tts_asr:app", "--host", "0.0.0.0", "--port","8000"]

