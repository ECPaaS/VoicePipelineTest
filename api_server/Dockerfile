FROM bitnami/python:3.11.11

RUN apt-get update -y && apt-get install -y ffmpeg

COPY . /api
WORKDIR /api

RUN sed -i '16 s/$/os.environ["COQUI_TOS_AGREED"] = "1"/' web_tts_asr.py

RUN pip install -r requirements.txt 
RUN pip install uvicorn 
RUN pip uninstall -y pydantic 
RUN pip install --no-cache-dir pydantic
RUN pip install fastapi torch



ENTRYPOINT ["python", "-m", "uvicorn", "web_tts_asr:app", "--host", "0.0.0.0", "--port","8000"]
