# ASR-LLM-TTS

## Step 1: Build asr-tts image
```
cd docker_build
sudo docker build -f Dockerfile -t test/asr-llm-tts:1.0 .
```

## Step 2: Deploy ollama and asr-tts pod
```
cd ..
kubectl apply -f ollama-deploy-svc.yml
kubectl apply -f deploy_tts_asr.yml
```

## Step 3: Run the recording file record.py on local pc
- Edit the URL and port for `ASR_LLM_TTS_URL` and `LLM_URL`
- In the `main()` function, there are two ways to call ASR-TTS and LLM APIs. Uncomment to run them together(combine) or separately

```
python record.py
```
