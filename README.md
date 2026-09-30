# HD Voice Stack

一个面向业务服务器调用的低成本语音服务：

- ASR：`Qwen/Qwen3-ASR-0.6B-hf`，支持请求级上下文热词。
- TTS：`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`，支持五种中文音色。
- Tesla T4 上 TTS 默认使用 FP32，避免该卡不支持原生 BF16、FP16 采样可能产生 NaN；可通过 `TTS_DTYPE` 调整。
- 浏览器只访问业务服务；业务服务通过带 Bearer Token 的 voice gateway 调用模型 worker。
- ASR 为整段上传/整段返回；TTS 同时支持整段 WAV 和按句生成、按句播放的伪流式响应。
- 限制 60 秒录音、20MB 上传和 500 字 TTS 文本；伪流式默认每段最多 60 字。

## 进程与端口

| 进程 | 地址 | 作用 |
| --- | --- | --- |
| business-api | `0.0.0.0:8443` HTTPS | 测试页和业务代理 API |
| voice-gateway | `127.0.0.1:18000` | 鉴权、限流、GPU 请求串行化 |
| asr-worker | `127.0.0.1:18001` | 语音识别、热词 prompt |
| tts-worker | `127.0.0.1:18002` | WAV 语音合成 |

## 本地检查

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/ruff check .
.venv/bin/pytest
```

模型 worker 的测试不会在本地加载模型。实际服务器使用三个 Conda 环境，避免 ASR 所需的 Transformers 5.13 与 `qwen-tts` 固定的 Transformers 4.57.3 冲突。

TTS 环境先安装运行时依赖，再以 `--no-deps` 安装 `qwen-tts`，避免引入仅用于官方 Gradio 演示页的依赖并与服务端固定的 FastAPI 版本发生解析回溯：

```bash
/data/miniconda3/envs/voice-tts/bin/pip install -r requirements-tts.txt
/data/miniconda3/envs/voice-tts/bin/pip install --no-deps qwen-tts==0.1.1
```

## Voice API

语音识别：

```bash
curl -H "Authorization: Bearer $VOICE_API_TOKEN" \
  -F 'file=@sample.wav' \
  -F 'language=Chinese' \
  -F 'hotwords=["元气森林","东方树叶"]' \
  http://127.0.0.1:18000/v1/audio/transcriptions
```

语音合成：

```bash
curl -H "Authorization: Bearer $VOICE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"input":"欢迎使用语音服务","voice":"Vivian","language":"Chinese","instruct":""}' \
  http://127.0.0.1:18000/v1/audio/speech -o speech.wav
```

按句伪流式语音合成：

```bash
curl --no-buffer \
  -H "Authorization: Bearer $VOICE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"input":"欢迎使用语音服务。请问您想喝什么？","voice":"Vivian","language":"Chinese","instruct":""}' \
  http://127.0.0.1:18000/v1/audio/speech/stream
```

流式接口返回 `application/x-ndjson`。事件依次为 `start`、一个或多个 `audio`、`done`；
每个 `audio` 事件携带一条独立 WAV 的 Base64 内容。流开始后的推理错误以 `error` 事件返回。
`TTS_STREAM_MAX_SENTENCE_CHARS` 控制无自然句末时的最大分段长度，范围为 20–200。

## 微调数据

每行一个 JSON 对象，音频必须为 16kHz 单声道 WAV：

```json
{"audio":"/data/audio/001.wav","text":"language Chinese<asr_text>我要一瓶元气森林","prompt":"Vocabulary: 元气森林.","speaker_id":"speaker-01","product":"元气森林"}
```

执行顺序：

```bash
python -m training.prepare_dataset --input raw.jsonl --output train.jsonl --audio-dir normalized
python -m training.validate_dataset train.jsonl
python -m training.train_lora --train train.jsonl --eval eval.jsonl --output checkpoints/drinks-v1
python -m training.evaluate_asr --manifest eval.jsonl --adapter checkpoints/drinks-v1/final-adapter --output report.json
```

只有独立测试集的领域词召回率达到目标，且通用集 CER 退化不超过 2 个百分点时，才设置 `ASR_ADAPTER_PATH` 并重启 ASR worker。

## 服务器部署

部署根目录为 `/data/voice-stack`。`deploy/voice.env.example` 是环境变量模板，`deploy/systemd` 包含四个服务和总 target。证书通过以下命令生成：

```bash
bash deploy/create_self_signed_cert.sh /data/voice-stack/config/tls 106.14.11.151
```

模型、环境、证书、token、数据和 checkpoint 均不提交到仓库。服务器只需在阿里云安全组放行 TCP 8443；18000–18002 不应对公网开放。
