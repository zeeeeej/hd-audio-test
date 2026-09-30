# HD Voice Stack

一个面向业务服务器调用的低成本语音服务：

- ASR：`Qwen/Qwen3-ASR-0.6B-hf`，支持请求级上下文热词。
- TTS：`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`，支持五种中文音色。
- Tesla T4 上 TTS 默认使用 FP32，避免该卡不支持原生 BF16、FP16 采样可能产生 NaN；可通过 `TTS_DTYPE` 调整。
- 浏览器只访问业务服务；业务服务通过带 Bearer Token 的 voice gateway 调用模型 worker。
- 网页提供冰柜饮料导购：录音 → 业务服务 → ASR → DeepSeek → 按句 TTS → 自动播放。
- 商品、价格、库存和购物规则使用本地演示资料，不执行下单、付款或开柜。
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

## 冰柜语音导购

业务服务统一编排 ASR、DeepSeek 和 TTS。浏览器不接触 DeepSeek 密钥或 voice token。
首次点击按钮模拟物理触发，每轮点击开始录音、点击停止后自动上传；最长 60 秒自动提交。
回答生成和播放期间不能继续录音，可点击“结束会话”停止请求及播放。暂不实现唤醒词、
静音检测、自动连续对话或语音打断。原有独立 ASR/TTS 入口在页面“Voice 服务测试”中。

本地启动业务页面（需要另行启动或连接可用的 voice gateway）：

```bash
# 项目根目录 .key 放置纯文本 DeepSeek key，已被 .gitignore 忽略。
# 也可由进程环境传入 DEEPSEEK_API_KEY，或设置 DEEPSEEK_KEY_FILE。
export VOICE_BASE_URL=http://127.0.0.1:18000
# VOICE_API_TOKEN 需与网关一致，请通过本地环境安全配置。
.venv/bin/uvicorn src.business.main:app --host 127.0.0.1 --port 8443 --workers 1
```

访问 `http://localhost:8443`。远程访问必须用 HTTPS 才能获取麦克风权限，现有 systemd
服务已配置 HTTPS。密钥读取优先级为非空 `DEEPSEEK_API_KEY` → `DEEPSEEK_KEY_FILE`
指定的文件 → 项目根目录 `.key`（仅未指定文件路径时）。不自动加载 `.env`。
服务启动不验证在线模型；缺少密钥时建立导购会话返回 503，独立语音测试仍可用。

默认使用 `deepseek-flash`，关闭思考模式，温度 0.3，JSON 输出最多 256 tokens，
整次模型调用最多 20 秒。先检查完整回答，再请求 TTS 按句生成；没有逐 token 播报。
历史 assistant 消息也使用 JSON 格式，仅保存已展示的短回答及意图，避免多轮 JSON 模式格式冲突。

| 限制 | 默认值 | 环境变量 |
| --- | --- | --- |
| 最大问答轮数 | 6 | `DIALOGUE_MAX_TURNS` |
| 最近完整问答历史 | 3 轮 | `DIALOGUE_HISTORY_TURNS` |
| 空闲结束期限 | 60 秒 | `DIALOGUE_IDLE_SECONDS` |
| 活跃会话上限 | 100 | `DIALOGUE_MAX_SESSIONS` |
| 单次识别文字上限 | 500 字符 | `DIALOGUE_MAX_INPUT_CHARS` |
| 全部消息内容预算 | 6,000 字符 | `DIALOGUE_CONTEXT_CHARS` |
| 回答总长／单句长／句数 | 60／30／2 | `DIALOGUE_REPLY_CHARS`／`DIALOGUE_SENTENCE_CHARS`／`DIALOGUE_REPLY_SENTENCES` |
| 首句优先目标（非硬上限） | 12 字符 | `DIALOGUE_FIRST_SENTENCE_TARGET_CHARS` |

首句优先直接回答核心问题，例如问价格答“演示价六元。”，追问库存答“演示库存八瓶。”。
第一句先给一个完整有效的信息点，其余必要内容放第二句，保持整体含义准确完整。
单个问题能在首句答完时只答一句；需要分句时，第二句可补全同一问题，也可回答用户同时询问的信息。
列举多款商品时首句最多介绍一种，例如“还有冰露饮用水。也有元气森林白桃味气泡水。”。
不主动添加未询问的规格、库存、缺货商品或例行追问；不得拆散金额与单位、否定条件及商品对应关系。
保留必要商品名和限定条件。首句目标范围为 1 至 `DIALOGUE_SENTENCE_CHARS`，允许必要信息超出目标，
仍受每句 30 字等现有硬限制约束；不会为压缩首句额外请求模型，也不会在 12 字处截断答案。
这项优化通过缩短首句待合成内容来减少等待，TTS 仍按完整句生成，实际首段耗时需按部署环境实测。

2026-09-30 在同一服务器、同一录音和 Vivian 音色下交替测试修改前后规则，各场景各版本 3 次：
问价首段耗时中位数从 11.39 秒降至 6.83 秒，降低约 40%；库存追问约 6.95 秒，基本持平。
统计从服务端发起 ASR 到收到首段音频，不含浏览器上传及播放时间。TTS 单次合成有明显波动，
这些小样本结果不代表固定延迟保证。对比测试通过在内存中执行候选规则完成。
首句规则已于同日发布，线上问价回答“演示价六元。”、库存追问回答“演示库存八瓶。”，
两轮真实语音验证通过。完整样本、代码校验值和发布验收见
[首句优化实测记录](docs/FIRST_SENTENCE_BENCHMARK.json)。

字符预算包含系统规则、演示资料、历史和当前问题，不等同于模型 token 数。超限时删除最早的
完整问答，系统资料过大则报告配置错误。回复只保留满足限制的连续完整句，无可用句时使用固定短提示。
购物范围和事实依据通过系统规则及模型意图分类约束，无关意图统一返回购物引导；这不等同于形式化的
事实校验，接入真实业务前仍需使用实际商品和场景问题做评估。

会话保存在单进程内存中，必须保持 `--workers 1`；服务重启清空历史，后续多进程部署需要共享会话存储。
处理中暂停空闲判断；录音开始通知提供“录音上限 + 5 秒”的有限保护，防止录音期间误过期。
模型回答发布后计入轮次，即使后续 TTS 失败也保留该轮；空识别、ASR 或模型失败不计轮次。
达到轮次上限或告别时仍播完本轮回答，然后要求重新开始。页面结束会话时取消业务侧等待和后续调用；
已经在模型 worker 内启动的 GPU 推理可能继续到当前句结束。

演示资料位于 `src/business/retail_catalog.json`，可用 `RETAIL_CATALOG_FILE` 指向其他本地 JSON。
每个商品包含 `id`、`name`、`size`、两位小数字符串 `price_yuan`、非负整数 `stock` 和 `tags`；
顶层包含 `mode: "demo"`、`notice`、`products` 和 `shopping_rules`。商品名与 `DEFAULT_HOTWORDS`
合并去重后传给 ASR。会话开始时固定资料快照，修改配置后新会话生效。真实业务可替换 `CatalogProvider`，
不能把现有演示价格和库存直接当作实时业务数据。

### 导购 API

| 方法与路径 | 作用 |
| --- | --- |
| `POST /api/dialogue/sessions` | 创建会话；返回 `session_id`、`limits`、`max_audio_seconds`、演示说明 |
| `POST /api/dialogue/sessions/{id}/recording` | 录音开始通知，刷新空闲时间并建立有限录音保护 |
| `POST /api/dialogue/sessions/{id}/turns` | multipart 上传 `file` 和可选 `voice`，返回完整链路事件流 |
| `DELETE /api/dialogue/sessions/{id}` | 幂等结束会话，取消未完成的编排并清理历史 |

一次成功请求的 NDJSON 事件顺序为 `stage(asr)` → `transcript` → `stage(deepseek)` →
`answer` → `stage(tts)` → 一个或多个 `audio` → `done`。
所有事件携带 `request_id`；`answer` 含 `text`、`intent`、`turn`、`remaining_turns`、
`session_ended`、`end_reason`；`audio` 含零起始 `index`、`text` 和独立 WAV 的 `audio_base64`。
`done` 含片段总数及 `timings`（ASR、模型、TTS、首段音频和总耗时，单位毫秒）。

流开始前返回标准 HTTP 错误；开始后失败返回 `error` 事件（含 `stage`、`error.code`、
`error.message`），不再发送 `done`。客户端必须把无 `done` 的断流视为失败。
同一会话并发提交返回 409，失效会话返回 410。TTS 失败时文字回答仍有效，可重新录音继续。
错误消息不回传模型原始响应或密钥；日志默认记录请求 ID、阶段、耗时及 token 用量，不记录完整对话。

### 导购验证与排查

自动测试使用模拟上游，不依赖模型或密钥。浏览器回归使用 Chrome 的模拟麦克风及模拟 API/WAV，
覆盖录音、追问、播放锁定、结束、录音上限、断流和会话超时。先按上文在本机启动业务服务，然后运行：

```bash
mkdir -p "$HOME/Documents/tmp/hd-deepseek-browser"
npm install --prefix "$HOME/Documents/tmp/hd-deepseek-browser" \
  --cache "$HOME/Documents/tmp/hd-deepseek-browser/npm-cache" playwright-core
TMPDIR="$HOME/Documents/tmp/hd-deepseek-browser" \
NODE_PATH="$HOME/Documents/tmp/hd-deepseek-browser/node_modules" \
APP_URL=http://127.0.0.1:8443 node tests/browser_dialogue.cjs
```

脚本默认使用 macOS Google Chrome；其他环境通过 `CHROME_BIN` 指定浏览器可执行文件。
真实端到端验收还需要可用的 ASR/TTS 服务：询问“推荐无糖茶”，追问“它多少钱”，再测试
未知商品、无关问题、告别和新顾客会话。麦克风不可用先检查 HTTPS 和浏览器权限；
`deepseek_not_configured` 检查密钥配置，`deepseek_invalid_response` 检查模型兼容性；
ASR/TTS 失败检查 `VOICE_BASE_URL`、网关 token 和 `/health/ready`。现有健康检查仍只探测语音服务，
不主动发起计费的 DeepSeek 请求。模型调用失败不自动重试，避免重复计费和延长等待。

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

服务器地址为 `106.14.11.151`，SSH 端口为 `21112`；网页 HTTPS 端口为 `8443`。
SSH 连接时使用 `ssh -p 21112 root@106.14.11.151`。本地连接信息保存在被 Git 忽略的
`.local.service` 中，文件权限为 `600`。

若公网 8443 尚不可访问，可通过 SSH 隧道访问网页：

```bash
ssh -p 21112 -N -L 28743:127.0.0.1:8443 root@106.14.11.151
```

保持该命令运行，浏览器打开 `https://localhost:28743`；当前服务使用自签名证书。
公网访问需要网络策略允许 TCP 8443。不要开放内部模型端口 18000–18002。

2026-09-30 已部署 `feature/deepseek-retail-dialogue` 工作区版本，真实三轮 ASR → DeepSeek → TTS
测试及浏览器自动播放验证通过。部署前备份与回滚脚本位于
`/data/voice-stack/backups/20260930-deepseek-retail/`；仅重启了 `voice-business`。
同日已发布首句优化，仅更新 `src/business/retail.py` 并重启业务服务；该次更新的备份和回滚脚本位于
`/data/voice-stack/backups/20260930-short-first-sentence/`。
随后已发布按完整信息点拆分首句的规则，备份位于
`/data/voice-stack/backups/20260930-semantic-first-sentence/`。线上“有没有其他饮料？”验证返回
“还有冰露饮用水。也有元气森林白桃味气泡水。”，首句 8 字，单次首段耗时 5.25 秒；
后续“第二种多少钱？”正确返回“演示价五元。”。
规则与验证记录见 [多商品分句验证](docs/SEMANTIC_FIRST_SENTENCE_EVALUATION.json)。

部署根目录为 `/data/voice-stack`。`deploy/voice.env.example` 是环境变量模板，`deploy/systemd` 包含四个服务和总 target。证书通过以下命令生成：

```bash
bash deploy/create_self_signed_cert.sh /data/voice-stack/config/tls 106.14.11.151
```

模型、环境、证书、token、数据和 checkpoint 均不提交到仓库。服务器只需在阿里云安全组放行 TCP 8443；18000–18002 不应对公网开放。
