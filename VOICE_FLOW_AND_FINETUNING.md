# Voice 服务执行流程与 ASR 微调说明

本文用尽量直白的方式说明当前 Voice 服务怎么工作，以及怎样准备饮料名称等业务词汇的语音识别微调数据。

## 1. 当前系统包含什么

当前系统主要包含四个进程：

| 服务 | 地址 | 作用 |
| --- | --- | --- |
| Business API | `0.0.0.0:8443` | 提供 HTTPS 测试页面和业务接口 |
| Voice Gateway | `127.0.0.1:18000` | 鉴权、排队并控制 GPU 并发 |
| ASR Worker | `127.0.0.1:18001` | 使用 Qwen3-ASR 将语音转换成文字 |
| TTS Worker | `127.0.0.1:18002` | 使用 Qwen3-TTS 将文字转换成 WAV 音频 |

浏览器和业务系统只访问 Business API。模型 Worker 不直接暴露到公网。

当前使用的模型：

- ASR：`Qwen/Qwen3-ASR-0.6B-hf`
- TTS：`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`

当前限制：

- 单段录音最长 60 秒
- 上传文件最大 20 MB
- TTS 单次最多 500 字
- GPU 请求串行执行，最多允许两个请求排队

## 2. 语音转文字的完整过程

整体调用链：

```text
浏览器录音
  → Business API 8443
  → Voice Gateway 18000
  → ASR Worker 18001
  → Qwen3-ASR
  → 原路返回识别文字
```

具体步骤如下。

### 2.1 浏览器录音

用户在测试页面点击“开始录音”。浏览器使用 `MediaRecorder` 录制语音，一般生成 WebM/Opus 音频。

用户还可以输入本次识别可能出现的热词，例如：

```text
元气森林
东方树叶
茉莉奶白
```

### 2.2 提交给业务服务

页面调用：

```text
POST /api/transcriptions
```

提交内容包括：

- 音频文件
- 语言，目前默认是 `Chinese`
- 热词列表

Business API 会先检查：

- 音频是否为空
- 文件类型是否支持
- 文件是否超过 20 MB
- 热词格式是否合法
- 热词数量和单个热词长度是否超限

### 2.3 业务服务调用 Voice Gateway

Business API 使用 Bearer Token 调用内部 Voice Gateway。

Gateway 负责：

- 检查 Token
- 限制排队请求数量
- 一次只放行一个请求使用 GPU
- 请求超时时返回明确错误

串行执行的目的，是避免 ASR、TTS 和服务器上已有的 GPU 服务同时抢占过多显存。

### 2.4 ASR Worker 预处理音频

ASR Worker 使用 FFmpeg 将上传音频统一转换成：

- 16 kHz 采样率
- 单声道
- PCM WAV
- 最长 60 秒

这样无论浏览器上传 WebM、MP3、MP4 还是 WAV，模型收到的格式都保持一致。

### 2.5 应用热词

用户传入的热词会转换成模型提示，例如：

```text
Vocabulary: 元气森林, 东方树叶, 茉莉奶白.
```

这个提示相当于在识别前告诉模型：

> 这次录音里可能出现这些词，请特别留意。

### 2.6 模型识别并返回

Qwen3-ASR 根据音频和热词提示生成文字。结果依次经过：

```text
ASR Worker → Voice Gateway → Business API → 浏览器
```

已验证的实际结果：

```text
录音：我要一瓶元气森林气泡水
结果：我要一瓶元气森林气泡水。
```

## 3. 文字转语音的完整过程

整体调用链：

```text
浏览器输入文字
  → Business API 8443
  → Voice Gateway 18000
  → TTS Worker 18002
  → Qwen3-TTS
  → 返回 WAV
  → 浏览器播放或下载
```

### 3.1 用户提交文字

用户输入最多 500 字的文本，并可以选择：

- 中文音色，如 Vivian、Serena、Dylan
- 语言，当前默认 `Chinese`
- 语气指令，例如“自然、清晰、热情”

页面调用：

```text
POST /api/speech
```

### 3.2 模型生成音频

TTS Worker 使用 Qwen3-TTS 生成 24 kHz 单声道 WAV，然后通过 Business API 返回浏览器。

这台 Tesla T4 上，TTS 当前使用 FP32。原因是 T4 不原生支持 BF16，而 FP16 生成时实际出现过 NaN。FP32 会稍慢，但稳定，并且当前显存仍然足够。

## 4. 热词和微调有什么区别

建议优先使用热词，热词解决不了的问题再使用微调。

### 4.1 热词

热词类似于识别前告诉工作人员：

> 今天客人可能会说“元气森林”和“茉莉奶白”，请特别留意。

优点：

- 不需要训练
- 新增饮料名称可以立即生效
- 没有训练成本
- 不容易影响普通语音识别能力
- 不满意时删除热词即可回退

适合：

- 新产品名称
- 品牌名
- 英文和中文混合名称
- 数量不多、变化频繁的业务词

### 4.2 微调

微调相当于给模型听很多业务录音，并告诉它每段录音的正确答案。

例如：

> 这段声音说的是“茉莉奶白”，不是“茉莉奈白”。

适合：

- 加了热词仍经常识别错误
- 门店噪声较大
- 顾客口音明显
- 多个商品名称读音接近
- 真实说法比较口语化
- 需要适应固定设备或固定场景的录音特点

微调有数据采集、人工标注、GPU 训练和效果回归风险，因此不应该作为新增商品名称的第一选择。

## 5. 当前微调范围

当前仓库只准备了 ASR 的 LoRA 微调流程，用于改善语音转文字。

当前没有实现 TTS 音色微调。

当前实际状态：

- 在线 ASR/TTS 已部署并验证
- ASR 热词功能已验证
- 数据整理、校验、LoRA 训练和评估脚本已经存在
- 还没有真实业务微调数据
- 还没有训练或上线 LoRA Adapter
- 当前线上 `ASR_ADAPTER_PATH` 为空，运行的是原始 Qwen3-ASR 模型

## 6. 微调数据怎么准备

### 6.1 先确定要解决的问题

先统计线上最常识别错误的商品名称，例如：

```text
元气森林
茉莉奶白
东方树叶
椰云拿铁
轻乳茶
```

不要一开始就盲目收集大量录音。应该先明确哪些词在什么环境下经常识别错。

### 6.2 录制真实业务说法

每个商品名称建议先准备约 30～100 条录音作为第一批试验数据。这个数量不是模型的硬性要求，而是一个便于验证流程的起点。

录音要尽量覆盖：

- 不同男女声音
- 不同年龄
- 普通话和常见口音
- 安静环境
- 门店背景音乐
- 咖啡机、冰块和顾客交谈声
- 单独说商品名
- 将商品名放在完整句子中

同一个商品可以录制：

```text
元气森林
我要一瓶元气森林
给我拿个元气森林气泡水
有没有冰的元气森林
```

不建议让一个人重复录制几百次。这样模型容易只记住这个人的声音，换一个顾客就可能失效。

### 6.3 人工标注正确文字

每段音频必须对应一条准确文本。原始 JSONL 中每行表示一段录音，例如：

```json
{"audio":"raw/001.m4a","text":"language Chinese<asr_text>我要一瓶元气森林气泡水","prompt":"Vocabulary: 元气森林, 气泡水.","speaker_id":"speaker-01","product":"元气森林"}
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `audio` | 原始录音文件路径 |
| `text` | 录音的准确文字，前面保留模型要求的格式 |
| `prompt` | 识别时提供的热词提示，可选 |
| `speaker_id` | 说话人编号，用于正确拆分数据 |
| `product` | 目标商品名称，用于计算商品名称召回率 |

`speaker_id` 和 `product` 不直接作为模型的学习内容，主要用于管理数据和验收效果。

标注时注意：

- 录音里说了什么就写什么，不要自行润色
- 商品名称必须使用业务系统认可的标准写法
- 不要把没有说出来的标点、语气词或商品名强行补进去
- 听不清的录音不要猜，应该返工或删除

### 6.4 按说话人拆分数据

建议将数据分为：

- `train`：约 80%，用于训练
- `dev`：约 10%，训练过程中观察效果
- `test`：约 10%，最后验收

拆分时最重要的是：同一个人的录音不要同时出现在训练集和测试集。

错误做法：

```text
speaker-01 的前 80 条放 train，后 20 条放 test
```

正确做法：

```text
speaker-01 只放 train
speaker-08 只放 dev
speaker-10 只放 test
```

否则测试结果会看起来很好，但只能说明模型记住了这个人的声音。

## 7. 数据处理步骤

### 7.1 统一音频格式

准备脚本会将音频转换成 16 kHz、单声道 WAV：

```bash
python -m training.prepare_dataset \
  --input raw_train.jsonl \
  --output train.jsonl \
  --audio-dir normalized/train
```

`dev` 和 `test` 也分别执行：

```bash
python -m training.prepare_dataset \
  --input raw_dev.jsonl \
  --output dev.jsonl \
  --audio-dir normalized/dev

python -m training.prepare_dataset \
  --input raw_test.jsonl \
  --output test.jsonl \
  --audio-dir normalized/test
```

脚本：`training/prepare_dataset.py`

### 7.2 检查数据

```bash
python -m training.validate_dataset train.jsonl
python -m training.validate_dataset dev.jsonl
python -m training.validate_dataset test.jsonl
```

校验内容包括：

- 音频文件是否存在
- 音频能否读取
- 是否为 16 kHz 单声道
- 时长是否在允许范围内
- 标注格式是否正确
- 有多少说话人
- 有多少种商品

脚本：`training/validate_dataset.py`

## 8. 训练前先测原始模型

训练之前，先用完全独立的测试集记录原始模型成绩：

```bash
python -m training.evaluate_asr \
  --manifest test.jsonl \
  --model /data/voice-stack/models/Qwen3-ASR-0.6B-hf \
  --output baseline-report.json
```

主要关注：

- `cer`：文字错误率，越低越好
- `product_recall`：商品名称识别正确率，越高越好

如果不记录训练前成绩，就无法判断训练后到底有没有提升。

## 9. 使用 LoRA 微调

执行：

```bash
python -m training.train_lora \
  --model /data/voice-stack/models/Qwen3-ASR-0.6B-hf \
  --train train.jsonl \
  --eval dev.jsonl \
  --output /data/voice-stack/checkpoints/drinks-v1
```

LoRA 可以理解为：

> 不重写整个模型，只训练一个小型业务知识补丁。

优点：

- 训练显存比全量微调小
- 训练时间更短
- Adapter 文件较小
- 上线和回滚比较简单

当前训练默认参数：

- 训练 3 轮
- 每次处理 1 条录音
- 累积 16 次后更新参数
- FP16 训练
- 使用梯度检查点节省显存
- 每 100 步保存和评估一次

训练结束后生成：

```text
/data/voice-stack/checkpoints/drinks-v1/final-adapter
```

脚本：`training/train_lora.py`

## 10. 训练后怎么验收

使用同一个独立测试集评估微调后的模型：

```bash
python -m training.evaluate_asr \
  --manifest test.jsonl \
  --model /data/voice-stack/models/Qwen3-ASR-0.6B-hf \
  --adapter /data/voice-stack/checkpoints/drinks-v1/final-adapter \
  --output tuned-report.json
```

比较：

```text
baseline-report.json
tuned-report.json
```

推荐的上线条件：

- 商品名称召回率明显提高
- 整体文字错误率没有明显变差
- 通用语音测试集的 CER 退化不超过 2 个百分点
- 不同说话人下都有效
- 在安静和噪声环境下都有效
- 没有把常见词错误地识别成商品名称

不能只看训练集或开发集成绩。最终判断必须使用训练过程从未见过的测试集。

脚本：`training/evaluate_asr.py`

## 11. 上线微调结果

验收通过后，在服务器配置中设置：

```text
ASR_ADAPTER_PATH=/data/voice-stack/checkpoints/drinks-v1/final-adapter
```

然后重启 ASR Worker：

```bash
systemctl restart voice-asr.service
```

重启后先检查：

```bash
curl http://127.0.0.1:18001/health/ready
```

再使用真实录音执行一轮冒烟测试。

## 12. 如何回滚

如果微调后效果不好：

1. 清空 `ASR_ADAPTER_PATH`
2. 重启 `voice-asr.service`

服务就会恢复使用原始 Qwen3-ASR 模型，不需要重新下载和部署基础模型。

## 13. 推荐的第一轮试验方式

建议第一轮不要直接收集几万条数据，可以按以下规模试跑：

1. 选择 5～10 个最容易识别错误的商品名称
2. 每个商品收集 30～50 条录音
3. 至少覆盖 5～10 个说话人
4. 保留部分完全不同说话人的录音作为测试集
5. 先测试原始模型和热词效果
6. 热词仍不理想时再训练 LoRA
7. 对比商品召回率和通用 CER
8. 达标后再扩大数据规模

这样可以先验证问题是否真的需要微调，避免在数据采集和训练上投入过多成本。

## 14. GPU 型号、BF16 与当前 T4 部署说明

模型部署不仅对显存大小有要求，也对 GPU 架构、支持的计算精度和注意力加速方式有要求。

这不代表 Voice 服务只能部署在某一种 GPU 上。准确地说，是不同 GPU 需要选择不同的计算精度，并会产生不同的速度、显存占用和稳定性。

### 14.1 为什么 Tesla T4 不原生支持 BF16

当前服务器使用 Tesla T4。T4 属于 NVIDIA Turing 架构，计算能力为 `7.5`。

T4 的 Tensor Core 原生支持：

- FP16
- FP32 累加
- INT8
- INT4

但 Turing 架构没有原生 BF16 Tensor Core。BF16 Tensor Core 是从后续的 Ampere 架构开始加入的。

服务器上的实际检测结果：

```text
GPU：Tesla T4
Compute Capability：(7, 5)
原生 BF16：False
```

可以使用下面的 PyTorch 代码检查一张 GPU 是否原生支持 BF16：

```python
import torch

print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
print(torch.cuda.is_bf16_supported(including_emulation=False))
```

参考资料：

- [NVIDIA T4 官方规格](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)
- [NVIDIA Ampere 调优指南](https://docs.nvidia.com/cuda/pdf/Ampere_Tuning_Guide.pdf)
- [PyTorch BF16 支持检测接口](https://docs.pytorch.org/docs/main/generated/torch.cuda.is_bf16_supported.html)

### 14.2 FP16、BF16 和 FP32 的简单区别

可以把计算精度理解为模型计算数字时使用的“数字格式”。

| 精度 | 特点 |
| --- | --- |
| FP16 | 占用显存小、速度快，但数值范围较小，部分生成模型可能出现溢出或 NaN |
| BF16 | 占用显存与 FP16 接近，但数值范围更接近 FP32，通常更适合生成式模型 |
| FP32 | 数值最稳定，但显存占用更高、速度更慢 |

Qwen3-TTS 官方示例通常使用：

```python
dtype=torch.bfloat16
attn_implementation="flash_attention_2"
```

参考：[Qwen3-TTS 官方微调说明](https://github.com/QwenLM/Qwen3-TTS/blob/main/finetuning/README.md)

### 14.3 当前服务器上的实际测试结果

当前服务器已经实际测试过不同组合：

| 功能 | 精度 | Tesla T4 上的结果 |
| --- | --- | --- |
| Qwen3-ASR | FP16 | 正常运行 |
| Qwen3-TTS | FP16 | 采样概率出现 NaN，CUDA 推理失败 |
| Qwen3-TTS | BF16 | T4 不原生支持 |
| Qwen3-TTS | FP32 | 正常、稳定，但速度稍慢、显存占用较高 |

因此当前部署配置为：

```text
ASR：FP16
TTS：FP32
Attention：PyTorch SDPA
```

TTS 使用 FP32 不是模型的硬性要求，而是针对 Tesla T4 做出的稳定性兼容方案。

### 14.4 当前是否必须更换 GPU

当前不必立即更换 GPU。

两个 0.6B 模型和已有 YOLO 服务可以同时放入这张 T4：

| 服务 | 大约显存占用 |
| --- | ---: |
| 已有 YOLO | 1.2 GB |
| Qwen3-ASR FP16 | 1.8 GB |
| Qwen3-TTS FP32 | 4.9 GB |
| 合计 | 约 7.9 GB |

这张 T4 实际可用显存约 15 GB，因此当前仍有余量。

当前 T4 适合：

- 功能验证
- 低并发业务
- 一次处理一条语音请求
- ASR 录音不超过 60 秒
- 对 TTS 延迟要求不高
- 使用 0.6B 模型

当前 Gateway 已经将 GPU 请求串行化，因此不会同时执行多个 ASR/TTS 推理请求。这也是 T4 能够稳定承载当前服务的重要条件。

### 14.5 什么情况下需要升级 GPU

出现以下需求时，建议升级：

- TTS 生成延迟不能接受
- 需要同时处理多个语音请求
- 要从 0.6B 升级到 1.7B 模型
- 要使用 BF16
- 要使用 FlashAttention 2
- 要提高 ASR/TTS 并发量
- 要进行更大规模的模型训练
- 服务器还要运行更多 GPU 服务

### 14.6 GPU 选择建议

| GPU | 架构 | 原生 BF16 | 对当前项目的建议 |
| --- | --- | --- | --- |
| T4 | Turing | 不支持 | 当前低并发方案可继续使用 |
| V100 | Volta | 不支持 | 不建议为 BF16 场景新采购 |
| A10 | Ampere | 支持 | 性价比较合适，适合正式部署 |
| A30 | Ampere | 支持 | 推理和训练都可以 |
| A100 | Ampere | 支持 | 性能强，成本较高 |
| L4 | Ada | 支持 | 比较适合推理服务 |
| L40S | Ada | 支持 | 适合更大模型和更高并发 |
| H100 | Hopper | 支持 | 性能很强，成本很高 |

针对当前业务规模，建议：

- 继续测试、低并发：保留 T4。
- 正式部署并希望降低 TTS 延迟：优先考虑 A10 24 GB 或 L4 24 GB。
- 后续升级 1.7B、提高并发或频繁微调：建议至少 24 GB 显存。
- 大规模训练：使用 A100 或同等级 GPU。

### 14.7 ASR 微调对 GPU 的要求

Qwen3-ASR 官方微调代码会根据 GPU 架构选择精度：

- 计算能力达到 8.0 或更高时使用 BF16
- 较老 GPU 使用 FP16

因此，0.6B ASR 的 LoRA 小规模微调可以先在 T4 上尝试，但训练速度会慢于 Ampere 或更新架构，并且需要控制音频长度、Batch Size 和梯度累积参数。

参考：[Qwen3-ASR 官方训练代码](https://github.com/QwenLM/Qwen3-ASR/blob/main/finetuning/qwen3_asr_sft.py)
