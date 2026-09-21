<div align="center">

<br/>

# any2jev

### 任何开源模型 &nbsp;→&nbsp; Jev 风格的决策模型

**单次前向传播。带类型的答案。校准过的概率。一个 token 都不生成。**

<br/>

[![CI](https://img.shields.io/github/actions/workflow/status/hwfengcs/any2jev/ci.yml?branch=main&label=ci&style=flat-square)](https://github.com/hwfengcs/any2jev/actions)
[![PyPI](https://img.shields.io/pypi/v/any2jev?style=flat-square&color=blue)](https://pypi.org/project/any2jev/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square)](LICENSE)
[![Jev API](https://img.shields.io/badge/API-%2Fv1%2Fsystemone%20兼容-orange?style=flat-square)](docs/architecture.md)

[**English**](README.md) · [原理](docs/architecture.md) · [数据格式](docs/data-format.md) · [Benchmark](#benchmark) · [参与贡献](CONTRIBUTING.md)

<br/>

<img src="https://raw.githubusercontent.com/hwfengcs/any2jev/main/docs/vs.gif" alt="同一套权重：提示输出 JSON vs any2jev 单次前向" width="100%">

<sub>同一套 <b>Qwen3-0.6B</b> 权重 · 同一张 RTX 2060 SUPER · 同一个请求。左：提示它输出 JSON，逐 token 解码。
右：经过 <code>any2jev train</code> 之后。真实计时，以 1/4 速度回放（<code>examples/record_vs.py</code>）。</sub>

</div>

<br/>

```bash
pip install "any2jev[serve]"
any2jev train --base Qwen/Qwen3-0.6B --data train.jsonl --val val.jsonl --out runs/my-jev   # 约 1 GPU 小时
any2jev serve runs/my-jev   # POST /v1/systemone · TypeSafe 官方 SDK 只改一个 base_url 就能连
```

## 三秒钟版本

[TypeSafe 的 Jev](https://typesafe.ai) 证明了一件事：路由、分诊、审核、排序、护栏、游戏 agent 这些场景，
你要的不是一个会**写**的模型，而是一个会**决定**的模型：毫秒级出结果，概率可以直接拿来设阈值。
Jev 不开放权重。**any2jev 就是那个转换器**：任意 Hugging Face 因果语言模型，加上几千条带标签的决策数据，
就变成一个单次前向回答 **Choice / Score / Noul** 问题的模型。

<!-- RESULTS:HERO -->
| 方案 | 怎么出答案 | 延迟¹ | 格式错误率² | 准确率² | ECE² | 峰值显存 | 每百万请求 GPU 时 |
|---|---|---|---|---|---|---|---|
| GPT-4o 级 API，JSON mode | 云端往返，逐 token 生成 | 秒级³ | JSON 合法，取值不受约束 | — | — | 云端 | 按 token 计费 |
| Qwen/Qwen3-0.6B，提示它输出 JSON | `generate()`，逐 token 解码 | 778 ms | 0.1% | 0.621 | 无概率 | 2.3 GB | ~216 h |
| Qwen/Qwen3-0.6B，直接读标签 logits | 每个问题一次 prefill，无需训练 | 137 ms | 0% | 0.531 | 0.111 | 2.3 GB | ~38 h |
| **any2jev（Qwen/Qwen3-0.6B）** | **单次前向，所有问题一起** | **42 ms** | **0%（类型由构造保证）** | **0.796** | **0.027** | **2.8 GB** | **~12 h** |

¹ 同一条 3 个问题的客服工单，NVIDIA GeForce RTX 2060 SUPER，fp32，多次运行取中位数（`examples/record_vs.py`）。  
² 1,000 个 held-out 问题（boolq、ag_news、banking77、sst5），`any2jev eval` 与 `examples/baseline_*.py` 使用同一份文件。  
³ 本仓库未测量。TypeSafe 在发布文章里给出前沿模型端到端 3 到 329 秒；用你自己的 key 跑 `examples/bench_cloud.py` 即可把真实数字填进这一行。
<!-- /RESULTS:HERO -->

上表的每个数字都来自本仓库里的脚本。我们没有测量任何云端模型，所以第一行不填数字；测量脚本已附上。

## 你会得到什么

```
                 ┌─────────────────────── 单次前向 ──────────────────────────────┐
  state ────────►│ <state> …  │ <q> 哪个团队? <opt>billing</opt><opt>shipping</opt> <decide> │──► {billing: 0.91, shipping: 0.09}, confidence 0.82
                 │            │ <q> 紧急吗?   <opt>no</opt><opt>yes</opt>          <decide> │──► noul 0.97
                 │            │ <q> 多生气?   <opt>calm</opt> … <opt>furious</opt> <decide> │──► score 1.4, {0: .05, 1: .5, 2: .45}
                 └──────────────── 问题看得到 state，看不到彼此 ────────────────────┘
```

| | |
|---|---|
| 🔪 **模型手术** | 卸掉词表头，挂一个 pointer head，用决策 token 给每个选项的隐状态打分。任何 `AutoModelForCausalLM` 都行；混合架构（Qwen3.5 的线性注意力）退化为每个问题一行因果序列。 |
| 🧱 **块因果打包** | state 只编码一次；每个问题只看得到 state 和自己，看不到其他问题，所以答案不依赖于同一次请求里还问了什么。有单测保证，不是口头承诺。 |
| 🎯 **冲着校准去的训练** | LoRA + 头 + 分隔符 embedding；交叉熵加可选的 Brier / 有序损失；训练时打乱选项顺序；最后在验证集上做温度缩放。 |
| 📏 **对得上宣传的评测** | 按问题类型分别报告准确率、NLL、Brier、ECE、AURC、5% 风险下的覆盖率，外加选项顺序敏感度测试和打包 vs 单独提问的隔离检查。 |
| 🔌 **Jev 兼容服务** | `POST /v1/systemone` 和 `GET /v1/models`，请求响应格式与 TypeSafe 完全一致，`typesafe-sdk` 原样可用。 |

## 40 毫秒一步的贪吃蛇

`examples/snake.py` 用 BFS 老师生成带标签的走法，`any2jev train` 把 Qwen3-0.6B 训成策略，游戏循环每一步就合法走法
问一个 Choice 问题。全程不生成文本。

<!-- RESULTS:SNAKE -->
![any2jev playing Snake](https://raw.githubusercontent.com/hwfengcs/any2jev/main/docs/snake.gif)

held-out 老师走法：准确率 0.953，ECE 0.034，400 次决策。
录制对局：final score 21 in 161 steps | median latency 41 ms
<!-- /RESULTS:SNAKE -->

## 60 秒试用，不用训练

预训练 adapter 已经放在 Hugging Face Hub 上。任何能接 checkpoint 路径的地方都能接 `hf://`：

```bash
pip install "any2jev[serve]"
any2jev ask hf://huaweifeng/any2jev-qwen3-0.6b \
    --state "My payouts have failed 3 days in a row, the bank says everything is fine. Fix this ASAP." \
    --choice "Which team should handle this? | billing, technical, sales" \
    --noul "Does this need urgent human attention?" \
    --score "How frustrated is the customer? | calm, frustrated, furious"
any2jev serve hf://huaweifeng/any2jev-qwen3-0.6b --port 8009
```

| checkpoint | 基座 | 训练数据 | held-out | 说明 |
|---|---|---|---|---|
| [`huaweifeng/any2jev-qwen3-0.6b`](https://huggingface.co/huaweifeng/any2jev-qwen3-0.6b) | Qwen3-0.6B | boolq、ag_news、banking77、sst5（5400 条） | acc 0.796 · ECE 0.027 | 本页所有数字背后的模型 |
| [`huaweifeng/any2jev-qwen3-0.6b-snake`](https://huggingface.co/huaweifeng/any2jev-qwen3-0.6b-snake) | Qwen3-0.6B | 4000 步 BFS 老师走法 | acc 0.953 | 驱动 `examples/snake.py` |
| [`huaweifeng/any2jev-qwen3.5-0.8b-synthetic`](https://huggingface.co/huaweifeng/any2jev-qwen3.5-0.8b-synthetic) | Qwen3.5-0.8B（混合架构） | 1600 条合成工单 | acc 0.997 | 验证线性注意力基座上的 rows 模式配方；不是通用模型 |

`hf://user/repo@revision` 同样支持 `train --base`、`eval` 和 Python 的 `DecisionModel.load()`。
`calibrate` 会原地修改本地 checkpoint；使用 Hub 模型时请先复制到本地目录。

每个仓库约 40 MB（LoRA adapter + pointer head + tokenizer + 配置）；基座权重首次使用时从它自己的 Hub 仓库下载。
训练数据是英文的，中文输入请用自己的数据重新训练。

## 快速开始

```bash
pip install "any2jev[serve]"            # 公开数据集加 [data]，官方 SDK 加 [compat]
# 或者直接装 GitHub 上的版本：  pip install "any2jev[serve] @ git+https://github.com/hwfengcs/any2jev"

# 1. 数据：2000 条合成客服工单（无需下载），或转换公开数据集
any2jev data synthetic --out data/synthetic --n 2000
any2jev data build --sources boolq,ag_news,banking77,sst5 --out data/public --n-per-source 1500

# 2. 训练：手术 + LoRA + 头 + 温度缩放一条命令
#    （Qwen3-0.6B 在一张 8GB 显卡上：2000 条短工单约 7 分钟，5400 条公开数据约 1 小时）
any2jev train --base Qwen/Qwen3-0.6B --data data/public/train.jsonl --val data/public/val.jsonl --out runs/qwen3-0.6b

# 3. 评测：校准、顺序敏感度、隔离
any2jev eval runs/qwen3-0.6b --data data/public/test.jsonl

# 4. 直接问，或者起服务
any2jev ask runs/qwen3-0.6b --state "我的提现连续失败三天了，赶紧处理" \
    --choice "该转给哪个团队？ | billing, technical, sales" --noul "这条消息紧急吗？" \
    --score "客户有多生气？ | 平静, 不满, 愤怒"
any2jev serve runs/qwen3-0.6b --port 8009
```

用官方 SDK，只改 `base_url`：

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8009", model="any2jev-latest")
r = client.system_one(
    state="鞋子晚到了两周而且尺码不对，另外我的卡上有两笔扣款。",
    questions={
        "department": Choice(instructions="该由哪个团队处理？",
                             criteria={"returns": "换货、退货、错发或损坏", "shipping": "物流状态、延误、丢件",
                                       "billing": "扣款、发票、支付问题"}),
        "escalate": Noul(instructions="需要人工紧急介入吗？"),
        "frustration": Score(instructions="客户有多生气？", criteria=["平静", "不满", "愤怒"]),
    })
r.choices["department"].probabilities
r.nouls["escalate"].noul
r.scores["frustration"].score
```

或者直接发 HTTP，和 Jev 的契约一模一样：

```bash
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d @examples/request.json
```

自己的数据是每行一个 JSON 对象，就是一条 `/v1/systemone` 请求加上每个问题的 `label`（[格式说明](docs/data-format.md)）。
模型训练时看到的字节和服务端喂给它的字节完全一致。

## Benchmark

Qwen3-0.6B，LoRA r=16，1 个 epoch，一张 RTX 2060 SUPER（8 GB）。held-out 测试集，温度在验证集上拟合。
`any2jev eval` 打印这些表；JSON 报告在 `runs/`，由 `scripts/fill_readme.py` 写到这里。
[docs/benchmarks.md](docs/benchmarks.md) 列出了每个数字由哪个脚本产生。

### 公开数据，按问题类型

<!-- RESULTS:PUBLIC -->
| 问题类型 | n | 系统 | acc | NLL | Brier | ECE | AURC | cov@5% |
|---|---|---|---|---|---|---|---|---|
| overall | 1000 | 提示输出 JSON（generate） | 0.621 (0.1% 格式错误) | — | — | — | — | — |
| overall | 1000 | 零样本 logits（基座） | 0.531 | 1.180 | 0.595 | 0.111 | 0.276 | 0.05 |
| overall | 1000 | 零样本 + 温度缩放 | 0.531 | 1.125 | 0.579 | 0.058 | 0.279 | 0.05 |
| overall | 1000 | **any2jev** | **0.796** | **0.506** | **0.284** | **0.027** | **0.062** | **0.57** |
| noul | 250 | 提示输出 JSON（generate） | 0.624 (0.0% 格式错误) | — | — | — | — | — |
| noul | 250 | 零样本 logits（基座） | 0.644 | 0.693 | 0.479 | 0.166 | 0.216 | 0.12 |
| noul | 250 | 零样本 + 温度缩放 | 0.644 | 0.631 | 0.443 | 0.119 | 0.216 | 0.12 |
| noul | 250 | **any2jev** | **0.844** | **0.386** | **0.242** | **0.090** | **0.058** | **0.58** |
| choice | 500 | 提示输出 JSON（generate） | 0.764 (0.2% 格式错误) | — | — | — | — | — |
| choice | 500 | 零样本 logits（基座） | 0.622 | 1.163 | 0.534 | 0.082 | 0.222 | 0.09 |
| choice | 500 | 零样本 + 温度缩放 | 0.622 | 1.118 | 0.532 | 0.087 | 0.226 | 0.09 |
| choice | 500 | **any2jev** | **0.906** | **0.272** | **0.142** | **0.020** | **0.018** | **0.88** |
| score | 250 | 提示输出 JSON（generate） | 0.332 (0.0% 格式错误) | — | — | — | — | — |
| score | 250 | 零样本 logits（基座） | 0.236 | 1.700 | 0.835 | 0.145 | 0.710 | 0.00 |
| score | 250 | 零样本 + 温度缩放 | 0.236 | 1.634 | 0.811 | 0.092 | 0.705 | 0.00 |
| score | 250 | **any2jev** | **0.528** | **1.093** | **0.611** | **0.056** | **0.424** | **0.01** |

选项顺序测试（23 个 Choice 问题）：argmax 在 96% 的问题上保持稳定，概率最大波动均值 0.088。
隔离检查：未执行（检查范围内没有包含多个问题的记录）。
验证集拟合的温度 T = 1.61。测试集 1000 条记录，1000 个问题。
<!-- /RESULTS:PUBLIC -->

合成客服工单（2000 条，4 类问题，规则标签）7 分钟训到准确率 1.000 / ECE 0.001；那是冒烟测试，不是 benchmark。

### 延迟

稳态、单请求，`examples/bench_latency.py`：

<!-- RESULTS:LATENCY -->
| 基座 | 精度 | 设备 | 输入 token | 问题数 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | float32 | cuda:0 | 116 | 3 | 41.8 | 53.2 |
| Qwen/Qwen3-0.6B | bfloat16 | cuda:0 | 116 | 3 | 61.6 | 64.2 |
| Qwen/Qwen3-0.6B | bfloat16 | cuda:0 | 353 | 12 | 133.4 | 136.8 |
| Qwen/Qwen3-0.6B | bfloat16 | cuda:0 | 361 | 3 | 132.8 | 133.7 |
<!-- /RESULTS:LATENCY -->

RTX 2060 SUPER（Turing）没有原生 bf16，所以 fp32 反而更快；Ampere 及以后 bf16 更快。延迟随输入 token 数增长，
而不随问题数增长：序列长度相同时，12 个问题和 3 个问题一样快。

### 验证过的基座

<!-- RESULTS:BASES -->
| 基座 | 架构 | 模式 | 训练数据 | acc | ECE | 训练耗时 | 可训练参数 |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 纯注意力 | packed | public (1000 q) | 0.796 | 0.027 | 61 min | 10.6 M |
| Qwen/Qwen3.5-0.8B | 混合（线性注意力 + 注意力） | rows | synthetic (594 q) | 0.997 | 0.003 | 26 min | 5.9 M |
<!-- /RESULTS:BASES -->

Qwen3.5 那一行是合成数据的冒烟测试，不是 benchmark：它证明混合架构（Gated DeltaNet）基座能走通同一套手术、训练和服务路径，
只是每个问题作为独立的因果序列运行，因为线性注意力层吃不了块因果 mask。rows 模式在这张卡上的延迟约为打包模式的 6 倍
（3 个问题的示例 259 ms，未安装 `flash-linear-attention` 的融合算子）。

`AutoModelForCausalLM` 能加载的都应该能用。Qwen、Llama 3、Gemma 的 tokenizer 内置了分隔符复用方案，其他 tokenizer
会新增五个 token。fp32 LoRA 训练、384 token state 的显存粗估：0.6B 约 8 GB，1.7B 约 16 GB（`--dtype bf16` 减半）。
验证了新的基座？带上 `eval.json` 开个 PR。

## 原理

详见 [docs/architecture.md](docs/architecture.md)。一句话版本：

1. `extract_backbone()` 保留 `*ForCausalLM` 的 decoder 栈，丢掉 `lm_head`。
2. 五个分隔符（`<state> <q> <opt> </opt> <decide>`）复用 tokenizer 里闲置的特殊 token，没有就新增。用户文本会被消毒，
   伪造不出分隔符。
3. state 和问题打包进一个序列，用**块因果 mask**；每个分支的位置编码从 state 之后重新计数。混合架构退化为每个问题一行。
4. **pointer head** 计算 `h(</opt>_k) · h(<decide>)`，softmax 时除以拟合出的温度。
5. Noul、Choice、Score 是同一个原语配不同的选项列表；Choice 的 `confidence` 参考 TypeSafe 公开演示，
   Score 使用近似公式（[说明](docs/architecture.md)）。

## 路线图

- [ ] 在监督配方之上加 RLCR 风格的 RL 阶段（奖励 `correct − (confidence − correct)²`）
- [ ] 服务端 state 前缀 KV 缓存（块因果 mask 保证精确）
- [ ] 大模型走 vLLM / SGLang 后端，小模型导出 ONNX
- [ ] 通过同一接口支持 encoder 基座（ModernBERT）
- [x] 在 Hub 上发布 Qwen3 0.6B 的预训练 adapter（[公开数据](https://huggingface.co/huaweifeng/any2jev-qwen3-0.6b)、[贪吃蛇](https://huggingface.co/huaweifeng/any2jev-qwen3-0.6b-snake)）
- [ ] Qwen3 1.7B / 4B 以及 Llama / Gemma 基座的 adapter

## 致谢

Jev 是 [TypeSafe AI](https://typesafe.ai) 的模型，本项目独立开发，与其无关。打包问题的架构参考了 Archer Hume 的
[*Jev's Architecture Unmasked*](https://archerhume.com/posts/jevs-architecture-unmasked) 和
[kev](https://github.com/jaredpalmer/kev)（Apache-2.0，固定 Qwen 家族）；any2jev 把这套配方泛化成面向任意基座的转换器，
并补上评测与校准工具链。校准目标的设计参考了 [rlcd-lite](https://github.com/arnabgho/rlcd-lite)。
生态全景见 [awesome-jev](https://github.com/OmniJev/awesome-jev)。

## 许可证

Apache-2.0。
