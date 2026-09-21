<div align="center">

# any2jev

**把任何开源模型变成 Jev 风格的 System One 决策模型。**
单次前向传播，带类型的答案，校准过的概率，不生成一个 token。

[English](README.md) · [原理](docs/architecture.md) · [数据格式](docs/data-format.md)

</div>

```
any2jev train --base Qwen/Qwen3-0.6B --data train.jsonl --val val.jsonl --out runs/my-jev
any2jev serve runs/my-jev            #  POST /v1/systemone，TypeSafe 官方 SDK 改个 base_url 直接用
```

TypeSafe 的 [Jev](https://typesafe.ai) 证明了一件事：一个"只做决定、不写文字"的模型，在路由、审核、分诊、
排序、护栏、游戏 agent 这类任务上比 LLM 快 40 到 200 倍，还便宜得多。但 Jev 是闭源的。
`any2jev` 是转换器：给它一个 Hugging Face 权重（Qwen、Llama、Gemma、SmolLM、Phi、Mistral……）和几千条
带标签的决策数据，就得到一个能在一次前向里回答 **Choice / Score / Noul** 问题、概率可以直接拿来设阈值的模型。

## 为什么不直接让 LLM 输出 JSON？

| | LLM + JSON schema | 零样本读 logits | **any2jev** |
|---|---|---|---|
| 延迟 | 秒级（自回归） | 一次前向 | **一次前向，所有问题同时出** |
| 输出 | 要解析、要校验的文本 | 标签 token 的概率 | **带类型的答案，永远合法** |
| 概率校准 | 无（"90% 确定"只是文字） | 无（基座原始 logits） | **训练 + 温度缩放，ECE 有报告** |
| 一个 state 问多个问题 | N 倍成本 | N 倍 prefill | **state 只编码一次，问题互相隔离** |
| 任意基座 | 是 | 大多数 | **是，LoRA + 0.5M 参数的头** |

## 你会得到什么

* **模型手术**：卸掉词表头，挂一个 pointer head，用决策 token 给每个选项的隐状态打分。任何
  `AutoModelForCausalLM` 都行，包括混合架构（Qwen3.5）。
* **块因果打包**：state 只编码一次；每个问题只看得到 state 和自己，看不到其他问题，所以答案不依赖于
  同一次请求里还问了什么（有单测保证，不是口头承诺）。
* **训练**：LoRA + 头 + 分隔符 embedding，交叉熵加可选的 Brier / 有序损失，训练时打乱选项顺序，最后在验证集上做
  **温度缩放**。
* **对得上宣传的评测**：按问题类型分别报告准确率、NLL、Brier、ECE、AURC、5% 风险下的覆盖率，外加选项顺序敏感度测试和
  打包 vs 单独提问的隔离检查。
* **Jev 兼容服务**：`POST /v1/systemone` 和 `GET /v1/models`，请求响应格式与 TypeSafe 完全一致，官方 `typesafe-sdk`
  只改 `base_url` 就能连。

## 快速开始

```bash
pip install any2jev[serve]              # 公开数据集加 [data]，官方 SDK 加 [compat]

# 1. 数据：2000 条合成客服工单（无需下载），或转换公开数据集
any2jev data synthetic --out data/synthetic --n 2000
any2jev data build --sources boolq,ag_news,banking77,sst5 --out data/public --n-per-source 1500

# 2. 训练：手术 + LoRA + 头 + 温度缩放一条命令（0.6B 在 8GB 显卡上约 7 分钟）
any2jev train --base Qwen/Qwen3-0.6B --data data/public/train.jsonl --val data/public/val.jsonl --out runs/qwen3-0.6b

# 3. 评测
any2jev eval runs/qwen3-0.6b --data data/public/test.jsonl

# 4. 直接问，或者起服务
any2jev ask runs/qwen3-0.6b --state "我的提现连续失败三天了，赶紧处理" \
    --choice "该转给哪个团队？ | billing, technical, sales" --noul "这条消息紧急吗？" \
    --score "客户有多生气？ | 平静, 不满, 愤怒"
any2jev serve runs/qwen3-0.6b --port 8009
```

用官方 SDK 调用本地服务：

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

## 结果

Qwen3-0.6B，LoRA r=16，1 个 epoch，一张 RTX 2060 SUPER（8 GB）。held-out 测试集，温度在验证集上拟合。

<!-- RESULTS:PUBLIC -->
_（公开数据集的实验正在跑，数字随后填入）_
<!-- /RESULTS:PUBLIC -->

合成客服工单（2000 条，4 类问题，规则标签）7 分钟训到准确率 1.000 / ECE 0.001，选项顺序测试 100% 稳定，
隔离检查里打包与单独提问的答案差异为 1e-6。这是冒烟测试，不是 benchmark。

延迟（稳态，单请求 3 个问题，`examples/bench_latency.py`）：

<!-- RESULTS:LATENCY -->
_（公开数据实验结束后测）_
<!-- /RESULTS:LATENCY -->

## 原理

详见 [docs/architecture.md](docs/architecture.md)。一句话版本：

1. `extract_backbone()` 保留 `*ForCausalLM` 的 decoder 栈，丢掉 `lm_head`。
2. 五个分隔符（`<state> <q> <opt> </opt> <decide>`）复用 tokenizer 里闲置的特殊 token，没有就新增。用户文本会被消毒，
   伪造不出分隔符。
3. state 和问题打包进一个序列，用**块因果 mask**；每个分支的位置编码从 state 之后重新计数。混合架构退化为每个问题一行。
4. **pointer head** 计算 `h(</opt>_k) · h(<decide>)`，softmax 时除以拟合出的温度。
5. Noul、Choice、Score 是同一个原语配不同的选项列表；`confidence` 用 TypeSafe 公开的公式。

## 路线图

- [ ] 在监督配方之上加 RLCR 风格的 RL 阶段（奖励 `correct − (confidence − correct)²`）
- [ ] 服务端 state 前缀 KV 缓存（块因果 mask 保证精确）
- [ ] 大模型走 vLLM / SGLang 后端，小模型导出 ONNX
- [ ] 通过同一接口支持 encoder 基座（ModernBERT）
- [ ] 在 Hub 上发布 Qwen3 0.6B / 1.7B / 4B 的预训练 adapter

## 致谢

Jev 是 [TypeSafe AI](https://typesafe.ai) 的模型，本项目独立开发，与其无关。打包问题的架构参考了 Archer Hume 的
[*Jev's Architecture Unmasked*](https://archerhume.com/posts/jevs-architecture-unmasked) 和
[kev](https://github.com/jaredpalmer/kev)（Apache-2.0，固定 Qwen 家族）；`any2jev` 把这套配方泛化成面向任意基座的转换器，
并补上评测与校准工具链。校准目标的设计参考了 [rlcd-lite](https://github.com/arnabgho/rlcd-lite)。
生态全景见 [awesome-jev](https://github.com/OmniJev/awesome-jev)。

## 许可证

Apache-2.0。
