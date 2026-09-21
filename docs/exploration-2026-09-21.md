# 2026-09-21：Jev 调研与 any2jev 改进记录

## 调研结论

阅读了以下公开资料，访问日期为 2026-09-21：

- [TypeSafe 发布文章](https://typesafe.ai/blog/introducing-system-one-models-and-jev)：Jev 面向结构化决策，直接并行输出概率，使用 RLCD 进行后训练。
- [官方 API](https://docs.typesafe.ai/api)：Noul、Choice、Score 的请求、概率输出和类型契约。
- [官方 AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer)：校准描述一组预测的频率性质，不能保证某一次决策正确。
- [官方 Confidence 文档](https://docs.typesafe.ai/confidence)：`confidence` 是从概率分布派生的统计量；公开交互示例使用近似的 Choice 公式，未公开完整 Score 公式。
- [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked)：通过实验推断共享 state、问题隔离和直接读出。文章明确区分公开声明、观测与架构假说。
- [Kev](https://github.com/jaredpalmer/kev)：社区开源实现，提供 Qwen 系列模型、训练与评测，并针对不同架构处理问题隔离。
- [awesome-jev](https://github.com/OmniJev/awesome-jev)：用于了解复现项目、评测和应用生态。

any2jev 的价值在于可复用的转换、训练和校准工具链。它使用社区架构思路，不能声称复现了未公开的 Jev 架构或 RLCD。输出格式正确、分类准确、概率校准和问题隔离是四个独立性质，应分别验证。

## 本轮修复

| 问题 | 修复与验证 |
|---|---|
| 等置信度样本被 adaptive ECE 漏掉 | 保留有效分箱；全错且置信度 0.9 的样本正确得到 ECE 0.9 |
| 同分样本的排列影响风险覆盖率 | 阈值同时接受整个同分组；AURC 使用这些可实现阈值的右阶梯面积 |
| 梯度累积跨过 epoch 尾部，重复读取记录 | 本轮结束时处理未满的累积窗口；回归测试从原先读取 7 条恢复为数据集的 5 条 |
| microbatch 问题数量不等时，训练权重不一致 | 按实际问题总数归一化梯度；与等效大 batch 比较梯度和预测分布 |
| 无效标签被静默转为真假或截断的整数 | 验证 Noul、Score 标签；保留 JSONL 文件名和行号的错误定位 |
| 空训练集可能进入无进展循环 | 加载模型前拒绝空训练、验证数据和无效训练参数 |
| 续训保留旧温度 | 更新权重前重置为 1，使用新的验证集重新校准 |
| 推理构建从不读取的 KV cache | packed、rows 前向均明确禁用；真实 Qwen3、Qwen3.5 检查概率一致性 |
| LoRA 独立模块增加推理开销 | `ask`、`serve` 和延迟脚本增加 `--merge`；只在内存中合并 |
| 超长问题导致 HTTP 500 | 转换为 HTTP 422；验证失败后下一次请求仍正常 |
| 二选一 Choice 未参与选项打乱和顺序评测 | 扩展到所有至少两个选项的 Choice |
| 未执行隔离检查却报告差异 0 | `n=0` 时记录 `null`，CLI 和 README 明确标为跳过；rows 模式也执行隔离检查 |
| 指定评测温度未用于辅助检查 | 顺序敏感度与隔离检查使用同一温度 |
| 滑动窗口基座的 packed mask 改变原生注意力语义 | Gemma2、Mistral 等启用滑动窗口时使用 rows；旧版 Transformers 的 GPT-2 也使用 rows |
| 声明支持的 Transformers 4.51.3 无法加载模型 | 使用兼容的新旧加载参数；CI 增加最低模型依赖版本组合 |
| `hf://` 只在部分 CLI 入口解析 | 在模型加载层统一解析，支持 Python、续训、评测和 `@revision` |
| 文档中的置信度、隔离和格式错误率表述不准确 | 修正来源说明、跳过检查的展示和小数精度；保留已有测量结果 |

`calibrate` 会修改 checkpoint 配置，因此只接受本地目录；Hub checkpoint 需先复制到本地。

## 验证结果

- 原测试集：42 项通过；改进后：84 项全部通过。
- 当前环境：Python 3.13.9、PyTorch 2.9.0+cu126、Transformers 5.5.0、PEFT 0.18.1。
- 独立兼容环境：Transformers 4.51.3、PEFT 0.14.0，同样 84 项全部通过。其他依赖沿用本机环境，这不是所有依赖同时取最低版本的测试。
- PEFT 0.14.0 沿用现有的冻结分隔符回退行为；完整的可训练分隔符配方使用新版 PEFT。
- `ruff check src tests examples scripts` 通过。
- wheel 与 sdist 构建成功，`twine check` 通过。
- TypeSafe 官方 SDK 真实本地 HTTP 往返测试通过。
- Qwen3、Llama、GPT-2、Gemma2、Mistral 使用小型随机模型验证架构行为；不代表它们的任务准确率已完成评测。

已有 Qwen3-0.6B 公开数据 checkpoint，重跑同一份 1000 问题测试集：

| 项目 | 结果 |
|---|---:|
| 准确率 | 0.796 |
| NLL | 0.505550 |
| Brier | 0.284324 |
| ECE | 0.027014 |
| 5% 风险下覆盖率 | 0.572 |
| 合并 LoRA 后改变最高概率选项的问题数 | 0 / 1000 |
| 合并前后最大概率差 | 7.09e-6 |
| 另取 18 条多问题合成记录，打包与单独提问最大概率差 | 3.28e-6 |

Qwen3.5-0.8B checkpoint 的 12 条合成记录也验证了 rows 路径：关闭 KV cache 前后概率完全一致，合并与独立提问的变化在该样本上极小。这里的样本只用于数值回归，不作为新 benchmark。

RTX 2060 SUPER、fp32、116 token、3 个问题，预热 5 次后各测量 60 次：

| 推理方式 | p50 | p95 | 峰值已分配显存 |
|---|---:|---:|---:|
| 关闭无用 KV cache，不合并 LoRA | 40.4 ms | 46.2 ms | 2918.4 MiB |
| 关闭无用 KV cache，合并 LoRA | 31.3 ms | 32.3 ms | 2291.5 MiB |

该请求上合并后的 p50 降低约 22.5%，峰值显存降低约 21.5%。延迟收益取决于设备、精度和输入长度；没有据此改写原有长输入或其他基座的性能数字。

## 复现命令

```bash
python -m pytest -q
python -m ruff check src tests examples scripts
python -m build

any2jev eval runs/qwen3-0.6b-public --data data/public/test.jsonl --out runs/recheck.json
python examples/bench_latency.py runs/qwen3-0.6b-public --n 60
python examples/bench_latency.py runs/qwen3-0.6b-public --n 60 --merge
any2jev serve runs/qwen3-0.6b-public --merge
```

完整本地实验产物放在被 Git 忽略的 `runs/exploration/`；[测量摘要](experiments/2026-09-21.json) 包含结果、环境和 checkpoint / 测试集的 SHA-256。原有 checkpoint、训练数据和已发布结果没有被覆盖。

## 后续优先级

下一步更有价值的是跨数据源、跨语言的泛化评测和真实业务数据训练。重复长 state 的场景可以继续研究前缀缓存，但需要分别证明全注意力与循环架构的缓存正确性。RLCD 的详细配方尚未公开，不能把未经对照实验的 RL 阶段直接宣传为校准提升。
