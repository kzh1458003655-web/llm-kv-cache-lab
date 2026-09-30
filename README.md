# 面向大语言模型推理的 KV Cache 管理策略优化与可视化系统

毕业设计项目：在成熟推理引擎上观察 KV Cache 的生命周期，修改一种可复用缓存的管理策略，并用真实模型、可重复的工作负载和可视化系统评估其影响。

> 当前状态：已进行真实 vLLM 模型实验，并实现一个可切换的混合模型缓存保留原型。2026-09-30 的小规模压力对照观察到缓存复用改善，**尚未证明稳定性能收益或生产负载优势；Web 页面尚未实现。**

- [混合模型前缀缓存可用性实验与公开数据](data/public/hybrid-prefix-retention/20260930/README.md)
- [保留原型、公开文档对照与容量边界](data/public/hybrid-prefix-retention/20260930-pressure/README.md)

## 项目目标

真实模型推理 → 缓存事件采集 → 前缀复用与淘汰策略分析 → 一项策略改动 → 同条件 Benchmark → 可视化展示。

重点是对实际推理系统中的缓存管理行为进行修改和验证，不重新实现完整推理引擎，也不预设改进策略一定在所有工作负载上优于原版。

## 毕设主体范围

1. 在 WSL/Linux 上运行 vLLM 和一个可在实验硬件上稳定运行的 Qwen 等开源模型。
2. 采集请求、KV Block 分配/释放、前缀命中和缓存淘汰事件；区分运行中使用的块与可复用的空闲缓存块。
3. 以开启 Prefix Cache 的原版 vLLM 为主要 Baseline，实现一种可切换的缓存管理策略。
4. 用无共享请求、固定共享前缀、多轮对话三类工作负载进行重复实验；长文档问答或 RAG 请求作为扩展。
5. 测量复用/重新计算的 Token 数、TTFT、吞吐及必要的显存指标，并报告收益和退化场景。
6. 提供模型交互、缓存事件回放与策略对比的 Web Demo。实时数据和回放数据会明确标注。

CUDA 算子、多卡、CPU/GPU Offloading 和 Prefill/Decode 分离不属于毕设最低完成范围。

## 仓库结构

```text
kv_cache_lab/          缓存事件格式与后续分析模块
tests/                 无 GPU 单元测试
docs/设计概览.md         系统模块及边界
docs/实验方案.md         Baseline、工作负载与测量约定
docs/开发路线.md         分阶段交付与完成标准
.github/workflows/     CPU 单元测试 CI
```

## 当前可运行内容

需要 Python 3.10+。在仓库根目录执行：

```bash
python -m unittest discover -s tests -v
```

需要检查覆盖率时，先安装开发依赖，再运行 `python -m coverage run -m unittest discover -s tests` 和 `python -m coverage report -m`。

`kv_cache_lab.events` 提供标准化的缓存事件和 JSONL 读写，供后续引擎埋点、Benchmark 和页面回放共用。它**不是**已完成的 vLLM 集成。

## 文档

- [服务器初步实验与定题结论（2026-09-30）](data/public/hybrid-prefix-retention/server-20260930)：19 次完整运行、2432 请求；热点收益、混合负载边界及尾延迟退化均保留。服务器已执行关机。

- [单卡服务器实验准备](server-kit/README.md)：固定 0.8B 模型、三场景、最多五轮成对对照；上传包在本机制作，服务器未租用时只做本地预检。
- [固定负载审核信息](data/public/hybrid-prefix-retention/server-prepared-20260930)：完整来源哈希、种子及请求哈希，不含原始文档和完整提示。

- [设计概览](docs/设计概览.md)
- [实验方案](docs/实验方案.md)
- [开发路线](docs/开发路线.md)
- [本地可行性实验（2026-09-27）](docs/feasibility-results-2026-09-27.md)：原版 vLLM 上的缓存压力验证；尚未实现改进策略。

## 参考背景

- [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)：分页式 KV Cache 和推理服务。
- [SGLang / RadixAttention](https://arxiv.org/abs/2312.07104)：前缀复用与结构化推理程序。
- MemServe、ShuffleInfer：后续跨设备缓存、调度与分离式推理扩展背景，不作为本阶段实现承诺。
