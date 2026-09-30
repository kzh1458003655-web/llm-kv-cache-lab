# 面向混合注意力模型的前缀缓存管理模块设计与实现

本科毕业设计项目，基于真实 vLLM 开发前缀缓存管理模块。在缓存容量有限时，根据复用价值决定哪些缓存优先保留、哪些优先回收，减少重复计算，并记录策略的开销和适用边界。

> 当前状态：开题前已完成机制原型和初步实验，正式模块待开发。2026-09-30 完成 19 次独立运行、2432 个成功请求；热点场景观察到复用与耗时改善，混合负载收益不一致，并有一次 p95 退化记录。完整结果见[服务器初步实验](data/public/hybrid-prefix-retention/server-20260930/README.md)。

## 毕业设计任务书

[任务书：课题目标、工作内容、完成标准与进度](docs/毕业设计任务书.md)

题目使用“设计与实现”，主体为缓存管理模块。具体工作包括策略接口、保留信息生命周期、缓存组关系、原生回收流程接入，以及行为观测与验收评测。

## 主体工作

1. 梳理混合模型的注意力 KV、状态检查点及 vLLM 缓存组的数据路径。
2. 将原型整理为可配置、可切换的缓存管理模块，限定候选扫描数量和附加记录开销。
3. 以基于复用次数的 `reuse2` 为基础，参考 Marconi 的复用可能性、计算收益与内存成本，实现至少一种候选保留策略。
4. 正确处理缓存淘汰、重置和对象再分配，沿用原生引用计数与匹配规则。
5. 用固定序列对照原版与候选策略，检查输出、缓存行为、冷请求开销和尾延迟。
6. 提供源码、配置、测量数据、复现说明及论文材料；时间线或 Web 页面可作为辅助展示。

## 已有实验说明

- 模型：Qwen3.5-0.8B；原型支持锁定版本 vLLM 0.30.0。
- 服务器实验：19 次完整独立运行，9 组策略对照，2432 个成功请求。
- 热点加一次性请求、生成 32 token：命中 token 从 52,224 增至 69,632，累计生成调用耗时降低 4.07%。
- 三组热点短输出耗时均降低；混合文档收益方向不一致；缓存充足时命中量相同。
- 一组热点短输出 p95 耗时约增加 20.3%，原因待归因。
- 九组策略对照的输出 token 哈希匹配 1152/1152；完整正确性和稳定性检查属于后续验收。

以上是公开文档配合构造访问顺序、串行离线生成的结果。当前测量采用固定配置与 eager 模式，计时为同步后的 `generate` 调用耗时；正式服务评测需要独立记录 TTFT、吞吐及基线调参情况。

## 仓库结构

```text
kv_cache_lab/          缓存事件、诊断观察器与保留策略原型
scripts/               负载准备、真实引擎实验与结果分析
server-kit/            单卡实验的环境准备和运行说明
data/public/           可公开的配置、测量记录与结果
tests/                 CPU 单元测试
docs/毕业设计任务书.md    当前题目、主体范围与完成标准
docs/设计概览.md         模块设计与已有实现状态
docs/实验方案.md         验收场景、指标口径与对照约定
docs/开发路线.md         后续工程阶段及交付物
```

## 使用说明

CPU 工具需要 Python 3.10+。已有单元测试的运行命令：

```bash
python -m unittest discover -s tests -v
```

真实推理需要单独配置 Linux/WSL、GPU 和固定依赖；参见[单卡实验准备](server-kit/README.md)。正式计时与诊断事件采集应分开运行，逐事件日志写入会影响耗时。

`kv_cache_lab.events` 提供标准化事件格式；`hybrid_observer` 提供真实缓存诊断；`hybrid_retention_prototype` 提供基于复用次数的原型。原型尚待整理为正式模块，具体状态见[设计概览](docs/设计概览.md)。

## 文档与数据

- [毕业设计任务书](docs/毕业设计任务书.md)
- [设计概览](docs/设计概览.md)
- [实验方案](docs/实验方案.md)
- [开发路线](docs/开发路线.md)
- [服务器初步实验及公开数据（2026-09-30）](data/public/hybrid-prefix-retention/server-20260930/README.md)
- [本地混合模型前缀缓存可用性实验](data/public/hybrid-prefix-retention/20260930/README.md)
- [本地保留原型与容量边界实验](data/public/hybrid-prefix-retention/20260930-pressure/README.md)
- [服务器负载准备审核](data/public/hybrid-prefix-retention/server-prepared-20260930)
- [租卡前本地预检](data/public/hybrid-prefix-retention/server-preflight-20260930/README.md)
- [较早的全注意力模型可行性实验（历史记录）](docs/feasibility-results-2026-09-27.md)

## 参考资料

- [Marconi](https://arxiv.org/abs/2411.19379)：混合模型缓存的准入与淘汰思路；当前 `reuse2` 是轻量原型，完整论文复现待单独实现与验证。
- [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)：分页式 KV Cache 和推理服务背景。
- [SGLang / RadixAttention](https://arxiv.org/abs/2312.07104)：前缀复用与结构化推理程序背景。
