# 混合模型前缀缓存可用性冒烟实验

## 状态

**两次独立重启均完成四个请求，共 8/8 成功。** 两轮复用数均为 `0、1088、0、1088`。使用原版缓存策略，尚未实现保留策略或证明优化空间。

| 顺序 | 请求 | 输入 token | 复用 token | 输出 token | 完整调用耗时 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 文档 A，问题 1 | 1467 | 0 | 16 | 1322.303 |
| 2 | 文档 A，问题 2 | 1468 | 1088 | 16 | 442.752 |
| 3 | 文档 B，问题 1 | 1467 | 0 | 16 | 619.680 |
| 4 | 文档 A，问题 3 | 1466 | 1088 | 16 | 487.555 |

以上来自 `run-04-offline` 的逐请求 JSON。首请求包含运行中的首次 JIT 编译，诊断采集也有开销，耗时不能用于证明缓存带来的加速比例。复用数来自完成后的 `RequestOutput.num_cached_tokens`，并与管理器 lookup 事件核对。

`run-05-offline` 独立重启的四次完整调用耗时按请求顺序为 901.64、434.70、559.85、452.79 ms。这两轮不是稳定延迟基准，未计算显著性或吞吐收益。

实际缓存配置为共享池 20 个物理块、块大小 544 token，四组分别为三组 Mamba（各 6 层）与一组完整注意力（6 层）。权重加载日志报告约 1.53 GiB。128 MiB 是缓存预算，不代表模型总显存占用。

保留了所有启动尝试：run01 在权重构造时 CUDA OOM；run02 因 Windows 可提交内存接近耗尽被手动停止；run03 权重已加载，随后因 FlashInfer JIT 找不到 nvcc 失败；run04 切换到现有 PyTorch 采样器后完成。多项配置发生变化，这些不是受控的性能对比，主机内存与首次 OOM 的因果关系尚未完全确定。

## 固定环境

- 模型：`Qwen/Qwen3.5-0.8B`
- 模型 revision：`2fc06364715b967f1860aea9cf38778875588b17`
- vLLM：`0.30.0`
- PyTorch：`2.13.0+cu130`
- Transformers：`5.17.0`
- GPU：`RTX 4060 Laptop GPU，8 GB`
- 系统：`WSL Ubuntu 24.04`

模型权重不纳入本仓库或实验输出。测试只使用脚本生成的合成英语文本，不包含个人文本。

## 成功配置的运行方式

在仓库根目录、已激活的 vLLM 环境中运行。输出目录必须不存在：

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/hybrid_offline_smoke.py \
  --model-dir /本机/Qwen3.5-0.8B模型目录 \
  --output-dir data/public/hybrid-prefix-retention/20260930/your-new-run
```

脚本使用 V1 runner、单进程、lazy 权重加载、PyTorch 采样器，关闭 CUDA graph，最大上下文 2048、最大批 token 1024、最大序列数 1，KV 缓存预算 128 MiB，Mamba `align` 模式，默认保留间隔 0，不启用 MTP。事件采集仅观察查找和登记，不计算“淘汰导致的重算量”。

模型权重 SHA256：`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`。

## 首次 HTTP 启动尝试（本机未成功）

在已激活的 vLLM 环境中设置本地模型目录，然后用 Bash 运行启动脚本：

```bash
export HYBRID_MODEL_DIR=/本机/Qwen3.5-0.8B模型目录
# 以下两项用于启用本实验的诊断采集器（在仓库根目录执行）：
export PYTHONPATH="$PWD/hybrid_observer_bootstrap:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HYBRID_PILOT_EVENTS="$PWD/data/public/hybrid-prefix-retention/20260930/events-run01.jsonl"
bash scripts/run_hybrid_pilot.sh
```

服务启动后，使用新建的结果目录运行探针；目录必须不存在，避免覆盖已有结果：

```bash
python scripts/hybrid_prefix_probe.py \
  --base-url http://127.0.0.1:8011 \
  --model hybrid-pilot \
  --output-dir data/public/hybrid-prefix-retention/20260930/run-01
```

探针逐请求保存 JSON，并在运行结束后写入 `summary.json`。请求出错时也会保存错误信息；出现请求失败会以非零状态退出。不要把本地模型路径、个人环境变量或模型权重放入公开结果。

## 实验含义与边界

探针按固定顺序串行发送四个请求：同一份合成文档的两个不同问题、另一份合成文档的一个问题、再回到第一份文档提出第三个问题。每次请求使用 OpenAI 兼容的 `/v1/completions` 接口，`temperature=0`、`max_tokens=16`、非流式输出。

这是**合成缓存可用性机制冒烟**：用于检查相同文档前缀能否被后续请求复用。离线脚本的 `elapsed_ms` 是完整生成调用耗时，HTTP 脚本的是完整 HTTP 请求耗时，两者**都不是 TTFT**，也不能混合比较。不能据此证明发生了有害淘汰、存在优化空间或代表真实负载效果。本实验不作为现实负载证据，16 token 输出也不用于评价回答质量或缓存正确性的完整覆盖。

## 公开与许可

本目录的合成输入与实验数据以 [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) 公开；项目代码使用仓库 MIT 许可。模型使用上游 Apache-2.0 许可，权重不随数据分发。公开日志仅替换个人路径，其余内容保留。失败和手动中断记录也公开，不筛选成功结果。
