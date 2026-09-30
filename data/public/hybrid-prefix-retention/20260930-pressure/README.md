# 混合模型检查点保留：小规模机制验证

日期：2026-09-30。状态：完成 5 次有效短运行，共 40/40 请求成功；另保留一次负载接口错误导致的失败运行（0 个请求）。不作为正式性能基准。

## 实际结论

在固定 128 MiB 缓存预算下，原版 vLLM 会在这两组热点扫描序列里回收已经复用过的前缀。一个约百行的 Python 原型可以改变回收顺序，保住这份前缀。**把原版缓存增到 256 MiB，同一 LooGLE 序列也不再丢失热点。** 因此本轮只支持“预算紧张时可以改善保留行为”，不支持通用加速、算法创新或已经达到发表条件。

| 场景 | 配置 | 请求成功 | 热点返回复用 token | 全请求缓存命中 token / 输入 token | 请求耗时之和 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 合成扫描 | 原版，128 MiB | 8/8 | 0 | 3264 / 11936 | 3958.85 |
| 合成扫描 | reuse2，128 MiB | 8/8 | 1088 | 4352 / 11936 | 4632.13 |
| LooGLE 文档摘录 | 原版，128 MiB | 8/8 | 0 | 3264 / 11727 | 5070.45 |
| LooGLE 文档摘录 | reuse2，128 MiB | 8/8 | 1088 | 4352 / 11727 | 4889.56 |
| LooGLE 文档摘录 | 原版，256 MiB | 8/8 | 1088 | 4352 / 11727 | 5016.11 |

这些耗时包含诊断采集和首次推理 JIT，不是 TTFT。每个配置只有一轮，不能判断可靠速度收益：合成对照原型慢了 17.0%，LooGLE 对照快了 3.6%，方向不一致。不能将缓存命中改善换算为吞吐提升。

原版两组 128 MiB 运行均出现同一证据链：warm3 查找命中的 5 个键（三个 Mamba 组各 1 个，完整注意力组 2 个）在随后分配时被移除，移除前引用数均为 0；hot_return 的命中为 0。原型在同一区间没有移除这 5 个键，hot_return 命中 1088。完成请求后的 `num_cached_tokens` 与 lookup 事件核对，不计算 E/P。

两组原型对照的每个请求、256 MiB 容量对照的每个请求，其生成文本均与相应原版一致。每请求只生成 16 token，这是有限的冒烟检查，不能证明完整正确性、QA 质量或长输出行为。

## 原型的具体改动

`kv_cache_lab/hybrid_retention_prototype.py`：只统计真正查找到的缓存键的复用次数；回收时最多检查 32 个空闲块，优先选择复用不足两次的块。所有块重新放回队列，再由 vLLM 原逻辑处理分配、引用数、哈希移除和通知。不创建新检查点，不修改模型计算和 CUDA。

原型只支持本机核对过的 vLLM 0.30.0，遇到大于 32 块的分配或存在 reuse watcher 时回退到原生分配。尚未适配所有 connector/MTP、多卡或并发场景。复用计数没有老化机制，哈希迁移可能清掉计数，元数据超过上限会清空保护；这是下一步工程问题，不是生产实现。

LooGLE 原型运行的 64 次选择，合计 CPU 时间 523049 ns（约 0.523 ms），最大扫描 19 个块、8 次改变选择顺序。这个计时仅覆盖选择逻辑，不含整个调度器、复用统计和文件采集开销。合成原型运行早于加入该计时与按 block ID 筛选的代码更新，未记录 CPU 计时；当前发布代码为本轮结束版本。

## 配置和负载

Qwen/Qwen3.5-0.8B，revision `2fc06364715b967f1860aea9cf38778875588b17`；vLLM 0.30.0，PyTorch 2.13.0+cu130，Transformers 5.17.0；RTX 4060 Laptop 8 GB，WSL Ubuntu 24.04。沿用前一轮成功的离线单进程 V1 runner、BF16、lazy 权重加载、PyTorch 采样器、eager；上下文 2048、批 token 1024、最大序列数 1、Mamba align、保留间隔默认 0，不启用 MTP。

128 MiB 时共享池 20 个物理块，256 MiB 时 40 个，四组块大小均为 544 token。128 MiB 是刻意设置的压力预算，不代表本机正常推理只能提供这些缓存，也不是全部 GPU 显存占用。

每组固定 8 个不同完整 Prompt：热点文档的 3 个不同问题 → 3 份新文档各一个问题 → 热点文档的另外两个问题。合成文本由脚本生成；LooGLE 使用公开文档的前 1400 token 和原问题，**到达顺序仍为人工构造，不是生产 trace或原版 LooGLE QA 评测**，截断也可能删去问题所需证据。

LooGLE 固定数据集 revision `4b50b4cb9333b48b3f7ddfd307c1d13369be5d1e`，来源：https://huggingface.co/datasets/bigainlco/LooGLE 。本地下载不完整：66796186 字节中仅使用 595 个完整记录、29 份文档，忽略唯一残缺末行。源 SHA256 为 `cf4e902b57b8fd1ff1fbb446330d537d8ecc988160b0ade1fcb2bf0219c6b149`。不将它标成完整数据集。公开结果保留记录 ID、输入/问题/前缀哈希和生成输出，不上传文档与问题原文；重新下载固定源后应核对所选记录与输入哈希。

## 复跑

在仓库根目录激活相同 vLLM 环境，使用不存在的输出目录，避免覆盖记录：

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/hybrid_offline_smoke.py --model-dir /本机/模型目录 \
  --workload pressure --policy stock --output-dir /新目录/合成原版
python scripts/hybrid_offline_smoke.py --model-dir /本机/模型目录 \
  --workload pressure --policy reuse2 --output-dir /新目录/合成原型
python scripts/hybrid_offline_smoke.py --model-dir /本机/模型目录 \
  --workload loogle --source-file /本机/shortdep_qa.jsonl \
  --policy stock --output-dir /新目录/文档原版
python scripts/hybrid_offline_smoke.py --model-dir /本机/模型目录 \
  --workload loogle --source-file /本机/shortdep_qa.jsonl \
  --policy reuse2 --output-dir /新目录/文档原型
# 原版容量对照：在原版命令中增加 --kv-cache-mib 256
python scripts/analyze_hybrid_pressure.py --run-dir /已完成的某次运行目录
```

`comparison.json` 由 `scripts/compare_hybrid_pressure.py --root 本目录` 生成。该脚本核对比较双方的顺序、输入哈希和 token 数。运行目录名称应与本目录相同，分析文件必须不存在；原始请求、事件和分析结果都保留。

日志只替换本地环境、模型、仓库和用户目录，未删除错误行。`loogle-stock` 记录一次负载返回格式不一致的失败，修复后用新目录 `loogle-stock-v2` 重跑。其余 5 次有效运行都保留，没有筛掉慢结果。

代码使用仓库 MIT 许可；本项目合成文本与测量记录使用 CC0-1.0；上游 LooGLE 数据不重新授权、不随仓库分发。模型权重遵循上游许可，不上传。
