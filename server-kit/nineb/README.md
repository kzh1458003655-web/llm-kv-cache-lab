# 固定 9B 在线实验包

主模型为 Qwen3.5-9B，revision、BF16、vLLM 版本和请求集已锁定。旧的 0.8B 脚本和数据保留为预实验。**本包已在本机做 CPU 检查，9B 的 GPU 启动与性能尚未验证。当前服务器保持关机。**

## 包含什么

- 9B 专用的 tokenizer/config 和模型文件哈希清单；4 个权重分片约 19.31GB，开机后按固定 revision 下载并核验。
- 20 个固定请求文件：两种文档负载 × 两档上下文截断长度 × 五个种子；每文件 512 请求。只准备这些输入，不要求把所有组合都跑一遍。
- 每轮生成 128 token，正式客户端并发上限 8/16，vLLM 最大活跃序列数固定为 16；实际并发单独记录。
- 原版/候选开关、流式 token-ID 首响应计时、缓存命中统计、输出哈希核对、结果导出和时间上限。
- 一个单独的检查点边界诊断，记录实际缓存组、查找和回收事件。诊断计时不进入性能结论。

请求使用公开 LooGLE 文档的截断片段，问题不重复，到达顺序人为生成；不属于生产 trace，也不用于评估问答质量。完整提示、源文本、权重和原始服务器日志不进入 Git。

## 开机后先做这几步

使用单卡约 32GiB 显存的 Linux 实例，Python 3.12，建议至少 80GB 可用磁盘。硬件显示名称与实际可见显存以 `nvidia-smi` 为准。安装和下载不计入压测脚本的时间上限，不能保证一小时完成全部轮次。

```bash
tar -xzf nineb-server-ready.tar.gz
cd nineb-server-ready
python3 scripts/nineb_artifacts.py verify-bundle --root .
bash server-kit/nineb/setup.sh
.venv-9b/bin/python scripts/nineb_artifacts.py download-model --manifest model-manifest.json --model-dir model
```

下载脚本复用哈希正确的已有文件；失败的本次新文件会删除，已有错误文件会报错并保留。下载速度取决于网络。不要在租期内尝试源码编译 vLLM。

## 初次校准

先以原版、并发上限 4、16 个请求启动，验证模型、流式返回、输出长度、原生缓存组和 CUDA graph 能否运行：

```bash
.venv-9b/bin/python scripts/run_9b_online.py --mode calibration --model model --manifest model-manifest.json --protocol configs/thesis-9b-protocol.json --requests prepared/hot_scan-ctx4096-rep0.jsonl --output data/runs/calibration/initial --concurrency 4 --rate 1 --count 16
```

这次自动使用 vLLM 的可用缓存预算，记录缓存组及实际张量字节。`rate=1` 只用于启动检查，正式到达率须经原版短测确定。检查日志里 CUDA graph 实际模式、缓存块大小、前缀匹配粒度和检查点保留配置；适用的原生选项写到一个 JSON 文件，再通过 `--native-options 文件名` 传入校准。仅支持已经核对的 `mamba_cache_mode`、`prefix_match_unit`、`prefix_cache_retention_interval`。

根据可用内存、实际活跃请求和工作集选出三档可行缓存预算，再对每一档做原版校准：`--kv-cache-bytes 实际字节数 --concurrency 16 --rate 选定到达率 --count 64`。到达率应能产生实际排队，同时使客户端等待可测、运行能完成。保存失败和成功尝试；不要根据候选收益挑选参数。

复制并填写 `server-kit/nineb/calibration-settings.template.json` 为 `calibration-settings.json`，记录原生选项检查、到达率依据及三档缓存压力证据。若必须关闭 CUDA graph，校准命令加 `--enforce-eager`，并在设置中固定同一值。

```bash
.venv-9b/bin/python scripts/freeze_9b_calibration.py --protocol configs/thesis-9b-protocol.json --prepared prepared --calibration-root data/runs/calibration --settings calibration-settings.json --output calibration-lock.json
```

冻结脚本要求三档预算都有相同环境、相同原生选项与到达率的成功原版校准，并实测至少 8 个客户端请求同时活跃。采样记录服务器运行/等待请求数，确认排队是否实际存在。正式对照拒绝变动后的模型、依赖、GPU 或请求文件。校准通过表示配置能运行，性能结论仍来自独立正式运行。

## 单独做混合模型边界诊断

从实际缓存组取一倍、两倍块大小的前后一个 token，串行发送同前缀、不同后缀的请求。以下 `实际roomy字节数`、`原生选项.json` 与校准选择保持一致，两次命令只有策略和输出目录不同：

```bash
.venv-9b/bin/python scripts/run_9b_online.py --mode diagnostic --boundary-probe --policy stock --model model --manifest model-manifest.json --protocol configs/thesis-9b-protocol.json --requests prepared/hot_scan-ctx4096-rep0.jsonl --output data/runs/boundary/stock --kv-cache-bytes 实际roomy字节数 --native-options 原生选项.json --rate 1
.venv-9b/bin/python scripts/run_9b_online.py --mode diagnostic --boundary-probe --policy reuse2 --model model --manifest model-manifest.json --protocol configs/thesis-9b-protocol.json --requests prepared/hot_scan-ctx4096-rep0.jsonl --output data/runs/boundary/reuse2 --kv-cache-bytes 实际roomy字节数 --native-options 原生选项.json --rate 1
.venv-9b/bin/python scripts/analyze_9b_boundaries.py --stock data/runs/boundary/stock --reuse2 data/runs/boundary/reuse2 --output data/runs/boundary/analysis.json
```

若诊断出现容量回收，先提高诊断缓存或缩短病例，不能把未复用的 token 全算成对齐损失。语义边界、周期保留和算子限制仍需结合引擎事件解释。

## 正式对照

先生成计划，不启动 GPU：

```bash
.venv-9b/bin/python scripts/run_9b_round.py --model model --manifest model-manifest.json --protocol configs/thesis-9b-protocol.json --prepared prepared --output data/runs/plan-preview
```

开跑时使用新输出目录和冻结文件：

```bash
.venv-9b/bin/python scripts/run_9b_round.py --model model --manifest model-manifest.json --protocol configs/thesis-9b-protocol.json --prepared prepared --lock calibration-lock.json --output data/runs/formal --minutes 45 --execute
```

计划选择 25 次运行，不跑全部参数组合：先覆盖热点、混合、大缓存对照和三次原版重复，再完成热点代表配置五组成对实验，最后补容量、并发和输入长度的单变量对照。每次独立重启，两策略使用同一请求文件和到达表。预算不足会保留部分完成状态，只分析完整成对结果。全部请求属于一次运行，不能把它们当成数百个独立实验。

TTFT 按首个含生成 token ID 的流式块计时；另记首段文字时间、客户端等待与从计划到达至首 token 的时间。TPOT 按首末 token 块间隔计算，受到流式合并影响。吞吐是选定到达率下的实际吞吐，不能直接当最大服务能力。请求失败、缺少 token ID、输出长度不符和配对哈希不一致都保留，查清前不声称正确性或性能提升。

## 保存、公开与关机

```bash
.venv-9b/bin/python scripts/export_9b_results.py --runs data/runs --output data/public/nineb-results
tar -czf nineb-results.tar.gz data/runs data/public/nineb-results calibration-lock.json
sha256sum nineb-results.tar.gz
```

先下载归档到本机、核对 SHA256 并确认测量文件完整，然后执行实例关机。压测入口只停止自己创建的服务进程，**不会停止平台计费**。实验时仍由主任务在下载验证后执行关机，不能在打包结束时立即关机。

公开导出仅含配置、测量、哈希、来源和审核后的诊断记录。原始日志留在私有归档；不上传 SSH 信息、密钥、密码或平台令牌。
