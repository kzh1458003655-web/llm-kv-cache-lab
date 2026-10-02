# 固定 9B 在线实验包

主模型为 Qwen3.5-9B，revision、BF16、vLLM 版本和请求集已锁定。旧的 0.8B 脚本和数据保留为预实验。**本包已在本机做 CPU 检查，9B 的 GPU 启动与性能尚未验证。当前服务器保持关机。**

## 包含什么

- 9B 专用的 tokenizer/config 和模型文件哈希清单；4 个权重分片约 19.31GB，开机后按固定 revision 下载并核验。
- 20 个固定请求文件：两种文档负载 × 两档上下文截断长度 × 五个种子；每文件 512 请求。只准备这些输入，不要求把所有组合都跑一遍。
- 每轮生成 128 token，正式客户端并发上限 8/16，vLLM 最大活跃序列数固定为 16；实际并发单独记录。
- 原版/候选开关、流式 token-ID 首响应计时、缓存命中统计、输出哈希核对、结果导出和时间上限。
- 一个单独的检查点边界诊断，记录实际缓存组、查找和回收事件。诊断计时不进入性能结论。

2026-10-02 审核后，开题前执行清单见 [nineb-opening-plan.json](../../configs/nineb-opening-plan.json)：两类主要负载各三组成对重复，另补全程无跨请求复用对照。主协议中代表配置五组重复是后续论文阶段目标，已经准备的第四、第五个种子保留备用。

请求使用公开 LooGLE 文档的截断片段，问题不重复，到达顺序人为生成；不属于生产 trace，也不用于评估问答质量。完整提示、源文本、权重和原始服务器日志不进入 Git。

## 开机后先做这几步

使用单卡约 32GiB 显存的 Linux 实例，Python 3.12，建议至少 80GB 可用磁盘。硬件显示名称与实际可见显存以 `nvidia-smi` 为准。安装和下载不计入压测脚本的时间上限，不能保证一小时完成全部轮次。

```bash
tar -xzf nineb-opening-ready.tar.gz
cd nineb-opening-ready
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

冻结脚本要求三档预算都有相同环境、相同原生选项与到达率的成功原版校准，并实测至少 8 个客户端请求同时活跃、至少 2 个服务器请求处于运行状态。中等缓存预算还须观测到服务器等待队列。正式对照拒绝变动后的模型、依赖、GPU 或请求文件。校准通过表示配置能运行，性能结论仍来自独立正式运行。

## 单独做混合模型边界诊断

总入口现在自动先运行四个短诊断，无须另手动运行本节命令：原版/候选各一次边界诊断，原版在中等/较大缓存各一次 64 请求串行回收诊断。边界诊断生成 16 token，其他负载仍生成 128 token；所有诊断计时不进入性能结论。以下命令仅供单独排查使用。

从实际缓存组取一倍、两倍块大小的前后一个 token，串行发送同前缀、不同后缀的请求。每个前缀长度使用独立 cache salt，同长度的一对请求共享 salt，使各用例不会互相命中。以下 `实际roomy字节数`、`原生选项.json` 与校准选择保持一致，两次命令只有策略和输出目录不同：

```bash
.venv-9b/bin/python scripts/run_9b_online.py --mode diagnostic --boundary-probe --policy stock --model model --manifest model-manifest.json --protocol configs/thesis-9b-protocol.json --requests prepared/hot_scan-ctx4096-rep0.jsonl --output data/runs/boundary/stock --kv-cache-bytes 实际roomy字节数 --native-options 原生选项.json --rate 1
.venv-9b/bin/python scripts/run_9b_online.py --mode diagnostic --boundary-probe --policy reuse2 --model model --manifest model-manifest.json --protocol configs/thesis-9b-protocol.json --requests prepared/hot_scan-ctx4096-rep0.jsonl --output data/runs/boundary/reuse2 --kv-cache-bytes 实际roomy字节数 --native-options 原生选项.json --rate 1
.venv-9b/bin/python scripts/analyze_9b_boundaries.py --stock data/runs/boundary/stock --reuse2 data/runs/boundary/reuse2 --output data/runs/boundary/analysis.json
```

若诊断出现容量回收，先检查是否影响当前用例，不能把未复用的 token 全算成对齐损失。语义边界、周期保留和算子限制仍需结合引擎事件解释。全注意力组的分块对齐参考值只是本模型内部的几何参照，并不等于实际跑过一个全注意力模型。压力诊断统计分组哈希回收及再次注册，刻画缓存内容反复保存的现象，不将这些计数直接当成重算 token 数或离线最优差距。

## 正式对照

先生成计划，不启动 GPU：

```bash
.venv-9b/bin/python scripts/run_9b_round.py --model model --manifest model-manifest.json --protocol configs/thesis-9b-protocol.json --prepared prepared --estimate-from data/runs/calibration --output data/runs/plan-preview
```

计划预览会依据校准的实际启动时间和单请求耗时粗估总时长，并给出增加 50% 余量的规划值，不含安装、权重传输和结果下载。只有校准后才能判断一次租机要多久。执行必须显式指定时间预算；下面的 45 分钟仅为命令示例，须按实际估时和允许租期填写：

```bash
.venv-9b/bin/python scripts/run_9b_round.py --model model --manifest model-manifest.json --protocol configs/thesis-9b-protocol.json --prepared prepared --lock calibration-lock.json --output data/runs/formal --minutes 45 --execute
```

计划为四个短诊断加 25 轮完整测量：

| 内容 | 完整测量轮数 | 用途 |
| --- | ---: | --- |
| 热点负载三组原版/候选 | 6 | 热点保护与一致性 |
| 混合负载三组原版/候选 | 6 | 另一类访问顺序下的表现 |
| 同一输入原版独立重复 | 3 | 机器和调度波动 |
| 全程无跨请求复用的原版/候选 | 2 | 策略额外开销；以每请求不同 salt 实现，必须测得缓存命中为零 |
| 同一热点负载的高压力、较大缓存对照 | 4 | 与主配置共同组成三档容量曲线 |
| 并发上限 16 对照 | 2 | 并发变化 |
| 约 2K 输入对照 | 2 | 输入长度变化 |

每次独立重启，两策略使用同一请求文件和到达表。原版/候选每组交替顺序；原版同输入重复可与同输入主配置的原版记录一起查看。扫描阶段的文档可能在其他片段重现，因此分析中称“扫描请求”，全程无复用情况由独立 salt 对照判断。较大缓存只表示最大可行测试预算，须结合回收数据判断是否真正无淘汰。

三组重复报告每组数值、方向和范围，用于开题初判，不给出强置信结论。p99 仅作诊断；512 个相关请求的尾部样本有限。预算不足时保存部分完成状态，只分析完整成对结果。全套完成须包括短诊断和两类负载的重复，不能把只跑完第一组称为全部完成。

TTFT 按首个含生成 token ID 的流式块计时；另记首段文字时间、客户端等待与从计划到达至首 token 的时间。TPOT 按首末 token 块间隔计算，受到流式合并影响。吞吐是选定到达率下的实际吞吐，不能直接当最大服务能力。请求失败、缺少 token ID、输出长度不符和配对哈希不一致都保留，查清前不声称正确性或性能提升。

## 保存、公开与关机

```bash
.venv-9b/bin/python scripts/export_9b_results.py --runs data/runs --output data/public/nineb-results
tar -czf nineb-results.tar.gz data/runs data/public/nineb-results calibration-lock.json
sha256sum nineb-results.tar.gz
```

先下载归档到本机、核对 SHA256 并确认测量文件完整，然后执行实例关机。压测入口只停止自己创建的服务进程，**不会停止平台计费**。实验时仍由主任务在下载验证后执行关机，不能在打包结束时立即关机。

公开导出包括嵌套的边界/压力分析文件，仅含配置、测量、哈希、来源和审核后的诊断记录。原始日志留在私有归档；不上传 SSH 信息、密钥、密码或平台令牌。

## 开题前的完成判据

需拿到真实 9B 的运行与策略接入证据、可解释的串行正确性检查、混合缓存边界及回收记录、两类负载的重复对照和开销/容量/并发边界。若串行输出不一致，总入口会暂停后续测量，先排查；若没有观察到性能收益，如实写出负结果和可实现模块范围。离线最优模拟器、多模型泛化、完整 Marconi 复现以及完整消融留到正式开发阶段，不作为开题前全部完成的条件。
