# Project Rules

## 当前授权：跨空间适应 V2（2026-09-28）
最新研究节奏：实质性改码前先说明独立方法思想、成立假设、结构风险、
替代解释和能区分解释的最小实验。实验验证判断，不能代替判断。
连续两三轮仅有局部补救时暂停改码，重审建模。当前进入方法重审，见
docs/METHOD_REVIEW.md；已有三批探索收尾，未形成完整下一方法前不追加训练。
用户授权以实际效果迭代任务框架、表示和算法，可重构，不被 V1 限制。
V1 规范和结果仅作为冻结历史记录；新实验遵守 docs/ANCHOR_V2.md。
实现采用最短正确路径、功能性注释和必要科学检查。
Leader 负责实现、训练和判断；遇到明确瓶颈可交给可见 GPT-6 Sol Worker
（medium）针对性查外部资料；通信简洁，不用隐藏 subagent 替代。
所有新设置和结果分别保存，不能用目标 query 调参后声称独立测试。
下面 V1 的算法范围和不得改实验语义要求仅适用于 V1 历史复现。

## Source of truth
`docs/UJI_FeMLoc_V1_SPEC.md` 是实验语义的唯一依据；`configs/v1.yaml`
保存 V1 fixed config。不得自行改变 task、support/query、AP 范围、参数更新、
loss、评价或基线定义。配置值不是永久研究规则，但正式运行必须保存配置副本。

## Implementation philosophy
- 只实现当前 V1，优先最短的正确、直接、可读的科研代码。
- 不添加未要求的 fallback、自动修复/重试、多后端兼容、插件、缓存或复杂 CLI。
- 不增加数据清理、算法、K 设置、多数据集或未来功能。
- 规范未定义且影响实验含义的问题，先报告用户；普通实现细节采用最简单合理方案并记录。
- 不以健壮性、性能或复现便利为理由改变实验含义。

## Organization
- 数据、任务、模型、训练、评价、配置分别放置；一个文件一个主要职责。
- 不把整个流程塞进单个脚本，不提前设计通用框架。
- 超参数读取配置。训练不得重新抽样/切分，具体 row ids 由 manifest 提供。
- SHA 选择集中在 `tasks/stable.py`，不扩散 hash/seed 检查。
- 结果只写入 `outputs/`，原始 CSV 只读。
- 代码注释以模块和主要函数的功能说明为主，避免逐句解释和大段易错点提示。

## Baselines
- 来源：andryr/indoor_localization，commit f34f1bf4560109254887e2f04166dd61e0d9dae9。
- 先提取原实现，保留来源，再最小改造接入 manifest；禁止凭印象重写近似网络。
- 全部算法共用同一个 support/query manifest。

## Tests
只检查必要科学不变量：位置隔离、T2 建筑隔离、query 标签不进 adapt、
各方法相同 row ids、关键 tensor shapes 和参数更新/重置语义。
不用通用测试框架，不扩展几十个单元测试。

## Scope
V1 = WKNN + MLP-Scratch + MLP-FT + FeMLoc-RI + FeMLoc。
GUFU、DANN、额外 AP 表示、时间适应不属于本版。
先完成数据→manifest→B0F1 WKNN/MLP→FeMLoc 前向与单轮训练，
实际确认后再执行完整训练。测试运行不能冒充正式结果。

## Operations
- 本项目后续 GitHub 推送统一经本机 SSH 临时隧道；不再尝试服务器直连。
- GitHub 凭证和认证配置限定在本项目 `.git/`，不得写进代码、日志或提交。
- 当前及后续授权的实验必须纳入自动值守，完成后自动汇报；启动新批次时同步更新值守范围。
- 值守以完成标记、结果文件及实际进程判断状态；失败保留日志并汇报，不盲目重启或改实验配置。
