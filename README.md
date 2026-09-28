# UJI-FeMLoc-V1

正式项目在 **ray-web**：`/home/panyushuo/projects/panyushuo/UJI-FeMLoc-V1`。
实验依据：`docs/UJI_FeMLoc_V1_SPEC.md`；项目规则：`AGENTS.md`；固定配置：`configs/v1.yaml`。

## Environment

服务器独立环境 `env/` 从已有 GSRF 环境克隆，不修改原环境；Python 3.10、PyTorch
2.8.0+cu128。具体安装清单保存于 `requirements.lock.txt`。训练使用空闲的 A4000 GPU 0，
主力服务器没有 Slurm。原始 CSV 使用已有服务器副本，只读。

## First-stage commands (run in the server project root)

```bash
env/bin/python -m scripts.build_manifest --task T1 --target B0F1 --repeat 0
env/bin/python -m tests.check_invariants outputs/manifests/T1_B0F1_e0.json
env/bin/python -u -m scripts.run outputs/manifests/T1_B0F1_e0.json --phase phase1
```

Manifest 中保存全部 1000 个源训练轮次和 AE 的实际 row ids；训练不再抽样。
`phase1` 只读取首个源训练轮次，目标仍按 V1 执行 AE200/监督500步。
它还运行 B0F1 的丰富标注 T0 WKNN/MLP。单轮源模型只验证链路，不能用来判断 FeMLoc 成败。
`full` 使用同一个 manifest 的全部1000轮，其他语义不变：

```bash
env/bin/python -u -m scripts.run outputs/manifests/T1_B0F1_e0.json --phase full
```

新目标通过 `build_manifest --task T2 --target B2F3 --repeat 0` 生成。
每个目标/重复独立适应；T2 可显式传入同源域、同重复的 `--source-checkpoint`，
共用预训练模型。程序核对实际源训练行号，不自动查找或替换 checkpoint。
批量运行采用顺序命令清单，不实现调度框架、断点恢复或自动重试。
manifest 和结果目录存在时不覆盖；无需为复现重新生成 manifest。
当前已接入自然条件 N；规范中的 C-F/C-B 受控筛选尚未实现，不能通过修改 profile
把自然条件结果标成受控结果。13目标×5重复的批量结果与跨建筑汇总尚未完成。

## Layout

- `data/`: CSV 清理、只读数据接口、预处理。
- `tasks/`: 唯一的稳定排序函数与 row-id manifest 构造。
- `models/`: 原仓库提取模型及 FeMLoc 网络。
- `training/`: 源训练、目标适应、基线训练。
- `evaluation/`: 冻结推理、标签合并与位置等权评价。
- `scripts/`: 简单入口。
- `tests/`: 必要科学不变量与原基线等价性检查。
- `vendor/andryr/`: 指定 commit 的原始 notebook。
- `outputs/`: manifests、检查记录、权重、逐扫描预测和指标。

每次运行保存 config、manifest 路径、真实源训练轮数、环境、预测 CSV 和 checkpoint。
目标 query 标签只在评价中连接；目标适应函数不能接收 query。
