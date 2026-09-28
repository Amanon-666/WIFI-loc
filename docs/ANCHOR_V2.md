# UJI 跨空间适应 V2

用户于 2026-09-28 授权实施并基于证据重构，不锁死算法。V1 文件与结果冻结。

## 首轮实现
复用现有目标 row ids、位置隔离、K=10/r=3 和源 episode 行号。
开发目标 B0F1/B1F3；源为 B0/B1 的其余七楼层，B2 不参与选择。
每位置三扫描先归一化后平均，参考库只使用 Support。
最近 h=5 个不同位置；距离为归一化 RSS 的 520 维 RMS 距离，参考库未见 AP 屏蔽。
距离相同时按参考点 (x,y) 升序；这是 V2 明确的确定性工程选择。
输入依次为五组 [距离,(参考x-参考库中心)/100,(参考y-中心)/100]。
Support 每个位置训练时剔除自身全部扫描和坐标，再重算参考库/AP/中心。
Query 独立使用完整 Support 库，Query 不相互连通，不参加适应。
中心是参考库独特位置均值；网络预测相对坐标，评价恢复米制坐标。

MLP 15→256→128→64→32→2，隐藏 ReLU，无 BN/Dropout，Xavier/零 bias。
Scratch/ERM/MAML 同初始化。ERM 每域 Support 和 Query MSE 各一半；域等权。
MAML 每域从共同参数出发，5 次 Support SGD(.01)，Query 梯度一阶传回，
外层 Adam(.001)，1000 轮。目标 SGD(.01)，0/5/10/20/50/100/200/500 快照。
完整网络参与适应，不重置坐标头。配置在 configs/anchor_v2.yaml。
raw 对照为同隐藏层的 520 输入网络；AP 表和坐标中心均来自该 episode Support。

## 方法来源与解释
MetaLoc §II-C1 的 RSS 参考点距离/坐标表示与 Algorithm 1 一阶 MAML 为来源。
三扫描原型、留一位置训练、网络尺寸和工程参数为本项目改造，不能称原样复现。
官方代码 StatFusion/MetaLoc master 2a3f7ae6dffcf7ebfc72a82b8091cc23db1aead9
主要是 CSI 分类，借用 fast weights/一阶更新语义，不搬分类与训练态 Query BN。
WKNN 沿用 models/wknn.py。所有算法同一目标 Support/Query；N 不表示排除采集混杂。
已有开发数据曾用于诊断；后续 B2 才用于候选冻结后的迁移评估。
源训练耗时和目标适应耗时分开报告；不预设 MAML 或 GGA 必须优于 ERM/WKNN。

## 运行与后续
env/bin/python -m scripts.check_anchor
env/bin/python -m scripts.run_anchor --config configs/anchor_v2.yaml
结果 outputs/anchor_v2_dev；完成标记只在全部方法成功后生成。
每轮改造在此文档追加证据与选择，旧结果不覆盖。不做无界参数搜索。
GGA、关系表示改造等由实际瓶颈决定；不得由梯度范数比例直接断言负向冲突。
