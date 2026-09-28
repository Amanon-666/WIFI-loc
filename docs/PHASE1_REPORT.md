# 第一阶段运行记录

执行位置：ray-web；项目目录 `/home/panyushuo/projects/panyushuo/UJI-FeMLoc-V1`。

数据与协议：T1 B0F0/F2/F3 → B0F1，episode 0，10位置×3扫描，固定285条query/14个query位置。
主方法及FT只执行1轮源训练；目标AE200步、监督500步。以下跨空间数字仅为链路核验结果。

| 方法 | 位置等权平均误差（UJI原始平面坐标） |
|---|---:|
| WKNN | 13.4264 |
| MLP-Scratch | 33.8957 |
| MLP-FT | 28.0665 |
| FeMLoc-RI | 13.9944 |
| FeMLoc | 15.5147 |
| T0-MLP | 6.2533 |
| T0-WKNN | 6.0948 |

通过的检查：位置隔离、T2源建筑排除、query标签不进入adapt接口、五方法相同行号、
原notebook MLP前向等价、原WKNN计算等价、FeMLoc式(11)更新、RI/MI目标私有模块同初值。

原始预测、checkpoint和指标：`outputs/phase1/T1_B0F1_e0/`。
检查记录：`outputs/checks/`。日志：`outputs/phase1_B0F1.log`。

运行耗时未经预热控制，首个神经网络运行包含CUDA冷启动，不能据此作方法速度排序。
第一阶段未调参、未改变网络、未丢弃预测困难的query。
