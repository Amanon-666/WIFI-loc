# 首个完整 T1 运行

服务器：ray-web。任务：B0F0/F2/F3 → B0F1，episode 0。
源训练1000轮；目标AE200步（FeMLoc/RI），监督适应500步；30条support，285条query。

| 方法 | 位置等权平均误差（UJI原始平面坐标） |
|---|---:|
| WKNN | 13.4264 |
| MLP-Scratch | 33.8957 |
| MLP-FT | 15.5190 |
| FeMLoc-RI | 13.9944 |
| FeMLoc | 13.9252 |

这是单目标、单次重复，不能据此断言方法普遍优劣。网络、数据划分和超参数均未因结果改变。
完整结果目录：`outputs/full/T1_B0F1_e0/`；日志：`outputs/full_B0F1_e0.log`。
全部五种方法使用同一manifest；神经网络checkpoint中的support/query行号及500步快照已核验。
待完成：其余目标/重复、完整跨建筑实验、受控补充实现、跨目标汇总。
