# UJI-FeMLoc V1 实验规范

本文件落实用户批准的 V1 方案及开工修订。实验语义由本文件固定，数值超参数
集中于 `configs/v1.yaml`，属于 V1 fixed config，不宣称永久最优。
本版是依据公开论文的重实现与 UJI 任务改造，不是已复现论文数值的声明。

## 1. 依据与范围

- FeMLoc: https://arxiv.org/html/2405.11079v1 （III/IV 节，表 II）。
- 基线：https://github.com/andryr/indoor_localization/tree/f34f1bf4560109254887e2f04166dd61e0d9dae9
- 方法：WKNN、MLP-Scratch、MLP-FT、FeMLoc-RI、FeMLoc。
- 已知 Building/Floor，输出 LONGITUDE/LATITUDE 两轴原始平面坐标；不做楼层分类。
- User/Phone/Time/SpaceID/RelativePosition 不输入模型。
- GUFU、DANN、额外自监督、更多 K、多数据集、时间适应均不在本版。

## 2. 样本、位置、采集组

原始样本为 `(row_id, rssi[520], xy[2], domain, condition)`。
row_id = trainingData.csv 从 1 开始、不含表头的数据行号。
domain = Building-Floor，字符串如 B0F1；位置 = (B,F,x,y)，不舍入合并。
位置 ID = 紧凑 JSON [domain,规范化十进制 x 字符串,规范化十进制 y 字符串]。
采集组 ID = 紧凑 JSON [position_id,USERID,PHONEID,TIMESTAMP]。
同组可以有多条不同 RSSI，不能自动合并。

仅使用 trainingData.csv。validationData.csv 不参与主任务或预处理拟合。
顺序：删除全 520 WAP=100 的记录；按全部 529 字段数值去重，保留最小 row_id；
只保留具有至少 3 个采集组的位置。不能按 RSSI 去重或新增异常值修复。
已核验：19937 原始行→19227 唯一有效行→926 合格位置/19201 行。
原始训练文件 SHA256：45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168。

## 3. 划分与 manifest

每域合格位置按 stable_order 排序，前 ceil(0.2*N) 个为固定 P_Q，其余 P_C。
目标 episode 从 P_C 取 K=10 个位置，每点 r=3 个不同采集组，各组一条扫描。
Query 为 P_Q 全部有效扫描。未被选入 support 的目标数据不可供训练读取。
重复编号 e=0..4。同域的 T1/T2 和全部算法共用目标 S/Q。

排序哈希为 SHA256(UTF8 compact JSON [protocol,seed,purpose,profile,domain,counter,object_id])。
protocol=UJI-FeMLoc-v1，seed=20260927，profile=N；域 ID、位置 ID、采集组 ID
按第 2 节构造，行 ID 使用整数。JSON ensure_ascii=False、separators=(',',':')。
按哈希升序、对象 ID 破同值。position_split 的 counter=-1；目标 support_position、
support_group、support_row 的 counter=e。算法名不进入选择键。
源训练用途为 source_support_position/group/row、source_query_position/group/row，
counter=[e,round]，round=0..999。
选择仅发生在任务生成模块。manifest 保存具体行号（包括源训练 batch、AE batch）；
训练程序只读这些行号，不重新排序/抽样。

AE 与 T0 的有放回均匀抽样：任务生成阶段用 NumPy Generator(PCG64)，种子由
同一哈希工具的用途键产生。逐条均匀选择位置→组→行，保存结果。此 RNG 是普通
实现细节，不能改变三层均匀抽样语义。

## 4. 任务

- T0：目标域 P_C 全部数据训练，在 P_Q 评价，仅 WKNN/MLP，不能传权重给 T1/T2。
- T1：(b,f) 的源域为同 b 下所有 f'!=f。
- T2：(b,f) 的源域为所有 b'!=b 的楼层，排除整个目标建筑。
- 正式范围：13 个 T1、13 个 T2，每个 5 次重复。同目标建筑 T2 可复用源模型，
  各目标楼层适应状态完全独立。
- 首批：T0 B0F1；T1 B0F0/F2/F3→B0F1；T2 B0/B1 的 8 个楼层→B2F3。
- 自然条件主实验不声称已经消除用户/设备/时间影响。
- 受控补充：B0F0→B0F1、B1F2→B2F3。共同条件 c=(user,phone,UTC 日期)，
  两侧合格位置均至少 30，最大化 min(源位置数,目标位置数)，并列按 user/phone/date
  排序。筛选后重新划分，版本分别 C-F/C-B，单独报告，不能称为纯空间因果效应。
  本机两侧位置数分别 52/61、48/60。受控补充在自然条件首批链路完成后接入。

## 5. 预处理

神经网络 z=0 if RSSI=100 else (RSSI+110)/110。合法观测范围 [-110,0]。
不从 query 估计统计量，不做设备校正、增强、mask 拼接、幂变换。
源 AP 表只从 P_C 扫描观测到的 AP 构建；目标 AP 表只从当次 support 构建；
按 WAP 原列顺序排列。FeMLoc 提取 m 列；基线保留 520 列，AP 表外全部置为缺失。
Query-only AP 不扩充输入，零共有 AP 的 query 也保留。
源坐标原点=源 P_C 不重复位置均值；目标原点=K 个 support 位置均值。
标签 y=(xy-origin)/100；输出 xy=origin+100*yhat。两轴 LONGITUDE/LATITUDE，
原值与减原点 float64，网络张量 float32。评价原始平面坐标距离，不转换地理坐标。

## 6. 网络与 AE

- Encoder E: m→1024→50，隐藏 ReLU，输出线性。
- Decoder D: 50→1024→m，隐藏 ReLU，输出 Sigmoid。
- Shared G: 50→256→128→64→32，隐藏 ReLU，输出线性。
- Mapper M: 32→64→32→2，隐藏 ReLU，输出线性。
- f=M(G(E(z)))。固定 latent=50（表 II），不按 AP 中位数重新算。
- 无 BN/Dropout/残差；Linear Xavier uniform(gain=1)，bias=0。
- 定位损失=逐元素平均 MSE，即 sum_i ||pred_i-y_i||²/(2*N)。
- AE 损失=全部 AP 元素平均 MSE，包括缺失对应的零。
- 源 AE：仅 P_C RSSI，200 步，batch32 三层均匀有放回抽样。
- 目标 AE：仅 30 条 support，200 步全 batch。
- AE Adam lr=.0095，更新 E/D；完成后丢弃 D 和 AE optimizer，定位阶段只有定位 MSE。
AE 时序、激活、初始化和优化器状态是补齐原文空缺的工程规定，不冒充原文实验细节。
30 条扫描训练大 encoder 的过拟合风险保留在结果分析中，不因此换网络。

## 7. 源 meta-training

每源域 E 经 AE 预训练，M 随机初始化，唯一 G 初始化为 theta0，并保存 theta0。
1000 轮，每轮全部源域参与，全部接收同一轮初 theta；私有 E/M 延续上轮状态。
每源域从 P_C 选10位置×3扫描为 S，从 P_Q 选10位置×3扫描为 Q。
内循环：同一份 S 全 batch 更新5步，联合更新 E/G/M。
Adam E lr=.0095，G/M lr=.0005；betas=(.9,.999)，eps=1e-8，weight_decay=0。
E/M Adam 状态跨轮保留；局部 G 参数每轮广播覆盖，G Adam 状态每轮新建。
每步同一前向先得到全部梯度，再更新各参数组。
在适应后的 E/G/M 上算 Q 定位损失对局部 G 的梯度，不更新 E/M，不穿过内循环求导。
服务器 theta_next=theta_round-.001*mean(client_query_grad)，无动量，无 Adam。
不是局部模型权重平均。全体完成后才更新服务器，随后进入下一轮。
返回第1000轮 theta_star，不用目标 query 选模型。私有状态保留仅用于源训练。

## 8. 目标适应

每域每 episode 全新建立 AP 表、原点、E/D/M；E 只在 support 上 AE 预训练。
FeMLoc G=theta_star；FeMLoc-RI G=theta0；两者复制相同目标 E/M 状态。
新建 Adam，E lr=.0095，G/M lr=.0005，全30条 support监督500步，更新全部三模块。
记录0/5/10/20/50/100/200/500步，0指AE完成后。500为主结果，不挑最好query步数。
Query 只在冻结推理中使用。适应接口无 query 参数。耗时包含AE与监督更新。
目标域、episode、任务之间不继承适应后的权重或优化器。

## 9. 基线

来源 notebook 原件保存在 vendor/andryr。MLP 类和 knn_weight 从原件提取，
修改清单保存在 docs/BASELINE_PROVENANCE.md，不能重写近似结构冒称提取。
WKNN：所有support扫描为库，3近邻，原RSSI尺度欧氏距离（缺失=-110），
w=1/(distance+1e-6)^2，坐标加权平均。距离并列按row_id升序；不按位置预聚合。
MLP：520→256→256→128→128→64→32→2，每隐藏 Linear→BN→LeakyReLU(.01)。
BN eps=1e-5，momentum=.1，affine/track_running_stats=True；初值scale1/bias0/mean0/var1。
Linear使用统一Xavier初始化。保留原层数和forward顺序。
Scratch：目标support上Adam .001，全batch500步，MSE；query用eval，不能更新BN。
FT：共享520→32 backbone，各源域独立32→2头。1000轮，按BF升序逐源域，
复用同份源episode，S普通监督5步+Q普通监督1步，Adam .001，状态连续。
每步只更新backbone及当前域头，backbone不广播重置，不做meta-gradient。
目标复制预训练backbone，目标头与Scratch初值相同；BN affine保留，running统计重置。
重新Adam .001，全部参数适应500步。
T0 MLP：P_C全部数据，5000步，batch32三层均匀有放回抽样，Adam .001。
不使用原仓库USERID阈值划分、坐标常量、ReduceLROnPlateau/early stopping、CUDA硬编码。

## 10. 初始化与记录

模型seed=上述H前8个十六进制字符转uint32；用途键model_shared/encoder/decoder/
mapper/baseline_backbone/baseline_head（均带model_前缀），计数器e，对象ID为模块名，
domain槽为按BF排序的源域ID列表或目标域ID。每模块独立初始化，不依赖创建顺序。
Scratch backbone初值=FT源训练开始前backbone初值。目标RI/MI私有模块复制同一状态。
每次输出保存config、manifest路径、运行范围、环境版本、checkpoint、预测和指标。
阶段1单轮结果必须标明phase1，不冒充1000轮正式结果；不实现复杂恢复/缓存框架。

## 11. 评价

扫描误差ei=||pred_xy-true_xy||2；先组内平均eg，再位置内组平均ep，再位置等权均值E。
同时报告ep中位数、P90（排序第ceil(.9*N)项）、扫描等权MDE、目标总适应耗时、
query可见AP中未被support AP表覆盖的比例（未覆盖观测总数/可见观测总数）。
每域5重复mean/sample SD，T1/T2分开；同建筑先楼层均值，再三个建筑等权均值。
受控补充单列。相同域/episode配对算算法差值。没有算法必须赢的验收要求。

## 12. 实现接口和必要检查

数据/data；划分/tasks；网络/models；源训练及adapt/training；评价/evaluation；
配置/configs；入口/scripts。单文件单主要职责，不扩展工厂/插件/复杂CLI。
adapt(checkpoint, support)：support含RSSI、坐标、row_ids；接口不能有query。
predict(state, query_features)：query仅RSSI和row_ids，不含坐标编码的位置ID。
evaluate(predictions, labels)：只有评价模块合并真实query坐标。
必要检查：position隔离、T2排除目标建筑、adapt无query标签、各算法同row_ids、
形状，以及meta共享/私有更新状态；不扩展通用测试套件。

## 13. 阶段交付

正式项目：ray-web:/home/panyushuo/projects/panyushuo/UJI-FeMLoc-V1。
第一阶段在该服务器GPU上建立并实际运行：读取数据、持久化manifest、B0F1 WKNN和MLP，
FeMLoc前向和一整轮训练。固定超参数仍保留在v1.yaml；缩短源轮数只由明确的
phase1运行范围表示并记录，不能暗改正式配置。确认后再进行完整1000轮/500步实验。
