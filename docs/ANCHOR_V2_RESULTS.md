# V2 初轮探索记录（2026-09-28）

三批×两开发楼层全部完成，每组WKNN与三个训练方法，共6组；单repeat0。
源是B0/B1的其他七楼层，1000轮；目标10位置×3扫描，500步。
这是混合源楼层开发，不能标为正式跨建筑结果。

|方法（500步）|B0F1/m|B1F3/m|
|---|---:|---:|
|WKNN（0步）|13.426|37.264|
|anchor-erm|17.111|43.337|
|anchor-maml|15.417|43.674|
|anchor-scratch|23.105|48.649|
|raw-erm|6.117|35.627|
|raw-maml|15.993|34.445|
|raw-scratch|16.276|51.892|
|scan-erm|13.931|38.066|
|scan-maml|14.406|38.413|
|scan-scratch|13.626|37.184|

## 判断
- Raw-ERM在B0F1较好，Raw-MAML在B1F3略优；无一致的MAML优势。
- 两开发目标Support AP均在源中出现，不能将Raw的结果外推到新建筑。
- Anchor/Scan同时变化多项因素，仅为方法探索，不支持单因素因果结论。
- 残差方案未获得稳定优于WKNN的收益，停止追加补丁。
- 原型均值退化的归因已撤回，原因见METHOD_REVIEW.md。
- 所有配置、source训练曲线、0/5/10/20/50/100/200/500预测与权重均保留。
- source_seconds包括源episode处理；adapt_seconds仅记录SGD与快照过程，
  不包含目标Support特征构造，不能作为完整部署耗时。

## 文件
- outputs/{anchor_v2_dev,anchor_scan_v2_dev,anchor_raw_v2_dev}/{target}_e0/
- outputs/anchor_v2_exploration.csv
- configs/anchor{,_scan,_raw}_v2.yaml

本轮没有在B2训练/评价定位模型；仅在方法审查中读取B2目标Support的AP覆盖情况。
后续转向先方法推导再实验；本报告不宣称已经交付跨建筑性能提升。
