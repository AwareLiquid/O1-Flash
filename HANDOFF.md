# O1-Flash — 会话交接文档

## 0. 当前快照（2026-09-18，初建）

- 独立仓库（本地 `E:\O1-Flash`，云端 `AwareLiquid/O1-Flash`），O 系列
  ARR 的 System One 化：纯决策读出，无文本生成
- 核心：`mt_flash/`（liquid_core / decision_heads / calibration / schema
  / model / state_tokenizer / config），15 测试全过
- 定位：Jev（TypeSafe 2026-09-15）并行决策理念 × O(1) 液态状态的端侧版

## 1. 已完成（真实验证过）

| 项 | 证据 |
|---|---|
| 液态递归核独立移植（MTLNNLayerV2 数学） | `tests/test_liquid_core.py`：O(1) 状态平坦/流式前缀=全量/重置确定性/选择性衰减 init 恒等 |
| 屏蔽扫描（pad 不污染状态） | 实测：无屏蔽时信号在前 0.48 → 屏蔽后 1.0（DESIGN §7） |
| 混合读出（联合训练修复） | 联合 3 问题：纯液态 0.19-0.31 → 混合 0.73/0.81/1.0（DESIGN §7） |
| **mean-pool 读出定型（2026-09-20）** | 变体对比见 DESIGN §8：mean-pool 0.977/0.973/1.0；default 配置 dept/sev 1.0、urgent 0.992 |
| 并行决策头（每问题独立读出，无跨问题通路） | `tests/test_decision_heads.py`：问题独立性/schema 有界/置信度=峰值度 |
| 校准参考 + 联合训练循环 | `tests/test_calibration.py` + `tests/test_train.py`：Brier/ECE 手算值 + 联合训练全问题过阈值 |
| 级联路由 FastSlowRouter | `tests/test_cascade.py`：阈值分区/边界校验 |
| 分块向量化扫描（静态衰减） | `tests/test_liquid_core.py::test_chunked_scan_matches_sequential`：与顺序循环等价（T=32 内 1e-3） |
| ONNX 固定 schema 导出 | `tests/test_export_onnx.py`：onnxruntime 实际推理与 PyTorch 决策级一致 |
| 真实基准（模板改写 + 留出集） | `tests/test_realistic_bench.py`：留出组合泛化超随机基线 |
| 全测 | `pytest tests/ -q` → 26 passed |

## 2. 进行中 / 待办

- [ ] 真实数据决策基准（模板库仍是合成；OOV 泛化需真实票集）
- [ ] 选择性衰减的向量化扫描（当前 selective 路径仍走顺序循环）
- [ ] ONNX 导出官方部署说明（量化 int8、多平台 runtime 示例）
- [ ] 训练参考的数据管线化（无 pad 屏蔽之外的增强）
- [ ] 混合读出的消融：原始嵌入并接是否可在流式下渐进式处理（当前仅 prefill）

## 2b. 训练诊断补充（2026-09-20，另一会话 A100 通道）

- **两层隔离**：① 线性探针（bag-of-bytes→dept）acc 1.000 → 数据可分；
  ② mean-pool + 线性头 on y → dept 0.996 → **编码器携带信号**，纯液态读出
  是瓶颈（与 §1 混合读出的修复一致）。
- **线性头参考上限**：encoder + 线性头（tiny 与 default 两档）三任务全 1.0
  （default d256/4L loss 3.45→0.04）——**线性头是读出侧的能力天花板**，
  混合读出（0.73/0.81）与 option-matching（0.22/0.27）之间的差距 =
  读出机制的可改进空间。
- 结论：信号损失不在编码器（y 可分），在读出对齐；混合读出的修复方向正确。

## 3. 要避免的坑

- **状态尺寸必须与 T 无关**：任何把 h 改成含 T 维度的"优化"都是架构回归，
  `test_state_size_flat_across_context_lengths` 就是那条红线
- **问题头之间不许加注意力**：跨问题通路会破坏独立性测试，也会破坏
  "加一个问题不涨价"的成本结构
- **置信度只认峰值度**：别换成 softmax max 等未定义语义的量
- 本仓不引入 M1 依赖；数学移植逐行对照，代码独立实现
