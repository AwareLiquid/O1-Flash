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
| 并行决策头 | `tests/test_decision_heads.py`：问题独立性（Jev 关键属性）/schema 有界/置信度=峰值度 |
| 校准参考 | `tests/test_calibration.py`：Brier/ECE 手算值 + 可反向传播 |
| 全测 | `pytest tests/ -q` → 15 passed |

## 2. 进行中 / 待办

- [ ] 训练脚本（`calibration.py` 是损失级参考，尚无数据管线/循环）
- [ ] Blelloch 并行扫描（长输入吞吐；数学等价，见 M1 pscan/chunkwise）
- [ ] 级联：快→慢路由（M1 `pipeline.py` DualSpeedSentry 是蓝本）
- [ ] ONNX 导出（M1 `export.py` 5MB 唤醒词经验可直接迁移）
- [ ] 决策基准（自建 eval：分类/打分/路由集，ECE 报告）

## 3. 要避免的坑

- **状态尺寸必须与 T 无关**：任何把 h 改成含 T 维度的"优化"都是架构回归，
  `test_state_size_flat_across_context_lengths` 就是那条红线
- **问题头之间不许加注意力**：跨问题通路会破坏独立性测试，也会破坏
  "加一个问题不涨价"的成本结构
- **置信度只认峰值度**：别换成 softmax max 等未定义语义的量
- 本仓不引入 M1 依赖；数学移植逐行对照，代码独立实现
