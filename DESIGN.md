# O1-Flash 设计文档 — JEV 并行理念映射

## 1. 为什么是 O 系列

M1 仓库三条产品线的成熟度盘点（2026-09-18）：

| 线 | 实测资产 | Flash 适配度 |
|---|---|---|
| M 系列（混合注意力+液态） | 跨窗回忆 0.56、EWC 抗遗忘 | 认知/Pro 线，不该做 Flash |
| **O 系列（ARR 无注意力）** | **O(1) 0.381MB 状态、8063×@1M、5MB ONNX 唤醒词、CPU 推理、电池流** | **唯一产品级实测，唯一候选** |
| M2（实验仓） | DPO/GRPO 参考、架构实验 | 不是模型 |

O 系列的短板（2.15× teacher PPL 的研究预览）恰好被 System One 定位绕开：
不做文本生成，PPL 不再是核心指标——核心指标变成决策准确率与置信度校准，
这两者恰好是 O(1) 状态 + 并行读出的主场。

## 2. JEV 并行理念 → O1-Flash 的映射

Jev（TypeSafe, 2026-09-15，Diogo Almeida）的四个核心理念与本仓库的落点：

| Jev 理念 | Jev 实现 | O1-Flash 落点 | 差异 |
|---|---|---|---|
| **共享状态编码一次** | KV cache（O(T)）| 液态递归携带状态 **O(1)** | 我们更强：状态大小与长度无关 |
| **问题独立并行读出** | 决策头一次前向，无跨问题注意力 | `decision_heads.py`，结构上无跨问题通路 | 等价（有测试锁定） |
| **类型化输出，schema 内不可能出错** | Choice/Score/Probability | `schema.py` 同名三类型，输出只可能是概率向量 | 等价 |
| **校准置信度（RLCD）** | 未公开方法 | `calibration.py`：Brier+CE 训练、ECE 报告（参考实现，诚实声明非 RLCD） | 我们是公开参考配方 |
| **级联快慢** | 快模型 + 慢 LLM 按置信度路由 | 设计对齐 M1 的 DualSpeedSentry/salience 点火管线（未在本仓实现） | 复用而非重造 |

## 3. 液态递归核（`liquid_core.py`）

M1 `mt_lnn/mt_lnn_v2.py` MTLNNLayerV2 的独立移植（逐行对照数学，非复制）：

1. 因式分解投影：D → r → P·d
2. 每原丝混合 W_mix (P,d,d)，跨尺度共享
3. 每 (proto, scale) 对角 sigmoid 调制 A_ps
4. τ 阶梯（softplus 参数化，几何扩展）+ 可选选择性衰减
5. 扫描：h_ps,t = decay·h_ps,t-1 + (1-decay)·A_ps,t —— **顺序循环（O(T) 算力，O(1) 状态）**
6. κ 门控 softmax 尺度混合（内容依赖）
7. 每维 SiLU 输出门（Mamba 式）

**已知差异**：M1 有 Blelloch 并行扫描/chunkwise（SSD 式）路径，本仓用顺序
循环——数学等价、状态契约相同，作为后续优化记录在案（见 HANDOFF 待办）。

**init 纪律**（与 M1 一致，测试锁定）：
- `selective_decay` 开启时 w_dt 零初始化、b_dt=softplus⁻¹(1) → dt≡1，
  与静态路径逐位一致（`test_selective_decay_inits_to_static_path`）
- κ 门零初始化 → 尺度混合从均匀开始
- 残差恒等：所有投影无偏置

## 4. 并行决策头（`decision_heads.py`）

```
state (B, D) ── choice_proj ──> key (B, D)
                                   │  bilinear: key · option_embedding
option text ── 共享字节嵌入 ──> opts (1, N, D) ──> logits (B, N) ── softmax
```

- **Choice**：选项经同一冻结嵌入层（无递归，便宜）→ 双线性打分 → 分布
- **Score**：state → 线性 → levels 分布（有序语义由标签序承载）
- **Probability**：state → sigmoid → p
- **置信度 = 分布峰值度**（Jev 定义）：扁平分布=低置信，无论谁赢
- **并行性由结构保证**：任何问题头只读共享 state，无跨问题注意力 →
  "加不加问题 B，A 的答案逐位一致"是可测试的属性，不是承诺

## 5. 校准（`calibration.py`）

RLCD 的公开参考配方（诚实：TypeSafe 方法未公开，我们给的是同名目标的
开源实现，不声称等价）：
- 损失 = Brier（proper scoring rule）+ 决策 CE
- 报告指标 = 决策 ECE（置信度=峰值度，正确性=argmax）
- 不做 RL——纯监督配方，README 已注明差异

## 6. 诚实边界

1. 本仓无训练权重：是架构+训练参考，与 M2 的 DPO/GRPO 参考同级
2. 无决策质量数字：不宣称与 Jev 的 67.8% 可比
3. 顺序扫描未优化：长输入的吞吐优化待做
4. 级联（快→慢路由）只做了设计对齐，未实现——M1 的 pipeline.py 是现成蓝本
