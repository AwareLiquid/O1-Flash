# JevBench 官方成绩汇总（2026-10-07）

官方 harness（fstandhartinger/jevbench）+ 原生 /v1/systemone 服务。全部
schema_validity = 1.0、operational_success = 1.0、charged $0。

## 三方对照（public tiers，官方 harness 实测）

| 模型 | easy acc | orig acc | 合计 | ECE | p50 | Speed 轴* |
|---|---|---|---|---|---|---|
| M2-2B SFT v2 | 0.333 (16/48) | 0.375 (27/72) | **0.358** | 0.50/0.34 | 0.92/0.71s | ~56 |
| M2-2B SFT v2b | 0.292 (14/48) | 0.333 (24/72) | 0.313 | 0.52/0.41 | 0.85/0.69s | ~57 |
| **O1-Flash 决策头 dv1** | 0.229 (11/48) | 0.306 (22/72) | 0.271 | 0.43/0.41 | **0.016/0.021s** | **~100** |

\* Speed 轴 = 100 − 20·log10(p50/0.1s)（官方公式），100 封顶。

## 读数

1. **速度/成本轴我们是天花板**：dv1 单前向 readout = **16-21ms**（比 2B-SFT 快
   40×，比榜首 Sage 的 0.15s 快 ~8×）→ **Speed 轴 100 封顶**；4.5M 端侧 →
   Cost 轴亦在 80-100 档。这两轴榜上是 52-61（他们守金矿不动）。
2. **准确率是墙**：dv1（4.5M、36k 合成/重排语料、8k 步）= 23-31%，**低于**
   2B-SFT 的 33-38%——合成语料（模板族）迁移到 JevBench 真实题不佳
   （intent 2/12、extraction 1-4/12、fact 6/12）。机会校正后 ≈ 0-5，
   远低于 50 门 → 官方复合会被门压制（与 Fastino GLiNER 340M 的处境同型）。
3. **校准待修**：ECE 0.41-0.52（温度缩放可修到 ~0.1 量级——已知手法）。
4. v2b（续训 5k 步的 2B-SFT）：生成更好了（loss 0.59）但 JevBench 准确率略降
   （0.358→0.313）——SFT 语料与决策题分布不同，训练更多 ≠ 决策更强。

## 结论与下一步

- **管线资产成立**：官方口径下我们已有 3 个模型的合法分数（可复现、可提交）。
- **决策头架构方向正确**（速度/成本达标），**待解的是"智力"**：
  正解组合 = **2B 冻结核 + 决策头**（单前向 readout ≈0.1-0.2s → Speed 仍 90-100，
  而智力来自 2B 的语言表征），或决策头在**真实分布语料**上更长时间训练。
- **马上可做**：dv1 的温度缩放（校准轴）、2B-core+head 教学（下一条线）。

产物：`/root/jev_runs/{dv1b_easy,dv1b_orig,v2b_easy,v2b_orig,...}/`（官方
results/raw/ledger/summary）；服务 `benchmarks`/`o1flash/jev_service_flash.py`；
训练器 `benchmarks/decision_train.py`。
