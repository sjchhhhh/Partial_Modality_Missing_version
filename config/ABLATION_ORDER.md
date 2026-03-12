# 消融顺序：逐步关闭约束，看谁导致性能下降

在 `config_regression.py` 的 **emt-dlfr → mosi（或 mosei/sims）** 里改下面几项，每次只改一项、跑一次、记一次结果。master 与 new 项目逻辑一致，你自己控制不同消融即可。

---

## 总 loss

`total = loss_pred_m + loss_recon_weight*loss_recon + uncertainty_reg_weight*loss_uncertainty_reg + loss_attra_weight*loss_attra`

---

## 消融步骤（按顺序做）

| 步骤 | 修改 | 含义 | 若性能变好说明 |
|------|------|------|----------------|
| **步1** | `uncertainty_reg_weight` → **0** | 关掉方案1（σ 校准正则） | 方案1 导致性能下降 |
| **步2** | `use_uncertainty_weighted_attra` → **False** | 高层对齐改为纯 SimSiam（与 based 一致） | 方案C 的不确定性加权导致下降 |
| **步3** | `loss_attra_weight` → **0** | 关掉整个高层对齐 | 高层对齐整体在拖后腿 |

---

## 建议

先做步1，若步1 变好则基本定位到方案1；再试步2；最后可用步3 看高层对齐是否值得保留。master 与 new 用同一套逻辑，仅在 config 里调不同数值即可做对比。
