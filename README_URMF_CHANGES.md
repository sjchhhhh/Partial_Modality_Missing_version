# old_urmf_fullseq 项目说明

基于 `old` 项目创建，做两处修改：

## 1. 不确定性输入：整个模态序列特征

- **文本**：对 BERT 完整序列 (B, T+1, text_out) 做不确定性估计，再 `_masked_mean_pool` 到 utterance 级
- **音频**：对 LSTM 时序输出 (B, T, audio_out) 做不确定性估计，再池化
- **视频**：同上

与 `old_full_unc` 一致，而非 `old` 仅用 utterance-level 表示。

## 2. 不确定性计算：与 Code_URMF-main 一致

- **公式**：`μ = Linear(x)`，`logvar = Linear(x)`，`σ² = exp(logvar)`
- **方差**：`σ² = exp(logvar)`，不使用 Softplus
- **KL 正则**：`L_KL = -(1 + logvar - μ² - exp(logvar)) / 2`（VAE 式）
- **配置**：`kl_reg_weight=1e-3` 控制 KL 损失权重

## 主要改动文件

- `models/subNets/UncertaintyEncoder.py`：改为输出 `(mu, logvar, sigma_sq)`，logvar 直接由 Linear 输出
- `models/missingTask/EMT_DLFR.py`：整序列输入 + `_masked_mean_pool` + σ² = exp(logvar)
- `trains/missingTask/EMT_DLFR.py`：增加 KL 正则
- `config/config_regression.py`：增加 `kl_reg_weight` 参数

## 运行

与 `old` 相同，例如：

```bash
bash scripts/mosi/run_once.sh
```
