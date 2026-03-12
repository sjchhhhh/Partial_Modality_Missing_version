# *_*coding:utf-8 *_*
"""
Uncertainty Encoder - 与 Code_URMF-main 一致的计算方式
- 使用 logvar（log σ²）表示不确定性，σ² = exp(logvar)
- 参考: Embracing Unimodal Aleatoric Uncertainty for Robust Multimodal Fusion (CVPR 2024)
"""
import torch
import torch.nn as nn

__all__ = ['UncertaintyEncoder']

class UncertaintyEncoder(nn.Module):
    """
    不确定性编码器（Code_URMF 风格）：
    - mu = Linear(x)
    - logvar = Linear(x)   # log(σ²)，无 Softplus
    - σ² = exp(logvar)
    
    与 Code_URMF 一致：直接输出 logvar，训练时用 KL 正则
    """
    def __init__(self, input_dim, output_dim, hidden_dim=None, use_batch_norm=False):
        super(UncertaintyEncoder, self).__init__()
        input_dim = int(round(input_dim))
        output_dim = int(round(output_dim))
        hidden_dim = int(round(hidden_dim or input_dim))
        
        if input_dim <= 0 or output_dim <= 0 or hidden_dim <= 0:
            raise ValueError(f"UncertaintyEncoder: 所有维度必须>0")
        
        # 均值分支：与 Code_URMF 一致，单层 Linear
        self.mu_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # logvar 分支：与 Code_URMF 一致，直接 Linear 输出 log(σ²)，无 Softplus
        self.logvar_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        
        if use_batch_norm:
            self.bn_mu = nn.BatchNorm1d(output_dim)
            self.bn_logvar = nn.BatchNorm1d(output_dim)
        else:
            self.bn_mu = None
            self.bn_logvar = None
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.mu_branch:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        # logvar  branch 最后一层初始化为较小值（与 Code_URMF 思想一致）
        linear_layers = [m for m in self.logvar_branch if isinstance(m, nn.Linear)]
        for i, module in enumerate(self.logvar_branch):
            if isinstance(module, nn.Linear):
                if module is linear_layers[-1]:
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                    nn.init.constant_(module.bias, -2.0)  # 初始 logvar 较小 -> exp(-2)≈0.14
                else:
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Args:
            x: (B, D) 或 (B, seq_len, D)
        Returns:
            mu: (B, D) 或 (B, seq_len, D)
            logvar: (B, D) 或 (B, seq_len, D)，即 log(σ²)
            sigma_sq: (B, D) 或 (B, seq_len, D)，即 exp(logvar)，便于下游使用
        """
        original_shape = x.shape
        if len(original_shape) == 3:
            B, seq_len, D = original_shape
            x = x.view(B * seq_len, D)
            need_reshape = True
        elif len(original_shape) == 2:
            need_reshape = False
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}")
        
        mu = self.mu_branch(x)
        logvar = self.logvar_branch(x)
        
        if self.bn_mu is not None:
            if need_reshape:
                mu = mu.view(B, seq_len, -1)
                logvar = logvar.view(B, seq_len, -1)
                mu = mu.permute(0, 2, 1).contiguous()
                logvar = logvar.permute(0, 2, 1).contiguous()
                mu = self.bn_mu(mu).permute(0, 2, 1).contiguous()
                logvar = self.bn_logvar(logvar).permute(0, 2, 1).contiguous()
            else:
                mu = self.bn_mu(mu)
                logvar = self.bn_logvar(logvar)
        else:
            if need_reshape:
                mu = mu.view(B, seq_len, -1)
                logvar = logvar.view(B, seq_len, -1)
        
        # 与 Code_URMF 一致：σ² = exp(logvar)
        sigma_sq = torch.exp(logvar)
        
        return mu, logvar, sigma_sq
