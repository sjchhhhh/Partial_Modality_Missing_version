# *_*coding:utf-8 *_*
"""
Mixture of Experts (MOE) Fusion with Uncertainty-aware Gating
Uses aleatoric uncertainty (sigma_sq) to guide expert selection and modality fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MOEFusion', 'Expert', 'ExpertTokenOnly', 'MOEFusionTokenGating']


class ExpertTokenOnly(nn.Module):
    """专家只接收该模态的 K 个 token 特征 (B, K*dim) -> (B, dim)"""
    def __init__(self, token_input_dim, dim, dropout=0.1):
        super(ExpertTokenOnly, self).__init__()
        token_input_dim = int(round(token_input_dim))
        dim = int(round(dim))
        self.net = nn.Sequential(
            nn.Linear(token_input_dim, dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class MOEFusionTokenGating(nn.Module):
    """
    门控输入：各模态 (pooled_μ, log(σ²+ε))，共 6*d_model
    专家输入：仅该模态的 K 个 token 特征 (B, K*d_model) -> (B, d_model)
    """
    def __init__(self, dim, K_text, K_audio, K_video, dropout=0.1, gating_hidden_dim=None):
        super(MOEFusionTokenGating, self).__init__()
        dim = int(round(dim))
        K_text, K_audio, K_video = int(round(K_text)), int(round(K_audio)), int(round(K_video))
        self.dim = dim
        self.text_expert = ExpertTokenOnly(K_text * dim, dim, dropout)
        self.audio_expert = ExpertTokenOnly(K_audio * dim, dim, dropout)
        self.video_expert = ExpertTokenOnly(K_video * dim, dim, dropout)
        gating_input_dim = dim * 6  # pooled_mu_t, logvar_t, pooled_mu_a, logvar_a, pooled_mu_v, logvar_v
        gating_hidden_dim = int(round(gating_hidden_dim or dim * 4))
        self.gating = nn.Sequential(
            nn.Linear(gating_input_dim, gating_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gating_hidden_dim, gating_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gating_hidden_dim // 2, 3),
            nn.Softmax(dim=-1)
        )
        for i, layer in enumerate(self.gating):
            if isinstance(layer, nn.Linear):
                if i == len(self.gating) - 2:
                    nn.init.normal_(layer.weight, 0.0, 0.01)
                    nn.init.zeros_(layer.bias)
                else:
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, tokens_text, tokens_audio, tokens_video,
                gate_pooled_mu_t, gate_logvar_t,
                gate_pooled_mu_a, gate_logvar_a,
                gate_pooled_mu_v, gate_logvar_v):
        """
        tokens_*: (B, K_*, dim)
        gate_pooled_mu_*, gate_logvar_*: (B, dim)
        Returns:
            gmc_tokens: (B, 3, dim)
        """
        B = tokens_text.shape[0]
        gmc_t = self.text_expert(tokens_text.reshape(B, -1))
        gmc_a = self.audio_expert(tokens_audio.reshape(B, -1))
        gmc_v = self.video_expert(tokens_video.reshape(B, -1))
        gmc_tokens = torch.stack([gmc_t, gmc_a, gmc_v], dim=1)  # (B, 3, dim)
        gate_in = torch.cat([
            gate_pooled_mu_t, gate_logvar_t,
            gate_pooled_mu_a, gate_logvar_a,
            gate_pooled_mu_v, gate_logvar_v
        ], dim=-1)
        modal_weights = self.gating(gate_in).unsqueeze(-1)  # (B, 3, 1)
        gmc_tokens = gmc_tokens * modal_weights
        return gmc_tokens

class Expert(nn.Module):
    """
    单个专家：负责一种融合策略
    每个专家可以专注于不同的模态组合或融合策略
    """
    def __init__(self, dim, dropout=0.1, expert_type='balanced'):
        """
        Args:
            dim: 特征维度（d_model）
            dropout: dropout概率
            expert_type: 专家类型
                - 'text_dominant': 文本主导
                - 'av_dominant': 音视频主导
                - 'balanced': 均衡融合
        """
        super(Expert, self).__init__()
        self.expert_type = expert_type
        
        # 确保dim是整数（如果传入的是浮点数如128.0，先四舍五入再转换）
        dim = int(round(dim))
        
        # 专家网络：融合三个模态的(mu, sigma_sq)
        # 输入：3个模态的(mu, sigma_sq) -> 6*dim + 3个reliability
        input_dim = int(round(dim * 6 + 3))  # 3*(mu+sigma_sq) + 3*reliability
        
        self.fusion_net = nn.Sequential(
            nn.Linear(input_dim, int(round(dim * 4))),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(int(round(dim * 4)), int(round(dim * 2))),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(int(round(dim * 2)), dim)  # 输出单个模态的融合特征
        )
        
        # 如果是文本主导或音视频主导，可以添加额外的注意力机制
        if expert_type == 'text_dominant':
            self.attention_weights = nn.Parameter(torch.tensor([0.6, 0.2, 0.2]))  # 文本权重高
        elif expert_type == 'av_dominant':
            self.attention_weights = nn.Parameter(torch.tensor([0.2, 0.4, 0.4]))  # 音视频权重高
        else:
            self.attention_weights = None  # 均衡融合，由网络学习
    
    def forward(self, mu_text, sigma_sq_text, mu_audio, sigma_sq_audio, mu_video, sigma_sq_video):
        """
        Args:
            mu_text, sigma_sq_text: 文本的均值和方差 (B, dim)
            mu_audio, sigma_sq_audio: 音频的均值和方差 (B, dim)
            mu_video, sigma_sq_video: 视频的均值和方差 (B, dim)
        Returns:
            fused: 融合后的特征 (B, dim)
        """
        B = mu_text.shape[0]
        
        # 计算可靠性分数：reliability = 1 / (1 + sigma_sq)
        # 不确定性小 -> 可靠性高
        rel_text = 1.0 / (1.0 + sigma_sq_text.mean(dim=-1, keepdim=True))  # (B, 1)
        rel_audio = 1.0 / (1.0 + sigma_sq_audio.mean(dim=-1, keepdim=True))
        rel_video = 1.0 / (1.0 + sigma_sq_video.mean(dim=-1, keepdim=True))
        
        # 拼接所有信息：3个模态的(mu, sigma_sq) + 3个reliability
        # 如果mu/sigma_sq是多维的，需要flatten
        if len(mu_text.shape) > 2:
            mu_text_flat = mu_text.view(B, -1)  # (B, dim)
            sigma_sq_text_flat = sigma_sq_text.view(B, -1)
        else:
            mu_text_flat = mu_text
            sigma_sq_text_flat = sigma_sq_text
            
        if len(mu_audio.shape) > 2:
            mu_audio_flat = mu_audio.view(B, -1)
            sigma_sq_audio_flat = sigma_sq_audio.view(B, -1)
        else:
            mu_audio_flat = mu_audio
            sigma_sq_audio_flat = sigma_sq_audio
            
        if len(mu_video.shape) > 2:
            mu_video_flat = mu_video.view(B, -1)
            sigma_sq_video_flat = sigma_sq_video.view(B, -1)
        else:
            mu_video_flat = mu_video
            sigma_sq_video_flat = sigma_sq_video
        
        # 拼接输入
        expert_input = torch.cat([
            mu_text_flat, sigma_sq_text_flat,      # 文本：均值 + 不确定性
            mu_audio_flat, sigma_sq_audio_flat,    # 音频
            mu_video_flat, sigma_sq_video_flat,    # 视频
            rel_text, rel_audio, rel_video  # 可靠性分数
        ], dim=-1)
        
        # 专家处理
        fused = self.fusion_net(expert_input)  # (B, dim)
        
        return fused


class MOEFusion(nn.Module):
    """
    Mixture of Experts融合模块（方案A）：每个模态对应一个专家，每个专家生成对应的gmc_token
    这样可以保持模态身份，符合EMT的设计理念
    """
    def __init__(self, dim, num_experts=3, dropout=0.1, gating_hidden_dim=None):
        """
        Args:
            dim: 特征维度（d_model）
            num_experts: 专家数量（默认3个，对应3个模态）
            dropout: dropout概率
            gating_hidden_dim: 门控网络隐藏层维度（如果None则使用dim*4）
        """
        super(MOEFusion, self).__init__()
        # 确保所有参数都是整数（如果传入的是浮点数如128.0，先四舍五入再转换）
        if num_experts is None:
            num_experts = 3  # 默认值
        num_experts = max(1, int(round(num_experts)))  # 至少为1，避免为0
        dim = int(round(dim))
        self.num_experts = num_experts
        self.dim = dim
        
        if gating_hidden_dim is not None:
            gating_hidden_dim = int(round(gating_hidden_dim))
        else:
            gating_hidden_dim = int(round(dim * 4))
        
        # 方案A：每个模态对应一个专家
        # 3个专家分别负责文本、音频、视频模态
        # 每个专家主要关注对应模态，但也会融合其他模态信息
        self.text_expert = Expert(dim, dropout, expert_type='text_dominant')  # 文本专家
        self.audio_expert = Expert(dim, dropout, expert_type='av_dominant')   # 音频专家
        self.video_expert = Expert(dim, dropout, expert_type='av_dominant')   # 视频专家
        
        # 保持向后兼容：experts列表（用于某些检查）
        self.experts = nn.ModuleList([self.text_expert, self.audio_expert, self.video_expert])
        
        # 门控网络：为每个模态选择专家权重（可选，用于进一步融合）
        # 输入：3个模态的(mu, sigma_sq) + 3个reliability = dim*6 + 3
        gating_input_dim = int(round(dim * 6 + 3))
        self.use_gating = True  # 是否使用门控网络（可以关闭，直接输出）
        if self.use_gating:
            self.gating = nn.Sequential(
                nn.Linear(gating_input_dim, gating_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(gating_hidden_dim, gating_hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(gating_hidden_dim // 2, 3),  # 为每个模态输出权重
                nn.Softmax(dim=-1)  # 每个模态的权重（和为1）
            )
        else:
            self.gating = None
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        优化后的权重初始化策略：
        1. 门控网络最后一层：初始化为接近均匀分布
        2. 专家网络：使用Xavier初始化，确保梯度流动
        """
        # 初始化门控网络（如果使用）
        if self.use_gating and self.gating is not None:
            for i, layer in enumerate(self.gating):
                if isinstance(layer, nn.Linear):
                    if i == len(self.gating) - 2:  # 最后一层Linear（在Softmax之前）
                        # 门控网络最后一层：初始化为接近均匀分布
                        # 使用较小的权重，让输出接近均匀分布（1/3）
                        nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                        nn.init.zeros_(layer.bias)
                    else:
                        # 其他层使用Xavier初始化
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
        
        # 初始化专家网络
        for expert in self.experts:
            for module in expert.fusion_net:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
    
    def forward(self, mu_text, sigma_sq_text, mu_audio, sigma_sq_audio, mu_video, sigma_sq_video):
        """
        方案A：每个模态对应一个专家，每个专家生成对应的gmc_token
        
        Args:
            mu_text, sigma_sq_text: 文本的均值和方差 (B, dim) 或 (B, 1, dim)
            mu_audio, sigma_sq_audio: 音频的均值和方差 (B, dim)
            mu_video, sigma_sq_video: 视频的均值和方差 (B, dim)
        Returns:
            gmc_tokens: 融合后的全局多模态上下文 (B, 3, dim)
                       gmc_tokens[:, 0, :] 对应文本模态
                       gmc_tokens[:, 1, :] 对应音频模态
                       gmc_tokens[:, 2, :] 对应视频模态
        """
        B = mu_text.shape[0]
        
        # 确保mu和sigma_sq是2维的（B, dim）
        if len(mu_text.shape) == 3:
            mu_text = mu_text.squeeze(1)  # (B, 1, dim) -> (B, dim)
            sigma_sq_text = sigma_sq_text.squeeze(1)
        if len(mu_audio.shape) == 3:
            mu_audio = mu_audio.squeeze(1)
            sigma_sq_audio = sigma_sq_audio.squeeze(1)
        if len(mu_video.shape) == 3:
            mu_video = mu_video.squeeze(1)
            sigma_sq_video = sigma_sq_video.squeeze(1)
        
        # 方案A：每个专家生成对应的gmc_token
        # 文本专家生成文本的gmc_token（主要基于文本，但融合了音频和视频信息）
        gmc_token_text = self.text_expert(mu_text, sigma_sq_text, mu_audio, sigma_sq_audio, mu_video, sigma_sq_video)  # (B, dim)
        
        # 音频专家生成音频的gmc_token（主要基于音频，但融合了文本和视频信息）
        gmc_token_audio = self.audio_expert(mu_text, sigma_sq_text, mu_audio, sigma_sq_audio, mu_video, sigma_sq_video)  # (B, dim)
        
        # 视频专家生成视频的gmc_token（主要基于视频，但融合了文本和音频信息）
        gmc_token_video = self.video_expert(mu_text, sigma_sq_text, mu_audio, sigma_sq_audio, mu_video, sigma_sq_video)  # (B, dim)
        
        # 堆叠成3个gmc_tokens，保持模态身份
        # gmc_tokens[:, 0, :] = 文本的gmc_token
        # gmc_tokens[:, 1, :] = 音频的gmc_token
        # gmc_tokens[:, 2, :] = 视频的gmc_token
        gmc_tokens = torch.stack([gmc_token_text, gmc_token_audio, gmc_token_video], dim=1)  # (B, 3, dim)
        
        # 可选：使用门控网络进一步调整（根据不确定性动态加权）
        if self.use_gating and self.gating is not None:
            # 计算可靠性分数
            rel_text = 1.0 / (1.0 + sigma_sq_text.mean(dim=-1, keepdim=True))  # (B, 1)
            rel_audio = 1.0 / (1.0 + sigma_sq_audio.mean(dim=-1, keepdim=True))
            rel_video = 1.0 / (1.0 + sigma_sq_video.mean(dim=-1, keepdim=True))
            
            # 门控网络输入：拼接所有模态的(mu, sigma_sq)和可靠性分数
            gating_input = torch.cat([
                mu_text, sigma_sq_text,      # 文本
                mu_audio, sigma_sq_audio,    # 音频
                mu_video, sigma_sq_video,    # 视频
                rel_text, rel_audio, rel_video  # 可靠性
            ], dim=-1)  # (B, dim*6 + 3)
            
            # 门控网络输出：为每个模态输出权重 (B, 3)
            modal_weights = self.gating(gating_input)  # (B, 3)
            
            # 根据不确定性调整每个模态的gmc_token
            # 不确定性小（可靠性高）的模态权重更大
            modal_weights_expanded = modal_weights.unsqueeze(-1)  # (B, 3, 1)
            gmc_tokens = gmc_tokens * modal_weights_expanded  # (B, 3, dim)
        
        return gmc_tokens
