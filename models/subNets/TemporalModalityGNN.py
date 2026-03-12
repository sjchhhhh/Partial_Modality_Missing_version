# *_*coding:utf-8 *_*
"""
Temporal Modality GNN: 模态时序补全（方案A：模态内补全）
在LSTM与不确定性估计之间，通过图神经网络进行模态内时序补全。
图结构：每个模态划分为K个时序片段，共3*K个节点；
边：仅模态内时序边（k↔k+1），每个模态独立根据前后帧补全。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['TemporalModalityGNN']


def _segment_pool(x, lengths, num_segments):
    """
    将时序特征划分为num_segments个片段，每片段内做masked mean pooling。
    x: (B, T, D), lengths: (B,), 输出 (B, num_segments, D)
    """
    B, T, D = x.shape
    device = x.device
    lengths = lengths.to(device)
    out = []
    for k in range(num_segments):
        t_start = k * T // num_segments
        t_end = (k + 1) * T // num_segments
        if t_end > t_start:
            chunk = x[:, t_start:t_end, :]
            seg_len = t_end - t_start
            valid = (torch.arange(seg_len, device=device).unsqueeze(0) < (
                lengths.clamp(0, T) - t_start
            ).clamp(0, seg_len).unsqueeze(1)).float().unsqueeze(-1)
            s = (chunk * valid).sum(dim=1) / (valid.sum(dim=1).clamp(min=1e-6))
        else:
            s = x[:, t_start, :]
        out.append(s)
    return torch.stack(out, dim=1)


class TemporalModalityGNN(nn.Module):
    """
    时序模态图网络（方案A：模态内时序补全）
    在 LSTM 与不确定性估计之间，对每个模态的时序特征独立进行图传播与补全。
    输入：text (B,T_t,D), audio (B,T_a,D), video (B,T_v,D)，已投影到 d_model。
    输出：补全后的 utterance 级特征 (B, D) x3，用于不确定性估计。
    
    图结构：每个模态内部构建时序图（相邻片段相连），不进行跨模态信息交换。
    """

    def __init__(self, d_model, num_segments=8, num_layers=2, dropout=0.1):
        super().__init__()
        d_model = int(round(d_model))
        num_segments = max(1, int(round(num_segments)))
        num_layers = max(1, int(round(num_layers)))
        self.d_model = d_model
        self.num_segments = num_segments
        self.num_layers = num_layers
        self.K = num_segments
        self.N = 3 * num_segments

        # 稠密邻接矩阵 A (N, N)，归一化
        A = self._build_adjacency()
        self.register_buffer('A', A)

        self.gcn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for _ in range(num_layers):
            self.gcn_layers.append(nn.Linear(d_model, d_model))
            self.norms.append(nn.LayerNorm(d_model))

    def _build_adjacency(self):
        """
        方案A：仅模态内时序补全
        只保留每个模态内部的时序边（k↔k+1），移除跨模态边
        每个模态独立根据前后帧进行补全
        """
        N = self.N
        K = self.K
        A = torch.zeros(N, N)
        # 模态内时序边：每个模态的相邻片段相连
        for m in range(3):
            base = m * K
            for k in range(K - 1):
                A[base + k, base + k + 1] = 1.0
                A[base + k + 1, base + k] = 1.0
        # 移除跨模态边（方案A：不进行跨模态补全）
        # 行归一化
        deg = A.sum(dim=1, keepdim=True).clamp(min=1e-6)
        A = A / deg
        return A

    def forward(
        self,
        text_temporal,
        audio_temporal,
        video_temporal,
        text_lengths,
        audio_lengths,
        video_lengths,
    ):
        """
        text_temporal: (B, T_t, D), audio: (B, T_a, D), video: (B, T_v, D)
        lengths: (B,) each.
        Returns:
            text_utt: (B, D), audio_utt: (B, D), video_utt: (B, D)
        """
        K = self.K
        device = text_temporal.device
        text_len = text_lengths.clamp(1)
        audio_len = audio_lengths.clamp(1)
        video_len = video_lengths.clamp(1)

        text_seg = _segment_pool(text_temporal, text_len, K)
        audio_seg = _segment_pool(audio_temporal, audio_len, K)
        video_seg = _segment_pool(video_temporal, video_len, K)

        h = torch.cat([text_seg, audio_seg, video_seg], dim=1)

        A = self.A.to(device)
        for i in range(self.num_layers):
            # GCN: H' = A @ H @ W, 残差 + LayerNorm
            msg = torch.matmul(A.unsqueeze(0), h)
            msg = self.gcn_layers[i](msg)
            h = self.norms[i](h + self.dropout(F.relu(msg)))

        t_utt = h[:, :K, :].mean(dim=1)
        a_utt = h[:, K : 2 * K, :].mean(dim=1)
        v_utt = h[:, 2 * K :, :].mean(dim=1)
        return t_utt, a_utt, v_utt
