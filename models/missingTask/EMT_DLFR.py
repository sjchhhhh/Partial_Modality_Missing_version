import os
import sys
import collections
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.subNets.BertTextEncoder import BertTextEncoder
from models.subNets.EMT import EMT
from models.subNets.UncertaintyEncoder import UncertaintyEncoder
from models.subNets.MOEFusion import MOEFusion

__all__ = ['EMT_DLFR', 'FeatureEncoder']

class FeatureEncoder(nn.Module):
    """
    特征编码器：将原始低维特征编码到更高维度
    用于音频和视频模态，在LSTM之前进行特征增强
    """
    def __init__(self, input_dim, output_dim, hidden_dim=None, dropout=0.1):
        """
        Args:
            input_dim: 输入特征维度（原始特征维度，如5或20）
            output_dim: 输出特征维度（编码后的维度，如d_model）
            hidden_dim: 隐藏层维度（如果None则使用output_dim）
            dropout: dropout概率
        """
        super(FeatureEncoder, self).__init__()
        input_dim = int(round(input_dim))
        output_dim = int(round(output_dim))
        hidden_dim = int(round(hidden_dim or output_dim))
        
        if input_dim <= 0 or output_dim <= 0 or hidden_dim <= 0:
            raise ValueError(f"FeatureEncoder: 所有维度必须>0, input_dim={input_dim}, output_dim={output_dim}, hidden_dim={hidden_dim}")
        
        # 分离Linear层和LayerNorm，以便正确处理3D tensor
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.ln2 = nn.LayerNorm(output_dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Args:
            x: 输入特征 (B, T, input_dim) 或 (B, input_dim)
        Returns:
            encoded: 编码后的特征 (B, T, output_dim) 或 (B, output_dim)
        """
        original_shape = x.shape
        need_reshape = len(original_shape) == 3
        
        if need_reshape:
            B, T, D = original_shape
            x = x.view(B * T, D)
        
        # 第一层
        x = self.linear1(x)
        if need_reshape:
            x = x.view(B, T, -1)
        x = self.ln1(x)
        if need_reshape:
            x = x.view(B * T, -1)
        x = self.relu(x)
        x = self.dropout(x)
        
        # 第二层
        x = self.linear2(x)
        if need_reshape:
            x = x.view(B, T, -1)
        x = self.ln2(x)
        
        if need_reshape:
            x = x.view(B, T, -1)
        
        return x


class EMT_DLFR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args  # 保存args引用
        self.aligned = args.need_data_aligned
        # unimodal encoders
        ## text encoder
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_finetune)

        ## audio-vision encoders
        audio_in, video_in = args.feature_dims[1:]
        
        # 确保所有维度参数都是整数
        audio_in = int(round(audio_in))
        video_in = int(round(video_in))
        
        # 检查 d_model 是否存在且有效
        if not hasattr(args, 'd_model'):
            raise ValueError(f"EMT_DLFR: args.d_model is not set. "
                           f"Available args attributes: {[k for k in dir(args) if not k.startswith('_')][:20]}. "
                           f"Please check the config file for 'd_model' parameter.")
        
        d_model_raw = args.d_model
        if d_model_raw is None:
            raise ValueError(f"EMT_DLFR: args.d_model is None. "
                           f"Please check the config file for 'd_model' parameter.")
        
        d_model_value = int(round(d_model_raw))
        if d_model_value <= 0:
            raise ValueError(
                f"EMT_DLFR: args.d_model must be > 0, got {d_model_raw} "
                f"(type: {type(d_model_raw)}, converted to {d_model_value}). "
                f"Please check the config file. "
                f"Model: {getattr(args, 'modelName', 'unknown')}, "
                f"Dataset: {getattr(args, 'datasetName', 'unknown')}"
            )
        
        # ============================================================
        #  模态编码：LSTM 之前的特征增强 & 不确定性建模
        #  这一部分就是「LSTM 前面的特征提取」，分三层意思：
        #  1）FeatureEncoder：先把原始低维特征(如 A:5 维, V:20 维)映射到统一的 d_model 维，
        #     本质是两层 MLP + LayerNorm + ReLU，相当于做一层前置表征增强；
        #  2）uncertainty_*_before_lstm：在时间序列维度上，对编码后的特征再估计一遍
        #     (mu, sigma^2)，得到「LSTM 之前的时序不确定性」；
        #  3）后面 LSTM 的输入会把 [feature, mu, sigma^2] 三块拼在一起，
        #     让 LSTM 在「已经增强的特征 + 对应不确定性」的空间里建模。
        # ============================================================
        self.audio_feature_encoder = FeatureEncoder(
            input_dim=audio_in,  # 原始音频特征维度（如5）
            output_dim=d_model_value,  # 编码到d_model维度
            hidden_dim=d_model_value,
            dropout=0.1
        )
        self.video_feature_encoder = FeatureEncoder(
            input_dim=video_in,  # 原始视频特征维度（如20）
            output_dim=d_model_value,  # 编码到d_model维度
            hidden_dim=d_model_value,
            dropout=0.1
        )
        
        # ========== LSTM前的不确定性估计（已注释） ==========
        # self.uncertainty_audio_before_lstm = UncertaintyEncoder(
        #     input_dim=d_model_value,
        #     output_dim=d_model_value,
        #     hidden_dim=d_model_value * 2,
        #     use_batch_norm=False
        # )
        # self.uncertainty_video_before_lstm = UncertaintyEncoder(
        #     input_dim=d_model_value,
        #     output_dim=d_model_value,
        #     hidden_dim=d_model_value * 2,
        #     use_batch_norm=False
        # )
        
        # ========== LSTM编码器：只吃编码后的特征 (d_model) ==========
        lstm_input_dim = d_model_value
        self.audio_model = AuViSubNetWithUncertainty(
            in_size=lstm_input_dim,
            hidden_size=int(round(args.a_lstm_hidden_size)),
            out_size=int(round(args.audio_out)) if args.audio_out is not None else None,
            num_layers=args.a_lstm_layers,
            dropout=args.a_lstm_dropout
        )
        self.video_model = AuViSubNetWithUncertainty(
            in_size=lstm_input_dim,
            hidden_size=int(round(args.v_lstm_hidden_size)),
            out_size=int(round(args.video_out)) if args.video_out is not None else None,
            num_layers=args.v_lstm_layers,
            dropout=args.v_lstm_dropout
        )
        
        # 不确定性估计（在LSTM之后，基于LSTM输出的utterance-level特征）
        # 输入维度是LSTM的输出维度（audio_out/video_out）
        # 输出维度是d_model（用于MOE融合）
        audio_out_dim = int(round(args.audio_out)) if args.audio_out is not None else int(round(args.a_lstm_hidden_size))
        video_out_dim = int(round(args.video_out)) if args.video_out is not None else int(round(args.v_lstm_hidden_size))
        
        # 验证维度有效性
        if audio_out_dim <= 0:
            raise ValueError(f"EMT_DLFR: args.audio_out must be > 0, got {args.audio_out}")
        if video_out_dim <= 0:
            raise ValueError(f"EMT_DLFR: args.video_out must be > 0, got {args.video_out}")
        
        uncertainty_hidden_dim_audio = int(round(audio_out_dim * 2))
        uncertainty_hidden_dim_video = int(round(video_out_dim * 2))
        if uncertainty_hidden_dim_audio <= 0:
            raise ValueError(f"EMT_DLFR: uncertainty_hidden_dim_audio must be > 0, got {uncertainty_hidden_dim_audio}. audio_out_dim={audio_out_dim}")
        if uncertainty_hidden_dim_video <= 0:
            raise ValueError(f"EMT_DLFR: uncertainty_hidden_dim_video must be > 0, got {uncertainty_hidden_dim_video}. video_out_dim={video_out_dim}")
        
        self.uncertainty_audio = UncertaintyEncoder(
            input_dim=audio_out_dim,  # LSTM输出维度
            output_dim=d_model_value,  # 输出d_model维度，用于MOE
            hidden_dim=uncertainty_hidden_dim_audio,
            use_batch_norm=False
        )
        self.uncertainty_video = UncertaintyEncoder(
            input_dim=video_out_dim,  # LSTM输出维度
            output_dim=d_model_value,  # 输出d_model维度，用于MOE
            hidden_dim=uncertainty_hidden_dim_video,
            use_batch_norm=False
        )

        # equalization
        self.proj_audio = nn.Linear(args.audio_out, args.d_model, bias=False) if args.audio_out != args.d_model else nn.Identity()
        self.proj_video = nn.Linear(args.video_out, args.d_model, bias=False) if args.video_out != args.d_model else nn.Identity()
        self.proj_text = nn.Linear(args.text_out, args.d_model, bias=False) if args.text_out != args.d_model else nn.Identity()

        # 文本的不确定性估计（在 BERT 之后，基于「已补全」的表示）
        self.uncertainty_text = UncertaintyEncoder(
            input_dim=args.text_out,  # BERT输出维度
            output_dim=args.d_model,
            hidden_dim=args.d_model * 2,
            use_batch_norm=False
        )
        # 文本 BERT 前的不确定性（已注释）
        # self.uncertainty_text_before_bert = UncertaintyEncoder(
        #     input_dim=args.text_out,
        #     output_dim=d_model_value,
        #     hidden_dim=args.text_out * 2,
        #     use_batch_norm=False
        # )
        
        # LSTM/BERT 之后 -> utterance 级 (μ, σ²) -> MOE 输入
        self.moe_fusion = MOEFusion(
            dim=args.d_model,
            num_experts=3,
            dropout=0.1,
            gating_hidden_dim=args.d_model * 4
        )

        # fusion: emt
        num_modality = 3
        self.fusion_method = args.fusion_method

        self.fusion = EMT(dim=args.d_model, depth=args.fusion_layers, heads=args.heads, num_modality=num_modality,
                          learnable_pos_emb=args.learnable_pos_emb, emb_dropout=args.emb_dropout, attn_dropout=args.attn_dropout,
                          ff_dropout=args.ff_dropout, ff_expansion=args.ff_expansion, mpu_share=args.mpu_share,
                          modality_share=args.modality_share, layer_share=args.layer_share,
                          attn_act_fn=args.attn_act_fn)

        # high-level: SimSiam + 不确定性加权（方案C）
        gmc_tokens_dim = num_modality * args.d_model
        self.gmc_tokens_projector = Projector(gmc_tokens_dim, gmc_tokens_dim)
        self.text_projector = Projector(args.text_out, args.text_out)
        self.audio_projector = Projector(args.audio_out, args.audio_out)
        self.video_projector = Projector(args.video_out, args.video_out)
        self.gmc_tokens_predictor = Predictor(gmc_tokens_dim, args.gmc_tokens_pred_dim, gmc_tokens_dim)
        self.text_predictor = Predictor(args.text_out, args.text_pred_dim, args.text_out)
        self.audio_predictor = Predictor(args.audio_out, args.audio_pred_dim, args.audio_out)
        self.video_predictor = Predictor(args.video_out, args.video_pred_dim, args.video_out)

        # low-level feature reconstruction
        self.recon_text = nn.Linear(args.d_model, args.feature_dims[0])
        self.recon_audio = nn.Linear(args.d_model, args.feature_dims[1])
        self.recon_video = nn.Linear(args.d_model, args.feature_dims[2])

        # final prediction module：三路 utterance 摘要 + gmc_tokens（与 based 一致）
        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        concat_dim = args.text_out + args.video_out + args.audio_out + gmc_tokens_dim
        self.post_fusion_layer_1 = nn.Linear(concat_dim, args.post_fusion_dim)
        self.post_fusion_shortcut = nn.Linear(concat_dim, args.post_fusion_dim)  # 残差 1：concat -> h1
        self.post_fusion_layer_2 = nn.Linear(args.post_fusion_dim, args.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(args.post_fusion_dim, 1)

    def forward_once(self, text, text_lengths, audio, audio_lengths, video, video_lengths, missing,
                     text_missing_mask=None, audio_missing_mask=None, vision_missing_mask=None):
        # ========== 设备检查：确保所有输入在GPU上 ==========
        # 确保所有特征tensor在相同的设备上（应该是GPU）
        assert audio.device == video.device == text.device, \
            f"设备不一致: audio={audio.device}, video={video.device}, text={text.device}"
        
        # 确保lengths在正确的设备上（与对应特征相同）
        # 注意：pack_padded_sequence需要lengths在CPU，但我们会单独处理
        device = text.device
        if text_lengths.device != device:
            text_lengths = text_lengths.to(device)
        if audio_lengths.device != device:
            audio_lengths = audio_lengths.to(device)
        if video_lengths.device != device:
            video_lengths = video_lengths.to(device)
        
        # ========== 文本：BERT + 整个模态序列的不确定性（Code_URMF 风格：logvar，整序列输入+池化） ==========
        text = self.text_model(text)  # (B, T+1, text_out)
        text_utt_original = text[:, 0].squeeze(1) if text.dim() == 3 else text[:, 0]
        text_for_recon = text[:, 1:].detach()
        text_seq = self.proj_text(text)  # (B, T+1, d_model) 供 EMT local
        text_lengths_full = (text_lengths + 1).clamp(max=text.shape[1])
        mu_text, logvar_text, _ = self.uncertainty_text(text)
        mu_text = _masked_mean_pool(mu_text, text_lengths_full)
        logvar_text = _masked_mean_pool(logvar_text, text_lengths_full)
        sigma_sq_text_out = torch.exp(logvar_text)  # Code_URMF: σ² = exp(logvar)
        
        # ========== 音频：feature encoder -> LSTM -> 整个模态序列的不确定性 ==========
        audio_encoded = self.audio_feature_encoder(audio)
        audio, audio_utt = self.audio_model(audio_encoded, audio_lengths, return_temporal=True)
        mu_audio, logvar_audio, _ = self.uncertainty_audio(audio)
        mu_audio = _masked_mean_pool(mu_audio, audio_lengths)
        logvar_audio = _masked_mean_pool(logvar_audio, audio_lengths)
        sigma_sq_audio_out = torch.exp(logvar_audio)
        
        # ========== 视频：同上 ==========
        video_encoded = self.video_feature_encoder(video)
        video, video_utt = self.video_model(video_encoded, video_lengths, return_temporal=True)
        mu_video, logvar_video, _ = self.uncertainty_video(video)
        mu_video = _masked_mean_pool(mu_video, video_lengths)
        logvar_video = _masked_mean_pool(logvar_video, video_lengths)
        sigma_sq_video_out = torch.exp(logvar_video)
        
        # ========== MOE 融合 ==========
        gmc_tokens = self.moe_fusion(mu_text, sigma_sq_text_out, mu_audio, sigma_sq_audio_out, mu_video, sigma_sq_video_out)  # (B, 3, D)
        
        # local unimodal features（与 master 一致：文本整序列含 CLS，音视频为投影序列）用于 EMT
        audio_seq = self.proj_audio(audio)
        video_seq = self.proj_video(video)
        text = text_seq   # (B, T+1, d_model)
        audio = audio_seq
        video = video_seq

        # get attention mask（文本有效长度为 1+content_len，即含 CLS）
        text_lengths_gpu = text_lengths.to(text.device) if text_lengths.device != text.device else text_lengths
        audio_lengths_gpu = audio_lengths.to(audio.device) if audio_lengths.device != audio.device else audio_lengths
        video_lengths_gpu = video_lengths.to(video.device) if video_lengths.device != video.device else video_lengths
        text_lengths_full = (text_lengths_gpu + 1).clamp(max=text.shape[1])
        modality_masks = [length_to_mask(seq_len, max_len=max_len)
                         for seq_len, max_len in zip([text_lengths_full, audio_lengths_gpu, video_lengths_gpu],
                                                     [text.shape[1], audio.shape[1], video.shape[1]])]

        # fusion
        gmc_tokens, modality_ouputs = self.fusion(gmc_tokens,[text, audio, video], modality_masks)
        gmc_tokens = gmc_tokens.reshape(gmc_tokens.shape[0], -1) # (B, 3*D)

        # high-level: SimSiam（方案C，与不确定性加权在 train 里结合）
        text_utt_2d = text_utt_original.view(text_utt_original.shape[0], -1) if len(text_utt_original.shape) > 2 else text_utt_original
        audio_utt_2d = audio_utt.view(audio_utt.shape[0], -1) if len(audio_utt.shape) > 2 else audio_utt
        video_utt_2d = video_utt.view(video_utt.shape[0], -1) if len(video_utt.shape) > 2 else video_utt
        z_gmc_tokens = self.gmc_tokens_projector(gmc_tokens)
        z_text = self.text_projector(text_utt_2d)
        z_audio = self.audio_projector(audio_utt_2d)
        z_video = self.video_projector(video_utt_2d)
        p_gmc_tokens = self.gmc_tokens_predictor(z_gmc_tokens)
        p_text = self.text_predictor(z_text)
        p_audio = self.audio_predictor(z_audio)
        p_video = self.video_predictor(z_video)

        # final prediction module：三路 utterance 摘要 + gmc_tokens（与 based 一致）
        text_utt_final = text_utt_original  # (B, text_out)
        audio_utt_final = audio_utt         # (B, audio_out)
        video_utt_final = video_utt         # (B, video_out)
        if len(text_utt_final.shape) != 2:
            text_utt_final = text_utt_final.view(text_utt_final.shape[0], -1)
        if len(audio_utt_final.shape) != 2:
            audio_utt_final = audio_utt_final.view(audio_utt_final.shape[0], -1)
        if len(video_utt_final.shape) != 2:
            video_utt_final = video_utt_final.view(video_utt_final.shape[0], -1)
        fusion_h = torch.cat([text_utt_final, audio_utt_final, video_utt_final, gmc_tokens], dim=-1)
        fusion_h = self.post_fusion_dropout(fusion_h)
        # 残差 1：h1 = layer_1(fusion_h) + shortcut(fusion_h)，维度均为 post_fusion_dim
        h1 = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False) + self.post_fusion_shortcut(fusion_h)
        # 残差 2：h2 = h1 + layer_2(h1)，同维残差
        h2 = F.relu(self.post_fusion_layer_2(h1) + h1, inplace=False)
        output_fusion = self.post_fusion_layer_3(h2)

        suffix = '_m' if missing else ''
        res = {
            f'pred{suffix}': output_fusion,
            # 方案1：不确定性校准正则所需
            f'sigma_sq_text{suffix}': sigma_sq_text_out,
            f'sigma_sq_audio{suffix}': sigma_sq_audio_out,
            f'sigma_sq_video{suffix}': sigma_sq_video_out,
            # Code_URMF 风格：logvar 用于 KL 正则
            f'logvar_text{suffix}': logvar_text,
            f'logvar_audio{suffix}': logvar_audio,
            f'logvar_video{suffix}': logvar_video,
            f'mu_text{suffix}': mu_text,
            f'mu_audio{suffix}': mu_audio,
            f'mu_video{suffix}': mu_video,
            # 方案B/C：高层对齐所需（方案C 还需 z/p 做 SimSiam）
            f'text_utt{suffix}': text_utt_final,
            f'audio_utt{suffix}': audio_utt_final,
            f'video_utt{suffix}': video_utt_final,
            f'gmc_tokens{suffix}': gmc_tokens,
            f'z_gmc_tokens{suffix}': z_gmc_tokens.detach(),
            f'p_gmc_tokens{suffix}': p_gmc_tokens,
            f'z_text{suffix}': z_text.detach(),
            f'p_text{suffix}': p_text,
            f'z_audio{suffix}': z_audio.detach(),
            f'p_audio{suffix}': p_audio,
            f'z_video{suffix}': z_video.detach(),
            f'p_video{suffix}': p_video,
        }

        # low-level feature reconstruction
        # 约定：recon_text 只对「内容」段 (B,49,D)->(B,49,768)，与 text_for_recon 对齐；不把 CLS 那一行拿去映射，否则需人为构造 CLS target（不推荐）
        if missing:
            text_recon = self.recon_text(modality_ouputs[0][:, 1:, :])  # 去掉 CLS，只重建内容
            audio_recon = self.recon_audio(modality_ouputs[1])
            video_recon = self.recon_video(modality_ouputs[2])
            res.update(
                {
                    'text_recon': text_recon,
                    'audio_recon': audio_recon,
                    'video_recon': video_recon,
                }
            )
        else:
            res.update({'text_for_recon': text_for_recon})

        return res

    def forward(self, text, audio, video, text_missing_mask=None, audio_missing_mask=None, vision_missing_mask=None):
        text, text_m = text
        audio, audio_m, audio_lengths = audio
        video, video_m, video_lengths = video

        # 计算text_lengths，确保在GPU上进行（text已经在GPU上）
        # text[:,1,:] 是第二个token（通常是SEP），用于计算长度
        mask_len = torch.sum(text[:,1,:], dim=1, keepdim=True)  # 在GPU上计算
        text_lengths = (mask_len.squeeze().int() - 2).to(text.device)  # 保持在GPU上，-2 for CLS and SEP

        # complete view（不使用缺失 mask）
        res = self.forward_once(text, text_lengths, audio, audio_lengths, video, video_lengths, missing=False)
        # incomplete view：传入缺失 mask，EMT 融合时缺失位置不参与 attention
        res_m = self.forward_once(text_m, text_lengths, audio_m, audio_lengths, video_m, video_lengths, missing=True,
                                  text_missing_mask=text_missing_mask, audio_missing_mask=audio_missing_mask,
                                  vision_missing_mask=vision_missing_mask)

        return {**res, **res_m}


class AuViSubNetWithUncertainty(nn.Module):
    """
    带不确定性输入的LSTM编码器
    输入包含：原始特征 + mu + sigma_sq
    """
    def __init__(self, in_size, hidden_size, out_size=None, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension (feature + mu + sigma_sq)
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super().__init__()
        # 确保所有维度参数都是整数
        in_size = int(round(in_size))
        hidden_size = int(round(hidden_size))
        out_size = int(round(out_size)) if out_size is not None else None
        
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        feature_size = hidden_size * 2 if bidirectional else hidden_size
        self.linear_1 = nn.Linear(feature_size, out_size) if feature_size != out_size and out_size is not None else nn.Identity()

    def forward(self, x, lengths, return_temporal=False):
        '''
        x: (batch_size, sequence_len, in_size) - 应该在GPU上
        lengths: (batch_size,) - 应该在GPU上，但pack_padded_sequence需要CPU版本
        '''
        # pack_padded_sequence要求lengths在CPU上（PyTorch限制），但实际计算会在x的设备上进行
        # 所以这里只转换lengths到CPU，x保持在GPU上
        lengths_cpu = lengths.cpu() if lengths.is_cuda else lengths
        packed_sequence = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_last_hidden_state, final_states = self.rnn(packed_sequence)

        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        if not return_temporal:
            return y_1
        else:
            unpacked_last_hidden_state, _ = pad_packed_sequence(packed_last_hidden_state, batch_first=True)
            last_hidden_state = self.linear_1(unpacked_last_hidden_state)
            return last_hidden_state, y_1


class AuViSubNet(nn.Module):
    """
    原始LSTM编码器（保留用于兼容性）
    """
    def __init__(self, in_size, hidden_size, out_size=None, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super().__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        feature_size = hidden_size * 2 if bidirectional else hidden_size
        self.linear_1 = nn.Linear(feature_size, out_size) if feature_size != out_size and out_size is not None else nn.Identity()

    def forward(self, x, lengths, return_temporal=False):
        '''
        x: (batch_size, sequence_len, in_size) - 应该在GPU上
        lengths: (batch_size,) - 应该在GPU上，但pack_padded_sequence需要CPU版本
        '''
        # pack_padded_sequence要求lengths在CPU上（PyTorch限制），但实际计算会在x的设备上进行
        # 所以这里只转换lengths到CPU，x保持在GPU上
        lengths_cpu = lengths.cpu() if lengths.is_cuda else lengths
        packed_sequence = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_last_hidden_state, final_states = self.rnn(packed_sequence)

        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        if not return_temporal:
            return y_1
        else:
            unpacked_last_hidden_state, _ = pad_packed_sequence(packed_last_hidden_state, batch_first=True)
            last_hidden_state = self.linear_1(unpacked_last_hidden_state)
            return last_hidden_state, y_1


# 对比学习相关类（SimSiam projector / predictor）
class Projector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, output_dim),
                                 nn.BatchNorm1d(output_dim, affine=False))

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, input_dim, pred_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, pred_dim, bias=False),
                                 nn.BatchNorm1d(pred_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(pred_dim, output_dim))

    def forward(self, x):
        return self.net(x)


def _masked_mean_pool(x, lengths, max_len=None):
    """对3D序列做按有效长度的masked mean pooling。x: (B, seq_len, D), lengths: (B,). 返回 (B, D)"""
    if max_len is None:
        max_len = x.size(1)
    mask = length_to_mask(lengths, max_len=max_len, dtype=x.dtype)
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / denom


def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    # 获取max_len：如果length在GPU上，需要先获取item()（Python int），但后续计算在GPU上
    if max_len is None:
        max_len = length.max().item()  # 获取Python int值
    # 确保所有计算在length所在的设备上进行（GPU或CPU）
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask