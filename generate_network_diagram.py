#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成EMT-DLFR网络结构图的Python脚本
使用matplotlib绘制网络流程图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig, ax = plt.subplots(1, 1, figsize=(20, 28))
ax.set_xlim(0, 20)
ax.set_ylim(0, 35)
ax.axis('off')

# 定义颜色
colors = {
    'input': '#E1F5FF',
    'encoder': '#FFF4E1',
    'projection': '#F0E1FF',
    'uncertainty': '#E1FFE1',
    'moe': '#FFE1F0',
    'emt': '#FFF1E1',
    'prediction': '#E1F5FF',
    'recon': '#FFE1E1',
    'loss': '#F5E1FF',
    'box': '#333333',
    'text': '#000000'
}

# 定义函数：绘制文本框
def draw_text_box(ax, x, y, width, height, text, color='lightblue', fontsize=9, bold=False):
    """绘制带文本的框"""
    # 绘制框
    box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                         boxstyle="round,pad=0.05", 
                         facecolor=color, edgecolor=colors['box'], linewidth=1.5)
    ax.add_patch(box)
    
    # 添加文本
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
            weight=weight, color=colors['text'], wrap=True)

# 定义函数：绘制箭头
def draw_arrow(ax, x1, y1, x2, y2, style='->', color='black', linewidth=1.5):
    """绘制箭头"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, color=color, 
                           linewidth=linewidth, zorder=1)
    ax.add_patch(arrow)

# 定义函数：绘制模块分组框
def draw_group_box(ax, x, y, width, height, title, color='lightgray'):
    """绘制分组框"""
    box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                         boxstyle="round,pad=0.3", 
                         facecolor=color, edgecolor=colors['box'], 
                         linewidth=2, alpha=0.3, linestyle='--')
    ax.add_patch(box)
    # 标题
    ax.text(x, y+height/2-0.2, title, ha='center', va='bottom', 
            fontsize=11, weight='bold', color=colors['box'])

# ==================== 绘制网络结构 ====================

# 1. 输入层 (y=33)
y_input = 33
draw_group_box(ax, 10, y_input+1, 18, 1.5, '输入层 (Input Layer)', colors['input'])
draw_text_box(ax, 3, y_input, 2, 0.6, 'Text\n(B,T,D)', colors['input'], 8)
draw_text_box(ax, 6, y_input, 2, 0.6, 'Audio\n(B,T,D)', colors['input'], 8)
draw_text_box(ax, 9, y_input, 2, 0.6, 'Video\n(B,T,D)', colors['input'], 8)
draw_text_box(ax, 12, y_input, 2, 0.6, 'Text_m\n(B,T,D)', colors['input'], 8)
draw_text_box(ax, 15, y_input, 2, 0.6, 'Audio_m\n(B,T,D)', colors['input'], 8)
draw_text_box(ax, 17, y_input, 2, 0.6, 'Video_m\n(B,T,D)', colors['input'], 8)

# 2. 单模态编码器 (y=30)
y_encoder = 30
draw_group_box(ax, 10, y_encoder+1, 18, 1.5, '单模态编码器 (Unimodal Encoders)', colors['encoder'])
draw_text_box(ax, 4, y_encoder, 2.5, 0.8, 'BERT\nText Encoder', colors['encoder'], 9, True)
draw_text_box(ax, 8, y_encoder, 2.5, 0.8, 'LSTM\nAudio Encoder', colors['encoder'], 9, True)
draw_text_box(ax, 12, y_encoder, 2.5, 0.8, 'LSTM\nVideo Encoder', colors['encoder'], 9, True)

# 箭头：输入到编码器
draw_arrow(ax, 3, y_input-0.3, 4, y_encoder+0.4)
draw_arrow(ax, 6, y_input-0.3, 8, y_encoder+0.4)
draw_arrow(ax, 9, y_input-0.3, 12, y_encoder+0.4)

# 编码器输出
y_output = 28.5
draw_text_box(ax, 2, y_output, 1.5, 0.5, 'text_utt\n(B,1,D)', colors['encoder'], 7)
draw_text_box(ax, 4, y_output, 1.5, 0.5, 'text\n(B,T,D)', colors['encoder'], 7)
draw_text_box(ax, 7, y_output, 1.5, 0.5, 'audio_utt\n(B,D)', colors['encoder'], 7)
draw_text_box(ax, 9, y_output, 1.5, 0.5, 'audio\n(B,T,D)', colors['encoder'], 7)
draw_text_box(ax, 12, y_output, 1.5, 0.5, 'video_utt\n(B,D)', colors['encoder'], 7)
draw_text_box(ax, 14, y_output, 1.5, 0.5, 'video\n(B,T,D)', colors['encoder'], 7)

draw_arrow(ax, 4, y_encoder-0.4, 2, y_output+0.25)
draw_arrow(ax, 4, y_encoder-0.4, 4, y_output+0.25)
draw_arrow(ax, 8, y_encoder-0.4, 7, y_output+0.25)
draw_arrow(ax, 8, y_encoder-0.4, 9, y_output+0.25)
draw_arrow(ax, 12, y_encoder-0.4, 12, y_output+0.25)
draw_arrow(ax, 12, y_encoder-0.4, 14, y_output+0.25)

# 3. 投影层 (y=27)
y_proj = 27
draw_group_box(ax, 10, y_proj+0.6, 18, 1, '投影层 (Projection Layer)', colors['projection'])
draw_text_box(ax, 3, y_proj, 2, 0.5, 'proj_text\nLinear', colors['projection'], 8)
draw_text_box(ax, 7, y_proj, 2, 0.5, 'proj_audio\nLinear', colors['projection'], 8)
draw_text_box(ax, 11, y_proj, 2, 0.5, 'proj_video\nLinear', colors['projection'], 8)

draw_arrow(ax, 2, y_output-0.25, 3, y_proj+0.25)
draw_arrow(ax, 4, y_output-0.25, 3, y_proj+0.25)
draw_arrow(ax, 7, y_output-0.25, 7, y_proj+0.25)
draw_arrow(ax, 9, y_output-0.25, 7, y_proj+0.25)
draw_arrow(ax, 12, y_output-0.25, 11, y_proj+0.25)
draw_arrow(ax, 14, y_output-0.25, 11, y_proj+0.25)

y_proj_out = 25.5
draw_text_box(ax, 2, y_proj_out, 1.2, 0.4, 'text_proj\n(B,d)', colors['projection'], 7)
draw_text_box(ax, 4, y_proj_out, 1.2, 0.4, 'text_seq\n(B,T,d)', colors['projection'], 7)
draw_text_box(ax, 7, y_proj_out, 1.2, 0.4, 'audio_proj\n(B,d)', colors['projection'], 7)
draw_text_box(ax, 9, y_proj_out, 1.2, 0.4, 'audio_seq\n(B,T,d)', colors['projection'], 7)
draw_text_box(ax, 12, y_proj_out, 1.2, 0.4, 'video_proj\n(B,d)', colors['projection'], 7)
draw_text_box(ax, 14, y_proj_out, 1.2, 0.4, 'video_seq\n(B,T,d)', colors['projection'], 7)

draw_arrow(ax, 3, y_proj-0.25, 2, y_proj_out+0.2)
draw_arrow(ax, 3, y_proj-0.25, 4, y_proj_out+0.2)
draw_arrow(ax, 7, y_proj-0.25, 7, y_proj_out+0.2)
draw_arrow(ax, 7, y_proj-0.25, 9, y_proj_out+0.2)
draw_arrow(ax, 11, y_proj-0.25, 12, y_proj_out+0.2)
draw_arrow(ax, 11, y_proj-0.25, 14, y_proj_out+0.2)

# 4. 不确定性估计 (y=24)
y_uncertainty = 24
draw_group_box(ax, 10, y_uncertainty+0.6, 18, 1, '不确定性估计 (Uncertainty Estimation)', colors['uncertainty'])
draw_text_box(ax, 3, y_uncertainty, 2, 0.5, 'Uncertainty\nEncoder (T)', colors['uncertainty'], 8)
draw_text_box(ax, 7, y_uncertainty, 2, 0.5, 'Uncertainty\nEncoder (A)', colors['uncertainty'], 8)
draw_text_box(ax, 11, y_uncertainty, 2, 0.5, 'Uncertainty\nEncoder (V)', colors['uncertainty'], 8)

draw_arrow(ax, 2, y_proj_out-0.2, 3, y_uncertainty+0.25)

draw_arrow(ax, 7, y_proj_out-0.2, 7, y_uncertainty+0.25)
draw_arrow(ax, 12, y_proj_out-0.2, 11, y_uncertainty+0.25)

y_unc_out = 22.5
draw_text_box(ax, 2, y_unc_out, 1, 0.4, 'μ_t, σ²_t', colors['uncertainty'], 7)
draw_text_box(ax, 6, y_unc_out, 1, 0.4, 'μ_a, σ²_a', colors['uncertainty'], 7)
draw_text_box(ax, 10, y_unc_out, 1, 0.4, 'μ_v, σ²_v', colors['uncertainty'], 7)

draw_arrow(ax, 3, y_uncertainty-0.25, 2, y_unc_out+0.2)
draw_arrow(ax, 7, y_uncertainty-0.25, 6, y_unc_out+0.2)
draw_arrow(ax, 11, y_uncertainty-0.25, 10, y_unc_out+0.2)

# 5. MOE融合 (y=21)
y_moe = 21
draw_group_box(ax, 10, y_moe+1, 18, 2, 'MOE融合 (MOE Fusion)', colors['moe'])

# 门控网络
draw_text_box(ax, 5, y_moe+0.5, 2.5, 0.6, 'Gating Network\n根据不确定性选择专家', colors['moe'], 8, True)

# 专家
draw_text_box(ax, 9, y_moe+0.5, 2, 0.5, 'Expert-1\n(text_dominant)', colors['moe'], 7)
draw_text_box(ax, 12, y_moe+0.5, 2, 0.5, 'Expert-2\n(av_dominant)', colors['moe'], 7)
draw_text_box(ax, 15, y_moe+0.5, 2, 0.5, 'Expert-3\n(balanced)', colors['moe'], 7)

# 加权融合
draw_text_box(ax, 10, y_moe-0.3, 3, 0.5, 'Weighted Fusion', colors['moe'], 8, True)

# gmc_tokens生成
draw_text_box(ax, 10, y_moe-1, 3, 0.5, 'gmc_proj', colors['moe'], 8)

# 箭头
draw_arrow(ax, 2, y_unc_out-0.2, 5, y_moe+0.8)
draw_arrow(ax, 6, y_unc_out-0.2, 5, y_moe+0.8)
draw_arrow(ax, 10, y_unc_out-0.2, 5, y_moe+0.8)

draw_arrow(ax, 2, y_unc_out-0.2, 9, y_moe+0.8)
draw_arrow(ax, 6, y_unc_out-0.2, 9, y_moe+0.8)
draw_arrow(ax, 10, y_unc_out-0.2, 9, y_moe+0.8)

draw_arrow(ax, 2, y_unc_out-0.2, 12, y_moe+0.8)
draw_arrow(ax, 6, y_unc_out-0.2, 12, y_moe+0.8)
draw_arrow(ax, 10, y_unc_out-0.2, 12, y_moe+0.8)

draw_arrow(ax, 2, y_unc_out-0.2, 15, y_moe+0.8)
draw_arrow(ax, 6, y_unc_out-0.2, 15, y_moe+0.8)
draw_arrow(ax, 10, y_unc_out-0.2, 15, y_moe+0.8)

draw_arrow(ax, 5, y_moe+0.2, 10, y_moe+0.2)
draw_arrow(ax, 9, y_moe+0.2, 10, y_moe+0.2)
draw_arrow(ax, 12, y_moe+0.2, 10, y_moe+0.2)
draw_arrow(ax, 15, y_moe+0.2, 10, y_moe+0.2)

draw_arrow(ax, 10, y_moe-0.55, 10, y_moe-1.25)

y_gmc = 18.5
draw_text_box(ax, 10, y_gmc, 2.5, 0.5, 'gmc_tokens\n(B, 3, d)', colors['moe'], 8, True)
draw_arrow(ax, 10, y_moe-1.25, 10, y_gmc+0.25)

# 6. EMT融合 (y=17)
y_emt = 17
draw_group_box(ax, 10, y_emt+0.8, 18, 1.2, 'EMT融合 (Efficient Multimodal Transformer)', colors['emt'])
draw_text_box(ax, 10, y_emt, 4, 0.6, 'EMT Module\nCross-Self Attention', colors['emt'], 9, True)

draw_arrow(ax, 10, y_gmc-0.25, 10, y_emt+0.3)
draw_arrow(ax, 4, y_proj_out-0.2, 10, y_emt+0.3)
draw_arrow(ax, 9, y_proj_out-0.2, 10, y_emt+0.3)
draw_arrow(ax, 14, y_proj_out-0.2, 10, y_emt+0.3)

y_emt_out = 15.5
draw_text_box(ax, 8, y_emt_out, 1.5, 0.4, 'gmc_out\n(B,3,d)', colors['emt'], 7)
draw_text_box(ax, 10, y_emt_out, 1.5, 0.4, 'text_out\n(B,T,d)', colors['emt'], 7)
draw_text_box(ax, 12, y_emt_out, 1.5, 0.4, 'audio_out\n(B,T,d)', colors['emt'], 7)
draw_text_box(ax, 14, y_emt_out, 1.5, 0.4, 'video_out\n(B,T,d)', colors['emt'], 7)

draw_arrow(ax, 10, y_emt-0.3, 8, y_emt_out+0.2)
draw_arrow(ax, 10, y_emt-0.3, 10, y_emt_out+0.2)
draw_arrow(ax, 10, y_emt-0.3, 12, y_emt_out+0.2)
draw_arrow(ax, 10, y_emt-0.3, 14, y_emt_out+0.2)

# 7. 最终预测 (y=14)
y_pred = 14
draw_group_box(ax, 10, y_pred+1, 18, 2, '最终预测 (Final Prediction)', colors['prediction'])

# 拼接
draw_text_box(ax, 10, y_pred+0.5, 3, 0.5, 'Concatenate\ntext_utt + audio_utt + video_utt + gmc', colors['prediction'], 8)

# MLP
draw_text_box(ax, 10, y_pred, 3, 0.5, 'post_fusion_layer_1\nLinear + ReLU', colors['prediction'], 7)
draw_text_box(ax, 10, y_pred-0.6, 3, 0.5, 'post_fusion_layer_2\nLinear + ReLU', colors['prediction'], 7)
draw_text_box(ax, 10, y_pred-1.2, 3, 0.5, 'post_fusion_layer_3\nLinear', colors['prediction'], 7)

draw_arrow(ax, 2, y_output-0.25, 10, y_pred+0.75)
draw_arrow(ax, 7, y_output-0.25, 10, y_pred+0.75)
draw_arrow(ax, 12, y_output-0.25, 10, y_pred+0.75)
draw_arrow(ax, 8, y_emt_out-0.2, 10, y_pred+0.75)

draw_arrow(ax, 10, y_pred+0.25, 10, y_pred-0.35)
draw_arrow(ax, 10, y_pred-0.35, 10, y_pred-0.95)
draw_arrow(ax, 10, y_pred-0.95, 10, y_pred-1.7)

y_final_pred = 11.5
draw_text_box(ax, 10, y_final_pred, 2, 0.5, 'pred\n(B, 1)', colors['prediction'], 9, True)
draw_arrow(ax, 10, y_pred-1.7, 10, y_final_pred+0.25)

# 8. 低层重建 (y=10)
y_recon = 10
draw_group_box(ax, 10, y_recon+1, 18, 1.5, '低层重建 (Low-level Reconstruction) - 仅missing view', colors['recon'])
draw_text_box(ax, 8, y_recon, 2, 0.5, 'recon_text\nLinear', colors['recon'], 8)
draw_text_box(ax, 10, y_recon, 2, 0.5, 'recon_audio\nLinear', colors['recon'], 8)
draw_text_box(ax, 12, y_recon, 2, 0.5, 'recon_video\nLinear', colors['recon'], 8)

draw_arrow(ax, 10, y_emt_out-0.2, 8, y_recon+0.25)
draw_arrow(ax, 12, y_emt_out-0.2, 10, y_recon+0.25)
draw_arrow(ax, 14, y_emt_out-0.2, 12, y_recon+0.25)

y_recon_out = 8.5
draw_text_box(ax, 8, y_recon_out, 1.5, 0.4, 'text_recon\n(B,T,D)', colors['recon'], 7)
draw_text_box(ax, 10, y_recon_out, 1.5, 0.4, 'audio_recon\n(B,T,D)', colors['recon'], 7)
draw_text_box(ax, 12, y_recon_out, 1.5, 0.4, 'video_recon\n(B,T,D)', colors['recon'], 7)

draw_arrow(ax, 8, y_recon-0.25, 8, y_recon_out+0.2)
draw_arrow(ax, 10, y_recon-0.25, 10, y_recon_out+0.2)
draw_arrow(ax, 12, y_recon-0.25, 12, y_recon_out+0.2)

# 9. 损失函数 (y=7)
y_loss = 7
draw_group_box(ax, 10, y_loss+1, 18, 2, '损失函数 (Loss Functions)', colors['loss'])

draw_text_box(ax, 5, y_loss+0.3, 2.5, 0.5, 'loss_pred_m\nL1 Loss', colors['loss'], 8)
draw_text_box(ax, 9, y_loss+0.3, 2, 0.5, 'loss_recon_text\nReconLoss', colors['loss'], 8)
draw_text_box(ax, 12, y_loss+0.3, 2, 0.5, 'loss_recon_audio\nReconLoss', colors['loss'], 8)
draw_text_box(ax, 15, y_loss+0.3, 2, 0.5, 'loss_recon_video\nReconLoss', colors['loss'], 8)

draw_text_box(ax, 10, y_loss-0.7, 4, 0.5, 'Total Loss = loss_pred_m + λ * loss_recon', colors['loss'], 9, True)

draw_arrow(ax, 10, y_final_pred-0.25, 5, y_loss+0.55)
draw_arrow(ax, 8, y_recon_out-0.2, 9, y_loss+0.55)
draw_arrow(ax, 10, y_recon_out-0.2, 12, y_loss+0.55)
draw_arrow(ax, 12, y_recon_out-0.2, 15, y_loss+0.55)

draw_arrow(ax, 5, y_loss+0.05, 10, y_loss-0.45)
draw_arrow(ax, 9, y_loss+0.05, 10, y_loss-0.45)
draw_arrow(ax, 12, y_loss+0.05, 10, y_loss-0.45)
draw_arrow(ax, 15, y_loss+0.05, 10, y_loss-0.45)

# 标题
ax.text(10, 34.5, 'EMT-DLFR 网络端到端结构图', ha='center', va='center', 
        fontsize=16, weight='bold', color='#000000')

# 保存图片
plt.tight_layout()
plt.savefig('network_architecture.jpg', dpi=300, bbox_inches='tight', format='jpg')
plt.savefig('network_architecture.png', dpi=300, bbox_inches='tight', format='png')
print("网络结构图已保存为 network_architecture.jpg 和 network_architecture.png")
