import os
import time
import logging
import math
import copy
import argparse
import numpy as np
import pickle as plk
from glob import glob
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.distributed as dist

from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop

logger = logging.getLogger('MSA')

class EMT_DLFR():
    def __init__(self, args):
        assert args.train_mode == 'regression'

        self.args = args
        self.args.tasks = "M"
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)
        # 仅在主进程上做 verbose 日志与结果汇总
        self.is_main_process = not dist.is_initialized() or dist.get_rank() == 0

    def do_train(self, model, dataloader):
        def count_parameters(model, name='.fusion'):
            answer = 0
            for n, p in model.named_parameters():
                if name in n:
                    answer += p.numel()
            return answer
        def count_parameters_2(model):
            answer = 0
            for n, p in model.named_parameters():
                if 'predictor' not in n and 'projector' not in n and 'recon' not in n:
                    answer += p.numel()
            return answer

        logger.info(f'The model during inference has {count_parameters_2(model)} parameters.')
        logger.info(f'The fusion module (emt) has {count_parameters(model)} parameters.')

        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # 处理 DataParallel / DDP 包裹
        raw_model = model.module if hasattr(model, 'module') else model
        bert_params = list(raw_model.Model.text_model.named_parameters())
        audio_params = list(raw_model.Model.audio_model.named_parameters())
        video_params = list(raw_model.Model.video_model.named_parameters())

        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        audio_params = [p for n, p in audio_params]
        video_params = [p for n, p in video_params]
        model_params_other = [
            p for n, p in list(raw_model.Model.named_parameters())
            if 'text_model' not in n and 'audio_model' not in n and 'video_model' not in n
        ]

        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert, 'lr': self.args.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.learning_rate_bert},
            {'params': audio_params, 'weight_decay': self.args.weight_decay_audio, 'lr': self.args.learning_rate_audio},
            {'params': video_params, 'weight_decay': self.args.weight_decay_video, 'lr': self.args.learning_rate_video},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other, 'lr': self.args.learning_rate_other}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)

        # criterion
        criterion_attra = nn.CosineSimilarity(dim=1)  # 方案C：SimSiam 余弦
        criterion_recon = ReconLoss(self.args.recon_loss)

        # initilize results
        logger.info("Start training...")
        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss', 'Loss(pred_m)'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        # loop util earlystop
        while True:
            epochs += 1
            # DDP: 确保各进程的 shuffle 一致
            if 'train_sampler' in dataloader:
                dataloader['train_sampler'].set_epoch(epochs)
            # train
            y_pred, y_true = [], []
            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs

            s_t = time.time()

            with tqdm(dataloader['train'], disable=not self.is_main_process) as td:
                for batch_idx, batch_data in enumerate(td, 1):
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    # complete view
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    # incomplete (missing) view
                    vision_m = batch_data['vision_m'].to(self.args.device)
                    audio_m = batch_data['audio_m'].to(self.args.device)
                    text_m = batch_data['text_m'].to(self.args.device)
                    vision_missing_mask = batch_data['vision_missing_mask'].to(self.args.device)
                    audio_missing_mask = batch_data['audio_missing_mask'].to(self.args.device)
                    text_missing_mask = batch_data['text_missing_mask'].to(self.args.device)
                    vision_mask = batch_data['vision_mask'].to(self.args.device)
                    audio_mask = batch_data['audio_mask'].to(self.args.device)

                    labels = batch_data['labels']['M'].to(self.args.device)

                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    # forward（与 based 一致，不传 missing mask，仅用 length mask）
                    outputs = model((text, text_m), (audio, audio_m, audio_lengths), (vision, vision_m, vision_lengths))
                    # store results
                    y_pred.append(outputs['pred_m'].cpu())
                    y_true.append(labels.cpu())
                    # compute loss
                    ## prediction loss
                    loss_pred_m = torch.mean(torch.abs(outputs['pred_m'].view(-1) - labels.view(-1)))
                    loss_pred = torch.mean(torch.abs(outputs['pred'].view(-1) - labels.view(-1)))
                    ## 方案C：SimSiam 高层对齐（可选不确定性加权，消融时用 use_uncertainty_weighted_attra=False 关掉加权）
                    ## 【改进】分层不确定性加权 + 模态对齐权重差异化（与 new 方案一致）
                    loss_attra = torch.tensor(0.0, device=labels.device)
                    if all(k in outputs for k in ['p_gmc_tokens', 'p_gmc_tokens_m', 'z_gmc_tokens', 'z_gmc_tokens_m']):
                        # 计算各模态的余弦相似度
                        cos_g = (F.cosine_similarity(outputs['p_gmc_tokens_m'], outputs['z_gmc_tokens'], dim=1) +
                                 F.cosine_similarity(outputs['p_gmc_tokens'], outputs['z_gmc_tokens_m'], dim=1)) * 0.5
                        cos_t = (F.cosine_similarity(outputs['p_text_m'], outputs['z_text'], dim=1) +
                                 F.cosine_similarity(outputs['p_text'], outputs['z_text_m'], dim=1)) * 0.5
                        cos_a = (F.cosine_similarity(outputs['p_audio_m'], outputs['z_audio'], dim=1) +
                                 F.cosine_similarity(outputs['p_audio'], outputs['z_audio_m'], dim=1)) * 0.5
                        cos_v = (F.cosine_similarity(outputs['p_video_m'], outputs['z_video'], dim=1) +
                                 F.cosine_similarity(outputs['p_video'], outputs['z_video_m'], dim=1)) * 0.5

                        use_w = getattr(self.args, 'use_uncertainty_weighted_attra', True)
                        use_per_modal_w = getattr(self.args, 'use_per_modal_uncertainty_weight', False)
                        has_sigma = all(k in outputs for k in ['sigma_sq_text_m', 'sigma_sq_audio_m', 'sigma_sq_video_m'])

                        if use_w and has_sigma:
                            alpha = getattr(self.args, 'uncertainty_attra_alpha', 0.1)

                            if use_per_modal_w:
                                # 【改进1】分层不确定性加权：每个模态根据自身不确定性单独加权
                                # GMC tokens 使用整体不确定性（三模态平均）
                                s_m_avg = (outputs['sigma_sq_text_m'].mean(dim=-1) +
                                           outputs['sigma_sq_audio_m'].mean(dim=-1) +
                                           outputs['sigma_sq_video_m'].mean(dim=-1)) / 3.0
                                w_g = 1.0 / (1.0 + alpha * s_m_avg.detach().clamp(min=0))

                                # 各模态使用自身的不确定性
                                w_t = 1.0 / (1.0 + alpha * outputs['sigma_sq_text_m'].mean(dim=-1).detach().clamp(min=0))
                                w_a = 1.0 / (1.0 + alpha * outputs['sigma_sq_audio_m'].mean(dim=-1).detach().clamp(min=0))
                                w_v = 1.0 / (1.0 + alpha * outputs['sigma_sq_video_m'].mean(dim=-1).detach().clamp(min=0))
                            else:
                                # 原始方式：统一加权
                                s_m = (outputs['sigma_sq_text_m'].mean(dim=-1) +
                                      outputs['sigma_sq_audio_m'].mean(dim=-1) +
                                      outputs['sigma_sq_video_m'].mean(dim=-1)) / 3.0
                                w_g = w_t = w_a = w_v = 1.0 / (1.0 + alpha * s_m.detach().clamp(min=0))

                            # 调试：每 epoch 首个 batch 打印一次，确认不确定性加权是否生效
                            if batch_idx == 1:
                                _s = (outputs['sigma_sq_text_m'].mean(dim=-1) + outputs['sigma_sq_audio_m'].mean(dim=-1) + outputs['sigma_sq_video_m'].mean(dim=-1)) / 3.0
                                logger.info(f'[UncertaintyWeight] use_w={use_w} alpha={alpha:.3f} mean(sigma_sq)={_s.mean().item():.4f} mean(w_g)={w_g.mean().item():.4f} min(w_g)={w_g.min().item():.4f}')

                            # 【改进2】模态对齐权重差异化：不同模态的重要性权重
                            modal_weights = getattr(self.args, 'modal_attra_weights', {'gmc': 1.0, 'text': 1.0, 'audio': 1.0, 'video': 1.0})
                            weight_gmc = modal_weights.get('gmc', 1.0)
                            weight_text = modal_weights.get('text', 1.0)
                            weight_audio = modal_weights.get('audio', 1.0)
                            weight_video = modal_weights.get('video', 1.0)

                            # 加权后的对齐损失
                            loss_attra = -(w_g * cos_g * weight_gmc +
                                          w_t * cos_t * weight_text +
                                          w_a * cos_a * weight_audio +
                                          w_v * cos_v * weight_video).mean()
                        else:
                            # 纯 SimSiam，无不确定性加权，但仍可使用模态权重差异化
                            if batch_idx == 1:
                                logger.info(f'[UncertaintyWeight] use_w={use_w} has_sigma={has_sigma} -> 未使用不确定性加权 (纯 SimSiam)')
                            modal_weights = getattr(self.args, 'modal_attra_weights', {'gmc': 1.0, 'text': 1.0, 'audio': 1.0, 'video': 1.0})
                            weight_gmc = modal_weights.get('gmc', 1.0)
                            weight_text = modal_weights.get('text', 1.0)
                            weight_audio = modal_weights.get('audio', 1.0)
                            weight_video = modal_weights.get('video', 1.0)
                            loss_attra = -(cos_g * weight_gmc + cos_t * weight_text +
                                          cos_a * weight_audio + cos_v * weight_video).mean()
                        # 与 Based 一致：对齐损失中加入完整视图预测损失，使完整视图表示更强
                        loss_attra = loss_attra + loss_pred
                    ## reconstruction loss (low-level)
                    # 约定：mask 只覆盖内容 tokens（49），不把 CLS 算进有效长度，与 text_recon / text_for_recon 一致
                    mask = text[:, 1, 1:] - text_missing_mask[:, 1:]  # '1:' for excluding CLS
                    loss_recon_text = criterion_recon(outputs['text_recon'], outputs['text_for_recon'], mask)
                    mask = audio_mask - audio_missing_mask
                    loss_recon_audio = criterion_recon(outputs['audio_recon'], audio[:,: batch_data['audio_lengths'].max()], mask[:,: batch_data['audio_lengths'].max()])
                    mask = vision_mask - vision_missing_mask
                    loss_recon_video = criterion_recon(outputs['video_recon'], vision[:,: batch_data['vision_lengths'].max()], mask[:,: batch_data['vision_lengths'].max()])
                    loss_recon = loss_recon_text + loss_recon_audio + loss_recon_video

                    ## 方案1：不确定性校准正则（鼓励缺失视图 σ² > 完整视图 σ²）
                    if all(k in outputs for k in ['sigma_sq_text', 'sigma_sq_text_m', 'sigma_sq_audio', 'sigma_sq_audio_m', 'sigma_sq_video', 'sigma_sq_video_m']):
                        margin = getattr(self.args, 'uncertainty_reg_margin', 1e-4)
                        loss_unc = 0.0
                        for mod in ['text', 'audio', 'video']:
                            s_full = outputs[f'sigma_sq_{mod}'].mean(dim=-1)
                            s_miss = outputs[f'sigma_sq_{mod}_m'].mean(dim=-1)
                            loss_unc = loss_unc + F.relu(s_full - s_miss + margin).mean()
                        loss_uncertainty_reg = loss_unc
                        reg_weight = getattr(self.args, 'uncertainty_reg_weight', 0.1)
                    else:
                        loss_uncertainty_reg = 0.0
                        reg_weight = 0.0

                    ## Code_URMF 风格：KL 正则（VAE 式，对各模态 mu/logvar）
                    kl_weight = getattr(self.args, 'kl_reg_weight', 1e-3)
                    loss_kl = torch.tensor(0.0, device=labels.device)
                    if kl_weight > 0 and all(k in outputs for k in ['logvar_text', 'logvar_text_m', 'mu_text', 'mu_text_m']):
                        def _kl_loss(mu, logvar):
                            """Code_URMF: -(1 + logvar - mu^2 - exp(logvar)) / 2"""
                            return -0.5 * (1 + logvar - mu.pow(2) - torch.exp(logvar)).sum(dim=-1).mean()
                        for suffix in ['', '_m']:
                            loss_kl = loss_kl + _kl_loss(outputs[f'mu_text{suffix}'], outputs[f'logvar_text{suffix}'])
                            loss_kl = loss_kl + _kl_loss(outputs[f'mu_audio{suffix}'], outputs[f'logvar_audio{suffix}'])
                            loss_kl = loss_kl + _kl_loss(outputs[f'mu_video{suffix}'], outputs[f'logvar_video{suffix}'])
                    else:
                        kl_weight = 0.0

                    ## 辅助损失：边界感知（在 ±0.5, ±1.5 附近加大惩罚，提升 Mult_acc_5/7）
                    boundary_loss_weight = getattr(self.args, 'boundary_loss_weight', 0.0)
                    if boundary_loss_weight > 0:
                        pred_m_flat = outputs['pred_m'].view(-1)
                        labels_flat = labels.view(-1)
                        boundaries = [-1.5, -0.5, 0.5, 1.5]  # -1.5 -0.5 0.5 1.5
                        loss_b = 0.0
                        for b in boundaries:
                            b_t = torch.tensor(b, device=pred_m_flat.device, dtype=pred_m_flat.dtype)
                            # label > b 时希望 pred > b，否则惩罚 relu(b - pred)；label < b 时希望 pred < b，否则惩罚 relu(pred - b)
                            loss_b = loss_b + torch.where(
                                labels_flat > b_t,
                                F.relu(b_t - pred_m_flat),
                                F.relu(pred_m_flat - b_t)
                            )
                        loss_boundary = loss_b.mean()
                    else:
                        loss_boundary = 0.0

                    ## total loss（方案1+方案2+方案C + 边界感知辅助 + Code_URMF KL正则）
                    loss_recon_weight = getattr(self.args, 'loss_recon_weight', 1.0)
                    loss_attra_weight = getattr(self.args, 'loss_attra_weight', 0.5)
                    loss = loss_pred_m + loss_recon_weight * loss_recon + reg_weight * loss_uncertainty_reg + loss_attra_weight * loss_attra + kl_weight * loss_kl
                    if boundary_loss_weight > 0:
                        loss = loss + boundary_loss_weight * loss_boundary
                    loss.backward()

                    if len(td) >= 2 and batch_idx % (len(td) // 2) == 0 and self.is_main_process:
                        logger.info(f'Epoch {epochs} | Batch {batch_idx:>3d} | [Train] Loss {loss:.4f}')
                    train_loss += loss.item()

                    # update parameters
                    if not left_epochs:
                        # update
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()

            e_t = time.time()
            if self.is_main_process:
                logger.info(f'One epoch time for training: {e_t - s_t:.3f}s.')

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)

            log_infos = [''] * 8
            log_infos[0] = log_infos[-1] = '-' * 100

            # validation
            s_t = time.time()

            val_results = self.do_test(model, dataloader['valid'], None, mode="VAL", epochs=epochs)  # 方案3 已注释

            e_t = time.time()
            if self.is_main_process:
                logger.info(f'One epoch time for validation: {e_t - s_t:.3f}s.')

            cur_valid = val_results[self.args.KeyEval]
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                self.best_epoch = best_epoch
                # 仅在主进程上保存模型；同时兼容 DataParallel / DDP
                if self.is_main_process:
                    state = model.cpu().state_dict()
                    if hasattr(model, 'module'):
                        state = {k.replace('module.', ''): v for k, v in state.items()}
                    torch.save(state, self.args.model_save_path)
                    model.to(self.args.device)
                # 同步各进程，确保在进入下一 epoch 前 checkpoint 已写完
                if dist.is_initialized():
                    dist.barrier()
                log_infos[5] = f'==> Note: achieve best [Val] results at epoch {best_epoch}'

            log_infos[1] = f"Seed {self.args.seed} ({self.args.seeds.index(self.args.seed)+1}/{self.args.num_seeds}) " \
                       f"| Epoch {epochs} (early stop={epochs-best_epoch}) | Train Loss {train_loss:.4f} | Val Loss {val_results['Loss']:.4f}"
            log_infos[2] = f"[Train] {dict_to_str(train_results)}"
            log_infos[3] = f"  [Val] {dict_to_str(val_results)}"

            # log information（仅主进程输出完整版日志）
            if self.is_main_process:
                for log_info in log_infos:
                    if log_info:
                        logger.info(log_info)

            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                if self.is_main_process:
                    logger.info(f"==> Note: since '{self.args.KeyEval}' does not improve in the past {self.args.early_stop} epochs, early stop the training process!")
                return

    def do_test(self, model, dataloader, criterion_attra=None, criterion_recon=None, mode="VAL", epochs=None):
        if epochs is None and self.is_main_process:
            logger.info("=" * 30 + f"Start Test of Seed {self.args.seed}" + "=" * 30)
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        eval_loss_pred = 0.0
        # criterion（与 Based 一致：Val/Test 也用 pred_m + attra + recon 算 Loss，便于和 Based 可比、早停一致）
        if criterion_attra is None: criterion_attra = nn.CosineSimilarity(dim=1)
        if criterion_recon is None: criterion_recon = ReconLoss(self.args.recon_loss)
        with torch.no_grad():
            with tqdm(dataloader, disable=not self.is_main_process) as td:
                for batch_idx, batch_data in enumerate(td, 1):
                    # complete view
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    # incomplete (missing) view
                    vision_m = batch_data['vision_m'].to(self.args.device)
                    audio_m = batch_data['audio_m'].to(self.args.device)
                    text_m = batch_data['text_m'].to(self.args.device)
                    vision_missing_mask = batch_data['vision_missing_mask'].to(self.args.device)
                    audio_missing_mask = batch_data['audio_missing_mask'].to(self.args.device)
                    text_missing_mask = batch_data['text_missing_mask'].to(self.args.device)
                    vision_mask = batch_data['vision_mask'].to(self.args.device)
                    audio_mask = batch_data['audio_mask'].to(self.args.device)

                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    labels = batch_data['labels']['M'].to(self.args.device).view(-1)
                    outputs = model((text, text_m), (audio, audio_m, audio_lengths), (vision, vision_m, vision_lengths))

                    # compute loss（与 Based 一致：含 loss_attra，便于与 Based 的 Loss 可比、早停一致）
                    ## task loss (prediction loss of incomplete view)
                    loss_pred_m = torch.mean(torch.abs(outputs['pred_m'].view(-1) - labels.view(-1)))
                    ## attraction loss (high-level)
                    loss_attra_gmc_tokens = -(criterion_attra(outputs['p_gmc_tokens_m'], outputs['z_gmc_tokens']).mean() +
                                              criterion_attra(outputs['p_gmc_tokens'], outputs['z_gmc_tokens_m']).mean()) * 0.5
                    loss_attra_text = -(criterion_attra(outputs['p_text_m'], outputs['z_text']).mean() +
                                        criterion_attra(outputs['p_text'], outputs['z_text_m']).mean()) * 0.5
                    loss_attra_audio = -(criterion_attra(outputs['p_audio_m'], outputs['z_audio']).mean() +
                                         criterion_attra(outputs['p_audio'], outputs['z_audio_m']).mean()) * 0.5
                    loss_attra_video = -(criterion_attra(outputs['p_video_m'], outputs['z_video']).mean() +
                                         criterion_attra(outputs['p_video'], outputs['z_video_m']).mean()) * 0.5
                    loss_pred = torch.mean(torch.abs(outputs['pred'].view(-1) - labels.view(-1)))
                    loss_attra = loss_attra_gmc_tokens + loss_attra_text + loss_attra_audio + loss_attra_video + loss_pred
                    ## reconstruction loss (low-level)
                    mask = text[:, 1, 1:] - text_missing_mask[:, 1:]  # '1:' for excluding CLS
                    loss_recon_text = criterion_recon(outputs['text_recon'], outputs['text_for_recon'], mask)
                    mask = audio_mask - audio_missing_mask
                    loss_recon_audio = criterion_recon(outputs['audio_recon'], audio[:,: batch_data['audio_lengths'].max()], mask[:,: batch_data['audio_lengths'].max()])
                    mask = vision_mask - vision_missing_mask
                    loss_recon_video = criterion_recon(outputs['video_recon'], vision[:,: batch_data['vision_lengths'].max()], mask[:,: batch_data['vision_lengths'].max()])
                    loss_recon = loss_recon_text + loss_recon_audio + loss_recon_video
                    ## total loss（与 Based 一致）
                    loss_recon_weight = getattr(self.args, 'loss_recon_weight', 1.0)
                    loss_attra_weight = getattr(self.args, 'loss_attra_weight', 0.5)
                    loss = loss_pred_m + loss_attra_weight * loss_attra + loss_recon_weight * loss_recon

                    if len(td) >= 2 and batch_idx % (len(td) // 2) == 0 and self.is_main_process:
                        logger.info(f'Epoch {epochs} | Batch {batch_idx:>3d} | [Val] Loss {loss:.4f}')
                    eval_loss += loss.item()
                    eval_loss_pred += loss_pred_m.item()
                    y_pred.append(outputs['pred_m'].cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)
        eval_loss_pred = eval_loss_pred / len(dataloader)
        # logger.info(mode+"-(%s)" % self.args.modelName + " >> loss: %.4f " % eval_loss)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        # logger.info('M: >> ' + dict_to_str(eval_results))
        eval_results['Loss'] = eval_loss
        eval_results['Loss(pred_m)'] = eval_loss_pred
        if epochs is None and self.is_main_process:  # for TEST
            logger.info(f"\n [Test] {dict_to_str(eval_results)}")
            logger.info(f"==> Note: achieve this results at epoch {getattr(self, 'best_epoch', None)} (best [Val]) / {getattr(self, 'best_test_epoch', None)} (best [Test])")
        return eval_results


class ReconLoss(nn.Module):
    def __init__(self, type):
        super().__init__()
        self.eps = 1e-6
        self.type = type
        if type == 'L1Loss':
            self.loss = nn.L1Loss(reduction='sum')
        elif type == 'SmoothL1Loss':
            self.loss = nn.SmoothL1Loss(reduction='sum')
        elif type == 'MSELoss':
            self.loss = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError

    def forward(self, pred, target, mask):
        """
            pred, target -> batch, seq_len, d
            mask -> batch, seq_len
        """
        mask = mask.unsqueeze(-1).expand(pred.shape[0], pred.shape[1], pred.shape[2]).float()

        loss = self.loss(pred*mask, target*mask) / (torch.sum(mask) + self.eps)

        return loss
