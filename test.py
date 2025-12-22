import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
import matplotlib.pyplot as plt


def min_max_normalize_along_axis(array, axis):
    min_vals = np.min(array, axis=axis, keepdims=True)
    max_vals = np.max(array, axis=axis, keepdims=True)
    normalized_array = (array - min_vals) / (max_vals - min_vals)
    return normalized_array


def test_epoch(epoch, model, test_loader, logger, device, mask, args):
    mse_loss = nn.MSELoss().to(device)
    model.eval()
    preds = np.array([])
    labels = np.array([])
    tol_loss, tol_sample = 0, 0

    feats = []

    with torch.no_grad():
        for i, (video_feat, audio_feat, flow_feat, label, missing_modalities, idx) in enumerate(test_loader):
            video_feat = video_feat.to(device)
            audio_feat = audio_feat.to(device)  # (b, t, c)
            flow_feat = flow_feat.to(device)  # (b, t, c)
            label = label.float().to(device)
            out = model(video_feat, audio_feat, flow_feat, mask)
            pred = out['output']

            if 'encode' in out.keys() and out['encode'] is not None:
                feats.append(out['encode'].mean(dim=1).cpu().detach().numpy())
                # feats.append(out['embed'].cpu().detach().numpy())

            loss = mse_loss(pred*args.score_range, label*args.score_range)
            tol_loss += (loss.item() * label.shape[0])
            tol_sample += label.shape[0]

            if len(preds) == 0:
                preds = pred.cpu().detach().numpy()
                labels = label.cpu().detach().numpy()
            else:
                preds = np.concatenate((preds, pred.cpu().detach().numpy()), axis=0)
                labels = np.concatenate((labels, label.cpu().detach().numpy()), axis=0)
    # print(preds)
    avg_coef, _ = spearmanr(preds, labels)
    avg_loss = float(tol_loss) / float(tol_sample)

    if logger is not None:
        logger.add_scalar('Test coef', avg_coef, epoch)
        logger.add_scalar('Test loss', avg_loss, epoch)
    # print(preds.tolist())
    # print(labels.tolist())
    return avg_loss, avg_coef

def make_block_keep_mask(T: int, miss_rate: float, seed: int):
    """返回 keep mask: [T], 1=保留, 0=缺失(连续块)"""
    rng = np.random.RandomState(seed)
    L = int(round(T * miss_rate))
    L = max(1, min(L, T))
    s = rng.randint(0, T - L + 1)
    m = np.ones(T, dtype=np.float32)
    m[s:s+L] = 0.0
    return torch.from_numpy(m)  # [T]

def apply_partial_block_missing(x: torch.Tensor, miss_rate: float, idx: torch.Tensor, base_seed: int):
    """
    x: [B,T,C]
    idx: [B] (int)
    对每个样本用 idx 做seed，生成一个连续缺失块
    """
    B, T, C = x.shape
    x = x.clone()
    for b in range(B):
        seed = base_seed + int(idx[b].item()) * 1000003 + int(miss_rate * 1000) * 97
        m = make_block_keep_mask(T, miss_rate, seed).to(x.device)  # [T]
        x[b] = x[b] * m.view(T, 1)
    return x

def test_epoch_partial(epoch, model, test_loader, logger, device, modal_combo, miss_rate, args, base_seed=1234):
    mse_loss = nn.MSELoss().to(device)
    model.eval()
    preds, labels = [], []
    tol_loss, tol_sample = 0.0, 0

    with torch.no_grad():
        for i, (video_feat, audio_feat, flow_feat, label, missing_modalities, idx) in enumerate(test_loader):
            video_feat = video_feat.to(device)
            audio_feat = audio_feat.to(device)
            flow_feat  = flow_feat.to(device)
            label = label.float().to(device)

            # 对选中的模态施加连续缺失
            if modal_combo[0] == 1:
                video_feat = apply_partial_block_missing(video_feat, miss_rate, idx, base_seed + 11)
            if modal_combo[1] == 1:
                audio_feat = apply_partial_block_missing(audio_feat, miss_rate, idx, base_seed + 23)
            if modal_combo[2] == 1:
                flow_feat  = apply_partial_block_missing(flow_feat,  miss_rate, idx, base_seed + 37)

            # 注意：这里 mask 仍传 [1,1,1]，因为模态“存在”，只是帧缺失
            out = model(video_feat, audio_feat, flow_feat, [1,1,1], missing_modalities=None)
            pred = out['output']

            loss = mse_loss(pred * args.score_range, label * args.score_range)
            tol_loss += loss.item() * label.shape[0]
            tol_sample += label.shape[0]

            preds.append(pred.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    coef, _ = spearmanr(preds, labels)
    avg_loss = float(tol_loss) / float(tol_sample)

    if logger is not None:
        logger.add_scalar(f'TestPartial/coef_r{miss_rate}_m{modal_combo}', coef, epoch)
        logger.add_scalar(f'TestPartial/loss_r{miss_rate}_m{modal_combo}', avg_loss, epoch)

    return avg_loss, coef