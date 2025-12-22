from torch.utils.data import Dataset
import numpy as np
import os
import torch
import torch.nn.functional as F
import pandas as pd

def generate_modality_mask():
    """Generate a mask combination that includes at least one modality"""
    combinations = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1]
    ]
    idx = torch.randint(0, 6, [1])
    return combinations[idx[0]]

class FS1000Dataset(Dataset):
    def __init__(self, video_feat_path, audio_feat_path, flow_feat_path, label_path, clip_num=26, action_type='Ball', train=True, args=None):
        self.train = train
        self.video_path = video_feat_path
        self.audio_path = audio_feat_path
        self.flow_path = flow_feat_path
        self.erase_path = video_feat_path + '_erTrue'
        score_idx = {'TES': 130, 'PCS': 60, 'SS': 10, 'TR': 10, 'PE': 10, 'CO': 10, 'IN': 10}
        self.score_range = score_idx[action_type]
        args.score_range = self.score_range
        self.clip_num = clip_num
        self.labels = self.read_label(label_path, action_type)

    def read_label(self, label_path, action_type):
        fr = open(label_path, 'r')
        idx = {'TES': 1, 'PCS': 2, 'SS': 3, 'TR': 4, 'PE': 5, 'CO': 6, 'IN': 7}
        labels = []
        score = []
        for i, line in enumerate(fr):
            line = line.strip().split()
            s = float(line[idx[action_type]])
            if action_type == "PCS":
                s = s / float(line[8])
            labels.append([line[0], s])
            score.append(s)
        print("max:", max(score))
        return labels

    def __getitem__(self, idx):
        video_feat = np.load(os.path.join(self.video_path, self.labels[idx][0] + '.npy'))
        audio_feat = np.load(os.path.join(self.audio_path, self.labels[idx][0] + '.npy'))
        flow_feat = np.load(os.path.join(self.flow_path, self.labels[idx][0] + '.npy'))

        # temporal random crop or padding
        video_feat = video_feat.mean(1)
        if self.train:
            if len(video_feat) > self.clip_num:
                st = np.random.randint(0, len(video_feat) - self.clip_num)
                video_feat = video_feat[st:st + self.clip_num]
                audio_feat = audio_feat[st:st + self.clip_num]
                flow_feat = flow_feat[st:st + self.clip_num]
                # erase_feat = erase_feat[st:st + self.clip_num]
            elif len(video_feat) < self.clip_num:
                new_feat = np.zeros((self.clip_num, video_feat.shape[1]))
                new_feat[:video_feat.shape[0]] = video_feat
                video_feat = new_feat
                new_feat = np.zeros((self.clip_num, audio_feat.shape[1]))
                new_feat[:audio_feat.shape[0]] = audio_feat
                audio_feat = new_feat
                new_feat = np.zeros((self.clip_num, flow_feat.shape[1]))
                new_feat[:flow_feat.shape[0]] = flow_feat
                flow_feat = new_feat
        else:
            if len(video_feat) > self.clip_num:
                st = (len(video_feat) - self.clip_num) // 2
                video_feat = video_feat[st:st + self.clip_num]
                audio_feat = audio_feat[st:st + self.clip_num]
                flow_feat = flow_feat[st:st + self.clip_num]
            elif len(video_feat) < self.clip_num:
                new_feat = np.zeros((self.clip_num, video_feat.shape[1]))
                new_feat[:video_feat.shape[0]] = video_feat
                video_feat = new_feat
                new_feat = np.zeros((self.clip_num, audio_feat.shape[1]))
                new_feat[:audio_feat.shape[0]] = audio_feat
                audio_feat = new_feat
                new_feat = np.zeros((self.clip_num, flow_feat.shape[1]))
                new_feat[:flow_feat.shape[0]] = flow_feat
                flow_feat = new_feat
        flow_feat = torch.from_numpy(flow_feat).float()
        audio_feat = torch.from_numpy(audio_feat).float()
        video_feat = torch.from_numpy(video_feat).float()
        return video_feat, audio_feat, flow_feat, self.normalize_score(self.labels[idx][1])
        # return video_feat, audio_feat, self.labels[idx][1]

    def __len__(self):
        return len(self.labels)

    def normalize_score(self, score):
        return score / self.score_range


class RGDataset(Dataset):
    def __init__(self, video_feat_path, audio_feat_path, flow_feat_path, label_path, clip_num=26, action_type='Ball', train=True, args=None):
        self.train = train
        self.video_path = os.path.join(video_feat_path, action_type + '_rgb_VST.npy')
        self.audio_path = os.path.join(audio_feat_path, action_type + '_audio_AST.npy')
        self.flow_path = os.path.join(flow_feat_path, action_type + '_flow_I3D.npy')
        self.erase_path = video_feat_path + '_erTrue'
        self.score_range = 25.
        args.score_range = self.score_range
        self.clip_num = clip_num
        self.labels = self.read_label(label_path, action_type)

    def read_label(self, label_path, action_type):
        fr = open(label_path, 'r')
        idx = {'Difficulty_Score': 1, 'Execution_Score': 2, 'Total_Score': 3}
        labels = []
        for i, line in enumerate(fr):
            if i == 0:
                continue
            line = line.strip().split()
            if action_type == 'all' or action_type == line[0].split('_')[0]:
                labels.append([line[0], float(line[idx['Total_Score']])])
        return labels

    def __getitem__(self, idx):
        video_feat = np.load(self.video_path, allow_pickle=True).item()[self.labels[idx][0]]
        audio_feat = np.load(self.audio_path, allow_pickle=True).item()[self.labels[idx][0]]
        flow_feat = np.load(self.flow_path, allow_pickle=True).item()[self.labels[idx][0]]
        # temporal random crop or padding
        if self.train:
            if len(video_feat) > self.clip_num:
                st = np.random.randint(0, len(video_feat) - self.clip_num)
                video_feat = video_feat[st:st + self.clip_num]
                audio_feat = audio_feat[st:st + self.clip_num]
                flow_feat = flow_feat[st:st + self.clip_num]
                # erase_feat = erase_feat[st:st + self.clip_num]
            elif len(video_feat) < self.clip_num:
                new_feat = np.zeros((self.clip_num, video_feat.shape[1]))
                new_feat[:video_feat.shape[0]] = video_feat
                video_feat = new_feat
                new_feat = np.zeros((self.clip_num, audio_feat.shape[1]))
                new_feat[:audio_feat.shape[0]] = audio_feat
                audio_feat = new_feat
                new_feat = np.zeros((self.clip_num, flow_feat.shape[1]))
                new_feat[:flow_feat.shape[0]] = flow_feat
                flow_feat = new_feat
        else:
            if len(video_feat) > self.clip_num:
                st = (len(video_feat) - self.clip_num) // 2
                video_feat = video_feat[st:st + self.clip_num]
                audio_feat = audio_feat[st:st + self.clip_num]
                flow_feat = flow_feat[st:st + self.clip_num]
                # erase_feat = erase_feat[st:st + self.clip_num]
            elif len(video_feat) < self.clip_num:
                new_feat = np.zeros((self.clip_num, video_feat.shape[1]))
                new_feat[:video_feat.shape[0]] = video_feat
                video_feat = new_feat
                new_feat = np.zeros((self.clip_num, audio_feat.shape[1]))
                new_feat[:audio_feat.shape[0]] = audio_feat
                audio_feat = new_feat
                new_feat = np.zeros((self.clip_num, flow_feat.shape[1]))
                new_feat[:flow_feat.shape[0]] = flow_feat
                flow_feat = new_feat
        flow_feat = torch.from_numpy(flow_feat).float()
        audio_feat = torch.from_numpy(audio_feat).float()
        video_feat = torch.from_numpy(video_feat).float()
        return video_feat, audio_feat, flow_feat, self.normalize_score(self.labels[idx][1])

    def __len__(self):
        return len(self.labels)

    def normalize_score(self, score):
        return score / self.score_range
    
    
class FisVDataset(Dataset):
    def __init__(self, video_feat_path, audio_feat_path, flow_feat_path, label_path, clip_num=26, action_type='TES', train=True, args=None):
        self.train = train
        self.video_path = os.path.join(video_feat_path, 'FISV_rgb_VST.npy')
        self.audio_path = os.path.join(audio_feat_path, 'FISV_audio_AST.npy')
        self.flow_path = os.path.join(flow_feat_path, 'FISV_flow_I3D.npy')
        self.erase_path = video_feat_path + '_erTrue'
        if action_type == 'TES':
            self.score_range = 45
            args.score_range = 45
        else:
            self.score_range = 40
            args.score_range = 40
        self.clip_num = clip_num
        self.labels = self.read_label(label_path, action_type)

    def read_label(self, label_path, action_type):
        fr = open(label_path, 'r')
        idx = {'TES': 1, 'PCS': 2}
        labels = []
        for i, line in enumerate(fr):
            if i == 0:
                continue
            line = line.strip().split()
            labels.append([line[0], float(line[idx[action_type]])])
        return labels



    def __getitem__(self, idx):
        video_feat = np.load(self.video_path, allow_pickle=True).item()[self.labels[idx][0]]
        audio_feat = np.load(self.audio_path, allow_pickle=True).item()[self.labels[idx][0]]
        flow_feat = np.load(self.flow_path, allow_pickle=True).item()[self.labels[idx][0]]
        missing_modalities = []
        # temporal random crop or padding
        if self.train:
            if len(video_feat) > self.clip_num:
                st = np.random.randint(0, len(video_feat) - self.clip_num)
                video_feat = video_feat[st:st + self.clip_num]
                audio_feat = audio_feat[st:st + self.clip_num]
                flow_feat = flow_feat[st:st + self.clip_num]
                # erase_feat = erase_feat[st:st + self.clip_num]
            elif len(video_feat) < self.clip_num:
                new_feat = np.zeros((self.clip_num, video_feat.shape[1]))
                new_feat[:video_feat.shape[0]] = video_feat
                video_feat = new_feat
                new_feat = np.zeros((self.clip_num, audio_feat.shape[1]))
                new_feat[:audio_feat.shape[0]] = audio_feat
                audio_feat = new_feat
                new_feat = np.zeros((self.clip_num, flow_feat.shape[1]))
                new_feat[:flow_feat.shape[0]] = flow_feat
                flow_feat = new_feat

            # 生成缺失帧的掩码和缺失模态信息
            # 策略设置，30%全模态缺失，50%部分帧缺失，20模态不缺失
            prob = np.random.rand()
            if prob < 0.3:
                # 随机选择一个模态缺失 0：video 1：audio 2：flow
                drop_idxs = generate_modality_mask()
                if drop_idxs[0] == 0:
                    video_feat = np.zeros_like(video_feat)
                    missing_modalities.append('video_full')
                if drop_idxs[1] == 0:
                    audio_feat = np.zeros_like(audio_feat)
                    missing_modalities.append('audio_full')
                if drop_idxs[2] == 0:
                    flow_feat = np.zeros_like(flow_feat)
                    missing_modalities.append('flow_full')
            elif prob < 0.8:
                """
                trigger_prob: 该模态发生缺失的概率 (例如 50% 概率会坏)
                min_rate, max_rate: 缺失率的随机范围
                """

                # 定义内部函数：支持独立概率判定 + 随机缺失率
                def apply_partial_missing_randomly(feat, trigger_prob=0.5, min_rate=0.1, max_rate=0.5):
                    # 1. 判定这个模态这次是否由于“运气好”而保持完整
                    if np.random.rand() > trigger_prob:
                        return feat, False  # 没触发缺失，直接返回原特征
                    # 2. 如果触发了，随机决定这次缺多少 (例如 10% - 50%)
                    missing_rate = np.random.uniform(min_rate, max_rate)
                    num_frames = feat.shape[0]
                    num_missing = int(num_frames * missing_rate)
                    if num_missing > 0:
                        # 连续块丢失 (Block Drop) - 模拟传感器短时故障
                        start_idx = np.random.randint(0, num_frames - num_missing + 1)
                        missing_indices = np.arange(start_idx, start_idx + num_missing)
                        feat_copy = feat.copy()
                        feat_copy[missing_indices] = 0
                        return feat_copy, True
                    return feat, False

                # 3. 对每个模态独立掷骰子
                # 设定每个模态有 50% 的概率发生部分缺失
                video_feat, v_miss = apply_partial_missing_randomly(video_feat, trigger_prob=0.5)
                audio_feat, a_miss = apply_partial_missing_randomly(audio_feat, trigger_prob=0.5)
                flow_feat, f_miss = apply_partial_missing_randomly(flow_feat, trigger_prob=0.5)
                # 记录部分缺失的模态
                if v_miss: missing_modalities.append('video_partial')
                if a_miss: missing_modalities.append('audio_partial')
                if f_miss: missing_modalities.append('flow_partial')
            else:
                pass
        else:
            if len(video_feat) > self.clip_num:
                st = (len(video_feat) - self.clip_num) // 2
                video_feat = video_feat[st:st + self.clip_num]
                audio_feat = audio_feat[st:st + self.clip_num]
                flow_feat = flow_feat[st:st + self.clip_num]
                # erase_feat = erase_feat[st:st + self.clip_num]
            elif len(video_feat) < self.clip_num:
                new_feat = np.zeros((self.clip_num, video_feat.shape[1]))
                new_feat[:video_feat.shape[0]] = video_feat
                video_feat = new_feat
                new_feat = np.zeros((self.clip_num, audio_feat.shape[1]))
                new_feat[:audio_feat.shape[0]] = audio_feat
                audio_feat = new_feat
                new_feat = np.zeros((self.clip_num, flow_feat.shape[1]))
                new_feat[:flow_feat.shape[0]] = flow_feat
                flow_feat = new_feat


        flow_feat = torch.from_numpy(flow_feat).float()
        audio_feat = torch.from_numpy(audio_feat).float()
        video_feat = torch.from_numpy(video_feat).float()
        return video_feat, audio_feat, flow_feat, self.normalize_score(self.labels[idx][1]), missing_modalities, idx

    def __len__(self):
        return len(self.labels)

    def normalize_score(self, score):
        return score / self.score_range