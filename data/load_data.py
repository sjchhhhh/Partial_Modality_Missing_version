import os
import logging
import pickle
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

__all__ = ['MMDataLoader']

logger = logging.getLogger('MSA')

class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATA_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims,
        }
        DATA_MAP[args.datasetName]()

    def __init_mosi(self):
        with open(self.args.dataPath, 'rb') as f:
            data = pickle.load(f)
        if self.args.use_bert:
            self.text = data[self.mode]['text_bert'].astype(np.float32)
        else:
            self.text = data[self.mode]['text'].astype(np.float32)
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.rawText = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']

        self.labels = {
            'M': data[self.mode][self.args.train_mode+'_labels'].astype(np.float32)
        }
        if self.args.datasetName == 'sims':
            for m in "TAV":
                self.labels[m] = data[self.mode][self.args.train_mode+'_labels_'+m]

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        if not self.args.need_data_aligned:
            self.audio_lengths = data[self.mode]['audio_lengths']
            self.vision_lengths = data[self.mode]['vision_lengths']
        else:
            text_lengths = np.sum(self.text[:,1], axis=1).astype(np.int16).tolist()
            self.audio_lengths, self.vision_lengths = text_lengths, text_lengths
        self.audio[self.audio == -np.inf] = 0

        if self.args.data_missing:
            # 统一训练：训练集按样本随机缺失率，不预生成；valid/test 仍用固定缺失率预生成
            random_missing = getattr(self.args, 'random_missing_rate', False) and self.mode == 'train'
            if random_missing:
                self._unified_missing_train = True
                # text_m / audio_m / vision_m 等在 __getitem__ 中按随机缺失率动态生成
            else:
                self._unified_missing_train = False
                self.text_m, self.text_length, self.text_mask, self.text_missing_mask = self.generate_m(self.text[:,0,:], self.text[:,1,:], None,
                                                                                            self.args.missing_rate[0], self.args.missing_seed[0], mode='text')
                Input_ids_m = np.expand_dims(self.text_m, 1)
                Input_mask = np.expand_dims(self.text_mask, 1)
                Segment_ids = np.expand_dims(self.text[:,2,:], 1)
                self.text_m = np.concatenate((Input_ids_m, Input_mask, Segment_ids), axis=1)

                self.audio_m, self.audio_length, self.audio_mask, self.audio_missing_mask = self.generate_m(self.audio, None, self.audio_lengths,
                                                                                            self.args.missing_rate[1], self.args.missing_seed[1], mode='audio')
                self.vision_m, self.vision_length, self.vision_mask, self.vision_missing_mask = self.generate_m(self.vision, None, self.vision_lengths,
                                                                                            self.args.missing_rate[2], self.args.missing_seed[2], mode='vision')
        if self.args.need_truncated:
            self.__truncated()

        if  self.args.need_normalized:
            self.__normalize()
    
    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()

    def generate_m(self, modality, input_mask, input_len, missing_rate, missing_seed, mode='text'):
        
        if mode == 'text':
            input_len = np.argmin(input_mask, axis=1)
        elif mode == 'audio' or mode == 'vision':
            input_mask = np.array([np.array([1] * length + [0] * (modality.shape[1] - length)) for length in input_len])

        np.random.seed(missing_seed)

        missing_mask = (np.random.uniform(size=input_mask.shape) > missing_rate) * input_mask
        
        assert missing_mask.shape == input_mask.shape
        
        if mode == 'text':
            # CLS SEG Token unchanged.
            for i, instance in enumerate(missing_mask):
                instance[0] = instance[input_len[i] - 1] = 1
            
            modality_m = missing_mask * modality + (100 * np.ones_like(modality)) * (input_mask - missing_mask) # UNK token: 100.
        elif mode == 'audio' or mode == 'vision':
            modality_m = missing_mask.reshape(modality.shape[0], modality.shape[1], 1) * modality
        
        return modality_m, input_len, input_mask, missing_mask


    def __truncated(self):
        # NOTE: Here for dataset we manually cut the input into specific length.
        def Truncated(modal_features, length):
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            padding = np.array([0 for i in range(modal_features.shape[2])])
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    if((instance[index] == padding).all()):
                        if(index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index+20])
                            break
                    else:                        
                        truncated_feature.append(instance[index:index+20])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature
                       
        text_length, audio_length, video_length = self.args.seq_lens
        self.vision = Truncated(self.vision, video_length)
        self.text = Truncated(self.text, text_length)
        self.audio = Truncated(self.audio, audio_length)

    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        # for visual and audio modality, we average across time
        # here the original data has shape (max_len, num_examples, feature_dim)
        # after averaging they become (1, num_examples, feature_dim)
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # remove possible NaN values
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

        if self.args.data_missing and not getattr(self, '_unified_missing_train', False):
            self.vision_m = np.transpose(self.vision_m, (1, 0, 2))
            self.audio_m = np.transpose(self.audio_m, (1, 0, 2))

            self.vision_m = np.mean(self.vision_m, axis=0, keepdims=True)
            self.audio_m = np.mean(self.audio_m, axis=0, keepdims=True)

            # remove possible NaN values
            self.vision_m[self.vision_m != self.vision_m] = 0
            self.audio_m[self.audio_m != self.audio_m] = 0

            self.vision_m = np.transpose(self.vision_m, (1, 0, 2))
            self.audio_m = np.transpose(self.audio_m, (1, 0, 2))

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if self.args.use_bert:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def _getitem_unified(self, index):
        """训练时按样本随机缺失率生成 text_m/audio_m/vision_m，使单次训练见到多种缺失率。
        同一 index 在所有 epoch 使用 (index, seed) 的确定性种子，保证「同一数据缺失要一样」：
        每个 epoch 拿到的数据集（每个样本的缺失率与缺失 mask）一致。"""
        pool = getattr(self.args, 'missing_rates_pool', None)
        if pool is None:
            pool = np.arange(0, 1.05, 0.1).round(2).tolist()
        pool = list(pool)
        run_seed = int(getattr(self.args, 'seed', 2222))
        # 确定性种子：同一样本 (index) 在所有 epoch 得到相同的缺失率与缺失 mask
        base = (int(index) * 2654435761 + run_seed) % (2**31)
        np.random.seed(base)
        r = float(np.random.choice(pool))
        rates = (r, r, r)
        seed_t = np.random.randint(0, 2**31)
        seed_a = np.random.randint(0, 2**31)
        seed_v = np.random.randint(0, 2**31)

        # text: 单样本 (1, L)
        text_m, _, text_mask, text_missing_mask = self.generate_m(
            self.text[index:index+1, 0, :], self.text[index:index+1, 1, :], None,
            rates[0], seed_t, mode='text')
        Input_ids_m = np.expand_dims(text_m, 1)
        Input_mask = np.expand_dims(text_mask, 1)
        Segment_ids = np.expand_dims(self.text[index:index+1, 2, :], 1)
        text_m = np.concatenate((Input_ids_m, Input_mask, Segment_ids), axis=1)[0]

        alen = self.audio_lengths[index:index+1] if hasattr(self.audio_lengths, '__getitem__') else [self.audio_lengths[index]]
        audio_m, _, audio_mask, audio_missing_mask = self.generate_m(
            self.audio[index:index+1], None, np.atleast_1d(alen),
            rates[1], seed_a, mode='audio')
        vlen = self.vision_lengths[index:index+1] if hasattr(self.vision_lengths, '__getitem__') else [self.vision_lengths[index]]
        vision_m, _, vision_mask, vision_missing_mask = self.generate_m(
            self.vision[index:index+1], None, np.atleast_1d(vlen),
            rates[2], seed_v, mode='vision')

        return {
            'text': torch.Tensor(self.text[index]),
            'text_m': torch.Tensor(text_m),
            'text_missing_mask': torch.Tensor(text_missing_mask[0]),
            'audio': torch.Tensor(self.audio[index]),
            'audio_m': torch.Tensor(audio_m[0]),
            'audio_lengths': self.audio_lengths[index],
            'audio_mask': audio_mask[0],
            'audio_missing_mask': audio_missing_mask[0],
            'vision': torch.Tensor(self.vision[index]),
            'vision_m': torch.Tensor(vision_m[0]),
            'vision_lengths': self.vision_lengths[index],
            'vision_mask': vision_mask[0],
            'vision_missing_mask': vision_missing_mask[0],
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        }

    def __getitem__(self, index):
        if getattr(self, '_unified_missing_train', False):
            return self._getitem_unified(index)
        if self.args.data_missing:
            sample = {
                'text': torch.Tensor(self.text[index]), # [batch_size, 3, 50]
                'text_m': torch.Tensor(self.text_m[index]), # [batch_size, 3, 50]
                'text_missing_mask': torch.Tensor(self.text_missing_mask[index]),
                'audio': torch.Tensor(self.audio[index]),
                'audio_m': torch.Tensor(self.audio_m[index]),
                'audio_lengths': self.audio_lengths[index],
                'audio_mask': self.audio_mask[index],
                'audio_missing_mask': self.audio_missing_mask[index],
                'vision': torch.Tensor(self.vision[index]),
                'vision_m': torch.Tensor(self.vision_m[index]),
                'vision_lengths': self.vision_lengths[index],
                'vision_mask': self.vision_mask[index],
                'vision_missing_mask': self.vision_missing_mask[index],
                'index': index,
                'id': self.ids[index],
                'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
            }
        else:
            sample = {
                'raw_text': self.rawText[index],
                'text': torch.Tensor(self.text[index]), 
                'audio': torch.Tensor(self.audio[index]),
                'vision': torch.Tensor(self.vision[index]),
                'index': index,
                'id': self.ids[index],
                'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
            } 
            if not self.args.need_data_aligned:
                sample['audio_lengths'] = self.audio_lengths[index]
                sample['vision_lengths'] = self.vision_lengths[index]
        return sample


def MMDataLoader(args, distributed: bool = False):
    """
    构建多模态数据加载器。
    - 当 distributed=True 时：使用 DistributedSampler 对 train 集做采样（单机多卡 / DDP）。
    - 当 distributed=False 时：退化为普通 DataLoader（与原实现保持一致）。
    """
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args.seq_lens = datasets['train'].get_seq_len()

    if distributed:
        train_sampler = DistributedSampler(datasets['train'], shuffle=True)
        dataLoader = {
            'train': DataLoader(
                datasets['train'],
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                sampler=train_sampler,
            ),
            'valid': DataLoader(
                datasets['valid'],
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
            ),
            'test': DataLoader(
                datasets['test'],
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
            ),
        }
        dataLoader['train_sampler'] = train_sampler
    else:
        dataLoader = {
            ds: DataLoader(
                datasets[ds],
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
            )
            for ds in datasets.keys()
        }

    return dataLoader