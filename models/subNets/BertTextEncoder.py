import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel
from config.get_data_root import data_root

__all__ = ['BertTextEncoder']

class BertTextEncoder(nn.Module):
    def __init__(self, language='en', use_finetune=False):
        """
        language: en / cn
        """
        super(BertTextEncoder, self).__init__()

        assert language in ['en', 'cn']

        tokenizer_class = BertTokenizer
        model_class = BertModel
        # NOTE: change the dataset_path to your own path!
        dataset_path = data_root
        if language == 'en':
            self.tokenizer = tokenizer_class.from_pretrained(os.path.join(dataset_path, 'pretrained_berts/bert_en'), do_lower_case=True)
            self.model = model_class.from_pretrained(os.path.join(dataset_path, 'pretrained_berts/bert_en'),)
        elif language == 'cn':
            self.tokenizer = tokenizer_class.from_pretrained(os.path.join(dataset_path, 'pretrained_berts/bert_cn'),)
            self.model = model_class.from_pretrained(os.path.join(dataset_path, 'pretrained_berts/bert_cn'),)

        
        self.use_finetune = use_finetune
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def from_text(self, text):
        """
        text: raw data
        """
        input_ids = self.get_id(text)
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
        return last_hidden_states.squeeze()
    
    def forward_embeddings_only(self, text):
        """
        仅返回 BERT 的 embedding 层输出（未经过 Transformer 层），用于在「BERT 补全前」估计不确定性。
        text: (batch_size, 3, seq_len) — input_ids, input_mask, segment_ids
        returns: (batch_size, seq_len, hidden_size)
        """
        input_ids = text[:, 0, :].long()
        segment_ids = text[:, 2, :].long()
        # BertEmbeddings(input_ids, token_type_ids=token_type_ids)；不经过 encoder
        return self.model.embeddings(input_ids=input_ids, token_type_ids=segment_ids)

    def forward(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:,0,:].long(), text[:,1,:].float(), text[:,2,:].long()
        if self.use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        return last_hidden_states
    
if __name__ == "__main__":
    bert_normal = BertTextEncoder()
