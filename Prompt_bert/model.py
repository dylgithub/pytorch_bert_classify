# coding: utf-8
# @File: model.py
# @Description:


import torch.nn as nn
from transformers import BertForMaskedLM


# Bert
class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义BERT模型
        self.bert = BertForMaskedLM.from_pretrained("../rbt3")

    def forward(self, input_ids, attention_mask, token_type_ids, label_id):
        # BERT的输出
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                labels=label_id)
        return bert_output
