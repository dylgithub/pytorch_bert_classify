# -*- coding: utf-8 -*-
import torch
from transformers import BertTokenizerFast
import numpy as np

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 返回cuda表示成功
# 或者
# print(torch.cuda.is_available())
# text = ["测试数据","测试数据2"]
# tokenizer = BertTokenizerFast.from_pretrained("rbt3")
# token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=30)["offset_mapping"]
# print(token2char_span_mapping)
data = [[[1, 0, 0], [1, 0, 0]],
        [[1, 1, 0], [1, 1, 0]]]
data_array = np.array(data)
# print(data_array.shape)
for ent_type_id, token_start_index, token_end_index in zip(*np.where(data_array > 0)):
    print(ent_type_id, token_start_index, token_end_index)