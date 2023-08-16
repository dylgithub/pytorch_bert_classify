# coding: utf-8
# @File: dataset.py
# @Description:

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertTokenizer
from tqdm import tqdm


class CNewsDataset(Dataset):
    def __init__(self, filename):
        # 数据集初始化
        self.tokenizer = BertTokenizer.from_pretrained('../rbt3')
        self.input_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.label_id = []
        self.load_data(filename)

    def load_data(self, filename):
        # 加载数据
        print('loading data from:', filename)
        label_list = ['房产', '财经', '教育', '科技', '时政', '体育', '游戏', '娱乐']
        with open(filename, 'r', encoding='utf-8') as rf:
            lines = rf.readlines()
        for line in tqdm(lines, ncols=100):
            text, label = line.strip().split('\t')
            text = '这是一篇关于[MASK][MASK]的新闻，' + text
            label = '这是一篇关于' + label_list[int(label)] + '的新闻，' + text
            token = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=30)
            label_input_ids = self.tokenizer(label, add_special_tokens=True, padding='max_length', truncation=True, max_length=30)
            self.input_ids.append(np.array(token['input_ids']))
            self.token_type_ids.append(np.array(token['token_type_ids']))
            self.attention_mask.append(np.array(token['attention_mask']))
            self.label_id.append(np.array(label_input_ids['input_ids']))

    def __getitem__(self, index):
        return self.input_ids[index], self.token_type_ids[index], self.attention_mask[index], self.label_id[index]

    def __len__(self):
        return len(self.input_ids)


if __name__ == '__main__':
    cd = CNewsDataset("../THUCNews/data/train.txt")
