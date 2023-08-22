# -*- coding: utf-8 -*-
# coding: utf-8
# @File: train.py
# @Description:

import os
import torch
import torch.nn as nn
import numpy as np
from transformers import BertConfig
from torch.utils.data import DataLoader
from model import BertClassifier
from dataset import CNewsDataset
from tqdm import tqdm
from config import Config
import cleanlab
from sklearn.model_selection import KFold

kfold = KFold(n_splits=2)


def predict(num_labels, valid_dataloader, device):
    bert_config = BertConfig.from_pretrained('../rbt3')

    # 定义模型
    model = BertClassifier(bert_config, num_labels)

    # 加载训练好的模型
    model.load_state_dict(torch.load('../models/best_model.pkl'))
    model = model.to(device)
    model.eval()
    eval_predict_list = []
    eval_predict_label = []
    with torch.no_grad():
        valid_bar = tqdm(valid_dataloader, ncols=100)
        for input_ids, token_type_ids, attention_mask, label_id in valid_bar:
            label_id = list(label_id.detach().cpu().numpy())
            eval_predict_label.extend(label_id)
            output = model(
                input_ids=input_ids.long().to(device),
                attention_mask=attention_mask.long().to(device),
                token_type_ids=token_type_ids.long().to(device),
            )
            output = torch.softmax(output, dim=-1)
            for _list in list(output.detach().cpu().numpy()):
                eval_predict_list.append(_list)
    return eval_predict_list, eval_predict_label


def main():
    # 参数设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = Config()
    # 获取到dataset
    all_dataset = CNewsDataset(config.all_data_location)

    # 读取BERT的配置文件
    bert_config = BertConfig.from_pretrained('../rbt3')

    # 初始化模型
    model = BertClassifier(bert_config, config.num_labels).to(device)

    # 优化器
    # optimizer = AdamW(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    pred_probs = []
    labels = []
    for fold_i, (train_ids, val_ids) in enumerate(kfold.split(all_dataset)):
        train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_ids)
        train_dataloader = torch.utils.data.DataLoader(all_dataset, batch_size=config.train_batch_size,
                                                       sampler=train_sampler)
        valid_dataloader = torch.utils.data.DataLoader(all_dataset, batch_size=config.eval_batch_size,
                                                       sampler=val_sampler)
        for epoch in range(1, config.epochs + 1):
            losses = 0  # 损失
            accuracy = 0  # 准确率
            model.train()
            train_bar = tqdm(train_dataloader, ncols=100)
            for input_ids, token_type_ids, attention_mask, label_id in train_bar:
                # 梯度清零
                model.zero_grad()
                train_bar.set_description('Epoch %i train' % epoch)

                # 传入数据，调用model.forward()
                # 注意这里的数据类型转换
                output = model(
                    input_ids=input_ids.long().to(device),
                    attention_mask=attention_mask.long().to(device),
                    token_type_ids=token_type_ids.long().to(device),
                )

                # 计算loss
                loss = criterion(output, label_id.to(device))
                losses += loss.item()

                pred_labels = torch.argmax(output, dim=1)  # 预测出的label
                acc = torch.sum(pred_labels == label_id.to(device)).item() / len(pred_labels)  # acc
                accuracy += acc

                loss.backward()
                optimizer.step()
                train_bar.set_postfix(loss=loss.item(), acc=acc)

            average_loss = losses / len(train_dataloader)
            average_acc = accuracy / len(train_dataloader)

            print('\tTrain ACC:', average_acc, '\tLoss:', average_loss)

            # 验证
            model.eval()
            losses = 0  # 损失
            accuracy = 0  # 准确率
            valid_bar = tqdm(valid_dataloader, ncols=100)
            for input_ids, token_type_ids, attention_mask, label_id in valid_bar:
                valid_bar.set_description('Epoch %i valid' % epoch)

                output = model(
                    input_ids=input_ids.long().to(device),
                    attention_mask=attention_mask.long().to(device),
                    token_type_ids=token_type_ids.long().to(device),
                )

                loss = criterion(output, label_id.to(device))
                losses += loss.item()

                pred_labels = torch.argmax(output, dim=1)  # 预测出的label
                acc = torch.sum(pred_labels == label_id.to(device)).item() / len(pred_labels)  # acc
                accuracy += acc
                valid_bar.set_postfix(loss=loss.item(), acc=acc)

            average_loss = losses / len(valid_dataloader)
            average_acc = accuracy / len(valid_dataloader)

            print('\tValid ACC:', average_acc, '\tLoss:', average_loss)

            if not os.path.exists('../models'):
                os.makedirs('../models')

            # 判断并保存验证集上表现最好的模型
            if average_acc > best_acc:
                best_acc = average_acc
                torch.save(model.state_dict(), '../models/best_model.pkl')
        # 加载最优模型获得预测值
        eval_predict_list, eval_predict_label = predict(config.num_labels, valid_dataloader, device)
        pred_probs.extend(eval_predict_list)
        labels.extend(eval_predict_label)
    pred_probs = np.array(pred_probs)
    res = cleanlab.filter.find_label_issues(labels=labels, pred_probs=pred_probs, n_jobs=1,
                                            filter_by="prune_by_noise_rate")
    print(res)

if __name__ == '__main__':
    # Valid ACC: 0.947 	Loss: 0.1618787732720375
    main()
