# coding: utf-8
# @File: train.py
# @Description:

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Prompt_bert.model import BertClassifier
from Prompt_bert.dataset import CNewsDataset
from tqdm import tqdm
from transformers import BertTokenizer
from Prompt_bert.config import Config
import numpy as np


def main():
    # 参数设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained('../rbt3')
    config = Config()
    # 获取到dataset
    train_dataset = CNewsDataset(config.train_data_location)
    valid_dataset = CNewsDataset(config.eval_data_location)
    # test_dataset = CNewsDataset('THUCNews/data/test.txt')

    # 生成Batch
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.eval_batch_size, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = BertClassifier().to(device)

    # 优化器
    # optimizer = AdamW(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    best_acc = 0

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
                label_id=label_id.long().to(device),
            )
            # (batch_size, seq_len, vocab_size)
            logits = output.logits
            # 获取loss
            loss = output.loss
            losses += loss.item()
            mask_position_index = torch.tensor(input_ids) == tokenizer.mask_token_id
            predict_logits = logits[mask_position_index].reshape(logits.size(0), -1, logits.size(-1))
            true_label_id = label_id[mask_position_index].reshape(logits.size(0), -1).cpu().numpy()
            predict_label_id = torch.argmax(predict_logits, dim=-1).detach().cpu().numpy()
            count = 0
            for index, true_label in enumerate(true_label_id):
                if set(true_label) == set(predict_label_id[index]):
                    count += 1
            acc = count / len(predict_label_id)
            accuracy += acc
            loss.backward()
            optimizer.step()
            train_bar.set_postfix(loss=loss.item(), acc=acc)
        average_loss = losses / len(train_dataloader)
        average_acc = accuracy / len(train_dataloader)

        print('\tTrain ACC:', average_acc, '\tLoss:', average_loss)
        #
        # # 验证
        model.eval()
        losses = 0  # 损失
        accuracy = 0  # 准确率
        valid_bar = tqdm(valid_dataloader, ncols=100)
        with torch.no_grad():
            for input_ids, token_type_ids, attention_mask, label_id in valid_bar:
                valid_bar.set_description('Epoch %i valid' % epoch)

                output = model(
                    input_ids=input_ids.long().to(device),
                    attention_mask=attention_mask.long().to(device),
                    token_type_ids=token_type_ids.long().to(device),
                    label_id=label_id.long().to(device),
                )
                # (batch_size, seq_len, vocab_size)
                logits = output.logits
                # 计算loss
                loss = output.loss
                losses += loss.item()
                mask_position_index = torch.tensor(input_ids) == tokenizer.mask_token_id
                predict_logits = logits[mask_position_index].reshape(logits.size(0), -1, logits.size(-1))
                true_label_id = label_id[mask_position_index].reshape(logits.size(0), -1).cpu().numpy()
                predict_label_id = torch.argmax(predict_logits, dim=-1).detach().cpu().numpy()
                count = 0
                for index, true_label in enumerate(true_label_id):
                    if set(true_label) == set(predict_label_id[index]):
                        count += 1
                acc = count / len(predict_label_id)
                accuracy += acc

            average_loss = losses / len(valid_dataloader)
            average_acc = accuracy / len(valid_dataloader)

            print('\tValid ACC:', average_acc, '\tLoss:', average_loss)

            if not os.path.exists('../models'):
                os.makedirs('../models')

            # 判断并保存验证集上表现最好的模型
            if average_acc > best_acc:
                best_acc = average_acc
                torch.save(model.state_dict(), '../models/best_model.pkl')


if __name__ == '__main__':
    # Valid ACC: 0.947 	Loss: 0.1618787732720375
    main()
