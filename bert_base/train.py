# coding: utf-8
# @File: train.py
# @Description:

import os
import torch
import torch.nn as nn
from transformers import BertConfig, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from bert_base.model import BertClassifier
from bert_base.dataset import CNewsDataset
from tqdm import tqdm
from bert_base.config import Config


def main():
    # 参数设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = Config()
    # 获取到dataset
    train_dataset = CNewsDataset(config.train_data_location)
    valid_dataset = CNewsDataset(config.eval_data_location)
    # test_dataset = CNewsDataset('THUCNews/data/test.txt')

    # 生成Batch
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.eval_batch_size, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    total_steps = len(train_dataloader) * config.epochs

    print("总训练步数为:{}".format(total_steps))
    # 读取BERT的配置文件
    bert_config = BertConfig.from_pretrained('../rbt3')

    # 初始化模型
    model = BertClassifier(bert_config, config.num_labels).to(device)
    param_optimizer = list(model.named_parameters())
    # 加这个效果会下降
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # 优化器
    # optimizer = AdamW(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=config.learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(config.warmup_proportion * total_steps),
    #                                             num_training_steps=total_steps)
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
            )

            # 计算loss
            loss = criterion(output, label_id.to(device))
            losses += loss.item()

            pred_labels = torch.argmax(output, dim=1)  # 预测出的label
            acc = torch.sum(pred_labels == label_id.to(device)).item() / len(pred_labels)  # acc
            accuracy += acc

            loss.backward()
            optimizer.step()
            # scheduler.step()
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


if __name__ == '__main__':
    # Valid ACC: 0.947 	Loss: 0.1618787732720375
    main()
