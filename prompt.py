# -*- coding: utf-8 -*-
import json
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments, BertTokenizer, BertForMaskedLM


def compute_metrics(pred):
    labels = pred.label_ids[:, 8]
    preds = pred.predictions[:, 8].argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


class Prompt:
    def build_dataset(self, texts, labels, tokenizer, max_len):
        data_dict = {'text': texts, 'label': labels}
        data_dataset = Dataset.from_dict(data_dict)

        def preprocess_function(examples):
            text_token = tokenizer(examples['text'], padding=True, truncation=True, max_length=max_len)
            text_token['labels'] = np.array(
                tokenizer(examples['label'], padding=True, truncation=True, max_length=max_len)[
                    "input_ids"])
            return text_token

        data_dataset = data_dataset.map(preprocess_function, batched=True)
        return data_dataset

    def data_process(self, input_file):
        texts = []
        labels = []
        label_list = ['房产', '财经', '教育', '科技', '时政', '体育', '游戏', '娱乐']
        with open(input_file, 'r', encoding='utf-8') as inp:
            for line in inp:
                data, label = line.strip().split('\t')
                text = '这是一篇关于[MASK][MASK]的新闻，' + data
                label = '这是一篇关于' + label_list[int(label)] + '的新闻，' + data
                texts.append(text)
                labels.append(label)
        return texts, labels

    def build_model(self, model_name="chinese-roberta-wwm-ext"):
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForMaskedLM.from_pretrained(model_name)
        return tokenizer, model

    def build_trainer(self, model, train_dataset, test_dataset, checkpoint_dir, learning_rate, batch_size, epochs_num):
        args = TrainingArguments(
            checkpoint_dir,
            evaluation_strategy="steps",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs_num,
            # weight_decay=decay,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
        )

        trainer = Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )
        return trainer


if __name__ == '__main__':
    prompt = Prompt()
    tokenizer, model = prompt.build_model('rbt3')
    train_texts, train_labels = prompt.data_process("THUCNews/data/dev.txt")
    test_texts, test_labels = prompt.data_process("THUCNews/data/test.txt")
    train_dataset = prompt.build_dataset(train_texts, train_labels, tokenizer, 40)
    test_dataset = prompt.build_dataset(test_texts, test_labels, tokenizer, 40)
    trainer = prompt.build_trainer(model, train_dataset, test_dataset, "prompt_checkpoint", 1e-5, 64, 3)
    trainer.train()
