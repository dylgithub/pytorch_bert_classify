# -*- coding: utf-8 -*-
class Config():
    def __init__(self):
        self.train_data_location = "../THUCNews/data/train.txt"
        self.eval_data_location = "../THUCNews/data/dev.txt"
        self.num_labels = 8
        self.train_batch_size = 64
        self.eval_batch_size = 64
        self.epochs = 10
        self.learning_rate = 5e-6  # Learning Rate不宜太大
