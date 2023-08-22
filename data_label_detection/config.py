# -*- coding: utf-8 -*-
class Config():
    def __init__(self):
        self.all_data_location = "../THUCNews/data/dev.txt"
        self.num_labels = 8
        self.train_batch_size = 64
        self.eval_batch_size = 64
        self.epochs = 1
        self.learning_rate = 5e-6  # Learning Rate不宜太大
