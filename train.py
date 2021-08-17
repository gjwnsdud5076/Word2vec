import numpy as np
from model import *
from optimizer import *
from util import *
from backward import *
import time
import matplotlib.pyplot as plt
from evaluation import *
import torch
import pickle

class Trainer:
    def __init__(self,model,optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_lst = []

    def train(self, batch_context, batch_target, max_grad=None):
        total_loss = 0
        loss_count = 0
        model, optimizer = self.model, self.optimizer

        loss = model.forward(batch_context, batch_target)
        model.backward()
        params, grads = remove_duplicate(model.params, model.grads)
        optimizer.update(params, grads)
        total_loss += loss
        loss_count += 1


    def plot(self, ylim=None):
        x = np.arange(cfg.batch)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_lst, label='train')
        plt.xlabel('batch')
        plt.ylabel('loss')
        plt.show()


if __name__ == '__main__':
    cfg = Config()
    path = cfg.train_path

    model = CBOW()
    optimizer = SGD(0.025)
    trainer = Trainer(model, optimizer)

    #context target batch만큼만 만들어서 돌리기
    for _ in range(cfg.max_epoch):
        file = cfg.train_path
        lines = []
        cnt = 0
        corpus = []

        for _ in range(99):
            with open(file,'r',encoding='UTF8')as f:
                lines = f.readlines()

            for line in lines: #한문장씩
                words = line.split()
                for word in words:
                    if word in word_to_id:
                        id = word_to_id[word]
                        corpus.append(id)

                corpus_len = len(corpus)
                context = []
                target = []
                for i in range(cfg.win_sz, corpus_len - cfg.win_sz):
                    con_tar = list(range(i - cfg.win_sz, i + cfg.win_sz + 1))
                    tar_pos = int(len(con_tar)/2)
                    target.append(con_tar[tar_pos])
                    del con_tar[tar_pos]
                    context.append(con_tar)
                    cnt += 1

                    if cnt == cfg.batch:
                        trainer.train(context, target)

                        context = []
                        target = []
                        cnt = 0

        trainer.plot()
        ae_analogy = Word_analogy_test(cfg.eval_show_num, model.word_vecs)
        ae_analogy.eval(cfg.eval_path)

    word_vecs = model.word_vecs











