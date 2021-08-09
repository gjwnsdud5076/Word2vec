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

    def train(self, x, t, max_grad=None):
        start_time = time.time()
        cfg = Config()

        for epoch in range(cfg.max_epoch):
        #don't need shuffle. already suffled data.
            for batch in range(cfg.batch):
                loss = self.model.forward(x[:,batch], t[:,batch])
                self.model.backward(1)
                params, grads = remove_duplicate(self.model.param, self.model.grad)
                self.optimizer.update(params, grads)

                elapsed_time = time.time() - start_time
                print('| epoch: %d, batch: %d, time: %d[s], loss: %.2f \n' %(epoch+1, batch+1, elapsed_time, loss))
                self.loss_lst.append(float(loss))

    def plot(self, ylim =None):
        x = np.arange(cfg.batch)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_lst, label='train')
        plt.xlabel('batch')
        plt.ylabel('loss')
        plt.show()


if __name__ == '__main__':
    cfg = Config()
    path = cfg.path

    model = CBOW()
    optimizer = SGD(0.025)
    trainer = Trainer(model, optimizer)

    context, target = model.context, model.target

    trainer.train(context, target)
    trainer.plot()

    word_vecs = model.word_vecs

    ### evaluation ###
    ae_analogy = Word_analogy_test(cfg.eval_show_num, word_vecs)
    ae_analogy.eval(cfg.eval_path)










