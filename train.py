import numpy as np
from model import *
from optimizer import *
from util import *
from backward import *
import time
import matplotlib.pyplot as plt
from evaluation import *

class Trainer:
    def __init__(self,model,optimizer):
        self.model = model
        self.optimizer = optimizer
        self.plot_x = []
        self.plot_y = []

    def train(self, x, t, max_grad=None):
        start_time = time.time()
        cfg = Config()
        data_sz = len(x)
        max_iter = data_sz // cfg.batch
        mid_loss = 0

        for epoch in range(cfg.max_epoch):
            idx = np.random.permutation(len(x))
            x = x[idx]
            t = t[idx] #섞기

            for iter in range(max_iter): #iter이랑 step이랑 같은말? o
                batch_x = x[iter * cfg.batch:(iter + 1) * cfg.batch]
                batch_t = t[iter * cfg.batch:(iter + 1) * cfg.batch]

                loss = self.model.forward(batch_x, batch_t)
                self.model.backward(1)
                params, grads = remove_duplicate(self.model.param, self.model.grad)
                self.optimizer.update(params, grads)
                mid_loss += loss

                if iter % cfg.eval_iter == 0 :
                    interval = time.time()-start_time
                    print(f' epoch: {epoch}, iter: {iter}, time: {interval}, loss:{mid_loss/cfg.eval_iter}, process: {(epoch*max_iter+iter)/(max_iter*cfg.max_epoch)} \n')
                    mid_loss = 0
                    self.plot_x.append(epoch*max_iter+iter)
                    self.plot_y.append(mid_loss/cfg.eval_iter)
            print(model.W_in)

        np.save("save/W_in", model.W_in)


    def plot(self):
        plt.plot(self.plot_x, self.plot_y)
        plt.show()



if __name__ == '__main__':
    cfg = Config()
    corpus, word_to_id, id_to_word = process(cfg.sentence)

    model = Simple_CBOW()
    optimizer = SGD(0.025)
    trainer = Trainer(model, optimizer)

    context, target = create_context_target(corpus,1)
    context = convert_to_onehot(context, cfg.V)
    target = convert_to_onehot(target, cfg.V)

    trainer.train(context, target)
    #trainer.plot()

    ### evaluation ###
    ae_analogy = Word_analogy_test(3, model.W_in)
    ae_analogy.eval('data/working_test.txt')




