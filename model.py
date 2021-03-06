import numpy as np
from backward import *
from util import *

class CBOW:
    def __init__(self):
        cfg = Config()

        self.D = cfg.D #D is projection layer size
        self.win_sz = cfg.win_sz
        self.batch_sz = cfg.batch

        self.word_to_id, self.id_to_word, self.id_to_freq, self.full_path = make_word_sys(cfg.train_path)

        self.V = len(self.word_to_id)

        self.W_in = 0.01 * np.random.randn(self.V, self.D).astype('f')
        self.W_out = 0.01 * np.random.randn(self.V, self.D).astype('f')

        self.ae_W_in = []

        for i in range(2*self.win_sz):
            self.ae_W_in.append(Embedding(self.W_in))
        self.ae_W_out = NegativeSampling(self.W_out, self.id_to_freq, power=0.75, sample_sz=5)

        layers = []
        for i in range(2*self.win_sz):
            layers.append(self.ae_W_in[i])
        layers.append(self.ae_W_out)

        self.params, self.grads = [], []

        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = self.W_in


    def forward(self, batch_context, batch_target):
        batch_sz = self.batch_sz
        proj_out = 0
        for i in range(2*self.win_sz):
            proj_out += self.ae_W_in[i].forward(batch_context[i])
        proj_out /= 2*self.win_sz
        out = self.ae_W_out.forward(proj_out, batch_target)
        out /= batch_sz
        return out

    def backward(self, dout=1):
        dout /= self.batch_sz
        do = self.ae_W_out.backward(dout)
        do *= 1/(2*self.win_sz)
        for layer in self.ae_W_in:
            layer.backward(do)
        return None

class Config:
    def __init__(self):
        self.D = 3
        self.win_sz = 5
        self.max_epoch = 5
        self.batch = 100
        self.eval_interval = 1
        self.train_path = "1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled"
        self.eval_path = "data/questions-words.txt"
        self.eval_show_num = 3
        self.width = 0

