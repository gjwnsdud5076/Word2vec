import numpy as np
from backward import *
from util import *

class CBOW:
    def __init__(self):
        cfg = Config()

        self.D = cfg.D #D is projection layer size
        self.win_sz = cfg.win_sz

        self.corpus, self.word_to_id, self.id_to_word, self.id_to_freq = make_word_sys(cfg.path, batch=cfg.batch)
        self.context, self.target = create_context_target(self.corpus, self.win_sz)

        self.V = len(self.word_to_id)
        cfg.V = self.V

        self.W_in = 0.01 * np.random.randn(self.V,self.D).astype('f')
        self.W_out = 0.01 * np.random.randn(self.V,self.D).astype('f') #이부분 VD 맞는지..

        self.ae_W_in = []

        for i in range(2*self.win_sz):
            self.ae_W_in.append(Embedding(self.W_in))

        self.ae_W_out = NegativeSampling(self.W_out,self.corpus,power=0.75,sample_sz=5)
        #self.sftloss = SoftmaxLoss()

        layers = []
        for i in range(2*self.win_sz):
            layers.append(self.ae_W_in[i])
        layers.append(self.ae_W_out)

        self.params, self.grads = [], []

        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = self.W_in


    def forward(self,context,target):
        proj_out = 0
        for i in range(2*self.win_sz):
            proj_out += self.ae_W_in[i].forward(context[i])
        proj_out /= 2*self.win_sz
        out = self.ae_W_out.forward(proj_out)

        return out

    def backward(self, dout=1):
        do = self.ae_W_out.backward(dout)
        do *= 1/(2*self.win_sz)
        for layer in self.ae_W_in:
            layer.backward(do)
        return None

class Config:
    def __init__(self):
        self.V= 7
        self.D = 3
        self.win_sz = 3
        self.max_epoch = 10
        self.batch = 9
        self.eval_iter = 1
        self.path = "1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled"
        self.eval_path = ""
        self.eval_show_num = 3

if __name__ == '__main__':
    cbow = CBOW()
    print(cbow.target)
    print(cbow.context)
    print(np.array(cbow.context).shape)

