import numpy as np
from backward import *
from util import *

class Simple_CBOW:
    def __init__(self):
        cfg = Config()
        #V = cfg.V
        self.D = cfg.D #D is projection layer size
        self.win_sz = cfg.win_sz

        #나중에 파일처리로 input 받을 부분
        sentence = cfg.sentence
        corpus, word_to_id, id_to_word = process(sentence)
        self.contexts, self.target = create_context_target(corpus, self.win_sz)

        self.V = len(word_to_id)
        cfg.V = self.V


        self.W_in = 0.01 * np.random.randn(self.V,self.D).astype('f')
        W_out = 0.01 * np.random.randn(self.D,self.V).astype('f')

        self.ae_W_in = []

        for i in range(2*self.win_sz):
            self.ae_W_in.append(MatMul(self.W_in))

        self.ae_W_out = MatMul(W_out)
        self.sftloss = SoftmaxLoss()

        layers = []
        for i in range(2*self.win_sz):
            layers.append(self.ae_W_in[i])
        layers.append(self.ae_W_out)

        self.params, self.grads = [], []
        #self.target = []

        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = self.W_in


    def forward(self,context,target):
        proj_out = np.zeros()
        for i in range(2*self.win_sz):
            proj_out += self.ae_W_in[i].forward(context[:,i])
        proj_out /= 2*self.win_sz
        out = self.ae_W_out.forward(proj_out)
        loss = self.sftloss.forward(out, target)

        return loss

    def backward(self, dout):
        dl = self.sftloss.backward(dout)
        dp = self.ae_W_out.backward(dl)
        dp *= 1/2
        for i in range(2*self.win_sz):
            self.ae_W_in[i].backward(dp)

        return None


class Config:
    def __init__(self):
        self.sentence = 'hi, I love dog and I love cat.'
        self.V= 7
        self.D = 3
        self.win_sz = 3
        self.max_epoch = 10
        self.batch = 3
        self.eval_iter = 1

if __name__ == '__main__':
    cbow = Simple_CBOW()
    print(cbow.target)
    print(cbow.contexts)
    print(np.array(cbow.contexts).shape)

