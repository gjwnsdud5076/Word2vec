import numpy as np
from functools import reduce
from util import *
from model import *

class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]

        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.matmul(x,W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        self.grads[0][...] = dW
        return dx




class SoftmaxLoss: #softmax with cross entropy loss
    def __init__(self):
        self.params, self.grads =[], []
        self.y = None #출력 vector
        self.t = None #정답 vector

    def forward(self, x, target):
        self.y = softmax(x)
        self.t = target

        lnsoft = []
        for i in range(x):
            lnsoft = reduce(lambda a, b: a+[np.log(b)], self.y, [])
        out = np.sum(lnsoft * target, axis = 1)

        return out

    def backward(self, dout=1):
        cfg = Config()
        return (self.y - self.t)/cfg.batch

class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        return W[idx]

    def backward(self, dout):
        dW, = self.grads
        dW[...]= 0
        np.add.at(dW, self.idx, dout)
        return None


class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.cache = None
        self.params = self.embed.params
        self.grads = self.embed.grads

    def forward(self, h, idx): #여기서 idx는 mini batch때문에 배열임.
        W_idx = self.embed.forward(idx)
        out = np.sum(W_idx*h, axis = 1) #책 167 참조

        self.cache = (W_idx, h)

        return out

    def backward(self, dout):
        W_idx, h = self.cache
        dout = dout.reshape(dout.shape[0],1) #TODO: 이거 역할 생각하기

        dh = dout * W_idx
        dW_idx = dout * h
        self.embed.backward(dW_idx)

        return dh

class SigmoidLoss:
    def __init__(self):
        self.params = []
        self.grams = []
        self.y = None
        self. t = None

    def forward(self, x, t):
        self.y = 1 / 1 + np.exp(-x)
        self.t = t

        lnsigmoid = np.log(self.y)
        out = lnsigmoid*t

        return out

    def backward(self, dout):
        batch_sz = dout.shape[0]

        return (self.y - self.t)/batch_sz


class NegativeSampling:
    def __init__(self, W, corpus, power=0.75, sample_sz=5):
        self.sample_sz = sample_sz
        self.ae_embedding_dot = [EmbeddingDot(W) for _ in range(sample_sz+1)]
        self.ae_sigmoid_loss = [SigmoidLoss() for _ in range(sample_sz+1)]
        self.params, self.grads = [], []
        self.corpus = corpus
        self.power = power

        for layer in self.ae_embedding_dot:
            self.params += layer.params
            self.grads += layer.grads


    def forward(self,h,target):
        batch_sz = h.shape[0]
        negative_sample = negative_sampler(self.corpus, self.power, self.sample_sz, target)

        out = self.ae_embedding_dot[0].forward(h, target)
        correct = np.ones(batch_sz)
        loss = self.ae_sigmoid_loss[0].forward(out, correct)

        for i in range(1,self.sample_sz+1):
            out = self.ae_embedding_dot[i].forward(h, negative_sample[:,i])
            incorrect = np.zeros(batch_sz)
            loss += self.ae_sigmoid_loss[i].forward(out, incorrect)

        return loss

    def backward(self,dout=1):
        dh = 0
        for embed, loss in zip(self.ae_embedding_dot,self.ae_sigmoid_loss):
            back_loss = loss.backward(dout)
            dh += embed.backward(back_loss)

        return dh















"""
class tmp:
    def __init__(self):
        self.W = [1, 2, 3]
        self.W_inst = [self.W for _ in range(3)]

        self.params = []
        for i in range(len(self.W_inst)):
            self.params += self.W_inst[i]

    def update(self):
        self.params += self.params

        return self.params
"""