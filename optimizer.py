import numpy as np

class Adagrad:
    def __init__(self, lr = 0.01):
        self.lr = lr
        self. h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = []
            for param in params:
                self.h.append(np.zeros_like(param))

        for i in range(len(params)):
            self.h[i] += grads[i]*grads[i]
            params[i] -= self.lr*grads[i]/np.sqrt(self.h[i]+1e-7) #여기에 1e - 7 은 0으로 나누지 않도록 하려고

class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr*grads[i]
