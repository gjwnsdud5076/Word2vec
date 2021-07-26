import numpy as np
from backward import *
from util import *

c0 = np.array([1,0,0])
c1 = np.array([0,0,1])

W0 = np.random.randn(3,7)
W0 = MatMul(W0)
W1 = np.random.randn(3,7)
W1 = MatMul(W1)

hidden_layer = 0.5*(W0.forward(c0.T)+W1.forward(c1.T))

W_out = np.random.rand(7,3)
W_out = MatMul(W_out)

output_layer = W_out.forward(hidden_layer)

print(output_layer)


def remove_duplicate(params, grads):
    '''
    매개변수 배열 중 중복되는 가중치를 하나로 모아
    그 가중치에 대응하는 기울기를 더한다.
    '''
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False #역할? 겹치는게 없을때까지 돌려
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 가중치 공유 시
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 경사를 더함
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 가중치를 전치행렬로 공유하는 경우(weight tying)
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads
