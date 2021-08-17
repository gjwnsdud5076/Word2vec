import numpy as np
from backward import *
from util import *
from model import *

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

if __name__ == '__main__':
    with open("cbow_params.pkl", 'rb') as f:
        load = pickle.load(f)
    word_to_id = load['word_to_id']
    id_to_word = load['id_to_word']
    id_to_freq = load['id_to_preq']  # f ㅎㅎ

    print(word_to_id['the'])



def create_context_target():
    cfg = Config()
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


                    con_tars = []
                    cnt = 0




