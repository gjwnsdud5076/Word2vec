import re
import numpy as np
from functools import reduce

word_to_id = {}
id_to_word = {}
corpus = []
def process(text):
    #text = text.lower()
    text = text.replace('.', ' .')
    text = text.replace('!', ' !')
    text = text.replace('?', ' ?')
    text_list = text.split( )

    word_list = []
    cnt = 0

    for word in text_list:
        if word not in word_list:
            word_list.append(word)
            word_to_id[word] = str(cnt)
            id_to_word[str(cnt)] = word
            cnt += 1

    corpus = np.array([word_to_id[word] for word in text_list])
    return corpus, word_to_id, id_to_word

def create_context_target(corpus, win_sz):
    contexts = []
    target = []
    length = len(corpus)
    for i in range(win_sz, length-win_sz):
        tmp = []
        for t in range(win_sz):
            tmp.append(corpus[i-win_sz+t])
        for t in range(win_sz):
            tmp.append(corpus[i+t+1])
        contexts.append(tmp)
        target.append(corpus[i])

    return contexts, target


def idx_to_onehot(V,idx):
    arr = [0 for i in range(V)]
    arr[int(idx)] = 1
    return arr

def convert_to_onehot(lst,V):
    if type(lst[0]) == list:
        return np.array([convert_to_onehot(lst[i],V) for i in range(len(lst))])
    else:
        return np.array([idx_to_onehot(V,lst[i]) for i in range(len(lst))])


def idx_to_vec(W_in, idx): #이거 굳이 이렇게 안해도 되니까 삭제할거면 삭제해
    return W_in[int(idx)]


def softmax(vec):
    rt_vec = reduce(lambda x,y: x + [np.exp(y)], vec, [])
    softmax_sum = reduce(lambda x,y : x+y, rt_vec, 0)
    rt_vec /= softmax_sum
    return rt_vec

def negative_sampler_uni(corpus, power, number, uni_target): #batch로 만들기
    size = max(corpus) + 1
    p = np.zeros(size)
    for idx in corpus:
        p[idx] += 1

    p[uni_target] = 0
    p /= p.sum()
    p = np.power(p, power)
    p /= p.sum()

    return list(np.random.choice(np.arange(size), p=p, size=number, replace=False))


def negative_sampler(corpus, power, number, target):
    batch_sz = target.shape[0]
    rt = []
    for i in range(batch_sz):
        rt.append(negative_sampler_uni(corpus, power, number, target[i]))

    return rt

def remove_duplicate(params, grads):
    params, grads = params[:], grads[:]
    ck_find = False

    while(1):
        L = len(params)
        for i in range(L-1):
            for j in range(i+1,L):
                if params[i] is params[j]:
                    ck_find = True
                    grads[i] += grads[j]
                    params.pop(j)
                    grads.pop(j)
                if ck_find:
                    break
            if ck_find:
                break

        if not ck_find:
            break

    return params, grads





