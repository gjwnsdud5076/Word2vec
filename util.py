import re
import numpy as np
from functools import reduce
import os
from tqdm import tqdm
from collections import Counter

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

def create_context_target(corpus, win_sz): #batch 고려
    corpus = np.array(corpus)
    contexts = 0
    length = len(corpus[0])
    target = corpus[:,win_sz:-win_sz]
    for t in range(win_sz,length-win_sz):
        idx = [i for i in range(t-win_sz, t+win_sz+1)]
        del idx[win_sz]
        if not contexts:
            contexts = [corpus[:,idx]]
        else:
            contexts = np.array(contexts+[corpus[:,idx]])
    return contexts, target.transpose()


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

def single_file_corpus(file):
    corpus = []
    with open(file,'r') as f:
        lines = f.readlines()
        for line in lines:
            for word in line.split():
                corpus.append(word_to_id[word])

    return corpus

def make_corpus(paths, batch = 9):
    corpuses = []
    for i in tqdm(range(batch), desc = "make corpus"):
        corpus = []
        for j in range(batch*i, batch*(i+1)):
            corpus += single_file_corpus(paths[j])

        corpuses.append(corpus)


def make_word_sys(path, batch = 9):
    full_path = []
    word = []
    word_to_id = {}
    id_to_word = {}
    id_to_freq = {}
    for pth, _, files in os.walk(path):
        for file in files:
            full_path.append(pth+ "/"+ file)

    counter = Counter()
    for file in tqdm(full_path, desc="create word matrix"):
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                word += line.split()

        counter.update(word)

        most_n = 700*10**3
        count = [('UNK', 0 )]
        count.extend(Counter.most_common(most_n))

        for word, freq in count:
            id_to_word[len(id_to_freq)] = word
            word_to_id[word] = len(id_to_freq)
            id_to_freq[len(id_to_freq)] = freq

        corpus = make_corpus(full_path)

    return corpus, word_to_id, id_to_word, id_to_freq

def to_cpu(x):
    import numpy
    if type(x) == numpy.ndarray:
        return x
    return np.asnumpy(x)


def to_gpu(x):
    import cupy #TODO: gpu 돌아가게 하기
    if type(x) == cupy.ndarray:
        return x
    return cupy.asarray(x)