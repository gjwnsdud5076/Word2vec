import re
import numpy as np
from functools import reduce
import os
from tqdm import tqdm
from collections import Counter
import pickle
import os

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
        if type(contexts) == int:
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

def negative_sampler_uni(p_table, number, uni_target): #batch로 만들기
    size = len(p_table)
    p_table[uni_target] = 0
    p_table /= sum(p_table)

    return list(np.random.choice(np.arange(size), p=p_table, size=number, replace=False))


def negative_sampler(id_to_freq, number, target):
    p_table = make_table(id_to_freq)
    batch_sz = len(target)
    rt = []
    for i in range(batch_sz):

        rt.append(negative_sampler_uni(p_table, number, target[i]))

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

def single_file_corpus(file, word_to_id):
    corpus = []
    with open(file,'r',encoding='UTF8') as f:
        lines = f.readlines()
        for line in lines:
            for word in line.split():
                if word in word_to_id.keys():
                    corpus.append(word_to_id[word])
                else:
                    corpus.append(0)

    return corpus

""" 이거 안씀
def make_corpus(paths, batch, word_to_id):
    corpuses = []
    assert len(paths)%batch == 0
    length = int(len(paths)/batch)
    for i in tqdm(range(batch), desc="make corpus"):
        corpus = []
        for j in range(length*i, length*(i+1)):
            corpus.append(single_file_corpus(paths[j],word_to_id))
        corpuses.append(corpus)
    return corpuses
"""

def make_word_sys(path):
    full_path = []
    word = []
    word_to_id = {}
    id_to_word = {}
    id_to_freq = {}
    for pth, _, files in os.walk(path):
        for file in files:
            full_path.append(pth+ "/"+ file)

    if os.path.isfile("cbow_params.pkl"):
        with open("cbow_params.pkl", 'rb') as f:
            load = pickle.load(f)
        word_to_id = load['word_to_id']
        id_to_word = load['id_to_word']
        id_to_freq = load['id_to_freq']

    else:
        counter = Counter()
        for file in tqdm(full_path, desc="create word matrix"):
            with open(file, 'r', encoding='UTF8') as f:
                lines = f.readlines()
                for line in tqdm(lines, desc="word list"):
                    line = line.lower()
                    word = line.split()
                    counter.update(word)

        count = [('UNK', 0)]

        for word in counter:
            if counter[word] < 5:
                continue
            else:
                tup = (word, counter[word])
                count.append(tup)

        for word, freq in count:
            id_to_word[len(id_to_freq)] = word
            word_to_id[word] = len(id_to_freq)
            id_to_freq[len(id_to_freq)] = freq


        params = {}
        params['word_to_id'] = word_to_id
        params['id_to_word'] = id_to_word
        params['id_to_freq'] = id_to_freq

        with open('cbow_params.pkl', 'wb') as f:
            pickle.dump(params, f, -1)


    return word_to_id, id_to_word, id_to_freq ,full_path

def make_table(id_to_freq):
    size = len(id_to_freq)+1
    total = sum(id_to_freq.values())
    id_to_p = [0 for _ in range(size)]

    for key, value in id_to_freq.items():
        p = float(value/total)
        id_to_p[key] = value
    a = int(sum(id_to_p))
    id_to_p = np.array(id_to_p)
    id_to_p = id_to_p/a
    id_to_p = np.power(id_to_p, 0.75)
    id_to_p /= sum(id_to_p)

    return id_to_p

