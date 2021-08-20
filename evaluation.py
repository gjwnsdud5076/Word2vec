#word analogy test
import numpy as np
from util import *
from model import *

class Word_analogy_test:
    def __init__(self, show_num, W_in, word_to_id): #, vec1, vec2, vec1_2):
        self.show_num = show_num
        self.idx_cossim_rank = {}
        self.W_in = W_in
        self.V = len(W_in)
        self.word_to_id = word_to_id

    def cos_simir(self, vec1, vec2):
        a = np.sqrt(np.inner(vec1,vec1))
        b = np.sqrt(np.inner(vec2,vec2))
        c = np.inner(vec1,vec2)

        return c/(a*b)

    def find_nearest(self,vec):
        for i in range(self.V):
            cos_sim = self.cos_simir(idx_to_vec(self.W_in,i), vec)
            if i < self.show_num:
                self.idx_cossim_rank[i] = cos_sim

            else:
                largest = max(self.idx_cossim_rank, key = self.idx_cossim_rank.get)
                if cos_sim < self.idx_cossim_rank[largest]:
                    self.idx_cossim_rank.pop(largest)
                    self.idx_cossim_rank[i] = cos_sim

        sorted_idx = sorted(self.idx_cossim_rank, key = self.idx_cossim_rank.get)
        ans = []
        for i in range(self.show_num):
            print(' ',id_to_word[str(sorted_idx[i])])
            ans.append(id_to_word[str(sorted_idx[i])])

        return ans

    def eval(self, file):
        corr_semantic = 0
        incor_semantic = 0
        corr_gram = 0
        incor_gram = 0
        typ = None
        vecs =[]
        with open(file, 'r', encoding='UTF8') as f:
            lst = f.readlines()
        conti = 1
        for line in lst:
            if line[:6] == ': gram':
                typ = 'gram'
            elif (line[0] == ':') & (line[:6] != ': gram'):
                typ = 'semantic'
            else:
                words = line.split()
                assert(len(words) == 4)
                for word in words:
                    if word not in word_to_id.keys():
                        conti = -1
                        continue
                    vecs.append(idx_to_vec(self.W_in, self.word_to_id[word]))
                if conti == -1:
                    conti = 1
                    continue
                vec = vecs[1] - vecs[0] + vecs[2]
                nearest = self.find_nearest(vec)
                if words[3] in nearest:
                    if typ == 'semantic':
                        corr_semantic += 1

                    elif typ == 'gram':
                        corr_gram += 1
                else:
                    if typ == 'semantic':
                        incor_semantic += 1
                    else:
                        incor_gram += 1
        try:
            print(f"semantic acc: {corr_semantic/(corr_semantic+incor_semantic)}, grammer acc: {corr_gram/(corr_gram+incor_gram)} \n")
        except:
            print(f"semantic correct: {corr_semantic}, incorrect: {incor_semantic}, grammer correct: {corr_gram}, incorrect: {incor_gram} \n")













