# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:51:31 2016

@author: hedi
"""
from collections import Counter
import seaborn as sns
import numpy as np
from fastsent import FastSent

def indexize(sentence, w2i):
    res = []
    for w in sentence:
        try:
            i = w2i[w]
        except:
            continue
        res.append(i)
    return res
    
class SentenceIt(object):
    def __init__(self, path):
        self.path = path
        self.n_data = 0
        
    def __iter__(self):
        for s in open(path,'r'):
            s = u"%s" % s.decode('utf-8')
            self.n_data += 1
            yield s.split()
            
class MinibatchSentenceIt(object):
    def __init__(self, path, batch_size, w2i):
        self.path = path
        self.batch_size = batch_size
        self.w2i = w2i
        self.n_empty = 0
        
    def __iter__(self):
        # we could make it better by:
            # - taking overlapping batches
            # - randomly sampling batches
        minibatch = []
        Ls = []
        for s in open(path,'r'):
            s = u"%s" % s.decode('utf-8')
            s = indexize(s.split(), self.w2i)
            if len(s) == 0:
                self.n_empty += 1
                continue
            minibatch.append(s)
            Ls.append(len(s))
            # Here it's savage, maybe find sth smarter and quicker
            if len(minibatch)==self.batch_size:
                M = max(Ls)
                padded = np.array([np.pad(m, (0,M-l), 'constant') for (m,l) in zip(minibatch, Ls)]).astype('int32')
                minibatch = []
                Ls = []
                yield padded
    
if __name__ == '__main__':
    path = 'bigpage_tokenized.txt'
    vocab = Counter()
    print "build vocab"
    sentences = SentenceIt(path)
    for s in sentences:
        for w in s:
            vocab[w] += 1
    n_data = sentences.n_data
    w2i = {}
    i2w = []
    f = open('vocab','w')
    for k,(w,_) in enumerate([('<pad>', 0)] + vocab.most_common()[:100000]):
        w2i[w] = k
        i2w.append(w)
        f.write("%s\n" % w.encode('utf-8'))
    f.close()
    batch_size = 64
    vocab_size = len(i2w)
    dim = 200
    n_epochs = 50
    saving_path = "chinese.fastsent"
    save_every = 1000
    
    batches = MinibatchSentenceIt(path, batch_size, w2i)
    print "create model"
    model = FastSent.create(vocab_size, dim)
    model.train(batches, 
                lr=0.025, 
                min_lr=0.0001, 
                n_epochs=n_epochs, 
                saving_path=saving_path, 
                save_every=save_every, 
                verbose=True)