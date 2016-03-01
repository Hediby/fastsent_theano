# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:51:31 2016

@author: hedi
"""
from collections import Counter
#import seaborn as sns
import numpy as np
from fastsent import FastSent
import utils
import sys
sys.path.insert(0, '/home/arame/hakken-api/models/')
import model

def indexize(sentence, w2i):
    res = []
    if sentence[0]=="#":
        try:
            return [w2i[sentence]]
        except:
            return []
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
            s = u"%s" % s.decode('utf-8')[:-1]
            if len(s)>0:
                self.n_data += 1
                yield s
            
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
            s = indexize(s, self.w2i)
            if len(s) == 0:
                self.n_empty += 1
                continue
            if s[0]=="#":
                l=1
            else:
                l=len(s)
            minibatch.append(s)
            
            Ls.append(l)
            # Here it's savage, maybe find sth smarter and quicker
            if len(minibatch)==self.batch_size:
                M = max(Ls)
                padded = np.array([np.pad(m, (0,M-l), 'constant') for (m,l) in zip(minibatch, Ls)], dtype='int32')
                minibatch = []
                Ls = []
                yield padded
    
if __name__ == '__main__':
    remote=True
    path = '/media/data/datasets/wikipedia/entities/bigpage_zh.txt_line_processed_extract' if remote else "dataset.txt"
    vocab = Counter()
    print "build vocab"
    Ls = []
    sentences = SentenceIt(path)
    for s in sentences:
        if s[0]=="#":
            Ls.append(1)
            vocab[s]+=1
        else:
            Ls.append(len(s))
            for w in s:
                vocab[w] += 1
    #sns.distplot(Ls)
    n_data = sentences.n_data
    w2i = {}
    i2cf = []
    i2w = []
    f = open('vocab','w')
    mc =  vocab.most_common()[:150000]
    cumFreq=0
    

    for k,(w,c) in enumerate([('<pad>', 0)] + mc):
        cumFreq+=int(c)
        w2i[w] = k
        i2cf.append(cumFreq)
        i2w.append(w)

        f.write("%s %f\n" % (w.encode('utf-8'),int(c))) 
    i2f=[cf/float(cumFreq) for cf in i2cf]
    f.close()

    words=w2i.keys()
    print words

    print i2f[:10]
    batch_size = 200
    vocab_size = len(i2w)
    n_epochs = 50000
    saving_path = "/media/data/datasets/models/word2vec_model/chinese/first" if remote else "chineseModel"
    save_every = 1000

    print w2i
    batches = MinibatchSentenceIt(path, batch_size, w2i)
    print "begin"
    for l in batches:
        print l

    i2e={}
    index_fixe=[]
    pretrainedFile="/media/data/datasets/models/word2vec_model/model_bridge/model_zh_ws5_pt_ne5_sa0.0001_mc40.vec"

    pretrained=model.model(pretrainedFile)
    for word,oldi in pretrained.words.items(): 
        if word in words:
            i=w2i[word]
            index_fixe.append(i)
            i2e[i]=pretrained.floats[oldi]
    print words[0:10]            
    print "create model"

    model = FastSent.createNeg(vocab_size, dim,i2f=i2f,index_fixe=index_fixe,i2e=i2e)
    model.train(batches, 
                lr=0.025, 
                min_lr=0.0001, 
                n_epochs=n_epochs, 
                saving_path=saving_path, 
                save_every=save_every, 
                verbose=True)
