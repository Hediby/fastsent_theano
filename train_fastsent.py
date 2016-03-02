# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:51:31 2016

@author: hedi & arame
"""
from collections import Counter
#import seaborn as sns
import numpy as np
from fastsent import FastSent,FastSentNeg
import sys

def indexize(s, w2i,sep=""):
    res = []
    if len(s)==0:
        return res
    if s[0]=="#":
        try:
            return [w2i[s]]
        except:
            return res
    if sep==" ":
        s=s.split(sep)
    for w in s:
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
    def __init__(self, path, batch_size, w2i,sep):
        self.path = path
        self.batch_size = batch_size
        self.w2i = w2i
        self.n_empty = 0
        self.i=0
        self.sep=sep
        
    def __iter__(self):
        # we could make it better by:
            # - taking overlapping batches
            # - randomly sampling batches
        minibatch = []
        Ls = []
        for s in open(path,'r'):
            s = u"%s" % s.decode('utf-8')[:-1]
            s = indexize(s, self.w2i,self.sep)
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
                self.i=self.i+1
                M = max(Ls)
                padded = np.array([np.pad(m, (0,M-l), 'constant') for (m,l) in zip(minibatch, Ls)], dtype='int32')                
                minibatch = []
                Ls = []
                print padded
                yield padded
    
if __name__ == '__main__':
    if len(sys.argv) < 1:
        print("usage : train_fastsent.py [sep] [remote]")
        sys.exit(1)
    np.random.seed(1234)
    
    if len(sys.argv)==1:
        sep=""
        remote=True
    elif len(sys.argv)==2:
        remote=True
        sep=" " if (sys.argv[1].lower()=='token') else ""
    elif len(sys.argv)==3:
        remote=(sys.argv[2].lower()[0]=='r')
        sep=" " if (sys.argv[1].lower()=='token') else ""
        

    if sep=="":
        path = '/media/data/datasets/wikipedia/entities/bigpage_zh.txt_line_processed_extract' if remote else "data/dataset.txt"
    else:
        path = '/media/data/datasets/wikipedia/entities/bigpage_zh_tokenized.txt_line_processed_extract' if remote else "data/dataset_tokenized.txt"
    vocab = Counter()
    print "build vocab"
    Ls = []

    sentences = SentenceIt(path)#.split(sep)
    for s in sentences:
        if s[0]=="#":
            Ls.append(1)
            vocab[s]+=1
        else:
            if sep==" ":
                s=s.split(sep)
            Ls.append(len(s))
            for w in s:
                vocab[w] += 1
    #sns.distplot(Ls)
    n_data = sentences.n_data
    print "data size: " + str(n_data)
    w2i = {}
    i2cf = []
    i2w = []
    f = open('vocab','w')
    mc =  vocab.most_common()[:150000]
    cumFreq=0
    

    for k,(w,c) in enumerate([('<pad>', 0)] + mc):
        cumFreq+=pow(int(c),3./4)
        w2i[w] = k
        i2cf.append(cumFreq)
        i2w.append(w)

        f.write("%s %f\n" % (w.encode('utf-8'),int(c))) 
    i2f=[cf/float(cumFreq)for cf in i2cf]

    f.close()

    words=w2i.keys()
    print "vocab size: " + str(len(words))
    print words[:10]

    print i2f#[:10]
    dim=200
    batch_size = 200 if remote else 5
    vocab_size = len(i2w)
    n_epochs = 50000 if remote else 2
    dim=200 if remote else 6
    saving_path = "/media/data/datasets/models/word2vec_model/chinese/first" if remote else "chineseModel"
    save_every = 100 if remote else 4

    batches = MinibatchSentenceIt(path, batch_size, w2i,sep)
    print "begin"


    i2e={}
    index_fixe=[0]
    pretrainedFile="/media/data/datasets/models/word2vec_model/model_bridge/model_zh_ws5_pt_ne5_sa0.0001_mc40.vec" if remote else "data/pretrained.txt"

    if remote:
        sys.path.insert(0, '/home/arame/hakken-api/models/')
        import model
        import utils
        pretrained=model.model(pretrainedFile,max_voc=20000)
        wordsModel=pretrained.words
        floatsModel=pretrained.floats
    else:
        import utils
        wordsModel,floatsModel=utils.loadModel(pretrainedFile)
    
    for word,oldi in wordsModel.items(): 
        if word in words:
            i=w2i[word]
            index_fixe.append(i)
            i2e[i]=floatsModel[oldi]
    
    print index_fixe[:10]
    print words[0:10]
    
    print "create model"
    lr=0.0025 if remote else 0.01
    neg_len=100 if remote else 10
    model = FastSentNeg.createNeg(vocab_size, dim,w2i=w2i,i2f=i2f,index_fixe=index_fixe,i2e=i2e,neg_len=neg_len)
    #model = FastSent.create(vocab_size, dim)
    model.train(batches, 
                lr=lr, 
                min_lr=lr/float(100), 
                n_epochs=n_epochs, 
                saving_path=saving_path, 
                save_every=save_every, 
                verbose=True)
