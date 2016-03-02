# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:28:10 2016

@author: hedi & arame
"""

import theano
import numpy as np
from time import time
import theano.tensor as T
import cPickle
from theano.tensor.nnet import softmax
#dtype = theano.config.dtype
from random import random
from bisect import bisect


def weighted_choice(i2f):
    x = random()
    i = bisect(i2f, x)
    return i
dtype='float32'


class Model(object):
    @classmethod
    def load(cls,path):
        params = cPickle.load(open(path,'r'))
        W, V, autoencode= params
        return cls(W, V, autoencode)
    
    @classmethod
    def create(cls, vocab_size, dim,w2i, autoencode=True):
        W = 0.001*np.random.randn(vocab_size, dim)
        V = 0.001*np.random.randn(vocab_size, dim)
        return cls(W,V,autoencode)
    
    @classmethod
    def createNeg(cls, vocab_size, dim,w2i, autoencode=True,i2f=None,index_fixe=[0],i2e=[],neg_len=10):
        W = np.vstack((np.zeros(dim),0.001*np.random.randn(vocab_size-1, dim))).astype(dtype)
        V = np.vstack((np.zeros(dim),0.001*np.random.randn(vocab_size-1, dim))).astype(dtype)
        for i,e in i2e.items():
            W[i]=e
            V[i]=e
        return cls(W,V,autoencode,w2i,i2f,index_fixe,neg_len=neg_len)    

    def save(self, saving_path):
        to_save = [self.w2i,self.W, self.V, self.autoencode]
        cPickle.dump(to_save, open(saving_path,'w'))
        return None


class FastSentNeg(Model):
    def __init__(self, W,V,w2i,autoencode,i2f,index_fixe,neg_len):
        self.W = theano.shared(W, name='W')
        self.V = theano.shared(V, name='V')
        self.w2i=w2i
        self.i2f=i2f
        self.autoencode = autoencode
        self.neg_len=neg_len

        self.params_name=["W_index","V_index"]
        
        indexes = T.imatrix('X')
        neg=T.ivector('neg')
        batch_mask = 1-T.eq(indexes, T.zeros_like(indexes))
        mask = batch_mask[1:-1]
        maskp = batch_mask[:-2]
        maskn = batch_mask[2:]
        X_indexes = indexes[1:-1]
        Yp_indexes = indexes[:-2]
        Yn_indexes = indexes[2:]

        batch_len,sent_len=X_indexes.shape

        indexX=X_indexes.flatten()
        W_index=self.W[indexX]
        X=W_index.reshape((batch_len,sent_len,-1))
        Xs = T.sum(X, axis=1)


        I = [Yp_indexes, Yn_indexes]
        M = [maskp, maskn]
        if self.autoencode:
            I.append(X_indexes)
            M.append(mask)
        I = T.concatenate(I, axis=1)
        M = T.concatenate(M, axis=1)
            
        max_len=I.shape[1]
        pos=I.flatten()
        index=T.concatenate([pos,neg])

        V_index=self.V[index]

        params = [W_index,V_index]

        activations = T.dot(Xs, V_index.T)

        chelou_line=T.arange(I.shape[0]).reshape((-1,1))

        pos_columns=T.arange(batch_len*max_len).reshape((batch_len,-1))
        neg_columns=(T.arange(batch_len*max_len,(batch_len*max_len+self.neg_len)))
        shaped_neg_columns=(neg_columns.reshape((1,-1)))
        rep_neg_columns=T.repeat(shaped_neg_columns,batch_len,axis=0)
        chelou_column=T.concatenate((pos_columns,rep_neg_columns),axis=1)

        
        mask_neg_zero=T.zeros_like(rep_neg_columns)
        mask_neg=(1-mask_neg_zero)
        mask_full = T.concatenate((M,mask_neg),axis=1)
        mask_zero=T.concatenate((M,mask_neg_zero),axis=1)

        acti_mask=activations[chelou_line,chelou_column]*mask_full
        
        prediction = softmax(acti_mask)

        cost = T.mean(T.sum(-T.log(prediction)*mask_zero, axis=1))
        
        lr = T.scalar('lr', dtype=dtype)
        
        grads = T.grad(cost, params)
        # Here is sgd, we could make it better
        sub_update={}

        for p,g,n in zip(params, grads,self.params_name):
            sub_update[n]=-lr*g
        
        updateV=T.inc_subtensor(self.V[index],sub_update["V_index"] )
        updateV=T.set_subtensor(updateV[index_fixe],self.V[index_fixe])  
        updateW=T.inc_subtensor(self.W[indexX],sub_update["W_index"])
        updateW=T.set_subtensor(updateW[index_fixe],self.W[index_fixe])  
        
        
        updates = []
        updates.append((self.V,updateV))
        updates.append((self.W,updateW))

        self._train = theano.function(inputs = [indexes, lr,neg],
                                      outputs = [cost], 
                                      updates=updates, 
                                      allow_input_downcast=True,
                                      on_unused_input='warn')
                                      

    def train(self, batch_iterator, lr, min_lr, n_epochs, 
              saving_path, save_every, verbose = True):
        n_iter = 0
        break_all = False
        for epoch in xrange(n_epochs+1):
            cost_vec=[]
            learning_rate = min_lr + (n_epochs-epoch)*(lr-min_lr)/n_epochs
            for batch in batch_iterator:
                b = batch
                tic = time()
                neg=np.array([weighted_choice(self.i2f) for i in range(self.neg_len)] )
                cost = self._train(b, learning_rate,neg)[0]
                toc = time() - tic
                cost_vec.append(cost)
                n_iter += 1
                if not n_iter%save_every:
                    if verbose:
                        print "\tSaving model"
                    self.save(saving_path)
                if verbose:
                    print "Epoch %d Update %d Cost %f Time %fs" % (epoch,
                                                                   n_iter, 
                                                                   cost, 
                                                                   toc)
                if np.isnan(cost):
                    break_all = True
                if break_all:
                    break
            if verbose:
                print "End Epoch %d Mean Cost %f " % (epoch, np.mean(np.array(cost_vec)))
            if break_all:
                break      


class FastSent(Model):
    def __init__(self, W,V,w2i,autoencode):
        self.w2i=w2i
        self.W = theano.shared(W, name='W')
        self.V = theano.shared(V, name='V')
        self.autoencode = autoencode
        self.params = [self.W,self.V]
        
        indexes = T.imatrix('X')
        batch_mask = 1-T.eq(indexes, T.zeros_like(indexes))
        mask = batch_mask[1:-1]
        maskp = batch_mask[:-2]
        maskn = batch_mask[2:]
        X_indexes = indexes[1:-1]
        Yp_indexes = indexes[:-2]
        Yn_indexes = indexes[2:]

        X = self.W[X_indexes]
        Xs = T.sum(X, axis=1)
        activations = T.dot(Xs, self.V.T)
        # Change this thing so we don't compute softmax for <pad> token
        prediction = softmax(activations)
        I = [Yp_indexes, Yn_indexes]
        M = [maskp, maskn]
        if self.autoencode:
            I.append(X_indexes)
            M.append(mask)
        y_true = T.concatenate(I, axis=1)
        mask = T.concatenate(M, axis=1)
        output = prediction[T.arange(y_true.shape[0]).reshape((-1,1)), 
                            y_true]
        
        cost = T.mean(T.sum(-T.log(output)*mask, axis=1))
        lr = T.scalar('lr', dtype=dtype)
        
        self.grads = T.grad(cost, self.params)
        # Here is sgd, we could make it better
        updates = [(p, p-lr*g) for p,g in zip(self.params, self.grads)]
        self._train = theano.function(inputs = [indexes, lr],
                                      outputs = [cost], 
                                      updates=updates, 
                                      allow_input_downcast=True)
                                     
                                      
    def train(self, batch_iterator, lr, min_lr, n_epochs, 
              saving_path, save_every, verbose = True):
        n_iter = 0
        break_all = False
        for epoch in xrange(n_epochs+1):
            learning_rate = min_lr + (n_epochs-epoch)*(lr-min_lr)/n_epochs
            for batch in batch_iterator:
                b = batch
                tic = time()
                cost = self._train(b, learning_rate)[0]
                toc = time() - tic
                n_iter += 1
                if not n_iter%save_every:
                    if verbose:
                        print "\tSaving model"
                    self.save(saving_path)
                if verbose:
                    print "Epoch %d Update %d Cost %f Time %fs" % (epoch,
                                                                   n_iter, 
                                                                   cost, 
                                                                   toc)
                if np.isnan(cost):
                    break_all = True
                if break_all:
                    break
            if break_all:
                break

     
if __name__=='__main__':
    vocab_size = 1000
    dim = 200
    batch_size = 50
    max_len = 15
    n_data = 1000
    
    np.random.seed(1234)
    lengths = np.random.randint(5,max_len+1, size=n_data)
    data = [np.random.randint(0,vocab_size,size=l).astype('float32') for l in lengths]
    
    i2f=[0]+[1./(f+1) for f in range(1,vocab_size)]
    i2cf=[]
    cumf=0
    for f in i2f:
        cumf+=f
        i2cf.append(cumf)
    i2f=[cf/cumf for cf in i2cf]
    print i2f[:20]

    model = FastSentNeg.create(vocab_size,dim,False,i2f=i2f)
    lr = 0.1
    
    n_batches = n_data/batch_size
    n_epochs = 100
    n_updates = 0
    break_all = False
    for epoch in xrange(n_epochs):
        mean_cost=[]
        for batch_id in xrange(n_batches):
            begin = batch_id*batch_size
            end = min((batch_id+1)*batch_size, n_data)
            minibatch = data[begin:end]
            Ls = [len(d) for d in minibatch]
            M = max(Ls)
            padded = np.array([np.pad(m, (0,M-l), 'constant') for (m,l) in zip(minibatch, Ls)]).astype('int32')
            tic = time()
            cost = model._train(padded,lr)[0]
            mean_cost.append(cost)
            toc = time()-tic
            n_updates += 1
            if np.isnan(cost):
                break_all = True
            if break_all:
                break
        print "Epoch %d Update %d Cost %f " % (epoch, n_updates, np.mean(np.array(mean_cost)))
        if break_all:
            break    
