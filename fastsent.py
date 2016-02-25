# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:28:10 2016

@author: hedi
"""

import theano
import numpy as np
from time import time
import theano.tensor as T
import seaborn as sns
import cPickle
from theano.tensor.nnet import softmax
#dtype = theano.config.dtype
dtype='float32'
class FastSent(object):
    @classmethod
    def load(cls,path):
        params = cPickle.load(open(path,'r'))
        W, V, autoencode= params
        return cls(W, V, autoencode)
    
    @classmethod
    def create(cls, vocab_size, dim, autoencode=True):
        W = 0.001*np.random.randn(vocab_size, dim).astype(dtype)
        V = 0.001*np.random.randn(vocab_size, dim).astype(dtype)
        return cls(W,V,autoencode)
        
    def __init__(self, W,V,autoencode):
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
                                      updates=updates, allow_input_downcast=True)
                                     
    def save(self, saving_path):
        to_save = [self.W, self.V, self.autoencode]
        cPickle.dump(to_save, open(saving_path,'w'))
        return None
                                      
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
    vocab_size = 100000
    dim = 300
    batch_size = 64
    max_len = 20
    n_data = 1000
    
    np.random.seed(1234)
    lengths = np.random.randint(10,max_len+1, size=n_data)
    data = [np.random.randint(0,vocab_size,size=l).astype('float32') for l in lengths]
   
    model = FastSent(vocab_size,dim,False)
    lr = 0.001
    
    n_batches = n_data/batch_size
    n_epochs = 500
    n_updates = 0
    break_all = False
    for epoch in xrange(n_epochs):
        for batch_id in xrange(n_batches):
            begin = batch_id*batch_size
            end = min((batch_id+1)*batch_size, n_data)
            minibatch = data[begin:end]
            Ls = [len(d) for d in minibatch]
            M = max(Ls)
            padded = np.array([np.pad(m, (0,M-l), 'constant') for (m,l) in zip(minibatch, Ls)]).astype('int32')
            tic = time()
            cost = model._train(padded,lr)[0]
            toc = time()-tic
            print "Epoch %d Update %d Cost %f Took %fs" % (epoch, n_updates, cost, toc)
            n_updates += 1
            if np.isnan(cost):
                break_all = True
            if break_all:
                break
        if break_all:
            break    